"""
Post-hoc CLIP text-alignment enrichment for explorer_data.pt files.

Loads an existing explorer_data.pt, computes per-feature CLIP text alignment
scores (via MEI images), and saves them back into the same file under:
    'clip_text_scores'      : Tensor (n_features, n_vocab) float16
    'clip_text_vocab'       : list[str]
    'clip_feature_embeds'   : Tensor (n_features, clip_proj_dim) float16
                              mean CLIP image embedding of each feature's top MEIs

This script does NOT need to re-run DINOv3 or the SAE — it only needs the
existing explorer_data.pt (for image paths and top-MEI indices) and CLIP.

Usage
-----
    python add_clip_embeddings.py \
        --data ../explorer_data_d32000_k160.pt \
        --vocab-file ../vocab/imagenet_labels.txt \
        --n-top-images 4 \
        --batch-size 32

    # Or use the built-in default vocabulary (ImageNet-1K labels + COCO categories):
    python add_clip_embeddings.py \
        --data ../explorer_data_d32000_k160.pt

The enriched file is saved to --output-path (defaults to overwriting --data
with a backup copy at <data>.bak).
"""

import argparse
import os
import shutil

import torch
import torch.nn.functional as F
from PIL import Image

# Allow running from scripts/ directory or project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from clip_utils import load_clip, compute_text_embeddings, compute_mei_text_alignment


# ---------------------------------------------------------------------------
# Default vocabulary
# ---------------------------------------------------------------------------

DEFAULT_VOCAB = [
    # COCO categories
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    # Texture / scene descriptors
    "grass", "sky", "water", "sand", "snow", "wood", "stone", "metal",
    "fabric", "fur", "feathers", "leaves", "clouds", "fire", "shadow",
    "stripes", "spots", "checkerboard pattern", "geometric pattern",
    # Orientation / structure cues (for patch features)
    "horizontal lines", "vertical lines", "diagonal lines", "curved lines",
    "edges", "corners", "grid", "dots", "concentric circles",
    # Color / illumination
    "red object", "blue object", "green object", "yellow object",
    "black and white", "bright highlight", "dark shadow", "gradient",
    # Scene types
    "indoor scene", "outdoor scene", "urban street", "nature landscape",
    "ocean", "mountain", "forest", "desert", "city buildings", "crowd",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add CLIP text alignment to explorer_data.pt")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to explorer_data.pt")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Output path (default: overwrite --data, keeping .bak)")
    parser.add_argument("--vocab-file", type=str, default=None,
                        help="Plain-text file with one concept per line.  "
                             "Default: built-in COCO+texture vocabulary.")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14",
                        help="HuggingFace CLIP model ID")
    parser.add_argument("--n-top-images", type=int, default=4,
                        help="Number of MEIs to average per feature for CLIP alignment")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for CLIP image encoding")
    parser.add_argument("--no-backup", action="store_true",
                        help="Skip creating a .bak copy before overwriting")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Primary image directory for resolving bare filenames")
    parser.add_argument("--extra-image-dir", type=str, action="append", default=[],
                        help="Additional image directory (repeatable)")
    args = parser.parse_args()

    image_bases = [b for b in ([args.image_dir] + args.extra_image_dir) if b]

    def resolve_path(p):
        if os.path.isabs(p) or not image_bases:
            return p
        for base in image_bases:
            full = os.path.join(base, p)
            if os.path.exists(full):
                return full
        return os.path.join(image_bases[0], p)  # fallback

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load explorer data ---
    print(f"Loading explorer data from {args.data}...")
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    image_paths = [resolve_path(p) for p in data['image_paths']]
    d_model = data['d_model']
    top_img_idx = data['top_img_idx']       # (n_features, n_top)
    n_top_stored = top_img_idx.shape[1]
    print(f"  d_model={d_model}, n_images={data['n_images']}, "
          f"top-{n_top_stored} images stored")

    # --- Load vocabulary ---
    if args.vocab_file:
        with open(args.vocab_file) as f:
            vocab = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(vocab)} concepts from {args.vocab_file}")
    else:
        vocab = DEFAULT_VOCAB
        print(f"Using default vocabulary ({len(vocab)} concepts)")

    # --- Load CLIP ---
    clip_model, clip_processor = load_clip(device, model_name=args.clip_model)

    # --- Precompute text embeddings ---
    print("Encoding text vocabulary with CLIP...")
    text_embeds = compute_text_embeddings(vocab, clip_model, clip_processor, device)
    print(f"  text_embeds: {text_embeds.shape}")

    # --- Collect MEI image paths per feature ---
    print("Collecting MEI image paths per feature...")
    n_use = min(args.n_top_images, n_top_stored)
    feature_mei_paths = []
    for feat in range(d_model):
        paths = []
        for j in range(n_use):
            idx = top_img_idx[feat, j].item()
            if idx >= 0:
                paths.append(image_paths[idx])
        feature_mei_paths.append(paths)

    # --- Compute per-feature CLIP image embeddings (mean of MEIs) ---
    print(f"Computing CLIP image embeddings for {d_model} features "
          f"(averaging {n_use} MEIs each)...")

    clip_proj_dim = clip_model.config.projection_dim
    feature_img_embeds = torch.zeros(d_model, clip_proj_dim, dtype=torch.float32)
    dead_count = 0

    for feat_start in range(0, d_model, args.batch_size):
        feat_end = min(feat_start + args.batch_size, d_model)
        for feat in range(feat_start, feat_end):
            paths = feature_mei_paths[feat]
            if not paths:
                dead_count += 1
                continue
            imgs = []
            for p in paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue
            if not imgs:
                dead_count += 1
                continue
            inputs = clip_processor(images=imgs, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            with torch.inference_mode():
                # Use vision_model + visual_projection directly to avoid
                # version differences in get_image_features() return type.
                vision_out = clip_model.vision_model(pixel_values=pixel_values)
                embeds = clip_model.visual_projection(vision_out.pooler_output)
                embeds = F.normalize(embeds, dim=-1)
                mean_embed = embeds.mean(dim=0)
                mean_embed = F.normalize(mean_embed, dim=-1)
            feature_img_embeds[feat] = mean_embed.cpu().float()

        if (feat_start // args.batch_size + 1) % 100 == 0:
            print(f"  [{feat_end}/{d_model}] features encoded", flush=True)

    print(f"  Done. Dead/missing features skipped: {dead_count}")

    # --- Compute alignment matrix ---
    print("Computing text alignment matrix...")
    # (n_features, clip_proj_dim) @ (clip_proj_dim, n_vocab) = (n_features, n_vocab)
    clip_text_scores = feature_img_embeds @ text_embeds.T   # float32
    print(f"  clip_text_scores: {clip_text_scores.shape}")

    # --- Save into explorer_data.pt ---
    output_path = args.output_path or args.data
    if output_path == args.data and not args.no_backup:
        bak_path = args.data + ".bak"
        print(f"Creating backup at {bak_path}...")
        shutil.copy2(args.data, bak_path)

    data['clip_text_scores']    = clip_text_scores.half()              # float16 to save space
    data['clip_feature_embeds'] = feature_img_embeds.half()            # float16
    data['clip_text_vocab']     = vocab

    print(f"Saving enriched explorer data to {output_path}...")
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
