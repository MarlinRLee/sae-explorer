"""
Extract vision backbone hidden states to float16 shards for SAE training.

Supports DINOv3 ViT-L/16 (default) and CLIP ViT-L/14 (--backbone clip).
Previously separate extract_activations.py (DINOv3) and
extract_clip_activations.py (CLIP) are now unified here.

Output shards are float16 tensors of shape (n_tokens, d_hidden) where
n_tokens = images_per_shard * tokens_per_image.

Usage
-----
    # DINOv3 spatial (final layer):
    python extract_activations.py --backbone dinov3 \\
        --data_dir /scratch.global/lee02328/train \\
        --output_dir /scratch.global/lee02328/data/DINOV3/train_spatial/final

    # DINOv3 intermediate layer 18:
    python extract_activations.py --backbone dinov3 --layer 18 \\
        --data_dir /scratch.global/lee02328/train \\
        --output_dir /scratch.global/lee02328/data/DINOV3/train_layer18/final

    # CLIP spatial:
    python extract_activations.py --backbone clip \\
        --data_dir /scratch.global/lee02328/train \\
        --output_dir /scratch.global/lee02328/data/CLIP/train_spatial/final

    # CLIP both (spatial + CLS to separate subdirs in one pass):
    python extract_activations.py --backbone clip --token-type both \\
        --data_dir /scratch.global/lee02328/train \\
        --output_dir /scratch.global/lee02328/data/CLIP/train \\
        --images_per_shard 2816
"""

import os
import re
import argparse
import glob
import sys

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from backbone_runners import load_batched_backbone


_IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.JPEG']


class ImageFolder(Dataset):
    """Load images from a flat directory (COCO) or class-subdirectory layout (ImageNet)."""
    def __init__(self, root, transform_fn, recursive=False):
        self.paths = sorted(
            p for ext in _IMG_EXTS
            for p in glob.glob(
                os.path.join(root, '**', ext) if recursive else os.path.join(root, ext),
                recursive=recursive,
            )
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")
        self._transform = transform_fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self._transform(img)


def _shard_index(output_dir):
    """Return the next shard index based on existing shards in output_dir."""
    existing = sorted(glob.glob(os.path.join(output_dir, "shard_*.pt")))
    if not existing:
        return 0
    last = os.path.basename(existing[-1])
    return int(re.search(r"shard_(\d+)", last).group(1)) + 1


def _flush_shard(buf, images_per_shard, output_dir, shard_idx):
    """Flush one shard from buf; return (leftover_tensor_or_None, new_shard_idx)."""
    all_tok    = torch.cat(buf, dim=0)          # (N_images, n_tok, d)
    shard_imgs = all_tok[:images_per_shard]
    leftover   = all_tok[images_per_shard:]
    data       = shard_imgs.reshape(-1, shard_imgs.shape[-1]).half()
    path       = os.path.join(output_dir, f"shard_{shard_idx:04d}.pt")
    torch.save(data, path)
    print(f"Saved {path} | shape {data.shape}", flush=True)
    return leftover if leftover.shape[0] > 0 else None, shard_idx + 1


def main():
    parser = argparse.ArgumentParser(description="Extract vision backbone hidden states to shards")
    parser.add_argument("--backbone", type=str, default="dinov3",
                        choices=["dinov3", "clip"],
                        help="Vision backbone (default: dinov3)")
    parser.add_argument("--data_dir",         type=str, required=True)
    parser.add_argument("--output_dir",       type=str, required=True)
    parser.add_argument("--batch_size",       type=int, default=64)
    parser.add_argument("--images_per_shard", type=int, default=2816,
                        help="Images per shard. For CLS use 100000; for spatial use 2816.")
    parser.add_argument("--num_workers",      type=int, default=8)
    parser.add_argument("--flat",             action="store_true",
                        help="Flat directory (COCO) vs class-subfolder (ImageNet)")
    parser.add_argument("--layer",            type=int, default=None,
                        help="Intermediate layer index to extract from (default: final layer)")
    parser.add_argument("--token-type",       type=str, default="spatial",
                        choices=["spatial", "cls", "all", "both"],
                        help="'spatial': patch tokens; 'cls': CLS token; "
                             "'all': CLS+registers+patches (DINOv3); "
                             "'both': save spatial AND CLS to separate subdirs (CLIP only)")
    args = parser.parse_args()

    token_type = args.token_type
    if token_type == "both" and args.backbone != "clip":
        parser.error("--token-type both is only valid for --backbone clip")
    if token_type == "all" and args.backbone == "clip":
        parser.error("--token-type all is not valid for CLIP (use 'both' for dual-output)")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Load backbone ---
    forward_fn, d_hidden, n_reg, transform_fn = load_batched_backbone(
        args.backbone, args.layer, device
    )

    # --- Dataset ---
    dataset = ImageFolder(args.data_dir, transform_fn, recursive=not args.flat)
    print(f"Found {len(dataset)} images | token_type={token_type}")

    # --- Output directory / resume ---
    if token_type == "both":
        spatial_dir = os.path.join(args.output_dir, "spatial")
        cls_dir     = os.path.join(args.output_dir, "cls")
        os.makedirs(spatial_dir, exist_ok=True)
        os.makedirs(cls_dir,     exist_ok=True)
        resume_dir  = spatial_dir
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        resume_dir  = args.output_dir

    start_shard = _shard_index(resume_dir)
    images_done = start_shard * args.images_per_shard
    if images_done > 0:
        print(f"Resuming: skipping {images_done} images (shard_{start_shard-1:04d} already done)")
        if images_done >= len(dataset):
            print("All images already processed!")
            return
        dataset = Subset(dataset, range(images_done, len(dataset)))

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # --- Extract ---
    spatial_buf, spatial_shard = [], start_shard
    cls_buf,     cls_shard     = [], start_shard
    spatial_count = cls_count  = 0

    with torch.inference_mode():
        for batch_i, imgs in enumerate(loader):
            imgs   = imgs.to(device, non_blocking=True)
            hidden = forward_fn(imgs)  # (B, all_tokens, d_hidden)

            # Token slicing (n_reg=0 for CLIP, handles both uniformly)
            if token_type == "spatial" or token_type == "both":
                spatial_tok = hidden[:, 1 + n_reg:, :].cpu()
                spatial_buf.append(spatial_tok)
                spatial_count += imgs.shape[0]

            if token_type == "cls" or token_type == "both":
                cls_tok = hidden[:, 0:1, :].cpu()
                cls_buf.append(cls_tok)
                cls_count += imgs.shape[0]

            if token_type == "all":
                spatial_buf.append(hidden.cpu())
                spatial_count += imgs.shape[0]

            # Flush spatial / all
            while spatial_count >= args.images_per_shard:
                out_dir   = spatial_dir if token_type == "both" else args.output_dir
                leftover, spatial_shard = _flush_shard(
                    spatial_buf, args.images_per_shard, out_dir, spatial_shard
                )
                spatial_count -= args.images_per_shard
                spatial_buf = [leftover] if leftover is not None else []

            # Flush CLS
            while cls_count >= args.images_per_shard:
                out_dir   = cls_dir if token_type == "both" else args.output_dir
                leftover, cls_shard = _flush_shard(
                    cls_buf, args.images_per_shard, out_dir, cls_shard
                )
                cls_count -= args.images_per_shard
                cls_buf = [leftover] if leftover is not None else []

            if (batch_i + 1) % 50 == 0:
                n_done = (batch_i + 1) * args.batch_size
                print(f"  Processed {n_done}/{len(dataset)} images | "
                      f"shards: spatial={spatial_shard} cls={cls_shard}", flush=True)

    # --- Save remaining ---
    BIG = len(dataset) * 10  # large enough to flush remainder in one call

    if spatial_buf:
        out_dir = spatial_dir if token_type == "both" else args.output_dir
        _flush_shard(spatial_buf, BIG, out_dir, spatial_shard)
        spatial_shard += 1

    if cls_buf:
        out_dir = cls_dir if token_type == "both" else args.output_dir
        _flush_shard(cls_buf, BIG, out_dir, cls_shard)
        cls_shard += 1

    print(f"\nDone!")
    if token_type in ("spatial", "all", "both"):
        out_dir = spatial_dir if token_type == "both" else args.output_dir
        print(f"  Spatial/all shards: {spatial_shard} -> {out_dir}")
    if token_type in ("cls", "both"):
        out_dir = cls_dir if token_type == "both" else args.output_dir
        print(f"  CLS shards:         {cls_shard} -> {out_dir}")


if __name__ == "__main__":
    main()
