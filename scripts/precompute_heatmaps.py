"""
Precompute per-feature patch-activation heatmaps for all image slots
(top-max / top-mean / 75th-percentile) stored in an explorer_data.pt file.

Run this once per trained model on a machine with a GPU so the Bokeh
explorer app can serve heatmaps without any live inference.

Output is saved as a sidecar next to the input file:
    <data>_heatmaps.pt

Contents of the sidecar:
    top_heatmaps   (d_model, n_top,  patch_grid²)  float16
    mean_heatmaps  (d_model, n_mean, patch_grid²)  float16
    p75_heatmaps   (d_model, n_p75,  patch_grid²)  float16

For CLS-token models patch_grid == 1, so heatmaps are scalars (still stored).

Only unique images are processed through the backbone — each image is run
through inference exactly once regardless of how many features reference it.

Optional: pass --save-patch-acts to also produce a GPU-free patch-explorer
sidecar:
    <data>_patch_acts.pt

Contents:
    img_indices   (n_unique,)                    int64   — global image index per row
    feat_indices  (n_unique, patch_grid², top_k) int16   — active feature indices per patch
    feat_values   (n_unique, patch_grid², top_k) float16 — corresponding activation values

When this file is present the Bokeh app's Patch Explorer can show complete
per-patch feature rankings without needing a GPU for any covered image.

Usage — one invocation per explorer_data file:

    # Final-layer DINOv3 spatial SAE:
    python precompute_heatmaps.py \\
        --data ../explorer_data_d32000_k160_val.pt \\
        --sae-path ../trained_models/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth \\
        --image-dir /scratch.global/lee02328/val \\
        --extra-image-dir /scratch.global/lee02328/coco/val2017

    # Intermediate-layer DINOv3 (e.g. layer 18):
    python precompute_heatmaps.py \\
        --data ../explorer_data_18.pt \\
        --sae-path ../trained_models_layer18/sae_1_SI-SAE_d20000_k80_per_init0.1_state_dict.pth \\
        --image-dir /scratch.global/lee02328/val \\
        --layer 18

    # CLIP spatial SAE:
    python precompute_heatmaps.py \\
        --data ../explorer_data_clip_spatial_d32000_k160_val.pt \\
        --sae-path ../trained_models_clip_spatial/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth \\
        --image-dir /scratch.global/lee02328/val

    # CLIP intermediate layer (e.g. layer 16):
    python precompute_heatmaps.py \\
        --data ../explorer_data_clip_layer16_val.pt \\
        --sae-path ../trained_models_clip_layer16/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth \\
        --image-dir /scratch.global/lee02328/val \\
        --layer 16
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from backbone_runners import load_batched_backbone
from precompute_utils import (
    resolve_path, extract_tokens,
    parse_top_k_from_path, load_sae,
)


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute heatmaps for explorer_data.pt")
    p.add_argument("--data", required=True,
                   help="Path to explorer_data.pt to add heatmaps to")
    p.add_argument("--sae-path", required=True,
                   help="SAE state-dict (.pth) that matches this dataset")
    p.add_argument("--image-dir", required=True,
                   help="Primary image directory (same as used during precompute)")
    p.add_argument("--extra-image-dir", default=None,
                   help="Secondary image directory (same as used during precompute)")
    p.add_argument("--layer", type=int, default=None,
                   help="Intermediate backbone layer index to extract from "
                        "(None = use final last_hidden_state). "
                        "Must match the layer used when training this SAE.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Images per GPU batch (default 32)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes (default 4)")
    p.add_argument("--output", default=None,
                   help="Output path (default: <data stem>_heatmaps.pt)")
    p.add_argument("--save-patch-acts", action="store_true",
                   help="Also save a _patch_acts.pt sidecar with sparse per-image patch "
                        "activations for every unique image processed.  Enables complete "
                        "GPU-free patch exploration in the Bokeh app for those images.  "
                        "Adds ~4 GB per spatial SAE (int16 feat-indices + float16 values, "
                        "shape [n_unique, n_patches, top_k]).")
    p.add_argument("--force-spatial", action="store_true",
                   help="For CLS-token SAEs: run the SAE on all spatial patch tokens instead "
                        "of just the CLS token.  Produces spatial heatmaps even though the "
                        "SAE was trained on CLS features.  The actual patch_grid is stored "
                        "in the output sidecar so the explorer app can render them correctly.")
    return p.parse_args()


# ---------- Helpers ----------

def build_inverted_index(top_idx, mean_idx, p75_idx):
    """
    Map global_image_idx → list of (set_name, feat, slot_j).

    Lets us scatter-write inference results into the output tensors one
    image at a time, without keeping all activations in RAM.
    """
    img_to_entries = defaultdict(list)
    for f in range(top_idx.shape[0]):
        for j in range(top_idx.shape[1]):
            i = int(top_idx[f, j].item())
            if i >= 0:
                img_to_entries[i].append(('top', f, j))
        for j in range(mean_idx.shape[1]):
            i = int(mean_idx[f, j].item())
            if i >= 0:
                img_to_entries[i].append(('mean', f, j))
        for j in range(p75_idx.shape[1]):
            i = int(p75_idx[f, j].item())
            if i >= 0:
                img_to_entries[i].append(('p75', f, j))
    return img_to_entries


# ---------- Main ----------

def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cpu':
        print("  Warning: running on CPU — this will be slow for large datasets.")

    # --- Load explorer data ---
    print(f"Loading {args.data} ...")
    data = torch.load(args.data, map_location='cpu', weights_only=False)

    d_model       = data['d_model']
    image_paths   = data['image_paths']
    patch_grid    = data['patch_grid']
    image_size    = data['image_size']
    token_type    = data.get('token_type', 'spatial')
    backbone_name = data.get('backbone', 'dinov3')

    # --force-spatial: run a CLS-trained SAE on all spatial patches instead.
    # Recompute patch_grid from backbone parameters so heatmaps are truly 2-D.
    if args.force_spatial and token_type == 'cls':
        token_type = 'spatial'
        if backbone_name == 'clip' or backbone_name.startswith('dinov2'):
            patch_grid = 16          # CLIP ViT-L/14 and DINOv2 ViT-B/14: 224 // 14 = 16
        else:
            patch_grid = image_size // 16   # DINOv3 ViT-L/16
        print(f"  --force-spatial: overriding token_type to 'spatial', patch_grid={patch_grid}")

    top_idx  = data['top_img_idx']
    mean_idx = data.get('mean_img_idx', data['top_img_idx'])
    p75_idx  = data['p75_img_idx']

    n_top  = top_idx.shape[1]
    n_mean = mean_idx.shape[1]
    n_p75  = p75_idx.shape[1]
    n_sq   = patch_grid * patch_grid  # 256 for spatial, 1 for CLS

    print(f"  d_model={d_model}, patch_grid={patch_grid} (n_sq={n_sq}), "
          f"token_type={token_type}, backbone={backbone_name}")
    print(f"  Image slots per feature: top={n_top}, mean={n_mean}, p75={n_p75}")

    # --- Build inverted index ---
    print("Building inverted index...")
    img_to_entries = build_inverted_index(top_idx, mean_idx, p75_idx)
    unique_images  = sorted(img_to_entries.keys())
    n_unique       = len(unique_images)
    # row in patch_acts tensors for each global image index
    img_to_row     = {img_i: row for row, img_i in enumerate(unique_images)}
    print(f"  {n_unique} unique images needed out of {len(image_paths)} in dataset")

    # --- Pre-allocate output tensors ---
    top_heatmaps  = torch.zeros(d_model, n_top,  n_sq, dtype=torch.float16)
    mean_heatmaps = torch.zeros(d_model, n_mean, n_sq, dtype=torch.float16)
    p75_heatmaps  = torch.zeros(d_model, n_p75,  n_sq, dtype=torch.float16)
    est_mb = (top_heatmaps.numel() + mean_heatmaps.numel() + p75_heatmaps.numel()) * 2 / 1e6
    print(f"  Output tensor size estimate: {est_mb:.0f} MB")

    # --- Load backbone ---
    forward_fn, d_hidden, n_reg, transform_fn = load_batched_backbone(
        backbone_name, args.layer, device
    )

    # --- Load SAE ---
    top_k_sae = parse_top_k_from_path(args.sae_path)
    print(f"Loading SAE {os.path.basename(args.sae_path)} (d={d_model}, k={top_k_sae})...")
    sae = load_sae(args.sae_path, d_hidden, d_model, top_k_sae, device)

    # --- Pre-allocate patch_acts tensors (optional) ---
    # feat_indices: int16 is safe for d_model <= 32767 (all current SAEs use d_model <= 32000)
    pa_feat_indices = None
    pa_feat_values  = None
    pa_img_indices  = None
    if args.save_patch_acts:
        est_mb = n_unique * n_sq * top_k_sae * 4 / 1e6  # int16 + float16 = 4 bytes each entry
        print(f"Allocating patch_acts tensors: ({n_unique}, {n_sq}, {top_k_sae})  "
              f"~{est_mb:.0f} MB")
        pa_feat_indices = torch.zeros(n_unique, n_sq, top_k_sae, dtype=torch.int16)
        pa_feat_values  = torch.zeros(n_unique, n_sq, top_k_sae, dtype=torch.float16)
        pa_img_indices  = torch.tensor(unique_images, dtype=torch.long)

    # --- Resolve base directories ---
    base_dirs = [args.image_dir, args.extra_image_dir]

    # --- Batched inference with scatter-write ---
    batch_size = args.batch_size
    n_done     = 0
    n_missing  = 0

    print(f"Running inference on {n_unique} unique images (batch_size={batch_size})...")

    with torch.inference_mode():
        for batch_start in range(0, n_unique, batch_size):
            batch_global_indices = unique_images[batch_start : batch_start + batch_size]

            batch_tensors  = []
            valid_in_batch = []   # (local_position, global_img_idx)

            for local_pos, global_img_i in enumerate(batch_global_indices):
                stored = image_paths[global_img_i]
                path   = resolve_path(stored, base_dirs)
                if path is None:
                    n_missing += 1
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    t   = transform_fn(img)
                    batch_tensors.append(t)
                    valid_in_batch.append((len(batch_tensors) - 1, global_img_i))
                except Exception as e:
                    print(f"  Warning: could not load {path}: {e}")
                    n_missing += 1

            if not batch_tensors:
                n_done += len(batch_global_indices)
                continue

            imgs    = torch.stack(batch_tensors).to(device)   # (bs, C, H, W)
            hidden  = forward_fn(imgs)                         # (bs, n_tokens, d_hidden)
            tokens  = extract_tokens(hidden, backbone_name, token_type, n_reg)
            bs_val  = tokens.shape[0]
            n_tok   = tokens.shape[1]

            _, z, _ = sae(tokens.reshape(-1, d_hidden))
            # z: (bs * n_tok, d_model) — reshape to (bs, n_tok, d_model)
            z_batch = z.cpu().float().numpy().reshape(bs_val, n_tok, d_model)

            # Scatter-write: for each image in this batch, fill in all the
            # heatmap slots that reference it.
            for local_pos, global_img_i in valid_in_batch:
                z_img = z_batch[local_pos]   # (n_patches, d_model)

                for (set_name, feat, slot_j) in img_to_entries[global_img_i]:
                    # Extract only this feature's activations across patches
                    hmap_vec = z_img[:, feat].astype(np.float16)  # (n_sq,) or (1,)
                    if set_name == 'top':
                        top_heatmaps[feat, slot_j] = torch.from_numpy(hmap_vec)
                    elif set_name == 'mean':
                        mean_heatmaps[feat, slot_j] = torch.from_numpy(hmap_vec)
                    else:  # 'p75'
                        p75_heatmaps[feat, slot_j] = torch.from_numpy(hmap_vec)

                # Collect full sparse patch activations for this image.
                # z_img has exactly top_k_sae non-zeros per patch (TopKSAE).
                if pa_feat_indices is not None:
                    row = img_to_row[global_img_i]
                    # For each patch, find the top-k feature indices and values.
                    # argpartition is O(n_sq * d_model) but d_model is large; use the
                    # non-zero mask trick since TopKSAE guarantees exactly top_k per patch.
                    top_col = np.argpartition(-z_img, top_k_sae, axis=1)[:, :top_k_sae]
                    top_val = z_img[np.arange(n_sq)[:, None], top_col]
                    pa_feat_indices[row] = torch.from_numpy(top_col.astype(np.int16))
                    pa_feat_values[row]  = torch.from_numpy(top_val.astype(np.float16))

            n_done += len(batch_global_indices)
            if (batch_start // batch_size + 1) % 20 == 0 or n_done >= n_unique:
                pct = 100 * n_done / n_unique
                print(f"  [{pct:5.1f}%] {n_done}/{n_unique} images  "
                      f"({n_missing} missing so far)", flush=True)

    if n_missing:
        print(f"Warning: {n_missing}/{n_unique} images could not be found — "
              f"those heatmap slots remain zero.")

    # --- Save sidecar ---
    output_path = args.output or (os.path.splitext(args.data)[0] + '_heatmaps.pt')
    print(f"Saving heatmaps to {output_path} ...")
    torch.save({
        'top_heatmaps':  top_heatmaps,
        'mean_heatmaps': mean_heatmaps,
        'p75_heatmaps':  p75_heatmaps,
        'patch_grid':    patch_grid,   # may differ from data['patch_grid'] if --force-spatial
    }, output_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done.  File: {output_path}  ({size_mb:.1f} MB)")
    print(f"  top_heatmaps  : {tuple(top_heatmaps.shape)}")
    print(f"  mean_heatmaps : {tuple(mean_heatmaps.shape)}")
    print(f"  p75_heatmaps  : {tuple(p75_heatmaps.shape)}")

    # --- Save patch_acts sidecar (optional) ---
    if pa_feat_indices is not None:
        pa_path = os.path.splitext(args.data)[0] + '_patch_acts.pt'
        print(f"Saving patch_acts to {pa_path} ...")
        torch.save({
            'img_indices':  pa_img_indices,   # (n_unique,) int64 global image indices
            'feat_indices': pa_feat_indices,  # (n_unique, n_patches, top_k) int16
            'feat_values':  pa_feat_values,   # (n_unique, n_patches, top_k) float16
            'd_model':  d_model,
            'top_k':    top_k_sae,
            'n_patches': n_sq,
        }, pa_path)
        pa_mb = os.path.getsize(pa_path) / 1e6
        print(f"  File: {pa_path}  ({pa_mb:.1f} MB)")
        print(f"  feat_indices: {tuple(pa_feat_indices.shape)}  (int16)")
        print(f"  feat_values : {tuple(pa_feat_values.shape)}  (float16)")
        print(f"  Covers {n_unique} images — patch explorer will work GPU-free for all of them.")


if __name__ == "__main__":
    main()
