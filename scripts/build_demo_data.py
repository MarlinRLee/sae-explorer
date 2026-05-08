"""Synthesize a minimal-but-valid demo dataset for the SAE Feature Explorer.

Generates two tiny ``.pt`` files (one without CLIP, one with), a matching
``_heatmaps.pt`` sidecar each, a directory of random RGB JPEGs, and a
``models.yaml`` registry pointing at them. Lets you run the explorer
end-to-end without any precompute pipeline:

    python scripts/build_demo_data.py [output_dir]
    bash scripts/run_explorer_local.sh --synthetic    # uses the output

The generated tensors are deterministic (fixed seeds) so the synthetic
data is also useful as a smoke-test fixture.

Layout produced under ``output_dir`` (default ``./demo_data``):
    explorer_data_demo.pt              — d_model=64, n_images=5 (primary)
    explorer_data_demo_heatmaps.pt     — sidecar with top/mean/p75 heatmaps
    explorer_data_demo2.pt             — d_model=32, n_images=5, with CLIP
    explorer_data_demo2_heatmaps.pt    — sidecar
    images/img_{0..4}.jpg              — five 224x224 RGB images
    models.yaml                        — registry; works with --registry
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image


# Default sizes — small enough that the explorer feels snappy, large
# enough that every code path (live mask, dead features, CLS + spatial,
# CLIP availability transition) is exercised.
DEFAULT_D_MODEL_A   = 64
DEFAULT_D_MODEL_B   = 32
DEFAULT_N_IMAGES    = 5
DEFAULT_PATCH_GRID  = 16
DEFAULT_IMAGE_SIZE  = 224
DEFAULT_N_TOP_SLOTS = 3
DEFAULT_CLIP_DIM    = 768
DEFAULT_CLIP_VOCAB  = ["dog", "cat", "tree", "sky", "water", "person",
                      "building", "road"]


def _make_random_jpegs(out_dir: str, n: int, size: int, seed: int = 0):
    """Write ``n`` deterministic random RGB 224x224 JPEGs into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        path = os.path.join(out_dir, f"img_{i}.jpg")
        Image.fromarray(arr).save(path, format="JPEG", quality=85)
        paths.append(path)
    return paths


def _make_explorer_pt(image_paths,
                      d_model: int,
                      n_top_slots: int = DEFAULT_N_TOP_SLOTS,
                      patch_grid: int = DEFAULT_PATCH_GRID,
                      image_size: int = DEFAULT_IMAGE_SIZE,
                      include_clip: bool = False,
                      seed: int = 0,
                      label: str = "demo"):
    """Build the contents of an ``explorer_data*.pt`` file."""
    rng = np.random.default_rng(seed)
    n_images = len(image_paths)

    # ~50% live (have UMAP coords) so the dead-feature path is exercised.
    live = rng.random(d_model) > 0.5

    # ~30% of features actually fire (frequency > 0); the rest are dead.
    fires = (rng.random(d_model) > 0.7) & live
    feature_frequency = np.where(fires, rng.integers(1, 200, d_model), 0).astype(np.int64)
    feature_mean_act  = np.where(fires, rng.random(d_model) * 0.5, 0.0).astype(np.float32)
    feature_p75_val   = np.where(fires, rng.random(d_model) * 0.3, 0.0).astype(np.float32)

    # UMAP coords: NaN for dead/non-live features (drives `live_mask`).
    umap_coords      = rng.random((d_model, 2)).astype(np.float32) * 10
    dict_umap_coords = rng.random((d_model, 2)).astype(np.float32) * 10
    umap_coords[~live]      = np.nan
    dict_umap_coords[~live] = np.nan

    # Top/mean/p75 image indices: random in [-1, n_images). -1 marks empty
    # slots so the renderer's no-image branch gets exercised.
    def _rand_slots():
        idx = rng.integers(-1, n_images, size=(d_model, n_top_slots), dtype=np.int64)
        # Dead features have no top images.
        idx[~fires] = -1
        act = rng.random((d_model, n_top_slots)).astype(np.float32) * 0.5
        act[idx < 0] = 0.0
        return idx, act

    top_idx,  top_act  = _rand_slots()
    mean_idx, mean_act_t = _rand_slots()
    p75_idx,  p75_act  = _rand_slots()

    entry = {
        'image_paths':       list(image_paths),
        'd_model':           int(d_model),
        'n_images':          int(n_images),
        'patch_grid':        int(patch_grid),
        'image_size':        int(image_size),
        'token_type':        'spatial',
        'backbone':          'demo',
        'top_img_idx':       torch.from_numpy(top_idx),
        'top_img_act':       torch.from_numpy(top_act),
        'mean_img_idx':      torch.from_numpy(mean_idx),
        'mean_img_act':      torch.from_numpy(mean_act_t),
        'p75_img_idx':       torch.from_numpy(p75_idx),
        'p75_img_act':       torch.from_numpy(p75_act),
        'feature_frequency': torch.from_numpy(feature_frequency),
        'feature_mean_act':  torch.from_numpy(feature_mean_act),
        'feature_p75_val':   torch.from_numpy(feature_p75_val),
        'umap_coords':       torch.from_numpy(umap_coords),
        'dict_umap_coords':  torch.from_numpy(dict_umap_coords),
    }

    if include_clip:
        n_vocab = len(DEFAULT_CLIP_VOCAB)
        clip_dim = DEFAULT_CLIP_DIM
        scores = rng.random((d_model, n_vocab)).astype(np.float32) - 0.5
        embeds = rng.normal(size=(d_model, clip_dim)).astype(np.float32)
        embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
        entry['clip_text_scores']    = torch.from_numpy(scores)
        entry['clip_text_vocab']     = list(DEFAULT_CLIP_VOCAB)
        entry['clip_feature_embeds'] = torch.from_numpy(embeds)

    return entry


def _make_heatmap_sidecar(d_model: int, n_top_slots: int, patch_grid: int, seed: int = 0):
    """Build the ``_heatmaps.pt`` sidecar matching an explorer .pt."""
    rng = np.random.default_rng(seed + 1)
    n_patches_sq = patch_grid * patch_grid
    shape = (d_model, n_top_slots, n_patches_sq)
    return {
        'top_heatmaps':  torch.from_numpy(rng.random(shape).astype(np.float16)),
        'mean_heatmaps': torch.from_numpy(rng.random(shape).astype(np.float16)),
        'p75_heatmaps':  torch.from_numpy(rng.random(shape).astype(np.float16)),
        'patch_grid':    int(patch_grid),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_dir", nargs="?", default="./demo_data",
                        help="Directory to write the synthetic dataset into")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = os.path.abspath(args.output_dir)
    img_dir = os.path.join(out, "images")
    os.makedirs(out, exist_ok=True)

    print(f"[build_demo_data] writing into {out}")

    image_paths = _make_random_jpegs(img_dir, DEFAULT_N_IMAGES, DEFAULT_IMAGE_SIZE,
                                     seed=args.seed)
    print(f"  {len(image_paths)} random JPEGs -> {img_dir}/")

    # Dataset 1: no CLIP. This is the eagerly-loaded primary.
    pt_a = _make_explorer_pt(image_paths,
                             d_model=DEFAULT_D_MODEL_A,
                             include_clip=False,
                             seed=args.seed,
                             label="demo")
    hm_a = _make_heatmap_sidecar(DEFAULT_D_MODEL_A, DEFAULT_N_TOP_SLOTS,
                                 DEFAULT_PATCH_GRID, seed=args.seed)
    p_a    = os.path.join(out, "explorer_data_demo.pt")
    p_a_hm = os.path.join(out, "explorer_data_demo_heatmaps.pt")
    torch.save(pt_a, p_a)
    torch.save(hm_a, p_a_hm)
    print(f"  -> {p_a}  (d_model={DEFAULT_D_MODEL_A}, no CLIP)")
    print(f"  -> {p_a_hm}")

    # Dataset 2: with CLIP. Lazy compare; opens the search panel when selected.
    pt_b = _make_explorer_pt(image_paths,
                             d_model=DEFAULT_D_MODEL_B,
                             include_clip=True,
                             seed=args.seed + 1,
                             label="demo CLIP")
    hm_b = _make_heatmap_sidecar(DEFAULT_D_MODEL_B, DEFAULT_N_TOP_SLOTS,
                                 DEFAULT_PATCH_GRID, seed=args.seed + 1)
    p_b    = os.path.join(out, "explorer_data_demo2.pt")
    p_b_hm = os.path.join(out, "explorer_data_demo2_heatmaps.pt")
    torch.save(pt_b, p_b)
    torch.save(hm_b, p_b_hm)
    print(f"  -> {p_b}  (d_model={DEFAULT_D_MODEL_B}, with CLIP)")
    print(f"  -> {p_b_hm}")

    # Synthetic registry — same shape as configs/models.yaml so the
    # explorer's --registry flow handles it identically to production.
    registry_path = os.path.join(out, "models.yaml")
    with open(registry_path, "w") as f:
        f.write(
            "defaults:\n"
            "  hf_data_repo:   ''\n"
            "  hf_images_repo: ''\n"
            "  images_tarball: ''\n"
            "models:\n"
            "  - id:         demo_no_clip\n"
            "    label:      Demo (no CLIP)\n"
            "    primary:    true\n"
            "    data_file:  explorer_data_demo.pt\n"
            "    sae_file:   ''\n"
            "    backbone:   demo\n"
            "    layer:      0\n"
            "    token_type: spatial\n"
            "  - id:         demo_clip\n"
            "    label:      Demo (CLIP)\n"
            "    data_file:  explorer_data_demo2.pt\n"
            "    sae_file:   ''\n"
            "    backbone:   demo\n"
            "    layer:      0\n"
            "    token_type: spatial\n"
        )
    print(f"  -> {registry_path}")

    print()
    print("Launch with:")
    print(f"  bokeh serve scripts/explorer_app.py --port 5006 --args \\")
    print(f"      --registry  {registry_path} \\")
    print(f"      --data-dir  {out} \\")
    print(f"      --image-dir {img_dir}")


if __name__ == "__main__":
    main()
