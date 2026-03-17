"""
Precompute SAE feature activation data for the interactive explorer.

Processes images through DINOv3 → SAE, then computes:
  - Per-feature top-N images (indices + max activation only, no heatmaps)
  - Per-feature 75th-percentile images (indices + acts)
  - UMAP embedding of SAE features (activation-based and dictionary-based)
  - Per-feature summary statistics

Heatmaps are NOT stored here; the explorer app computes them on-the-fly
when you click a feature by re-running DINOv3+SAE for that image.

Use --interleave-classes with a large dataset (e.g. ImageNet train) so that
images are sampled spread across all classes from the start.  Combine with
--coverage-threshold to stop early once enough features have fired at least
once, avoiding a full pass over millions of images.

Usage:
    python precompute_explorer_data.py \
        --image-dir /scratch.global/lee02328/imagenet/train \
        --extra-image-dir /scratch.global/lee02328/coco/train2017 \
        --sae-path ../trained_models/sae_1_SI-SAE_d32000_k160_per_init0.1_state_dict.pth \
        --output-path ../explorer_data.pt \
        --recursive \
        --interleave-classes \
        --coverage-threshold 0.90
"""

import argparse
import heapq
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import umap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from backbone_runners import load_batched_backbone
from precompute_utils import (
    RESOLUTION, INPUT_SIZE,
    ImageFolder, extract_tokens, load_sae,
)


# ---------- Class-interleaved sampling ----------
def interleave_by_class(paths):
    """
    Reorder paths so that images from every class are spread out evenly.

    ImageNet-style layout: each class is a subdirectory.  We group by parent
    directory and round-robin across groups so that after N images we have
    sampled ~N/n_classes images from each class rather than exhausting one
    class before touching the next.  This maximises feature coverage when
    early stopping is used.
    """
    from collections import defaultdict
    class_buckets = defaultdict(list)
    for p in paths:
        class_buckets[os.path.dirname(p)].append(p)

    # Sort within each class for reproducibility
    buckets = [sorted(v) for v in class_buckets.values()]
    # Sort bucket list so output is deterministic
    buckets.sort(key=lambda b: b[0])

    interleaved = []
    max_len = max(len(b) for b in buckets)
    for i in range(max_len):
        for b in buckets:
            if i < len(b):
                interleaved.append(b[i])
    return interleaved


# ---------- Per-Feature Image Tracker ----------
class FeatureImageTracker:
    """
    For each feature, track top-N images (unique) by max-patch activation.
    Does NOT store heatmaps — the explorer computes them on-the-fly.

    p75 statistics are computed at finalize() time from the shared UMAP
    reservoir rather than maintained per-feature during the pass, which
    avoids a ~29K-iteration Python loop per image.
    """

    def __init__(self, n_features, top_n=16):
        self.n_features = n_features
        self.top_n = top_n

        # Per-feature: min-heap of (max_act, img_idx), size ≤ top_n
        self.feature_heaps = [[] for _ in range(n_features)]

        # Numpy arrays for O(1) pre-filtering of heap updates.
        # Once a heap is full, only images that beat the current minimum
        # (heap[0][0]) need to enter the Python loop at all.
        self.heap_min_vals = np.full(n_features, -np.inf, dtype=np.float32)
        self.heap_sizes    = np.zeros(n_features, dtype=np.int32)

        # Parallel heaps ranked by mean patch activation
        self.mean_heaps          = [[] for _ in range(n_features)]
        self.mean_heap_min_vals  = np.full(n_features, -np.inf, dtype=np.float32)
        self.mean_heap_sizes     = np.zeros(n_features, dtype=np.int32)

        # Per-feature stats (populated in bulk from the main loop)
        self.act_counts = np.zeros(n_features, dtype=np.int64)
        self.act_sums   = np.zeros(n_features, dtype=np.float64)
        self.img_counts = np.zeros(n_features, dtype=np.int64)

    def _update_heap_set(self, heaps, min_vals, sizes, image_idx, acts, track_img_counts=False):
        """Push one image into a set of per-feature min-heaps."""
        active_feats = np.nonzero(acts > 0)[0]
        active_vals  = acts[active_feats]
        if track_img_counts:
            self.img_counts[active_feats] += 1

        # Vectorized pre-filter: skip features whose heap is full and the
        # new value can't displace the current minimum.
        needs = (sizes[active_feats] < self.top_n) | (active_vals > min_vals[active_feats])
        update_feats = active_feats[needs]
        update_vals  = active_vals[needs]

        for i in range(len(update_feats)):
            feat = int(update_feats[i])
            val  = float(update_vals[i])
            heap = heaps[feat]
            sz   = sizes[feat]
            if sz < self.top_n:
                heapq.heappush(heap, (val, image_idx))
                sizes[feat] = sz + 1
                if sz + 1 == self.top_n:
                    min_vals[feat] = heap[0][0]
            else:
                heapq.heapreplace(heap, (val, image_idx))
                min_vals[feat] = heap[0][0]

    def update_heaps(self, image_idx, max_acts):
        self._update_heap_set(
            self.feature_heaps, self.heap_min_vals, self.heap_sizes,
            image_idx, max_acts, track_img_counts=True,
        )

    def update_mean_heaps(self, image_idx, mean_acts):
        self._update_heap_set(
            self.mean_heaps, self.mean_heap_min_vals, self.mean_heap_sizes,
            image_idx, mean_acts,
        )

    def _heaps_to_tensors(self, heaps, n_top):
        """Convert a list of per-feature min-heaps to (idx, act) tensors."""
        n       = self.n_features
        idx_t   = torch.full((n, n_top), -1, dtype=torch.long)
        act_t   = torch.zeros(n, n_top)
        for feat in range(n):
            heap = heaps[feat]
            if heap:
                for j, (act, img_i) in enumerate(sorted(heap, reverse=True)[:n_top]):
                    idx_t[feat, j] = img_i
                    act_t[feat, j] = act
        return idx_t, act_t

    def finalize(self, n_top, umap_data, umap_img_idx, n_p75=8):
        """
        Assemble final tensors.

        Parameters
        ----------
        n_top        : int — number of top images to store per feature
        umap_data    : np.ndarray (n_reservoir, n_features) float32
                       max activations for reservoir images (used for p75)
        umap_img_idx : np.ndarray (n_reservoir,) int64
                       global image indices corresponding to umap_data rows
        """
        n     = self.n_features
        n_res = len(umap_img_idx)

        top_img_idx, top_img_act   = self._heaps_to_tensors(self.feature_heaps, n_top)
        self.feature_heaps = None

        mean_img_idx, mean_img_act = self._heaps_to_tensors(self.mean_heaps, n_top)
        self.mean_heaps = None

        # --- p75 statistics from UMAP reservoir (fully vectorized) ---
        p75_img_idx    = torch.full((n, n_p75), -1, dtype=torch.long)
        p75_img_act    = torch.zeros(n, n_p75)
        feature_p75_val = torch.zeros(n)

        if n_res >= 4:
            # p75 per feature over activating (non-zero) reservoir images only.
            p75_vals = np.zeros(n, dtype=np.float32)
            for feat in range(n):
                col = umap_data[:, feat]
                nonzero = col[col > 0]
                if len(nonzero) >= 1:
                    p75_vals[feat] = np.percentile(nonzero, 75)
            feature_p75_val = torch.from_numpy(p75_vals)

            # For each feature find the n_p75 reservoir images closest to p75.
            # argpartition over the reservoir axis is O(n_res * n_features) in C.
            dists   = np.abs(umap_data - p75_vals[np.newaxis, :])  # (n_res, n_features)
            k       = min(n_p75, n_res)
            near_idx = np.argpartition(dists, k, axis=0)[:k]       # (k, n_features)
            feat_idx = np.arange(n)
            for j in range(k):
                rows = near_idx[j]                                    # (n_features,)
                p75_img_idx[:, j] = torch.from_numpy(
                    umap_img_idx[rows].astype(np.int64))
                p75_img_act[:, j] = torch.from_numpy(
                    umap_data[rows, feat_idx].astype(np.float32))

        # --- Summary stats ---
        feature_frequency = torch.from_numpy(self.act_counts.copy()).float()
        feature_mean_act  = torch.zeros(n)
        nonzero = self.act_counts > 0
        feature_mean_act[nonzero] = torch.from_numpy(
            (self.act_sums[nonzero] / self.act_counts[nonzero]).astype(np.float32))

        return {
            'top_img_idx':       top_img_idx,
            'top_img_act':       top_img_act,
            'mean_img_idx':      mean_img_idx,
            'mean_img_act':      mean_img_act,
            'p75_img_idx':       p75_img_idx,
            'p75_img_act':       p75_img_act,
            'feature_frequency': feature_frequency,
            'feature_mean_act':  feature_mean_act,
            'feature_p75_val':   feature_p75_val,
        }


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Precompute SAE explorer data")
    parser.add_argument("--image-dir", type=str, required=True,
                        help="Primary image directory (e.g. ImageNet train)")
    parser.add_argument("--extra-image-dir", type=str, default=None,
                        help="Additional image directory to include (e.g. COCO train)")
    parser.add_argument("--sae-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="../explorer_data.pt")
    parser.add_argument("--thumbnail-dir", type=str, default=None,
                        help="If set, save resized JPEG thumbnails for all images referenced "
                             "in feature display slots and the UMAP reservoir, and store "
                             "basenames (not absolute paths) in image_paths. Required for "
                             "HF-hosted deployment where original image paths are unavailable.")
    parser.add_argument("--thumbnail-size", type=int, default=256,
                        help="Pixel size of saved thumbnails (default 256)")
    parser.add_argument("--d-model", type=int, default=20000)
    parser.add_argument("--top-k", type=int, default=40, help="SAE sparsity (top-k)")
    parser.add_argument("--top-n", type=int, default=16,
                        help="Number of top unique images to store per feature")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--umap-subset", type=int, default=5000)
    parser.add_argument("--umap-reservoir", type=int, default=50000,
                        help="Max images to keep in reservoir for activation-based UMAP and p75")
    parser.add_argument("--reservoir-max", type=int, default=500,
                        help="Kept for backward compatibility; no longer used.")
    parser.add_argument("--recursive", action="store_true",
                        help="Search subdirectories for images (ImageNet-style layout)")
    parser.add_argument("--interleave-classes", action="store_true",
                        help="Reorder images so classes are sampled round-robin "
                             "(recommended when using --coverage-threshold on ImageNet train)")
    parser.add_argument("--coverage-threshold", type=float, default=0.0,
                        help="Stop early when this fraction of features have fired at least "
                             "once (0 = disabled, e.g. 0.90 for 90%% coverage)")
    parser.add_argument("--coverage-check-interval", type=int, default=100,
                        help="Check coverage every N batches (default 100)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Extract from this intermediate DINOv3 layer instead of the final "
                             "layer (0-indexed into hidden_states; 0=embedding output, "
                             "1=after block 0, …). Default: None = use last_hidden_state.")
    parser.add_argument("--token-type", type=str, default="spatial",
                        choices=["spatial", "cls", "all"],
                        help="Which tokens to feed through the SAE per image: "
                             "'spatial' = patch tokens only (default); "
                             "'cls' = CLS token only (1 token/image, no patchwise heatmap); "
                             "'all' = CLS + register + patch tokens.")
    parser.add_argument("--backbone", type=str, default="dinov3",
                        choices=["dinov3", "clip", "dinov2"],
                        help="Vision backbone: 'dinov3' (default), 'clip' (ViT-L/14), or 'dinov2' (ViT-B/14).")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    token_type = args.token_type
    print(f"Token type: {token_type}")
    if args.layer is not None:
        print(f"Will extract from intermediate layer {args.layer}")
    else:
        print("Will extract from final layer (last_hidden_state)")

    # --- Load backbone ---
    backbone_name = args.backbone
    _get_hidden, d_brain, num_register_tokens, _transform = \
        load_batched_backbone(backbone_name, args.layer, device)
    if backbone_name == "clip" or backbone_name.startswith("dinov2"):
        _backbone_input_size = 224
        _backbone_resolution = 16
    else:
        _backbone_input_size = INPUT_SIZE
        _backbone_resolution = RESOLUTION

    # --- Load SAE ---
    print(f"Loading SAE from {args.sae_path}...")
    sae = load_sae(args.sae_path, d_brain, args.d_model, args.top_k, device)
    print(f"  SAE loaded: d_model={args.d_model}, top_k={args.top_k}")

    # --- Dataset ---
    roots = [args.image_dir]
    if args.extra_image_dir:
        roots.append(args.extra_image_dir)
    dataset = ImageFolder(roots, recursive=args.recursive, transform_fn=_transform)
    n_images = len(dataset)
    print(f"Found {n_images} images across {len(roots)} director(y/ies)")

    if args.interleave_classes:
        print("Interleaving images by class for spread-out sampling...")
        dataset.paths = interleave_by_class(dataset.paths)
        print(f"  Done — {len(dataset.paths)} paths reordered")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # --- Tracker ---
    tracker = FeatureImageTracker(args.d_model, top_n=args.top_n)
    image_paths = []

    umap_reservoir_size  = args.umap_reservoir
    umap_reservoir       = torch.zeros(umap_reservoir_size, args.d_model, dtype=torch.float16)
    umap_reservoir_img_idx = np.full(umap_reservoir_size, -1, dtype=np.int64)
    umap_reservoir_count = 0
    umap_n_seen          = 0

    # --- Process images ---
    print(f"Processing {n_images} images...")
    global_img_idx = 0

    with torch.inference_mode():
        for batch_i, (imgs, paths) in enumerate(loader):
            image_paths.extend(list(paths))
            bs = imgs.shape[0]
            imgs = imgs.to(device, non_blocking=True)

            hidden = _get_hidden(imgs)
            tokens = extract_tokens(hidden, backbone_name, token_type, num_register_tokens)

            n_patches   = tokens.shape[1]
            flat_tokens = tokens.reshape(-1, d_brain)

            z_pre, z, x_hat = sae(flat_tokens)
            z_per_image = z.reshape(bs, n_patches, args.d_model)

            # Single GPU→CPU transfer per batch; precompute per-image max and median activations
            z_batch_cpu    = z_per_image.cpu().numpy()   # (bs, n_patches, d_model)
            max_acts_batch = z_batch_cpu.max(axis=1)     # (bs, d_model)
            mean_acts_batch = z_batch_cpu.mean(axis=1)  # (bs, d_model)

            # Batch stats update — one numpy call for the whole batch instead
            # of separate calls per image inside the loop below.
            tracker.act_counts += (z_batch_cpu > 0).sum(axis=(0, 1))
            tracker.act_sums   += z_batch_cpu.sum(axis=(0, 1))

            for i in range(bs):
                img_idx  = global_img_idx + i
                max_acts = max_acts_batch[i]
                mean_acts = mean_acts_batch[i]

                tracker.update_heaps(img_idx, max_acts)
                tracker.update_mean_heaps(img_idx, mean_acts)

                umap_n_seen += 1
                if umap_reservoir_count < umap_reservoir_size:
                    umap_reservoir[umap_reservoir_count] = torch.from_numpy(max_acts).half()
                    umap_reservoir_img_idx[umap_reservoir_count] = img_idx
                    umap_reservoir_count += 1
                else:
                    j = np.random.randint(0, umap_n_seen)
                    if j < umap_reservoir_size:
                        umap_reservoir[j] = torch.from_numpy(max_acts).half()
                        umap_reservoir_img_idx[j] = img_idx

            global_img_idx += bs

            if (batch_i + 1) % 50 == 0:
                print(f"  Processed {global_img_idx}/{n_images} images", flush=True)

            # --- Early stopping by feature coverage ---
            if args.coverage_threshold > 0 and (batch_i + 1) % args.coverage_check_interval == 0:
                n_covered = int((tracker.img_counts > 0).sum())
                coverage = n_covered / args.d_model
                print(f"  Coverage check: {n_covered}/{args.d_model} features active "
                      f"({coverage:.1%}) after {global_img_idx} images", flush=True)
                if coverage >= args.coverage_threshold:
                    print(f"  Reached coverage threshold {args.coverage_threshold:.1%} — "
                          f"stopping early after {global_img_idx} images.")
                    break

    n_processed = global_img_idx
    print(f"Done processing {n_processed} images "
          f"({'full dataset' if n_processed == n_images else f'early stop at {n_processed}/{n_images}'})")
    umap_reservoir       = umap_reservoir[:umap_reservoir_count]
    umap_reservoir_img_idx = umap_reservoir_img_idx[:umap_reservoir_count]

    # --- Finalize ---
    print("Finalizing per-feature statistics...")
    n_top_save = min(15, args.top_n)
    umap_data_np = umap_reservoir.float().numpy()  # (n_reservoir, d_model) float32
    results = tracker.finalize(
        n_top=n_top_save,
        umap_data=umap_data_np,
        umap_img_idx=umap_reservoir_img_idx,
        n_p75=15,
    )
    del tracker

    # --- UMAP (activation patterns) ---
    print("Computing UMAP of SAE features (activation patterns)...")
    act_matrix = torch.from_numpy(umap_data_np).T  # (d_model, n_reservoir)
    live_mask    = results['feature_frequency'] > 0
    live_indices = live_mask.nonzero(as_tuple=False).squeeze(-1)
    n_live       = live_indices.shape[0]
    print(f"  Live features in precompute dataset: {n_live} / {args.d_model}")
    print(f"  UMAP reservoir: {umap_reservoir_count} images sampled from {umap_n_seen}")

    umap_indices = live_indices

    umap_data_sub = act_matrix[umap_indices].numpy()
    del act_matrix

    _umap_kw = dict(n_components=2, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)

    umap_coords_subset  = umap.UMAP(**_umap_kw).fit_transform(umap_data_sub)
    del umap_data_sub

    umap_coords = np.full((args.d_model, 2), np.nan, dtype=np.float32)
    umap_coords[umap_indices.numpy()] = umap_coords_subset
    results['umap_coords']           = torch.from_numpy(umap_coords)
    results['umap_feature_indices']  = umap_indices

    # --- UMAP (dictionary geometry) — use ALL features ---
    print("Computing UMAP of dictionary atoms (all features)...")
    dictionary = sae.get_dictionary().detach().cpu().float()
    dict_umap_indices = torch.arange(args.d_model)

    dict_umap_data          = dictionary[dict_umap_indices].numpy()
    dict_umap_coords_subset = umap.UMAP(**_umap_kw).fit_transform(dict_umap_data)

    dict_umap_coords = np.full((args.d_model, 2), np.nan, dtype=np.float32)
    dict_umap_coords[dict_umap_indices.numpy()] = dict_umap_coords_subset
    results['dict_umap_coords']          = torch.from_numpy(dict_umap_coords)
    results['dict_umap_feature_indices'] = dict_umap_indices

    # --- Metadata ---
    # n_patches depends on token type: 1 for CLS, RESOLUTION² for spatial, full seq for all
    _res = _backbone_resolution
    if token_type == "spatial":
        n_patches_stored = _res * _res
        patch_grid_stored = _res
    elif token_type == "cls":
        n_patches_stored = 1
        patch_grid_stored = 1
    else:  # "all"
        n_patches_stored = 1 + num_register_tokens + _res * _res
        patch_grid_stored = _res   # spatial layout only applies to patch portion

    results['image_paths'] = image_paths
    results['d_model']     = args.d_model
    results['top_k']       = args.top_k
    results['n_images']    = n_processed
    results['n_patches']   = n_patches_stored
    results['patch_grid']  = patch_grid_stored
    results['token_type']  = token_type
    results['image_size']  = _backbone_input_size
    results['backbone']    = backbone_name
    results['dictionary']  = dictionary

    # --- Thumbnails (for HF-hosted deployment) ---
    if args.thumbnail_dir:
        os.makedirs(args.thumbnail_dir, exist_ok=True)
        print(f"Saving {args.thumbnail_size}px thumbnails to {args.thumbnail_dir} ...")

        # Collect unique image indices from all display slots + UMAP reservoir
        referenced = set()
        for key in ('top_img_idx', 'mean_img_idx', 'p75_img_idx'):
            t = results[key]
            referenced.update(int(x) for x in t.view(-1).tolist() if x >= 0)
        referenced.update(
            int(x) for x in umap_reservoir_img_idx[:umap_reservoir_count] if x >= 0
        )
        referenced = sorted(referenced)
        print(f"  {len(referenced)} unique images referenced across feature slots + UMAP reservoir")

        n_saved = n_skipped = n_failed = 0
        sz = args.thumbnail_size
        for img_i in referenced:
            src = image_paths[img_i]
            dst = os.path.join(args.thumbnail_dir, os.path.basename(src))
            if os.path.exists(dst):
                n_skipped += 1
                continue
            try:
                thumb = Image.open(src).convert("RGB").resize((sz, sz), Image.BILINEAR)
                thumb.save(dst, format="JPEG", quality=85)
                n_saved += 1
            except Exception as e:
                print(f"  Warning: could not thumbnail {src}: {e}")
                n_failed += 1

        print(f"  Thumbnails: {n_saved} saved, {n_skipped} already existed, {n_failed} failed")

        # Switch image_paths to basenames so the hosted app resolves via --image-dir
        image_paths = [os.path.basename(p) for p in image_paths]
        results['image_paths'] = image_paths
        print(f"  image_paths: replaced with {len(image_paths)} basenames")

    output_path = args.output_path
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(results, output_path)
    print(f"Saved explorer data to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
