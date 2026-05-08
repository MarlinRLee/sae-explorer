"""Per-image SAE activation reconstruction without a GPU.

Two paths, in priority order:

1. ``patch_acts`` sidecar — complete (n_patches, d_sae) activations for
   any image covered by ``precompute_heatmaps.py --save-patch-acts``.
2. Heatmap reconstruction — partial activations recovered from the
   ``top/mean/p75 heatmaps`` sidecars, populated only for features whose
   top/mean/p75 image lists include the requested image.

Both paths are GPU-free and read from the dataset dict produced by
:func:`scripts.explorer_app._load_dataset_dict`. An LRU cache lives on
each dataset dict (the ``inference_cache`` entry) to amortise repeated
lookups for the same image.
"""

import numpy as np

from . import runtime


def _reconstruct_z_from_heatmaps(img_idx: int, ds: dict):
    """Reconstruct ``(n_patches², d_sae)`` float32 activations from
    pre-computed heatmaps — no GPU required.

    For each ``(feature, slot)`` pair where top/mean/p75 image indices
    equal ``img_idx``, copy ``heatmap[feat, slot, :]`` into
    ``z[:, feat]``. Returns None if the image does not appear in any
    pre-computed slot for this dataset.
    """
    z = None
    for hm_key, idx_key in [
        ('top_heatmaps',  'top_img_idx'),
        ('mean_heatmaps', 'mean_img_idx'),
        ('p75_heatmaps',  'p75_img_idx'),
    ]:
        hm  = ds.get(hm_key)   # (d_sae, n_slots, n_patches²) float16 tensor
        idx = ds.get(idx_key)  # (d_sae, n_slots) int tensor
        if hm is None or idx is None:
            continue
        if z is None:
            d_sae, _, n_patches_sq = hm.shape
            z = np.zeros((n_patches_sq, d_sae), dtype=np.float32)
        idx_np = idx.numpy()
        feat_ids, slot_ids = np.where(idx_np == img_idx)
        if len(feat_ids) == 0:
            continue
        # hm[feat_ids, slot_ids, :] -> (K, n_patches²); transpose to (n_patches², K)
        vals = hm[feat_ids, slot_ids, :].float().numpy()
        # np.maximum handles the (rare) case where the same image appears
        # in multiple sets (top & mean) for the same feature.
        z[:, feat_ids] = np.maximum(z[:, feat_ids], vals.T)
    if z is None or not np.any(z):
        return None
    return z


def compute_patch_activations(img_idx: int):
    """Return (n_patches, d_sae) float32 for the active dataset, or None.

    Reads from the active ``runtime.state`` and ``runtime.args`` for the
    LRU cache size limit. The cache is keyed per-dataset (lives on the
    dataset dict) so dataset switches don't invalidate prior work.
    """
    state = runtime.state
    args  = runtime.args
    ds    = state.ds
    cache = ds['inference_cache']
    if img_idx in cache:
        cache.move_to_end(img_idx)
        return cache[img_idx]

    z_np = None

    # 1. Try patch_acts lookup (complete activations for covered images).
    pa = ds.get('patch_acts')
    if pa is not None:
        row = pa['img_to_row'].get(img_idx)
        if row is not None:
            fi = pa['feat_indices'][row].numpy()              # (n_patches, top_k) int16
            fv = pa['feat_values'][row].float().numpy()       # (n_patches, top_k) float32
            n_p = fi.shape[0]
            z_np = np.zeros((n_p, ds['d_model']), dtype=np.float32)
            # d_model <= 32000, so all feature indices fit in signed int16 (max 32767).
            z_np[np.arange(n_p)[:, None], fi.astype(np.int32)] = fv

    # 2. Fall back to heatmap reconstruction (partial activations).
    if z_np is None:
        z_np = _reconstruct_z_from_heatmaps(img_idx, ds)

    if z_np is not None:
        cache[img_idx] = z_np
        if len(cache) > args.inference_cache_size:
            cache.popitem(last=False)
    return z_np
