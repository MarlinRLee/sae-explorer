"""Per-image SAE activation reconstruction for the patch explorer.

Two paths, in priority order:

1. ``patch_acts`` sidecar — pre-computed (n_patches, d_sae) activations
   for any image covered by ``precompute_heatmaps.py --save-patch-acts``.
2. On-demand backbone + SAE inference — load the backbone (DINOv2 / v3 /
   CLIP) and SAE on first patch-select for the dataset, then run a fresh
   forward pass per image. Both models stay in memory for subsequent
   images. First call pays the model-load cost (a few hundred MB +
   ~1-10 s on CPU); subsequent images run in ~1-3 s on CPU.

An LRU cache lives on each dataset dict (the ``inference_cache`` entry)
so repeated patch-explorer lookups for the same image are free.
"""

import os
import sys

import numpy as np

from . import runtime


# Project layout: this file lives at scripts/explorer/, with the shared
# inference helpers (backbone_runners, precompute_utils) at src/. We
# import them lazily so unit tests can exercise the panels without
# pulling torch onto the path at module-import time.
_PROJECT_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'src')
)


def _ensure_src_on_path() -> None:
    if _PROJECT_SRC not in sys.path:
        sys.path.insert(0, _PROJECT_SRC)


def _load_inference_pieces(ds: dict):
    """Lazy-load the backbone forward fn + SAE for this dataset, cache
    them on ``ds``, return a tuple of (forward_fn, transform_fn, n_reg,
    sae, torch_module) or ``None`` if any piece is unavailable (no SAE
    file on disk, missing deps). Once a load fails for a dataset the
    sentinel ``None`` is cached so we don't retry on every patch-select.
    """
    if '_inference_pieces' in ds:
        return ds['_inference_pieces']

    sae_file = ds.get('sae_file')
    if not sae_file:
        print("  [patch-activations] no sae_file on dataset entry; "
              "on-demand inference disabled.")
        ds['_inference_pieces'] = None
        return None
    sae_path = os.path.join(runtime.args.data_dir, sae_file)
    if not os.path.exists(sae_path):
        print(f"  [patch-activations] SAE not found at {sae_path}; "
              "on-demand inference disabled.")
        ds['_inference_pieces'] = None
        return None

    _ensure_src_on_path()
    try:
        import torch as _torch
        from backbone_runners import load_batched_backbone
        from precompute_utils import load_sae, parse_top_k_from_path
    except Exception as e:
        print(f"  [patch-activations] dependency import failed: {e}")
        ds['_inference_pieces'] = None
        return None

    backbone = ds.get('backbone', 'dinov3')
    layer    = ds.get('layer')
    device   = _torch.device('cpu')
    try:
        print(f"  [patch-activations] loading backbone={backbone} "
              f"layer={layer} + SAE {os.path.basename(sae_path)} on {device} ...")
        forward_fn, d_hidden, n_reg, transform_fn = load_batched_backbone(
            backbone, layer, device,
        )
        top_k = parse_top_k_from_path(sae_path)
        sae   = load_sae(sae_path, d_hidden, ds['d_model'], top_k, device)
    except Exception as e:
        print(f"  [patch-activations] inference setup failed: {e}")
        ds['_inference_pieces'] = None
        return None

    ds['_inference_pieces'] = (forward_fn, transform_fn, n_reg, sae, _torch)
    return ds['_inference_pieces']


def _run_inference(img_idx: int, ds: dict):
    """Run the backbone + SAE on one image and return ``(n_patches,
    d_sae)`` float32, or ``None`` if inference isn't available."""
    pieces = _load_inference_pieces(ds)
    if pieces is None:
        return None
    forward_fn, transform_fn, n_reg, sae, _torch = pieces

    _ensure_src_on_path()
    from precompute_utils import extract_tokens

    pil = runtime.load_image(img_idx)
    backbone   = ds.get('backbone', 'dinov3')
    token_type = ds.get('token_type', 'spatial')
    with _torch.no_grad():
        img_tensor = transform_fn(pil).unsqueeze(0)
        hidden = forward_fn(img_tensor)                       # (1, n_tokens, d_hidden)
        tokens = extract_tokens(hidden, backbone, token_type, n_reg)
        d_hidden = tokens.shape[-1]
        flat = tokens.reshape(-1, d_hidden)                   # (n_patches, d_hidden)
        _, z, _ = sae(flat)                                   # (n_patches, d_model)
    return z.cpu().float().numpy()


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

    # 2. On-demand backbone + SAE forward — gives the complete
    #    (n_patches, d_sae) activations for any image at the cost of
    #    keeping the models in memory.
    if z_np is None:
        z_np = _run_inference(img_idx, ds)

    if z_np is not None:
        cache[img_idx] = z_np
        if len(cache) > args.inference_cache_size:
            cache.popitem(last=False)
    return z_np
