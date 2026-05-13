"""Dataset loading helpers for the explorer.

Pulled out of ``scripts/explorer_app.py`` so that the entry script only
composes panels — disk I/O, JSON sidecars, and registry walking all live
here.

Public surface:

- ``load_dataset(path, label, sae_url, names_file)`` — load one
  ``explorer_data*.pt`` plus its JSON sidecars and heatmap / patch_acts
  ``.pt`` sidecars; return the unified dict the rest of the app expects.

- ``build_dataset_list(registry, local_dir, names_file)`` — eagerly load
  the primary entry and return lazy placeholders for the compares; use
  with ``ensure_loaded`` to materialize a compare on first access.

- ``ensure_loaded(datasets, idx)`` — replace a placeholder with a fully
  loaded dataset dict in place.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import List, Optional

import torch

from .registry import ModelEntry, Registry
from .persistence import _maybe_pull_and_merge_local


# ---------- JSON sidecar helpers ----------

def _load_json_dict(path: str, kind: str) -> dict:
    """Load a sidecar JSON keyed by feature-id-as-str. Missing -> {}.

    ``kind`` is just used in the warning message ("authors", "history",
    etc.) so a malformed file is easy to track down.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
        if kind == 'history':
            return {int(k): list(v) for k, v in raw.items()}
        return {int(k): v for k, v in raw.items()}
    except Exception as e:
        print(f"  Warning: could not read {kind} file {path}: {e}")
        return {}


# ---------- Single-dataset load ----------

def load_dataset(path: str,
                 label: str,
                 sae_url: Optional[str] = None,
                 names_file: Optional[str] = None) -> dict:
    """Load one ``explorer_data*.pt`` plus all sibling JSON / .pt sidecars.

    ``names_file`` overrides the default
    ``<path-without-ext>_feature_names.json``; pass it through from
    ``args.names_file`` when invoked from the entry script.
    """
    print(f"Loading [{label}] from {path} ...")
    d = torch.load(path, map_location='cpu', weights_only=False)
    cs = d.get('clip_text_scores', None)

    base = os.path.splitext(path)[0]
    if names_file is None:
        names_file = base + '_feature_names.json'
    auto_interp_file       = base + '_auto_interp.json'
    fn_authors_file        = os.path.splitext(names_file)[0]       + '_authors.json'
    ai_authors_file        = os.path.splitext(auto_interp_file)[0] + '_authors.json'
    fn_history_file        = os.path.splitext(names_file)[0]       + '_history.json'
    ai_history_file        = os.path.splitext(auto_interp_file)[0] + '_history.json'

    # Seed each sidecar from the canonical HF dataset copy (no-op when
    # HF_TOKEN / HF_DATASET_REPO aren't set). This is the boot half of
    # the round-trip sync: on Spaces the local container has only the
    # bundled snapshot, so without this every cold start would clobber
    # remote labels on the first save.
    for _p in (names_file, auto_interp_file,
               fn_authors_file, ai_authors_file,
               fn_history_file, ai_history_file):
        _maybe_pull_and_merge_local(_p)

    feat_names    = _load_json_dict(names_file,         'feature names')
    auto_interp   = _load_json_dict(auto_interp_file,   'auto-interp')
    fn_authors    = _load_json_dict(fn_authors_file,    'authors')
    ai_authors    = _load_json_dict(ai_authors_file,    'authors')
    fn_history    = _load_json_dict(fn_history_file,    'history')
    ai_history    = _load_json_dict(ai_history_file,    'history')

    if auto_interp:
        print(f"  Loaded {len(auto_interp)} auto-interp labels from "
              f"{os.path.basename(auto_interp_file)}")
    if fn_authors:
        print(f"  Loaded {len(fn_authors)} feature-name authors from "
              f"{os.path.basename(fn_authors_file)}")
    if ai_authors:
        print(f"  Loaded {len(ai_authors)} auto-interp authors from "
              f"{os.path.basename(ai_authors_file)}")
    if fn_history:
        print(f"  Loaded feature-name history for {len(fn_history)} feat(s) from "
              f"{os.path.basename(fn_history_file)}")
    if ai_history:
        print(f"  Loaded auto-interp history for {len(ai_history)} feat(s) from "
              f"{os.path.basename(ai_history_file)}")

    entry = {
        'label':                      label,
        'path':                       path,
        'image_paths':                d['image_paths'],
        'd_model':                    d['d_model'],
        'n_images':                   d['n_images'],
        'patch_grid':                 d['patch_grid'],
        'image_size':                 d['image_size'],
        'token_type':                 d.get('token_type', 'spatial'),
        'backbone':                   d.get('backbone', 'dinov3'),
        'top_img_idx':                d['top_img_idx'],
        'top_img_act':                d['top_img_act'],
        'mean_img_idx':               d.get('mean_img_idx', d['top_img_idx']),
        'mean_img_act':               d.get('mean_img_act', d['top_img_act']),
        'p75_img_idx':                d['p75_img_idx'],
        'p75_img_act':                d['p75_img_act'],
        'feature_frequency':          d['feature_frequency'],
        'feature_mean_act':           d['feature_mean_act'],
        'feature_p75_val':            d['feature_p75_val'],
        'umap_coords':                d['umap_coords'].numpy(),
        'dict_umap_coords':           d['dict_umap_coords'].numpy(),
        'clip_scores':                cs,
        'clip_vocab':                 d.get('clip_text_vocab', None),
        'clip_embeds':                d.get('clip_feature_embeds', None),
        'clip_scores_f32':            cs.float() if cs is not None else None,
        'inference_cache':            OrderedDict(),
        'names_file':                 names_file,
        'auto_interp_file':           auto_interp_file,
        'feature_names':              feat_names,
        'auto_interp_names':          auto_interp,
        'feature_name_authors_file':  fn_authors_file,
        'auto_interp_authors_file':   ai_authors_file,
        'feature_name_authors':       fn_authors,
        'auto_interp_authors':        ai_authors,
        'feature_names_history_file': fn_history_file,
        'auto_interp_history_file':   ai_history_file,
        'feature_names_history':      fn_history,
        'auto_interp_history':        ai_history,
    }

    # ---- Heatmap sidecar ----
    sidecar = base + '_heatmaps.pt'
    if os.path.exists(sidecar):
        print(f"  Loading pre-computed heatmaps from {os.path.basename(sidecar)} ...")
        hm = torch.load(sidecar, map_location='cpu', weights_only=True)
        entry['top_heatmaps']       = hm.get('top_heatmaps')
        entry['mean_heatmaps']      = hm.get('mean_heatmaps')
        entry['p75_heatmaps']       = hm.get('p75_heatmaps')
        entry['heatmap_patch_grid'] = hm.get('patch_grid', d['patch_grid'])
        has_hm = 'yes (no GPU needed for heatmaps)'
    else:
        entry['top_heatmaps']       = None
        entry['mean_heatmaps']      = None
        entry['p75_heatmaps']       = None
        entry['heatmap_patch_grid'] = d['patch_grid']
        has_hm = 'no'

    # ---- Patch-activations sidecar ----
    pa_sidecar = base + '_patch_acts.pt'
    if os.path.exists(pa_sidecar):
        print(f"  Loading pre-computed patch acts from {os.path.basename(pa_sidecar)} ...")
        pa = torch.load(pa_sidecar, map_location='cpu', weights_only=True)
        img_to_row = {int(idx): row for row, idx in enumerate(pa['img_indices'].tolist())}
        entry['patch_acts'] = {
            'feat_indices': pa['feat_indices'],   # (n_unique, n_patches, top_k) int16
            'feat_values':  pa['feat_values'],    # (n_unique, n_patches, top_k) float16
            'img_to_row':   img_to_row,
        }
        print(f"    patch_acts: {len(img_to_row)} images covered (GPU-free patch explorer)")
    else:
        entry['patch_acts'] = None

    entry['sae_url'] = sae_url

    print(f"  d={entry['d_model']}, n={entry['n_images']}, token={entry['token_type']}, "
          f"backbone={entry['backbone']}, clip={'yes' if cs is not None else 'no'}, "
          f"heatmaps={has_hm}, patch_acts={'yes' if entry['patch_acts'] else 'no'}")
    return entry


# ---------- Registry-driven build ----------

def _entry_paths(reg: Registry, entry: ModelEntry, local_dir: str):
    """Resolve the .pt path and SAE download URL for a registry entry."""
    return os.path.join(local_dir, entry.data_file), reg.sae_url(entry)


def build_dataset_list(reg: Registry,
                       local_dir: str,
                       names_file: Optional[str] = None) -> List[dict]:
    """Eagerly load the primary, return lazy placeholders for compares.

    Compare placeholders carry only ``label``, ``path``, ``sae_url``, and
    ``_lazy: True``. Use :func:`ensure_loaded` to materialize one in
    place on first access.
    """
    primary_path, primary_sae_url = _entry_paths(reg, reg.primary, local_dir)
    datasets: List[dict] = [load_dataset(
        primary_path, reg.primary.label,
        sae_url=primary_sae_url, names_file=names_file,
    )]
    for entry in reg.compares:
        path, sae_url = _entry_paths(reg, entry, local_dir)
        datasets.append({
            'label':   entry.label,
            'path':    path,
            '_lazy':   True,
            'sae_url': sae_url,
        })
    return datasets


def ensure_loaded(datasets: List[dict], idx: int) -> None:
    """Replace a lazy placeholder at ``datasets[idx]`` with the real entry."""
    ds = datasets[idx]
    if ds.get('_lazy', False):
        print(f"[Lazy load] Loading '{ds['label']}' on first access ...")
        datasets[idx] = load_dataset(
            ds['path'], ds['label'], sae_url=ds.get('sae_url'),
        )
