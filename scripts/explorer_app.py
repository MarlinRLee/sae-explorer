"""
Interactive SAE Feature Explorer - Bokeh Server App.

Visualizes SAE features with:
  - UMAP scatter plot of features (activation-based and dictionary-based)
  - Click a feature to see its top-activating images with heatmap overlays
  - 75th percentile images for distribution understanding
  - Patch explorer: click patches of any image to find active features
  - Feature naming: assign names to features, saved to JSON, searchable

All display is driven by pre-computed sidecars (_heatmaps.pt, _patch_acts.pt).
No GPU or model weights are required at serve time.

Launch:
    bokeh serve explorer_app.py --port 5006 --allow-websocket-origin="*" \
        --session-token-expiration 86400 \
        --args \
          --data ../explorer_data_d32000_k160_val.pt \
          --image-dir /scratch.global/lee02328/val \
          --extra-image-dir /scratch.global/lee02328/coco/val2017 \
          --primary-label "DINOv3 L24 Spatial (d=32K)" \
          --compare-data ../explorer_data_18.pt \
          --compare-labels "DINOv3 L18 Spatial (d=20K)"

Then SSH tunnel: ssh -L 5006:<node>:5006 <user>@<login-node>
Open: http://localhost:5006/explorer_app
"""

import argparse
import os
import io
import json
import base64
import threading
from collections import OrderedDict
from functools import partial

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from clip_utils import load_clip, compute_text_embeddings

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.events import MouseMove
from bokeh.models import (
    ColumnDataSource, HoverTool, Div, Select, TextInput, Button,
    DataTable, TableColumn, NumberFormatter, IntEditor,
    Slider, Toggle, CustomJS,
)
from bokeh.plotting import figure
from bokeh.palettes import Turbo256
from bokeh.transform import linear_cmap


# ---------- Parse args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--image-dir", type=str, required=True,
                    help="Primary image directory used during precompute")
parser.add_argument("--extra-image-dir", type=str, default=None,
                    help="Additional image directory used during precompute")
parser.add_argument("--thumb-size", type=int, default=256)
parser.add_argument("--inference-cache-size", type=int, default=64,
                    help="Number of images to keep in the patch-activations LRU cache")
parser.add_argument("--names-file", type=str, default=None,
                    help="Path to JSON file for saving feature names "
                         "(default: <data>_feature_names.json)")
parser.add_argument("--compare-data", type=str, nargs="*", default=[],
                    help="Additional explorer_data.pt files to show in cross-dataset "
                         "comparison panel (e.g. layer 18, CLS SAE)")
parser.add_argument("--compare-labels", type=str, nargs="*", default=[],
                    help="Display labels for each --compare-data file")
parser.add_argument("--primary-label", type=str, default="Primary",
                    help="Display label for the primary --data file")
parser.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14",
                    help="HuggingFace CLIP model ID for free-text search "
                         "(only loaded on first out-of-vocab query)")
parser.add_argument("--google-api-key", type=str, default=None,
                    help="Google API key for Gemini auto-interp button "
                         "(default: GOOGLE_API_KEY env var)")
parser.add_argument("--sae-url", type=str, default=None,
                    help="Download URL for the primary dataset's SAE weights — "
                         "shown as a link in the summary panel")
parser.add_argument("--compare-sae-urls", type=str, nargs="*", default=[],
                    help="Download URLs for each --compare-data dataset's SAE weights (in order)")
args = parser.parse_args()


# ---------- Lazy CLIP model (loaded on first free-text query) ----------
# _clip_handle[0] is None until the first out-of-vocab query is issued.
_clip_handle = [None]   # (model, processor, device)

def _get_clip():
    """Load CLIP once and cache it."""
    if _clip_handle[0] is None:
        _dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[CLIP] Loading {args.clip_model} on {_dev} (first free-text query)...")
        _m, _p = load_clip(_dev, model_name=args.clip_model)
        _clip_handle[0] = (_m, _p, _dev)
        print("[CLIP] Ready.")
    return _clip_handle[0]


# ---------- Load all datasets into a unified list ----------

def _load_dataset_dict(path, label, sae_url=None):
    """Load one explorer_data.pt file and return a unified dataset dict."""
    print(f"Loading [{label}] from {path} ...")
    d = torch.load(path, map_location='cpu', weights_only=False)
    cs = d.get('clip_text_scores', None)
    names_file = (args.names_file if path == args.data and args.names_file
                  else os.path.splitext(path)[0] + '_feature_names.json')
    feat_names = {}
    if os.path.exists(names_file):
        with open(names_file) as _nf:
            feat_names = {int(k): v for k, v in json.load(_nf).items()}
    auto_interp_file = os.path.splitext(path)[0] + '_auto_interp.json'
    auto_interp = {}
    if os.path.exists(auto_interp_file):
        with open(auto_interp_file) as _af:
            auto_interp = {int(k): v for k, v in json.load(_af).items()}
        print(f"  Loaded {len(auto_interp)} auto-interp labels from "
              f"{os.path.basename(auto_interp_file)}")
    entry = {
        'label':             label,
        'path':              path,
        'image_paths':       d['image_paths'],
        'd_model':           d['d_model'],
        'n_images':          d['n_images'],
        'patch_grid':        d['patch_grid'],
        'image_size':        d['image_size'],
        'token_type':        d.get('token_type', 'spatial'),
        'backbone':          d.get('backbone', 'dinov3'),
        'top_img_idx':       d['top_img_idx'],
        'top_img_act':       d['top_img_act'],
        'mean_img_idx':      d.get('mean_img_idx', d['top_img_idx']),
        'mean_img_act':      d.get('mean_img_act', d['top_img_act']),
        'p75_img_idx':       d['p75_img_idx'],
        'p75_img_act':       d['p75_img_act'],
        'feature_frequency': d['feature_frequency'],
        'feature_mean_act':  d['feature_mean_act'],
        'feature_p75_val':   d['feature_p75_val'],
        'umap_coords':       d['umap_coords'].numpy(),
        'dict_umap_coords':  d['dict_umap_coords'].numpy(),
        'clip_scores':       cs,
        'clip_vocab':        d.get('clip_text_vocab', None),
        'clip_embeds':       d.get('clip_feature_embeds', None),
        'clip_scores_f32':   cs.float() if cs is not None else None,
        'inference_cache':   OrderedDict(),
        'names_file':        names_file,
        'auto_interp_file':  auto_interp_file,
        'feature_names':     feat_names,
        'auto_interp_names': auto_interp,
    }
    # Load pre-computed heatmaps sidecar if present
    sidecar = os.path.splitext(path)[0] + '_heatmaps.pt'
    if os.path.exists(sidecar):
        print(f"  Loading pre-computed heatmaps from {os.path.basename(sidecar)} ...")
        hm = torch.load(sidecar, map_location='cpu', weights_only=True)
        entry['top_heatmaps']      = hm.get('top_heatmaps')
        entry['mean_heatmaps']     = hm.get('mean_heatmaps')
        entry['p75_heatmaps']      = hm.get('p75_heatmaps')
        # patch_grid stored in sidecar may differ from data (e.g. --force-spatial on CLS SAE)
        entry['heatmap_patch_grid'] = hm.get('patch_grid', d['patch_grid'])
        has_hm = 'yes (no GPU needed for heatmaps)'
    else:
        entry['top_heatmaps']      = None
        entry['mean_heatmaps']     = None
        entry['p75_heatmaps']      = None
        entry['heatmap_patch_grid'] = d['patch_grid']
        has_hm = 'no'

    # Load pre-computed patch activations sidecar if present.
    # Enables complete GPU-free patch exploration for any image covered by the file.
    pa_sidecar = os.path.splitext(path)[0] + '_patch_acts.pt'
    if os.path.exists(pa_sidecar):
        print(f"  Loading pre-computed patch acts from {os.path.basename(pa_sidecar)} ...")
        pa = torch.load(pa_sidecar, map_location='cpu', weights_only=True)
        img_to_row = {int(idx): row for row, idx in enumerate(pa['img_indices'].tolist())}
        entry['patch_acts'] = {
            'feat_indices': pa['feat_indices'],  # (n_unique, n_patches, top_k) int16
            'feat_values':  pa['feat_values'],   # (n_unique, n_patches, top_k) float16
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


_all_datasets = []
_active = [0]   # index of the currently displayed dataset

# Primary dataset — always loaded eagerly
_all_datasets.append(_load_dataset_dict(args.data, args.primary_label, sae_url=args.sae_url))

# Compare datasets — stored as lazy placeholders; loaded on first access
for _ci, _cpath in enumerate(args.compare_data):
    _clabel = (args.compare_labels[_ci]
               if args.compare_labels and _ci < len(args.compare_labels)
               else os.path.basename(_cpath))
    _csae = (args.compare_sae_urls[_ci]
             if args.compare_sae_urls and _ci < len(args.compare_sae_urls)
             else None)
    _all_datasets.append({'label': _clabel, 'path': _cpath, '_lazy': True, 'sae_url': _csae})


def _ensure_loaded(idx):
    """Load dataset at idx if it is still a lazy placeholder."""
    ds = _all_datasets[idx]
    if ds.get('_lazy', False):
        print(f"[Lazy load] Loading '{ds['label']}' on first access ...")
        _all_datasets[idx] = _load_dataset_dict(ds['path'], ds['label'], sae_url=ds.get('sae_url'))


def _apply_dataset_globals(idx):
    """Swap every module-level data variable to point at dataset[idx]."""
    global image_paths, d_model, n_images, patch_grid, image_size, heatmap_patch_grid
    global top_img_idx, top_img_act, mean_img_idx, mean_img_act
    global p75_img_idx, p75_img_act
    global top_heatmaps, mean_heatmaps, p75_heatmaps
    global feature_frequency, feature_mean_act, feature_p75_val
    global umap_coords, dict_umap_coords
    global freq, mean_act, log_freq, freq_np, mean_act_np, p75_np
    global live_mask, live_indices, dict_live_mask, dict_live_indices
    global umap_backup
    global _clip_scores, _clip_vocab, _clip_embeds, _clip_scores_f32, HAS_CLIP
    global _compare_datasets
    global feature_names, _names_file, auto_interp_names, _auto_interp_file

    ds = _all_datasets[idx]
    image_paths        = ds['image_paths']
    d_model            = ds['d_model']
    n_images           = ds['n_images']
    patch_grid         = ds['patch_grid']
    image_size         = ds['image_size']
    top_img_idx        = ds['top_img_idx']
    top_img_act        = ds['top_img_act']
    mean_img_idx       = ds['mean_img_idx']
    mean_img_act       = ds['mean_img_act']
    p75_img_idx        = ds['p75_img_idx']
    p75_img_act        = ds['p75_img_act']
    top_heatmaps        = ds.get('top_heatmaps')
    mean_heatmaps       = ds.get('mean_heatmaps')
    p75_heatmaps        = ds.get('p75_heatmaps')
    heatmap_patch_grid  = ds.get('heatmap_patch_grid', patch_grid)
    feature_frequency  = ds['feature_frequency']
    feature_mean_act   = ds['feature_mean_act']
    feature_p75_val    = ds['feature_p75_val']
    umap_coords        = ds['umap_coords']
    dict_umap_coords   = ds['dict_umap_coords']
    _clip_scores       = ds['clip_scores']
    _clip_vocab        = ds['clip_vocab']
    _clip_embeds       = ds['clip_embeds']
    _clip_scores_f32   = ds['clip_scores_f32']
    HAS_CLIP           = _clip_scores is not None and _clip_vocab is not None
    _compare_datasets  = [d for i, d in enumerate(_all_datasets) if i != idx]
    feature_names      = ds['feature_names']
    _names_file        = ds['names_file']
    auto_interp_names  = ds['auto_interp_names']
    _auto_interp_file  = ds['auto_interp_file']

    # Derived arrays used by UMAP, feature list, and callbacks
    freq          = feature_frequency.numpy()
    mean_act      = feature_mean_act.numpy()
    log_freq      = np.log10(freq + 1)
    freq_np       = freq
    mean_act_np   = mean_act
    p75_np        = feature_p75_val.numpy()
    live_mask         = ~np.isnan(umap_coords[:, 0])
    live_indices      = np.where(live_mask)[0]
    dict_live_mask    = ~np.isnan(dict_umap_coords[:, 0])
    dict_live_indices = np.where(dict_live_mask)[0]
    umap_backup = dict(
        act_x=umap_coords[live_mask, 0].tolist(),
        act_y=umap_coords[live_mask, 1].tolist(),
        act_feat=live_indices.tolist(),
        dict_x=dict_umap_coords[dict_live_mask, 0].tolist(),
        dict_y=dict_umap_coords[dict_live_mask, 1].tolist(),
        dict_feat=dict_live_indices.tolist(),
    )


# Initialise all globals from the primary dataset
_apply_dataset_globals(0)


def _save_names():
    with open(_names_file, 'w') as _f:
        json.dump({str(k): v for k, v in sorted(feature_names.items())}, _f, indent=2)
    print(f"Saved {len(feature_names)} feature names to {_names_file}")
    _schedule_hf_push(_names_file)


def _save_auto_interp():
    with open(_auto_interp_file, 'w') as _f:
        json.dump({str(k): v for k, v in sorted(auto_interp_names.items())}, _f, indent=2)
    print(f"Saved {len(auto_interp_names)} auto-interp labels to {_auto_interp_file}")
    _schedule_hf_push(_auto_interp_file)


_hf_push_pending = [None]   # holds the active debounce timeout handle

def _schedule_hf_push(names_file_path):
    """Debounce HF dataset upload: waits 2 s after the last save, then pushes in a thread.
    No-op if HF_TOKEN / HF_DATASET_REPO are not set (i.e. running locally)."""
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo  = os.environ.get("HF_DATASET_REPO")
    if not (hf_token and hf_repo):
        return

    # Cancel any already-pending push for this session.
    if _hf_push_pending[0] is not None:
        try:
            curdoc().remove_timeout_callback(_hf_push_pending[0])
        except Exception:
            pass

    def _push_thread():
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=names_file_path,
                path_in_repo=os.path.basename(names_file_path),
                repo_id=hf_repo,
                repo_type="dataset",
                token=hf_token,
                commit_message="Update feature names",
            )
            print(f"  Pushed {os.path.basename(names_file_path)} to HF dataset {hf_repo}")
        except Exception as e:
            print(f"  Warning: could not push feature names to HF: {e}")

    def _fire():
        _hf_push_pending[0] = None
        threading.Thread(target=_push_thread, daemon=True).start()

    _hf_push_pending[0] = curdoc().add_timeout_callback(_fire, 2000)


def _display_name(feat: int) -> str:
    """Return the label to show in tables: manual label takes priority over auto-interp."""
    m = feature_names.get(feat)
    if m:
        return m
    a = auto_interp_names.get(feat)
    return f"[auto] {a}" if a else ""


def _reconstruct_z_from_heatmaps(img_idx, ds):
    """Reconstruct (n_patches², d_sae) float32 from pre-computed heatmaps — no GPU needed.

    For each (feature, slot) pair where top/mean/p75 image indices equal img_idx,
    we copy heatmap[feat, slot, :] into z[:, feat].  Returns None if this image
    does not appear in any pre-computed slot.
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
        # hm[feat_ids, slot_ids, :] → (K, n_patches²); transpose to (n_patches², K)
        vals = hm[feat_ids, slot_ids, :].float().numpy()
        # np.maximum handles the (rare) case where the same image appears in
        # multiple sets (top & mean) for the same feature
        z[:, feat_ids] = np.maximum(z[:, feat_ids], vals.T)
    if z is None or not np.any(z):
        return None
    return z


def compute_patch_activations(img_idx):
    """Return (n_patches, d_sae) float32 for the active dataset, or None.

    Priority order (both GPU-free):
      1. LRU cache
      2. Pre-computed patch_acts lookup — complete activations for covered images
      3. Heatmap reconstruction — partial (only features that stored this image)
    Uses a per-dataset LRU cache.
    """
    ds    = _all_datasets[_active[0]]
    cache = ds['inference_cache']
    if img_idx in cache:
        cache.move_to_end(img_idx)
        return cache[img_idx]

    z_np = None

    # 2. Try patch_acts lookup (complete activations for covered images)
    pa = ds.get('patch_acts')
    if pa is not None:
        row = pa['img_to_row'].get(img_idx)
        if row is not None:
            fi = pa['feat_indices'][row].numpy()   # (n_patches, top_k) int16
            fv = pa['feat_values'][row].float().numpy()  # (n_patches, top_k) float32
            n_p = fi.shape[0]
            z_np = np.zeros((n_p, ds['d_model']), dtype=np.float32)
            # d_model <= 32000, so all feature indices fit in signed int16 (max 32767)
            z_np[np.arange(n_p)[:, None], fi.astype(np.int32)] = fv

    # 3. Fall back to heatmap reconstruction (partial activations)
    if z_np is None:
        z_np = _reconstruct_z_from_heatmaps(img_idx, ds)

    if z_np is not None:
        cache[img_idx] = z_np
        if len(cache) > args.inference_cache_size:
            cache.popitem(last=False)
    return z_np


# ---------- Alpha colormap ----------
def create_alpha_cmap(base='jet'):
    base_cmap = plt.cm.get_cmap(base)
    colors = base_cmap(np.arange(base_cmap.N))
    colors[:, -1] = np.linspace(0.0, 1.0, base_cmap.N)
    return mcolors.LinearSegmentedColormap.from_list('alpha_cmap', colors)

ALPHA_JET = create_alpha_cmap('jet')
ALPHA_VIRIDIS = create_alpha_cmap('viridis')


# ---------- Image helpers ----------
THUMB = args.thumb_size


def load_image(img_idx):
    """Load an image by index, return PIL Image."""
    path = image_paths[img_idx]
    fname = os.path.basename(path)
    for base in [args.image_dir] + ([args.extra_image_dir] if args.extra_image_dir else []):
        candidate = os.path.join(base, fname)
        if os.path.exists(candidate):
            return Image.open(candidate).convert("RGB")
    return Image.open(path).convert("RGB")


def load_compare_image(path, size):
    """Load an image for a comparison dataset, resolving via the same base dirs."""
    fname = os.path.basename(path)
    for base in [args.image_dir] + ([args.extra_image_dir] if args.extra_image_dir else []):
        candidate = os.path.join(base, fname)
        if os.path.exists(candidate):
            return Image.open(candidate).convert("RGB").resize((size, size), Image.BILINEAR)
    if os.path.exists(path):
        return Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return Image.new("RGB", (size, size), (180, 180, 180))


def render_heatmap_overlay(img_idx, heatmap_16x16, size=THUMB, cmap=ALPHA_JET, alpha=1.0):
    """Render image with heatmap overlay."""
    img = load_image(img_idx).resize((size, size), Image.BILINEAR)
    img_arr = np.array(img).astype(np.float32) / 255.0

    heatmap = heatmap_16x16.numpy() if isinstance(heatmap_16x16, torch.Tensor) else heatmap_16x16
    heatmap = heatmap.astype(np.float32)
    heatmap_up = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_CUBIC)

    hmax = heatmap_up.max()
    heatmap_norm = heatmap_up / hmax if hmax > 0 else heatmap_up

    overlay = cmap(heatmap_norm)
    ov_alpha = overlay[:, :, 3:4] * alpha
    blended = img_arr * (1 - ov_alpha) + overlay[:, :, :3] * ov_alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def render_zoomed_patch(img_idx, heatmap_16x16, size=THUMB, pg=None):
    """Zoom into the highest-activating patch region."""
    if pg is None:
        pg = heatmap_patch_grid
    heatmap = heatmap_16x16.numpy() if isinstance(heatmap_16x16, torch.Tensor) else heatmap_16x16
    peak_idx = np.argmax(heatmap)
    peak_row, peak_col = divmod(int(peak_idx), pg)

    img = load_image(img_idx).resize((image_size, image_size), Image.BILINEAR)
    patch_px = image_size // pg
    zoom_patches = int(zoom_slider.value)   # controlled by UI slider
    half = (zoom_patches * patch_px) // 2
    cy = peak_row * patch_px + patch_px // 2
    cx = peak_col * patch_px + patch_px // 2

    y0 = max(0, cy - half); y1 = min(image_size, cy + half)
    x0 = max(0, cx - half); x1 = min(image_size, cx + half)
    return img.crop((x0, y0, x1, y1)).resize((size, size), Image.BILINEAR)


def _load_image_from_ds(ds, img_i):
    """Like load_image() but uses the given dataset's image_paths."""
    path = ds['image_paths'][img_i]
    fname = os.path.basename(path)
    for base in [args.image_dir] + ([args.extra_image_dir] if args.extra_image_dir else []):
        candidate = os.path.join(base, fname)
        if os.path.exists(candidate):
            return Image.open(candidate).convert("RGB")
    return Image.open(path).convert("RGB")


def _render_overlay_from_ds(ds, feat, slot, size=THUMB, alpha=None):
    """Return (PIL overlay image, caption) for ds/feat/slot, or None on failure."""
    if alpha is None:
        alpha = heatmap_alpha_slider.value
    try:
        img_i = int(ds['top_img_idx'][feat, slot].item())
        if img_i < 0:
            return None
        plain = _load_image_from_ds(ds, img_i).resize((size, size), Image.BILINEAR)
        hm_tensor = ds.get('top_heatmaps')
        if hm_tensor is not None:
            pg = ds.get('heatmap_patch_grid', 16)
            hmap = hm_tensor[feat, slot].float().numpy().reshape(pg, pg)
            img_arr = np.array(plain).astype(np.float32) / 255.0
            hmap_up = cv2.resize(hmap, (size, size), interpolation=cv2.INTER_CUBIC)
            hmax = hmap_up.max()
            hmap_norm = hmap_up / hmax if hmax > 0 else hmap_up
            overlay = ALPHA_JET(hmap_norm)
            ov_a = overlay[:, :, 3:4] * alpha
            blended = np.clip((img_arr * (1 - ov_a) + overlay[:, :, :3] * ov_a) * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(blended), f"img {img_i}"
        return plain, f"img {img_i}"
    except Exception:
        return None


def pil_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _status_html(state, msg):
    """Return a styled HTML status banner."""
    styles = {
        'idle':    'background:#f5f5f5;border-left:4px solid #bbb;color:#666',
        'loading': 'background:#fff8e0;border-left:4px solid #f0a020;color:#7a5000',
        'ok':      'background:#e8f4e8;border-left:4px solid #2a8a2a;color:#1a5a1a',
        'dead':    'background:#fce8e8;border-left:4px solid #c03030;color:#8a1a1a',
    }
    style = styles.get(state, styles['idle'])
    return f'<div style="{style};padding:7px 12px;border-radius:3px;font-size:13px">{msg}</div>'


def make_image_grid_html(images_info, title, cols=9):
    if not images_info:
        return (f'<h3 style="margin:4px 0 6px 0;color:#444;border-bottom:2px solid #e8e8e8;'
                f'padding-bottom:4px">{title}</h3>'
                f'<p style="color:#aaa;font-style:italic;margin:4px 0">No examples available</p>')
    thumb_w = min(THUMB, 224)
    html = (f'<h3 style="margin:4px 0 8px 0;color:#333;border-bottom:2px solid #e0e0e0;'
            f'padding-bottom:4px">{title}</h3>')
    html += '<div style="display:flex;flex-wrap:wrap;gap:8px;padding:2px 0 10px 0">'
    for img, caption in images_info:
        url = pil_to_data_url(img)
        parts = caption.split('<br>')
        cap_html = ''.join(f'<div>{p}</div>' for p in parts)
        html += (f'<div style="text-align:center;width:{thumb_w}px">'
                 f'<img src="{url}" width="{thumb_w}" height="{thumb_w}"'
                 f' style="border:1px solid #d0d0d0;border-radius:5px;display:block"/>'
                 f'<div style="font-size:10px;color:#555;margin-top:3px;line-height:1.4">'
                 f'{cap_html}</div></div>')
    html += '</div>'
    return html


def make_compare_aggregations_html(top_infos, mean_infos, p75_infos, feat, n_each=6, model_label=None):
    """Figure-ready side-by-side comparison of the first two aggregation methods.

    Only Top (Max Activation) and Mean Activation are shown so that a screenshot
    of this element stands alone as a clean figure panel.
    """
    col_thumb = min(THUMB, 160)

    # Only the first two methods are shown in the figure
    sections = [
        ("Top Activation",  "#2563a8", top_infos),
        ("Mean Activation", "#1a7a4a", mean_infos),
    ]

    cols_per_row = 2
    strip_w = cols_per_row * col_thumb + (cols_per_row - 1) * 6

    # Outer container — white background, no border decoration so the figure can
    # be cropped cleanly.  A subtle bottom-padding keeps images from being clipped.
    html = (
        '<div style="font-family:Arial,Helvetica,sans-serif;background:#ffffff;'
        'padding:16px 20px 14px 20px;display:inline-block">'
        # Title row
        f'<div style="font-size:13px;font-weight:bold;color:#222;margin-bottom:14px;'
        f'letter-spacing:0.1px">'
        + (f'{model_label} — ' if model_label else '')
        + f'Feature {feat}</div>'
        '<div style="display:flex;gap:24px;align-items:flex-start">'
    )

    for method_name, color, infos in sections:
        shown = (infos or [])[:n_each]

        html += (
            f'<div style="display:inline-flex;flex-direction:column">'
            # Bold, clearly-coloured column header
            f'<div style="background:{color};color:#ffffff;font-size:13px;font-weight:bold;'
            f'text-align:center;padding:6px 0;border-radius:5px;margin-bottom:10px;'
            f'letter-spacing:0.4px;width:{strip_w}px;box-sizing:border-box">{method_name}</div>'
            f'<div style="display:grid;grid-template-columns:repeat({cols_per_row},{col_thumb}px);gap:6px">'
        )
        if not shown:
            html += '<div style="color:#aaa;font-style:italic;font-size:11px;padding:8px">No images</div>'
        for img, caption in shown:
            url = pil_to_data_url(img)
            parts = caption.split('<br>')
            cap_html = '<br>'.join(parts)
            html += (
                f'<div style="text-align:center">'
                f'<img src="{url}" width="{col_thumb}" height="{col_thumb}"'
                f' style="border:1px solid #ccc;border-radius:3px;display:block"/>'
                f'<div style="font-size:9px;color:#555;margin-top:3px;line-height:1.35">'
                f'{cap_html}</div></div>'
            )
        html += '</div></div>'

    html += '</div></div>'
    return html


def make_cross_sae_comparison_html(ds_a, feat_a, ds_b, feat_b, n=4, size=160, alpha=1.0):
    """
    Two side-by-side 2×2 grids: left = SAE A / feat_a, right = SAE B / feat_b.
    """
    def _collect(ds, feat):
        items = []
        for slot in range(min(n, ds['top_img_idx'].shape[1])):
            result = _render_overlay_from_ds(ds, feat, slot, size=size, alpha=alpha)
            if result:
                items.append(result)
            if len(items) == n:
                break
        return items

    items_a = _collect(ds_a, feat_a)
    items_b = _collect(ds_b, feat_b)

    def _strip_dim(label):
        """Remove parenthetical dimension info like '(d=32K)' or '(d=32K, k=160)'."""
        out = label
        while '(' in out and ')' in out:
            l, r = out.index('('), out.index(')')
            out = out[:l].rstrip() + out[r+1:]
        return out.strip(' —').strip()

    def _grid_html(items, model_label, feat_num, color):
        header = (
            f'<div style="background:{color};color:#fff;text-align:center;'
            f'padding:5px 6px 4px 6px;border-radius:4px;margin-bottom:6px;line-height:1.4">'
            f'<div style="font-size:12px;font-weight:bold">{model_label}</div>'
            f'<div style="font-size:10px;opacity:0.88">Feature {feat_num}</div>'
            f'</div>'
        )
        grid = '<div style="display:grid;grid-template-columns:repeat(2,{s}px);gap:4px">'.format(s=size)
        for img, cap in items:
            url = pil_to_data_url(img)
            grid += (f'<div style="text-align:center">'
                     f'<img src="{url}" width="{size}" height="{size}"'
                     f' style="border:1px solid #ccc;border-radius:3px;display:block"/>'
                     f'<div style="font-size:9px;color:#555;margin-top:2px">{cap}</div></div>')
        grid += '</div>'
        return f'<div style="display:flex;flex-direction:column">{header}{grid}</div>'

    label_a = _strip_dim(ds_a['label'])
    label_b = _strip_dim(ds_b['label'])
    col_a = _grid_html(items_a, label_a, feat_a, "#2563a8")
    col_b = _grid_html(items_b, label_b, feat_b, "#b85c00")

    return (
        '<div style="display:flex;gap:16px;padding:8px;background:#fafafa;'
        'border:1px solid #ddd;border-radius:6px">'
        + col_a + col_b + '</div>'
    )


# ---------- UMAP data source ----------
# live_mask / live_indices / freq / mean_act / log_freq / umap_backup are all
# already set by _apply_dataset_globals(0) above — just build the source from them.
umap_source = ColumnDataSource(data=dict(
    x=umap_coords[live_mask, 0],
    y=umap_coords[live_mask, 1],
    feature_idx=live_indices.tolist(),
    frequency=freq[live_mask].tolist(),
    log_freq=log_freq[live_mask].tolist(),
    mean_act=mean_act[live_mask].tolist(),
))


# ---------- UMAP figure ----------
color_mapper = linear_cmap(
    field_name='log_freq', palette=Turbo256,
    low=0, high=float(np.nanmax(log_freq[live_mask])) if live_mask.any() else 1,
)

umap_fig = figure(
    title="UMAP of SAE Features (by activation pattern)",
    width=700, height=650,
    tools="pan,wheel_zoom,box_zoom,reset,tap",
    active_scroll="wheel_zoom",
)
umap_scatter = umap_fig.scatter(
    'x', 'y', source=umap_source, size=4, alpha=0.6,
    color=color_mapper,
    selection_color="red", selection_alpha=1.0, nonselection_alpha=0.3,
)

# Scale point size with zoom: bigger when zoomed in
_zoom_cb = CustomJS(args=dict(renderer=umap_scatter, x_range=umap_fig.x_range), code="""
    const span = x_range.end - x_range.start;
    if (window._umap_base_span === undefined) {
        window._umap_base_span = span;
    }
    const zoom = window._umap_base_span / span;
    const new_size = Math.min(12, Math.max(3, 3 * Math.pow(zoom, 0.1)));
    renderer.glyph.size = new_size;
    renderer.nonselection_glyph.size = new_size;
    renderer.selection_glyph.size = new_size;
""")
umap_fig.x_range.js_on_change('start', _zoom_cb)
umap_fig.x_range.js_on_change('end', _zoom_cb)

umap_fig.add_tools(HoverTool(tooltips=[
    ("Feature", "@feature_idx"),
    ("Frequency", "@frequency{0}"),
    ("Mean Act", "@mean_act{0.000}"),
]))


# ---------- Dataset / model selector ----------
dataset_select = Select(
    title="Active model:",
    value="0",
    options=[(str(i), ds['label']) for i, ds in enumerate(_all_datasets)],
    width=250,
)


def _on_dataset_switch(attr, old, new):
    global _active_feats
    idx = int(new)
    _ensure_loaded(idx)
    _active[0] = idx
    _apply_dataset_globals(idx)

    # Rebuild UMAP scatter
    umap_source.data = dict(
        x=umap_coords[live_mask, 0],
        y=umap_coords[live_mask, 1],
        feature_idx=live_indices.tolist(),
        frequency=freq[live_mask].tolist(),
        log_freq=log_freq[live_mask].tolist(),
        mean_act=mean_act[live_mask].tolist(),
    )
    umap_source.selected.indices = []
    umap_type_select.value = "Activation Pattern"
    umap_fig.title.text = f"UMAP — {_all_datasets[idx]['label']}"

    # Rebuild feature list
    _search_filter[0] = None
    _apply_order(_get_sorted_order())

    # Rebuild active-feature pool for random button
    _active_feats = [int(i) for i in range(d_model) if feature_frequency[i].item() > 0]

    # Update summary panel
    summary_div.text = _make_summary_html()

    # Show/hide patch explorer depending on token type and data availability.
    ds = _all_datasets[idx]
    has_heatmaps   = ds.get('top_heatmaps') is not None
    has_patch_acts = ds.get('patch_acts') is not None
    can_explore = (
        ds.get('token_type', 'spatial') == 'spatial'
        and (has_heatmaps or has_patch_acts)
    )
    patch_fig.visible = can_explore
    patch_info_div.visible = can_explore
    if not can_explore:
        if ds.get('token_type') == 'cls':
            reason = "CLS token — no patch grid"
        else:
            reason = "no pre-computed heatmaps or patch_acts for this model"
        patch_info_div.text = (
            f'<p style="color:#888;font-style:italic">Patch explorer unavailable: {reason}.</p>')
        patch_info_div.visible = True

    # Update CLIP search hint
    if HAS_CLIP and 'clip_result_div' in dir():
        clip_result_div.text = ""
        clip_result_source.data = dict(
            feature_idx=[], clip_score=[], frequency=[], mean_act=[], name=[])

    # Clear feature display
    feature_input.value = ""
    stats_div.text = "<h3>Select a feature to explore</h3>"
    status_div.text = _status_html('idle', 'Model switched — select a feature to explore.')
    for div in [top_heatmap_div, top_zoom_div,
                mean_heatmap_div, mean_zoom_div,
                p75_heatmap_div, p75_zoom_div]:
        div.text = ""


dataset_select.on_change('value', _on_dataset_switch)


# ---------- Detail panels ----------
status_div = Div(
    text=_status_html('idle', 'Select a feature on the UMAP or from the list to begin.'),
    width=900,
)
stats_div = Div(text="<h3>Click a feature on the UMAP to explore it</h3>", width=900)
top_heatmap_div = Div(text="", width=900)
top_zoom_div = Div(text="", width=900)
mean_heatmap_div = Div(text="", width=900)
mean_zoom_div = Div(text="", width=900)
p75_heatmap_div = Div(text="", width=900)
p75_zoom_div = Div(text="", width=900)
compare_agg_div = Div(text="", width=1400)   # side-by-side aggregation comparison

# Name editing widget (defined here so update_feature_display can reference it)
name_input = TextInput(
    title="Feature name (auto-saved):",
    placeholder="Enter a name for this feature...",
    width=420,
)

# Gemini auto-interp button
_gemini_api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
gemini_btn = Button(
    label="Label with Gemini",
    width=140,
    button_type="warning",
    disabled=(_gemini_api_key is None),
)
gemini_status_div = Div(text=(
    "<i style='color:#aaa'>No GOOGLE_API_KEY set</i>"
    if _gemini_api_key is None else ""
), width=300)

# Zoom slider — controls neighbourhood size in the zoomed-patch view
zoom_slider = Slider(
    title="Zoom window (patches)", value=3, start=1, end=8, step=1, width=220,
)

# Heatmap opacity slider — controls alpha of the overlay in render_heatmap_overlay
heatmap_alpha_slider = Slider(
    title="Heatmap opacity", value=1.0, start=0.0, end=1.0, step=0.05, width=220,
)

# View selector: which image ranking to show in the detail panel
view_select = Select(
    title="Image ranking:",
    value="Top (max activation)",
    options=["Top (max activation)", "Mean activation", "75th percentile",
             "Compare aggregations"],
    width=220,
)

N_DISPLAY = 9
_render_token = [0]  # incremented on each new feature selection to cancel stale renders


def update_feature_display(feature_idx):
    feat = int(feature_idx)
    _render_token[0] += 1
    my_token = _render_token[0]

    freq_val = feature_frequency[feat].item()
    mean_val = feature_mean_act[feat].item()
    p75_val = feature_p75_val[feat].item()
    dead = "DEAD FEATURE" if freq_val == 0 else ""

    feat_name = feature_names.get(feat, "")
    auto_name = auto_interp_names.get(feat, "")
    name_parts = []
    if feat_name:
        name_parts.append(
            f'<div style="color:#1a6faf;font-style:italic;margin:2px 0 3px 0">'
            f'&#x1F3F7;&#xFE0E; {feat_name}'
            f'<span style="font-size:10px;color:#999;margin-left:6px">(manual)</span></div>'
        )
    if auto_name:
        name_parts.append(
            f'<div style="color:#5a9a5a;font-style:italic;margin:2px 0 3px 0">'
            f'&#x1F916; {auto_name}'
            f'<span style="font-size:10px;color:#999;margin-left:6px">(auto-interp)</span></div>'
        )
    name_display = "".join(name_parts)

    stats_div.text = f"""
    <h2 style="margin:4px 0">Feature {feat} <span style="color:red">{dead}</span></h2>
    {name_display}
    <table style="font-size:14px;border-collapse:collapse;">
    <tr><td style="padding:2px 12px"><b>Patch activation count:</b></td><td>{int(freq_val):,}</td></tr>
    <tr><td style="padding:2px 12px"><b>Mean activation:</b></td><td>{mean_val:.4f}</td></tr>
    <tr><td style="padding:2px 12px"><b>75th pctl value:</b></td><td>{p75_val:.4f}</td></tr>
    </table>
    """
    name_input.value = feat_name

    if freq_val == 0:
        status_div.text = _status_html(
            'dead', f'Feature {feat} is dead — it never activated on the precompute set.')
        for div in [top_heatmap_div, top_zoom_div,
                    mean_heatmap_div, mean_zoom_div,
                    p75_heatmap_div, p75_zoom_div,
                    compare_agg_div]:
            div.text = ""
        return

    status_div.text = _status_html(
        'loading', f'&#x23F3; Rendering heatmaps for feature {feat}...')

    def _render():
        # Bail out if the user has already clicked a different feature.
        if _render_token[0] != my_token:
            return

        def _patch_stats(hmap_flat):
            max_act = float(hmap_flat.max())
            activating = hmap_flat[hmap_flat > 0]
            mean_act_val = float(activating.mean()) if len(activating) > 0 else 0.0
            return max_act, mean_act_val

        def _render_one(img_idx_tensor, act_tensor, ranking_idx, heatmap_tensor=None):
            img_i = img_idx_tensor[feat, ranking_idx].item()
            if img_i < 0:
                return None, None
            try:
                # Use pre-computed heatmap
                if heatmap_tensor is not None and heatmap_patch_grid > 1:
                    hmap = heatmap_tensor[feat, ranking_idx].float().numpy()
                    hmap = hmap.reshape(heatmap_patch_grid, heatmap_patch_grid)
                else:
                    hmap = None

                if hmap is None:
                    plain = load_image(img_i).resize((THUMB, THUMB), Image.BILINEAR)
                    act_val = float(act_tensor[feat, ranking_idx].item())
                    caption = f"act={act_val:.4f}  img {img_i}"
                    return (plain, caption), (plain, caption)
                max_act, mean_act_val = _patch_stats(hmap.flatten())
                overlay = render_heatmap_overlay(img_i, hmap, size=THUMB,
                                                   alpha=heatmap_alpha_slider.value)
                caption = f"img {img_i}"
                return (overlay, caption), (render_zoomed_patch(img_i, hmap, size=THUMB), caption)
            except Exception as e:
                ph = Image.new("RGB", (THUMB, THUMB), "gray")
                return (ph, f"Error: {e}"), (ph, f"Error: {e}")

        # --- Top images ---
        heatmap_infos, zoom_infos = [], []
        for j in range(min(N_DISPLAY, top_img_idx.shape[1])):
            hm, zm = _render_one(top_img_idx, top_img_act, j, top_heatmaps)
            if hm is None:
                break
            heatmap_infos.append(hm)
            zoom_infos.append(zm)

        top_heatmap_div.text = make_image_grid_html(
            heatmap_infos, f"Top by Max Activation — Heatmap overlay (feature {feat})")
        top_zoom_div.text = make_image_grid_html(
            zoom_infos, f"Top by Max Activation — Zoomed to peak patch (feature {feat})")

        # --- Mean-ranked images ---
        mean_hm_infos, mean_zm_infos = [], []
        for j in range(min(N_DISPLAY, mean_img_idx.shape[1])):
            hm, zm = _render_one(mean_img_idx, mean_img_act, j, mean_heatmaps)
            if hm is None:
                break
            mean_hm_infos.append(hm)
            mean_zm_infos.append(zm)

        mean_heatmap_div.text = make_image_grid_html(
            mean_hm_infos, f"Top by Mean Activation — Heatmap overlay (feature {feat})")
        mean_zoom_div.text = make_image_grid_html(
            mean_zm_infos, f"Top by Mean Activation — Zoomed to peak patch (feature {feat})")

        # --- 75th percentile images ---
        p75_hm_infos = []
        p75_zm_infos = []
        for j in range(min(N_DISPLAY, p75_img_idx.shape[1])):
            img_i = p75_img_idx[feat, j].item()
            if img_i < 0:
                break
            act = p75_img_act[feat, j].item()
            if act == 0:
                continue
            hm, zm = _render_one(p75_img_idx, p75_img_act, j, p75_heatmaps)
            if hm is None:
                break
            p75_hm_infos.append(hm)
            p75_zm_infos.append(zm)

        p75_heatmap_div.text = make_image_grid_html(
            p75_hm_infos, f"75th Percentile — Heatmap overlay (feature {feat})")
        p75_zoom_div.text = make_image_grid_html(
            p75_zm_infos, f"75th Percentile — Zoomed to peak patch (feature {feat})")

        # Side-by-side aggregation comparison (paper-ready screenshot view)
        compare_agg_div.text = make_compare_aggregations_html(
            heatmap_infos, mean_hm_infos, p75_hm_infos, feat,
            model_label=_all_datasets[_active[0]]['label'])

        status_div.text = _status_html('ok', f'&#x2713; Feature {feat} ready.')
        _update_view_visibility()

    curdoc().add_next_tick_callback(_render)


# ---------- View visibility ----------
def _update_view_visibility():
    v = view_select.value
    is_compare = (v == "Compare aggregations")
    top_heatmap_div.visible  = (v == "Top (max activation)")
    top_zoom_div.visible     = (v == "Top (max activation)")
    mean_heatmap_div.visible = (v == "Mean activation")
    mean_zoom_div.visible    = (v == "Mean activation")
    p75_heatmap_div.visible  = (v == "75th percentile")
    p75_zoom_div.visible     = (v == "75th percentile")
    compare_agg_div.visible  = is_compare

view_select.on_change('value', lambda attr, old, new: _update_view_visibility())
_update_view_visibility()  # set initial state


def _on_zoom_change(attr, old, new):
    """Re-render the current feature when zoom window changes."""
    try:
        feat = int(feature_input.value)
        if 0 <= feat < d_model:
            update_feature_display(feat)
    except ValueError:
        pass

zoom_slider.on_change('value', _on_zoom_change)


def _on_alpha_change(attr, old, new):
    """Re-render the current feature when heatmap opacity changes."""
    try:
        feat = int(feature_input.value)
        if 0 <= feat < d_model:
            update_feature_display(feat)
    except ValueError:
        pass

heatmap_alpha_slider.on_change('value', _on_alpha_change)


# ---------- Callbacks ----------
def on_umap_select(attr, old, new):
    if new:
        feature_idx = umap_source.data['feature_idx'][new[0]]
        feature_input.value = str(feature_idx)
        update_feature_display(feature_idx)

umap_source.selected.on_change('indices', on_umap_select)


# UMAP type toggle
umap_type_select = Select(
    title="UMAP Type", value="Activation Pattern",
    options=["Activation Pattern", "Dictionary Geometry"], width=200,
)

def on_umap_type_change(attr, old, new):
    if new == "Activation Pattern":
        umap_source.data = dict(
            x=umap_backup['act_x'],
            y=umap_backup['act_y'],
            feature_idx=umap_backup['act_feat'],
            frequency=freq[live_mask].tolist(),
            log_freq=log_freq[live_mask].tolist(),
            mean_act=mean_act[live_mask].tolist(),
        )
        umap_fig.title.text = "UMAP of SAE Features (by activation pattern)"
    else:
        dict_freq = freq[dict_live_mask]
        dict_log_freq = log_freq[dict_live_mask]
        dict_mean_act = mean_act[dict_live_mask]
        umap_source.data = dict(
            x=umap_backup['dict_x'],
            y=umap_backup['dict_y'],
            feature_idx=umap_backup['dict_feat'],
            frequency=dict_freq.tolist(),
            log_freq=dict_log_freq.tolist(),
            mean_act=dict_mean_act.tolist(),
        )
        umap_fig.title.text = "UMAP of SAE Features (by dictionary geometry)"

umap_type_select.on_change('value', on_umap_type_change)


# Direct feature input
feature_input = TextInput(title="Feature Index:", value="", width=120)
go_button = Button(label="Go", width=60)
random_btn = Button(label="Random", width=70)

def on_go_click():
    try:
        feat = int(feature_input.value)
        if 0 <= feat < d_model:
            update_feature_display(feat)
            feat_list = umap_source.data['feature_idx']
            if feat in feat_list:
                umap_source.selected.indices = [feat_list.index(feat)]
        else:
            stats_div.text = f"<h3>Feature {feat} out of range (0-{d_model-1})</h3>"
    except ValueError:
        stats_div.text = "<h3>Please enter a valid integer</h3>"

go_button.on_click(on_go_click)

_active_feats = [int(i) for i in range(d_model) if feature_frequency[i].item() > 0]


def _on_random():
    import random
    if not _active_feats:
        return
    feat = random.choice(_active_feats)
    feature_input.value = str(feat)
    update_feature_display(feat)
    feat_list = umap_source.data['feature_idx']
    if feat in feat_list:
        umap_source.selected.indices = [feat_list.index(feat)]

random_btn.on_click(_on_random)


# ---------- Sorted feature list ----------
sort_select = Select(
    title="Sort features by:", value="Frequency (high→low)",
    options=[
        "Frequency (high→low)", "Frequency (low→high)",
        "Mean Activation (high→low)", "Mean Activation (low→high)",
        "75th Pctl (high→low)", "75th Pctl (low→high)",
        "Feature Index", "Name (A→Z)",
    ],
    width=220,
)

_init_order = np.argsort(-freq_np)
feature_list_source = ColumnDataSource(data=dict(
    feature_idx=_init_order.tolist(),
    frequency=freq_np[_init_order].tolist(),
    mean_act=mean_act_np[_init_order].tolist(),
    p75_val=p75_np[_init_order].tolist(),
    name=[_display_name(int(i)) for i in _init_order],
))

feature_table = DataTable(
    source=feature_list_source,
    columns=[
        TableColumn(field="feature_idx", title="Feature", width=60),
        TableColumn(field="frequency", title="Freq", width=70,
                    formatter=NumberFormatter(format="0,0")),
        TableColumn(field="mean_act", title="Mean Act", width=80,
                    formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="p75_val", title="P75", width=70,
                    formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="name", title="Name", width=200),
    ],
    width=500, height=500, sortable=False, index_position=None,
)

# Search state: None = no filter, otherwise a set of matching feature indices
_search_filter = [None]


def _get_sorted_order():
    sort_key = sort_select.value
    if sort_key == "Frequency (high→low)":
        order = np.argsort(-freq_np)
    elif sort_key == "Frequency (low→high)":
        order = np.argsort(freq_np)
    elif sort_key == "Mean Activation (high→low)":
        order = np.argsort(-mean_act_np)
    elif sort_key == "Mean Activation (low→high)":
        order = np.argsort(mean_act_np)
    elif sort_key == "75th Pctl (high→low)":
        order = np.argsort(-p75_np)
    elif sort_key == "75th Pctl (low→high)":
        order = np.argsort(p75_np)
    elif sort_key == "Name (A→Z)":
        # Named features first (alphabetical), then unnamed by index
        order = sorted(range(d_model), key=lambda i: (feature_names.get(i, "\xff"), i))
        order = np.array(order)
    else:
        order = np.arange(d_model)
    if _search_filter[0] is not None:
        mask = np.isin(order, list(_search_filter[0]))
        order = order[mask]
    return order


def _apply_order(order):
    feature_list_source.data = dict(
        feature_idx=order.tolist(),
        frequency=freq_np[order].tolist(),
        mean_act=mean_act_np[order].tolist(),
        p75_val=p75_np[order].tolist(),
        name=[_display_name(int(i)) for i in order],
    )


def _resort_feature_list(attr, old, new):
    _apply_order(_get_sorted_order())


def _update_table_names():
    """Refresh the name column after saving or deleting a feature name."""
    order = np.array(feature_list_source.data['feature_idx'])
    feature_list_source.data = dict(
        feature_idx=order.tolist(),
        frequency=freq_np[order].tolist(),
        mean_act=mean_act_np[order].tolist(),
        p75_val=p75_np[order].tolist(),
        name=[_display_name(int(i)) for i in order],
    )


sort_select.on_change('value', _resort_feature_list)

def _on_table_select(attr, old, new):
    if new:
        feat = feature_list_source.data['feature_idx'][new[0]]
        feature_input.value = str(feat)
        update_feature_display(feat)
        feat_list = umap_source.data['feature_idx']
        if feat in feat_list:
            umap_source.selected.indices = [feat_list.index(feat)]

feature_list_source.selected.on_change('indices', _on_table_select)


# ---------- Auto-save name on typing ----------
def on_name_change(attr, old, new):
    try:
        feat = int(feature_input.value)
    except ValueError:
        return
    name = new.strip()
    if name:
        feature_names[feat] = name
    elif feat in feature_names:
        del feature_names[feat]
    _save_names()
    _update_table_names()

name_input.on_change('value', on_name_change)


# ---------- Gemini auto-interp button ----------
_N_GEMINI_IMAGES    = 6
_GEMINI_MODEL       = "gemini-2.5-flash"
_GEMINI_HM_ALPHA    = 0.25   # heatmap overlay opacity sent to Gemini

def _resolve_img_path(stored_path):
    """Mirror the path resolution from auto_interp_vlm.py."""
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    basename = os.path.basename(stored_path)
    for base in filter(None, [args.image_dir, args.extra_image_dir]):
        candidate = os.path.join(base, basename)
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(stored_path):
        return stored_path
    return None


def _gemini_label_thread(feat, mei_items, doc):
    """Run in a worker thread: call Gemini and push the result back to the doc.

    mei_items: list of (path_str, heatmap_np_or_None) where heatmap is (H, W) float32.
    """
    try:
        from google import genai
        from google.genai import types

        SYSTEM_PROMPT = (
            "You are labeling features of a Sparse Autoencoder (SAE) trained on a "
            "vision transformer. Each SAE feature is a sparse direction in activation "
            "space that fires strongly on certain visual patterns."
        )
        USER_PROMPT = (
            "The images below are the top maximally-activating images for one SAE feature. "
            "In 2–5 words, give a precise label for the visual concept this feature detects. "
            "Be specific — prefer 'dog snout close-up' over 'dog', or 'brick wall texture' "
            "over 'texture'. "
            "Reply with ONLY the label, no explanation, no punctuation at the end."
        )

        client = genai.Client(api_key=_gemini_api_key)
        parts = []
        for path, _heatmap in mei_items[:_N_GEMINI_IMAGES]:
            resolved = _resolve_img_path(path)
            if resolved is None:
                continue
            try:
                img = Image.open(resolved).convert("RGB").resize((224, 224), Image.BILINEAR)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
            except Exception:
                continue

        if not parts:
            def _no_images():
                gemini_btn.disabled = False
                gemini_status_div.text = "<span style='color:#c00'>No images could be loaded.</span>"
            doc.add_next_tick_callback(_no_images)
            return

        parts.append(types.Part.from_text(text=USER_PROMPT))
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        label = response.text.strip().strip(".,;:\"'")

        def _apply_label(feat=feat, label=label):
            auto_interp_names[feat] = label
            _save_auto_interp()
            _update_table_names()
            # Refresh the stats panel so the [auto] label appears immediately
            try:
                update_feature_display(feat)
            except Exception:
                pass
            gemini_btn.disabled = False
            gemini_status_div.text = (
                f"<span style='color:#1a6faf'><b>Labeled:</b> {label}</span>"
            )
            print(f"  [Gemini] feat {feat}: {label}")

        doc.add_next_tick_callback(_apply_label)

    except Exception as e:
        err = str(e)
        def _show_err(err=err):
            gemini_btn.disabled = False
            gemini_status_div.text = f"<span style='color:#c00'>Error: {err[:120]}</span>"
            print(f"  [Gemini] feat {feat} error: {err}")
        doc.add_next_tick_callback(_show_err)


def _on_gemini_click():
    try:
        feat = int(feature_input.value)
    except ValueError:
        gemini_status_div.text = "<span style='color:#c00'>Select a feature first.</span>"
        return

    if feature_frequency[feat].item() == 0:
        gemini_status_div.text = "<span style='color:#c00'>Dead feature — no images.</span>"
        return

    n_top_stored = top_img_idx.shape[1]
    mei_items = []
    for j in range(n_top_stored):
        idx = top_img_idx[feat, j].item()
        if idx >= 0:
            hm = None
            if top_heatmaps is not None:
                hm = top_heatmaps[feat, j].float().numpy().reshape(heatmap_patch_grid, heatmap_patch_grid)
            mei_items.append((image_paths[idx], hm))

    if not mei_items:
        gemini_status_div.text = "<span style='color:#c00'>No MEI paths found.</span>"
        return

    gemini_btn.disabled = True
    gemini_status_div.text = "<i style='color:#888'>Calling Gemini…</i>"

    doc = curdoc()
    t = threading.Thread(
        target=_gemini_label_thread,
        args=(feat, mei_items, doc),
        daemon=True,
    )
    t.start()


if _gemini_api_key:
    gemini_btn.on_click(_on_gemini_click)


# ---------- Search by name ----------
search_input = TextInput(
    title="Search feature names:",
    placeholder="Type to search...",
    width=220,
)
search_btn = Button(label="Search", width=70, button_type="primary")
clear_search_btn = Button(label="Clear", width=60)
search_result_div = Div(text="", width=360)


def _do_search():
    query = search_input.value.strip().lower()
    if not query:
        _search_filter[0] = None
        search_result_div.text = ""
        _apply_order(_get_sorted_order())
        return
    matches = {i for i, name in feature_names.items() if query in name.lower()}
    _search_filter[0] = matches
    _apply_order(_get_sorted_order())
    if matches:
        search_result_div.text = (
            f'<span style="color:#1a6faf"><b>{len(matches)}</b> feature(s) matching '
            f'&ldquo;{query}&rdquo;</span>'
        )
    else:
        search_result_div.text = (
            f'<span style="color:#c00">No features named &ldquo;{query}&rdquo;</span>'
        )


def _do_clear_search():
    search_input.value = ""
    _search_filter[0] = None
    search_result_div.text = ""
    _apply_order(_get_sorted_order())


search_btn.on_click(_do_search)
clear_search_btn.on_click(_do_clear_search)


# Summary — regenerated on every dataset switch
def _make_summary_html():
    ds = _all_datasets[_active[0]]
    n_umap_act     = int(live_mask.sum())
    n_live_dict    = int(dict_live_mask.sum())
    n_truly_active = int((freq > 0).sum())
    n_dead         = d_model - n_truly_active
    tok_label = ("CLS global" if ds.get('token_type') == 'cls'
                 else f"{patch_grid}×{patch_grid} = {patch_grid**2} patches")
    backbone_label = ds.get('backbone', 'dinov3').upper()
    clip_label = "yes" if ds['clip_scores'] is not None else "no"
    hm_label   = "yes" if ds.get('top_heatmaps') is not None else "no"
    pa = ds.get('patch_acts')
    pa_label   = f"yes ({len(pa['img_to_row'])} images)" if pa is not None else "no — run --save-patch-acts"
    sae_url    = ds.get('sae_url')
    dl_row     = (f'<tr><td><b>SAE weights:</b></td>'
                  f'<td><a href="{sae_url}" download style="color:#1a6faf">⬇ Download</a></td></tr>'
                  if sae_url else '')
    return f"""
<div style="background:#f0f4f8;padding:12px;border-radius:6px;margin-bottom:8px;">
<h2 style="margin:0 0 8px 0">SAE Feature Explorer</h2>
<table style="font-size:13px;">
<tr><td><b>Active model:</b></td><td><b style="color:#1a6faf">{ds['label']}</b></td></tr>
<tr><td><b>Backbone:</b></td><td>{backbone_label}</td></tr>
<tr><td><b>Token type:</b></td><td>{ds.get('token_type','spatial')}</td></tr>
<tr><td><b>Dictionary size:</b></td><td>{d_model:,}</td></tr>
<tr><td><b>Active (fired ≥1):</b></td><td>{n_truly_active:,} ({100*n_truly_active/d_model:.1f}%)</td></tr>
<tr><td><b>Dead:</b></td><td>{n_dead:,} ({100*n_dead/d_model:.1f}%)</td></tr>
<tr><td><b>Images:</b></td><td>{n_images:,}</td></tr>
<tr><td><b>Tokens/image:</b></td><td>{tok_label}</td></tr>
{dl_row}
</table>
</div>"""

summary_div = Div(text=_make_summary_html(), width=700)


# ---------- Patch Explorer ----------
# Click patches of an image to find the top active SAE features for that region.
# Activations are served from pre-computed sidecars (no GPU required at serve time).

_PATCH_FIG_PX = 400

_pr = [r for r in range(patch_grid) for _ in range(patch_grid)]
_pc = [c for _ in range(patch_grid) for c in range(patch_grid)]

patch_grid_source = ColumnDataSource(data=dict(
    x=[c + 0.5 for c in _pc],
    y=[patch_grid - r - 0.5 for r in _pr],
    row=_pr,
    col=_pc,
))

patch_bg_source = ColumnDataSource(data=dict(
    image=[], x=[0], y=[0], dw=[patch_grid], dh=[patch_grid],
))

patch_fig = figure(
    width=_PATCH_FIG_PX, height=_PATCH_FIG_PX,
    x_range=(0, patch_grid), y_range=(0, patch_grid),
    tools=["tap", "reset"],
    title="Click or drag to paint patch selection",
    toolbar_location="above",
    visible=False,
)

# Paint-on-drag selection: any patch the mouse passes over while the button
# is held gets added to the selection.  We track button state with a
# document-level mousedown/mouseup listener (set up lazily on first move).
_paint_js = CustomJS(args=dict(source=patch_grid_source, pg=patch_grid), code="""
    if (!window._patch_paint_init) {
        window._patch_paint_init = true;
        window._patch_btn_held  = false;
        document.addEventListener('mousedown', () => { window._patch_btn_held = true;  });
        document.addEventListener('mouseup',   () => { window._patch_btn_held = false; });
    }
    if (!window._patch_btn_held) return;

    const x = cb_obj.x, y = cb_obj.y;
    if (x === null || y === null || x < 0 || x >= pg || y < 0 || y >= pg) return;

    const col       = Math.floor(x);
    const row       = pg - 1 - Math.floor(y);
    const flat_idx  = row * pg + col;

    const sel = source.selected.indices.slice();
    if (sel.indexOf(flat_idx) === -1) {
        sel.push(flat_idx);
        source.selected.indices = sel;
    }
""")
patch_fig.js_on_event(MouseMove, _paint_js)
patch_fig.image_rgba(
    source=patch_bg_source,
    image='image', x='x', y='y', dw='dw', dh='dh',
)
patch_fig.rect(
    source=patch_grid_source,
    x='x', y='y', width=0.95, height=0.95,
    fill_color='yellow', fill_alpha=0.0,
    line_color='white', line_alpha=0.35, line_width=0.5,
    selection_fill_color='red', selection_fill_alpha=0.45,
    nonselection_fill_alpha=0.0, nonselection_line_alpha=0.35,
)
patch_fig.axis.visible = False
patch_fig.xgrid.visible = False
patch_fig.ygrid.visible = False

patch_img_input = TextInput(title="Image Index:", value="0", width=120)
load_patch_btn = Button(label="Load Image", width=90, button_type="primary")
clear_patch_btn = Button(label="Clear", width=60)

patch_feat_source = ColumnDataSource(data=dict(
    feature_idx=[], patch_act=[], frequency=[], mean_act=[],
))
patch_feat_table = DataTable(
    source=patch_feat_source,
    columns=[
        TableColumn(field="feature_idx", title="Feature",  width=65),
        TableColumn(field="patch_act",   title="Patch Act", width=85,
                    formatter=NumberFormatter(format="0.0000")),
        TableColumn(field="frequency",   title="Freq",      width=65,
                    formatter=NumberFormatter(format="0,0")),
        TableColumn(field="mean_act",    title="Mean Act",  width=80,
                    formatter=NumberFormatter(format="0.0000")),
    ],
    width=310, height=350, index_position=None, sortable=False, visible=False,
)
patch_info_div = Div(
    text="<i>Load an image, then click patches to find top features.</i>",
    width=310,
)

_current_patch_img = [None]
_current_patch_z = [None]  # cached (n_patches, d_model) for the loaded image


def _pil_to_bokeh_rgba(pil_img, size):
    pil_img = pil_img.resize((size, size), Image.BILINEAR).convert("RGBA")
    arr = np.array(pil_img, dtype=np.uint8)
    out = np.empty((size, size), dtype=np.uint32)
    view = out.view(dtype=np.uint8).reshape((size, size, 4))
    view[:, :, :] = arr
    return out[::-1].copy()


def _do_load_patch_image():
    try:
        img_idx = int(patch_img_input.value)
    except ValueError:
        patch_info_div.text = "<b style='color:red'>Invalid image index</b>"
        return
    if not (0 <= img_idx < n_images):
        patch_info_div.text = f"<b style='color:red'>Index out of range (0–{n_images - 1})</b>"
        return

    _current_patch_img[0] = img_idx
    try:
        pil = load_image(img_idx)
        bokeh_arr = _pil_to_bokeh_rgba(pil, _PATCH_FIG_PX)
        patch_bg_source.data = dict(
            image=[bokeh_arr], x=[0], y=[0], dw=[patch_grid], dh=[patch_grid],
        )
    except Exception as e:
        patch_info_div.text = f"<b style='color:red'>Error loading image: {e}</b>"
        return

    patch_info_div.text = "<i>Loading patch activations...</i>"
    try:
        z_np = compute_patch_activations(img_idx)
        _current_patch_z[0] = z_np
    except Exception as e:
        patch_info_div.text = f"<b style='color:red'>Error: {e}</b>"
        _current_patch_z[0] = None
        return

    patch_fig.visible = True
    patch_grid_source.selected.indices = []
    patch_feat_source.data = dict(feature_idx=[], patch_act=[], frequency=[], mean_act=[])

    if z_np is None:
        patch_feat_table.visible = False
        patch_info_div.text = (
            f"<b style='color:#888'>Image {img_idx} has no pre-computed patch activations "
            f"and no GPU runner is available.  Run precompute_heatmaps.py with "
            f"--save-patch-acts to enable GPU-free exploration for all images.</b>"
        )
        return

    patch_feat_table.visible = True
    _ds = _all_datasets[_active[0]]
    _pa = _ds.get('patch_acts')
    if _pa is not None and img_idx in _pa['img_to_row']:
        source = "patch_acts (complete)"
    else:
        source = "heatmap reconstruction (partial)"
    patch_info_div.text = (
        f"Image {img_idx} loaded ({source}). "
        f"Drag to select a region, or click individual patches."
    )


load_patch_btn.on_click(_do_load_patch_image)


def _do_clear_patches():
    patch_grid_source.selected.indices = []
    patch_feat_source.data = dict(feature_idx=[], patch_act=[], frequency=[], mean_act=[])
    patch_info_div.text = "<i>Selection cleared.</i>"

clear_patch_btn.on_click(_do_clear_patches)


def _get_top_features_for_patches(patch_indices, top_n=20):
    """Sum SAE activations over selected patches; return top features."""
    z_np = _current_patch_z[0]
    if z_np is None:
        return [], [], [], []

    # z_np: (n_patches, d_model) — vectorized sum over selected patches
    z_selected = z_np[patch_indices]          # (n_sel, d_model)
    feat_sums = z_selected.sum(axis=0)        # (d_model,)

    top_feats = np.argsort(-feat_sums)[:top_n]
    top_feats = top_feats[feat_sums[top_feats] > 0]   # keep only nonzero

    feats = top_feats.tolist()
    acts  = feat_sums[top_feats].tolist()
    freqs = [int(feature_frequency[f].item()) for f in feats]
    means = [float(feature_mean_act[f].item()) for f in feats]
    return feats, acts, freqs, means


def _on_patch_select(attr, old, new):
    if _current_patch_img[0] is None:
        return
    if not new:
        patch_feat_source.data = dict(feature_idx=[], patch_act=[], frequency=[], mean_act=[])
        patch_info_div.text = "<i>Selection cleared.</i>"
        return

    # Convert selected rect indices to flat patch indices
    rows = [patch_grid_source.data['row'][i] for i in new]
    cols = [patch_grid_source.data['col'][i] for i in new]
    patch_indices = [r * patch_grid + c for r, c in zip(rows, cols)]

    feats, acts, freqs, means = _get_top_features_for_patches(patch_indices)
    patch_feat_source.data = dict(feature_idx=feats, patch_act=acts, frequency=freqs, mean_act=means)
    patch_info_div.text = (
        f"{len(new)} patch(es) selected → {len(feats)} feature(s) found. "
        f"Click a row below to explore the feature."
    )

patch_grid_source.selected.on_change('indices', _on_patch_select)


def _on_patch_feat_table_select(attr, old, new):
    if not new:
        return
    feat = patch_feat_source.data['feature_idx'][new[0]]
    feature_input.value = str(feat)
    update_feature_display(feat)
    feat_list = umap_source.data['feature_idx']
    if feat in feat_list:
        umap_source.selected.indices = [feat_list.index(feat)]

patch_feat_source.selected.on_change('indices', _on_patch_feat_table_select)


# ---------- CLIP Text Search ----------
if HAS_CLIP:
    clip_query_input = TextInput(
        title="Search features by text (CLIP):",
        placeholder="e.g. 'dog', 'red stripes', 'water'...",
        width=280,
    )
    clip_search_btn   = Button(label="Search", width=70, button_type="primary")
    clip_result_div   = Div(text="", width=360)
    clip_top_k_input  = TextInput(title="Top-K results:", value="20", width=70)

    clip_result_source = ColumnDataSource(data=dict(
        feature_idx=[], clip_score=[], frequency=[], mean_act=[], name=[],
    ))
    clip_result_table = DataTable(
        source=clip_result_source,
        columns=[
            TableColumn(field="feature_idx", title="Feature", width=65),
            TableColumn(field="clip_score",  title="CLIP score", width=85,
                        formatter=NumberFormatter(format="0.0000")),
            TableColumn(field="frequency",   title="Freq",  width=65,
                        formatter=NumberFormatter(format="0,0")),
            TableColumn(field="mean_act",    title="Mean Act", width=80,
                        formatter=NumberFormatter(format="0.0000")),
            TableColumn(field="name",        title="Name", width=160),
        ],
        width=470, height=300, index_position=None, sortable=False,
    )

    def _do_clip_search():
        query = clip_query_input.value.strip()
        if not query:
            clip_result_div.text = "<i>Enter a text query above.</i>"
            return

        try:
            top_k = max(1, int(clip_top_k_input.value))
        except ValueError:
            top_k = 20

        # Check if query matches a vocab term exactly (case-insensitive)
        vocab_lower = [v.lower() for v in (_clip_vocab or [])]
        if _clip_vocab and query.lower() in vocab_lower:
            col = vocab_lower.index(query.lower())
            scores_vec = _clip_scores_f32[:, col]
        elif _clip_embeds is not None:
            # Free-text: encode on-the-fly with CLIP, dot with feature image embeds
            clip_result_div.text = "<i>Encoding query with CLIP…</i>"
            try:
                clip_m, clip_p, clip_dev = _get_clip()
                q_embed = compute_text_embeddings([query], clip_m, clip_p, clip_dev)
                scores_vec = (_clip_embeds.float() @ q_embed.T).squeeze(-1)
            except Exception as exc:
                clip_result_div.text = f"<span style='color:#c00'>CLIP error: {exc}</span>"
                return
        else:
            clip_result_div.text = (
                f"<span style='color:#c00'>Query not in vocab and no feature embeddings "
                f"available. Try one of: {', '.join((_clip_vocab or [])[:8])}…</span>"
            )
            return

        top_indices = torch.topk(scores_vec, k=min(top_k, len(scores_vec))).indices.tolist()

        clip_result_source.data = dict(
            feature_idx=top_indices,
            clip_score=[float(scores_vec[i]) for i in top_indices],
            frequency=[int(feature_frequency[i].item()) for i in top_indices],
            mean_act=[float(feature_mean_act[i].item()) for i in top_indices],
            name=[_display_name(int(i)) for i in top_indices],
        )
        clip_result_div.text = (
            f'<span style="color:#1a6faf"><b>{len(top_indices)}</b> features for '
            f'&ldquo;{query}&rdquo;</span>'
        )

    clip_search_btn.on_click(_do_clip_search)
    clip_query_input.on_change('value', lambda attr, old, new: None)  # enter triggers nothing; use button

    def _on_clip_result_select(attr, old, new):
        if not new:
            return
        feat = clip_result_source.data['feature_idx'][new[0]]
        feature_input.value = str(feat)
        update_feature_display(feat)
        feat_list = umap_source.data['feature_idx']
        if feat in feat_list:
            umap_source.selected.indices = [feat_list.index(feat)]

    clip_result_source.selected.on_change('indices', _on_clip_result_select)

    clip_search_panel = column(
        row(clip_query_input, clip_top_k_input, clip_search_btn),
        clip_result_div,
        clip_result_table,
    )
else:
    clip_search_panel = Div(
        text="<i style='color:#aaa'>CLIP text search unavailable — "
             "run <code>scripts/add_clip_embeddings.py</code> to enable.</i>",
        width=470,
    )


# ---------- Layout ----------
controls = row(umap_type_select, feature_input, go_button, random_btn)

name_panel = column(
    name_input,
    row(gemini_btn, gemini_status_div),
)

search_panel = column(
    row(search_input, search_btn, clear_search_btn),
    search_result_div,
)

feature_list_panel = column(sort_select, search_panel, feature_table)


def _make_collapsible(title, body, initially_open=False):
    """Wrap a widget in a toggle-able collapsible section."""
    btn = Toggle(
        label=("▼  " if initially_open else "▶  ") + title,
        active=initially_open,
        button_type="light",
        width=500,
        height=30,
    )
    body.visible = initially_open
    btn.js_on_click(CustomJS(args=dict(body=body, btn=btn, title=title), code="""
        body.visible = btn.active;
        btn.label = (btn.active ? '▼  ' : '▶  ') + title;
    """))
    return column(btn, body)


patch_explorer_panel = column(
    row(patch_img_input, load_patch_btn, clear_patch_btn),
    patch_fig,
    patch_info_div,
    patch_feat_table,
)

summary_section = _make_collapsible("SAE Summary",      summary_div)
patch_section   = _make_collapsible("Patch Explorer",   patch_explorer_panel)
clip_section    = _make_collapsible("CLIP Text Search", clip_search_panel)

left_panel = column(dataset_select, controls, umap_fig, feature_list_panel)

middle_panel = column(
    status_div,
    stats_div,
    name_panel,
    row(view_select, zoom_slider, heatmap_alpha_slider),
    compare_agg_div,
    top_heatmap_div, top_zoom_div,
    mean_heatmap_div, mean_zoom_div,
    p75_heatmap_div, p75_zoom_div,
)

# --- Cross-SAE comparison section ---
cmp_ds_a = Select(title="SAE A:", value="0",
    options=[(str(i), ds['label']) for i, ds in enumerate(_all_datasets)])
cmp_feat_a = TextInput(title="Feature (SAE A):", value="0", width=100)
cmp_ds_b = Select(title="SAE B:", value=str(min(1, len(_all_datasets)-1)),
    options=[(str(i), ds['label']) for i, ds in enumerate(_all_datasets)])
cmp_feat_b = TextInput(title="Feature (SAE B):", value="0", width=100)
cmp_alpha_slider = Slider(title="Heatmap opacity", value=1.0, start=0.0, end=1.0, step=0.05, width=220)
cmp_btn = Button(label="Generate Comparison", button_type="primary", width=200)
cmp_output_div = Div(text="", width=400)

def _on_cmp_generate():
    if not cmp_output_div.text:
        return
    try:
        idx_a = int(cmp_ds_a.value)
        idx_b = int(cmp_ds_b.value)
        fa = int(cmp_feat_a.value)
        fb = int(cmp_feat_b.value)
        _ensure_loaded(idx_a)
        _ensure_loaded(idx_b)
        ds_a = _all_datasets[idx_a]
        ds_b = _all_datasets[idx_b]
        cmp_output_div.text = make_cross_sae_comparison_html(
            ds_a, fa, ds_b, fb, alpha=cmp_alpha_slider.value)
    except Exception as e:
        cmp_output_div.text = f'<p style="color:red">Error: {e}</p>'

def _on_cmp_btn():
    try:
        idx_a = int(cmp_ds_a.value)
        idx_b = int(cmp_ds_b.value)
        fa = int(cmp_feat_a.value)
        fb = int(cmp_feat_b.value)
        _ensure_loaded(idx_a)
        _ensure_loaded(idx_b)
        ds_a = _all_datasets[idx_a]
        ds_b = _all_datasets[idx_b]
        cmp_output_div.text = make_cross_sae_comparison_html(
            ds_a, fa, ds_b, fb, alpha=cmp_alpha_slider.value)
    except Exception as e:
        cmp_output_div.text = f'<p style="color:red">Error: {e}</p>'

cmp_btn.on_click(lambda: _on_cmp_btn())
cmp_alpha_slider.on_change('value', lambda attr, old, new: _on_cmp_generate())

cmp_section = _make_collapsible("Cross-SAE Comparison", column(
    row(cmp_ds_a, cmp_feat_a),
    row(cmp_ds_b, cmp_feat_b),
    row(cmp_alpha_slider, cmp_btn),
    cmp_output_div,
))

right_panel = column(summary_section, patch_section, clip_section, cmp_section)

layout = row(left_panel, middle_panel, right_panel)
curdoc().add_root(layout)
curdoc().title = "SAE Feature Explorer"

print("Explorer app ready!")
