"""Heatmap-overlay + hover-thumbnail rendering helpers.

These are the heavy image-rendering primitives the explorer calls on
every feature click and every UMAP hover. They late-bind to the active
``runtime.state`` / ``runtime.ui`` / ``runtime.args`` so they can be
imported during package load (before the bootstrap finishes).

Module-level state living here:

- ``_render_executor`` — a single shared ``ThreadPoolExecutor``. Stateless,
  safe to share across Bokeh sessions. Used to parallelise the 27
  (3 aggregations × N_DISPLAY) heatmap renders triggered by every feature
  click.

- ``_umap_mei_cache`` — feat-id → list of data-URL strings. Read on every
  UMAP hover, populated by :func:`_prewarm_hover_cache_async` and on
  demand by hover handlers. Kept here (not on Context / per-session)
  because warm thumbnails are valid as long as the dataset isn't
  switched, and the dataset-switch path clears the cache explicitly.
"""

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image

from . import runtime
from .images import ALPHA_JET, _load_image_from_ds


# ---------- Shared resources ----------
_render_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="render")
_umap_mei_cache: dict = {}  # feat_idx -> list of up-to-3 data URLs

# Hover-thumbnail constants
UMAP_THUMB_PX = 72
UMAP_THUMBS_PER_FEAT = 3


# ---------- Heatmap-overlay rendering ----------
def render_heatmap_overlay(img_idx: int, heatmap_grid, size: int,
                           cmap=ALPHA_JET, alpha: float = 1.0,
                           zoom_patches=None, pg=None) -> Image.Image:
    """Render an image with a heatmap overlay, optionally cropped around
    the peak patch.

    ``zoom_patches`` is the window size in patch units around the peak
    patch. When ``None`` or >= the patch grid, the full MEI is rendered
    without cropping. ``pg`` defaults to the active state's
    ``heatmap_patch_grid``.
    """
    state = runtime.state
    if pg is None:
        pg = state.heatmap_patch_grid

    heatmap = heatmap_grid.numpy() if isinstance(heatmap_grid, torch.Tensor) else heatmap_grid
    heatmap = heatmap.astype(np.float32)

    full_img = runtime.load_image(img_idx).resize(
        (state.image_size, state.image_size), Image.BILINEAR)

    if zoom_patches is None or zoom_patches >= pg:
        x0, y0, x1, y1 = 0, 0, state.image_size, state.image_size
    else:
        peak_idx = int(np.argmax(heatmap))
        peak_row, peak_col = divmod(peak_idx, pg)
        patch_px = state.image_size // pg
        half = (int(zoom_patches) * patch_px) // 2
        cy = peak_row * patch_px + patch_px // 2
        cx = peak_col * patch_px + patch_px // 2
        y0 = max(0, cy - half); y1 = min(state.image_size, cy + half)
        x0 = max(0, cx - half); x1 = min(state.image_size, cx + half)

    img_crop = full_img.crop((x0, y0, x1, y1)).resize((size, size), Image.BILINEAR)
    img_arr = np.array(img_crop).astype(np.float32) / 255.0

    heatmap_full = cv2.resize(heatmap, (state.image_size, state.image_size),
                              interpolation=cv2.INTER_CUBIC)
    heatmap_crop = heatmap_full[y0:y1, x0:x1]
    heatmap_up = cv2.resize(heatmap_crop, (size, size), interpolation=cv2.INTER_CUBIC)

    hmax = heatmap_up.max()
    heatmap_norm = heatmap_up / hmax if hmax > 0 else heatmap_up

    overlay = cmap(heatmap_norm)
    ov_alpha = overlay[:, :, 3:4] * alpha
    blended = img_arr * (1 - ov_alpha) + overlay[:, :, :3] * ov_alpha
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def _render_overlay_from_ds(ds: dict, feat: int, slot: int,
                            size: int, alpha: float):
    """Return ``(PIL overlay, caption)`` for a (dataset, feat, slot) triple,
    or ``None`` if the slot is empty / rendering fails.

    Note: this is the cross-dataset variant — used for the cross-SAE
    comparison panel — so it reads its dataset dict explicitly rather
    than from the active state. ``alpha`` must be passed by the caller
    (no slider read).
    """
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
            blended = np.clip((img_arr * (1 - ov_a) + overlay[:, :, :3] * ov_a) * 255,
                              0, 255).astype(np.uint8)
            return Image.fromarray(blended), f"img {img_i}"
        return plain, f"img {img_i}"
    except Exception:
        return None


# ---------- UMAP hover thumbnails ----------
def _mei_data_urls(feat: int, n: int = UMAP_THUMBS_PER_FEAT, hm_tensor=None):
    """Return up to ``n`` hover-thumbnail data URLs for a feature.

    Single-pass: load the image straight at thumb resolution, blend the
    heatmap upscaled directly to the same resolution, JPEG-encode at
    lower quality. Avoids the 224-px intermediate that
    :func:`render_heatmap_overlay` performs.
    """
    state = runtime.state
    hm_tensor = hm_tensor if hm_tensor is not None else state.top_heatmaps
    urls = []
    for slot in range(n):
        try:
            if slot >= state.top_img_idx.shape[1]:
                break
            img_i = int(state.top_img_idx[feat, slot].item())
            if img_i < 0:
                continue
            img = runtime.load_image(img_i).resize(
                (UMAP_THUMB_PX, UMAP_THUMB_PX), Image.BILINEAR)
            if hm_tensor is not None and state.heatmap_patch_grid > 1:
                hmap = hm_tensor[feat, slot].float().numpy().reshape(
                    state.heatmap_patch_grid, state.heatmap_patch_grid).astype(np.float32)
                hmap_up = cv2.resize(
                    hmap, (UMAP_THUMB_PX, UMAP_THUMB_PX),
                    interpolation=cv2.INTER_LINEAR)
                hmax = float(hmap_up.max())
                hmap_norm = hmap_up / hmax if hmax > 0 else hmap_up
                overlay = ALPHA_JET(hmap_norm)
                img_arr = np.asarray(img, dtype=np.float32) / 255.0
                ov_a = overlay[:, :, 3:4] * 0.6
                blended = img_arr * (1 - ov_a) + overlay[:, :, :3] * ov_a
                final = Image.fromarray(np.clip(blended * 255, 0, 255).astype(np.uint8))
            else:
                final = img
            buf = io.BytesIO()
            final.save(buf, format="JPEG", quality=65)
            urls.append(
                "data:image/jpeg;base64," +
                base64.b64encode(buf.getvalue()).decode("utf-8")
            )
        except Exception:
            continue
    return urls


def _prewarm_hover_cache_async() -> None:
    """Background-fill ``_umap_mei_cache`` for the top-K most-frequent
    live features so the first hover doesn't stall.

    K is read from ``runtime.args.hover_prewarm_count``. The caller
    (dataset-switch path) is responsible for clearing the cache before
    invoking this. A token bump (``runtime.ui.prewarm_token``) cancels
    any in-flight prewarm so cache entries can never come from a stale
    dataset's tensors.
    """
    args  = runtime.args
    state = runtime.state
    ui    = runtime.ui

    top_k = int(getattr(args, 'hover_prewarm_count', 0))
    if top_k <= 0 or not ui.active_feats:
        return

    ui.prewarm_token += 1
    my_token = ui.prewarm_token

    # Sort once on the calling (doc) thread — quick and avoids touching
    # `state` from the worker.
    feats = sorted(ui.active_feats,
                   key=lambda i: -float(state.feature_frequency[i].item()))[:top_k]
    label = state.ds.get('label', '?')

    def _worker():
        n_done = 0
        for f in feats:
            if ui.prewarm_token != my_token:
                print(f"[prewarm] '{label}' cancelled at {n_done}/{len(feats)}")
                return
            if f in _umap_mei_cache:
                continue
            try:
                _umap_mei_cache[f] = _mei_data_urls(f)
                n_done += 1
            except Exception:
                pass
        print(f"[prewarm] '{label}' done — cached {n_done} hover thumbnails")

    threading.Thread(target=_worker, daemon=True,
                     name=f"prewarm-{my_token}").start()
