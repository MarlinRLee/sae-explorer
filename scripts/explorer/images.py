"""Image loading + colormap helpers used across the explorer.

Pure helpers — no Bokeh, no `state`, no module-level imports of the rest of
the explorer. Designed to be imported early so that downstream modules
(rendering, html_views, selection, panels) can rely on them.

Module-level configuration:
    IMAGE_DIRS  — tuple of image directories searched, in order, by
                  `_resolve_image_path`. Set by the bootstrap after argparse
                  via ``images.IMAGE_DIRS = (primary_dir, extra_dir)``.
    THUMB       — default thumbnail size in pixels. Used as the default for
                  the `size` argument of helpers that emit thumbnails.
                  Set by the bootstrap from ``args.thumb_size``.
"""

import io
import base64

import numpy as np
import matplotlib

# Heatmap colormaps are computed at import time on the Agg backend so this
# module is safe to import on a headless server.
matplotlib.use('Agg')
import matplotlib.colors as mcolors  # noqa: E402  (must follow `use('Agg')`)
import matplotlib.pyplot as plt       # noqa: E402
from PIL import Image                  # noqa: E402


# ---------- Configuration (set by the bootstrap) ----------
# IMAGE_DIRS is the search path used by _resolve_image_path. It must be a
# tuple/list of strings; empty entries are filtered out at call time.
IMAGE_DIRS: tuple = ()

# Default thumbnail size for image grids. Configurable from the CLI via
# --thumb-size on the bootstrap, then assigned here.
THUMB: int = 256


# ---------- Alpha colormaps ----------
def create_alpha_cmap(base: str = 'jet'):
    """Return a copy of ``base`` colormap with linear-alpha replacing solid
    alpha. Used so heatmap overlays fade with intensity instead of clipping
    abruptly at zero."""
    base_cmap = plt.cm.get_cmap(base)
    colors = base_cmap(np.arange(base_cmap.N))
    colors[:, -1] = np.linspace(0.0, 1.0, base_cmap.N)
    return mcolors.LinearSegmentedColormap.from_list('alpha_cmap', colors)


ALPHA_JET = create_alpha_cmap('jet')
ALPHA_VIRIDIS = create_alpha_cmap('viridis')


# ---------- Path resolution ----------
# Stored paths in the explorer_data sidecars reflect the precompute machine.
# Try each dir in IMAGE_DIRS, then the absolute path itself.
# `_missing_image_warned` keeps the warning to once per stored path so a
# missing-file storm can't fill the log.
import os  # noqa: E402  (kept near use site)
_missing_image_warned: set = set()


def _resolve_image_path(stored_path: str):
    """Resolve a stored path against the configured IMAGE_DIRS.

    Returns an existing path or None. Tries each directory in order, then
    falls back to the stored path itself (useful when the precompute and
    serve machines share a filesystem layout).
    """
    fname = os.path.basename(stored_path)
    for base in filter(None, IMAGE_DIRS):
        candidate = os.path.join(base, fname)
        if os.path.exists(candidate):
            return candidate
    if os.path.exists(stored_path):
        return stored_path
    return None


def _open_image(stored_path: str) -> Image.Image:
    """Resolve and open an image. Returns a gray placeholder (with one-time
    log) if the file cannot be found or opened — never raises."""
    resolved = _resolve_image_path(stored_path)
    if resolved is None:
        if stored_path not in _missing_image_warned:
            _missing_image_warned.add(stored_path)
            print(f"  Warning: image not found in any image dir: {stored_path}")
        return Image.new("RGB", (THUMB, THUMB), (180, 180, 180))
    try:
        return Image.open(resolved).convert("RGB")
    except Exception as e:
        print(f"  Warning: could not open {resolved}: {e}")
        return Image.new("RGB", (THUMB, THUMB), (180, 180, 180))


def _load_image_from_ds(ds: dict, img_i: int) -> Image.Image:
    """Open an image given the dataset dict and an image index into its
    ``image_paths`` list. Used by cross-dataset rendering paths where the
    active state is *not* the dataset whose image we need."""
    return _open_image(ds['image_paths'][img_i])


# ---------- Encoding helpers ----------
def pil_to_data_url(img: Image.Image, quality: int = 85) -> str:
    """JPEG-encode + base64 to a `data:` URL suitable for inline <img src>."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _pil_to_bokeh_rgba(pil_img: Image.Image, size: int) -> np.ndarray:
    """Resize a PIL image to (size, size) and return a Bokeh-compatible
    flat uint32 RGBA buffer (top-down → bottom-up flipped). Used by the
    patch explorer to render the active image as a Bokeh image_rgba glyph."""
    pil_img = pil_img.resize((size, size), Image.BILINEAR).convert("RGBA")
    arr = np.array(pil_img, dtype=np.uint8)
    out = np.empty((size, size), dtype=np.uint32)
    view = out.view(dtype=np.uint8).reshape((size, size, 4))
    view[:, :, :] = arr
    return out[::-1].copy()
