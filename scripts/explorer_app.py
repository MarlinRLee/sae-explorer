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

The model list (primary + compares) comes from a YAML registry — see
``configs/models.yaml`` and ``scripts/explorer/registry.py``.

Launch:
    bokeh serve scripts/explorer_app.py --port 5006 \
        --allow-websocket-origin="*" --session-token-expiration 86400 \
        --args \
          --registry  configs/models.yaml \
          --data-dir  ./local_data \
          --image-dir ./local_images

Open: http://localhost:5006/explorer_app
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
# Make the sibling `explorer/` package and the project's src/ directory both
# importable from this entry script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'src'))
from clip_utils import load_clip, compute_text_embeddings
from explorer import images as _images_mod
from explorer.images import (
    ALPHA_JET, ALPHA_VIRIDIS, create_alpha_cmap,
    _resolve_image_path, _open_image, _load_image_from_ds,
    pil_to_data_url, _pil_to_bokeh_rgba, _missing_image_warned,
)
from explorer.state import _build_parser, _parse_args, _validate_args, _UI, _State
from explorer.persistence import _save_json_atomic, _archive_label, _now_iso
from explorer import persistence as _persistence_mod
from explorer.html_views import (
    _status_html, make_image_grid_html, make_compare_aggregations_html,
    make_cross_sae_comparison_html,
)
from explorer import runtime as _runtime
from explorer.runtime import load_image
from explorer.activations import _reconstruct_z_from_heatmaps, compute_patch_activations
from explorer.rendering import (
    _render_executor, _umap_mei_cache,
    UMAP_THUMB_PX, UMAP_THUMBS_PER_FEAT,
    render_heatmap_overlay, _render_overlay_from_ds,
    _mei_data_urls, _prewarm_hover_cache_async,
)

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
# `_build_parser`, `_parse_args`, `_validate_args` live in explorer/state.py.
# Bokeh serve runs this file as a script, so parsing at import is unavoidable.
args = _parse_args()
_validate_args(args)


# ---------- Ephemeral UI session state ----------
# `_UI` is defined in explorer/state.py and imported above.
ui = _UI()


# `_render_executor`, `render_heatmap_overlay`, `_mei_data_urls`,
# `_prewarm_hover_cache_async`, `_render_overlay_from_ds`,
# `_umap_mei_cache`, `UMAP_THUMB_PX`, `UMAP_THUMBS_PER_FEAT` all live in
# explorer/rendering.py and are imported above.


# `_get_clip` lives in explorer/clip_loader.py.
from explorer.clip_loader import get_clip as _get_clip


# ---------- Load all datasets into a unified list ----------
# Heavy lifting (JSON sidecars, .pt sidecars, lazy-load placeholders) lives
# in explorer/loaders.py. The registry drives the model list.
from explorer.registry import load_registry
from explorer.loaders import build_dataset_list, ensure_loaded as _loaders_ensure_loaded

_registry = load_registry(args.registry)
_all_datasets = build_dataset_list(_registry, args.data_dir, names_file=args.names_file)


def _ensure_loaded(idx):
    """Load dataset at idx if it is still a lazy placeholder."""
    _loaders_ensure_loaded(_all_datasets, idx)


# `_State` is defined in explorer/state.py and imported above.
state = _State(_all_datasets)
state.apply(0)

# Publish the runtime references so other explorer/ modules can late-bind
# to the active state, ui, and datasets without import cycles.
_runtime.args = args
_runtime.state = state
_runtime.ui = ui
_runtime.datasets = _all_datasets
_runtime.ensure_loaded = _ensure_loaded


# `_save_json_atomic`, `_schedule_hf_push`, `_archive_label`, `_now_iso`
# all live in explorer/persistence.py. The thin wrappers below tie the
# save primitives to the active `state` and `ui`. Each save writes the
# labels JSON, the matching authors sidecar (who created each label),
# and the history sidecar (every label that has been overwritten or
# deleted) so the history is preserved without showing stale entries
# in the UI.
def _save_names():
    _save_json_atomic(state.names_file,
                      {str(k): v for k, v in sorted(state.feature_names.items())})
    _save_json_atomic(state.feature_name_authors_file,
                      {str(k): v for k, v in sorted(state.feature_name_authors.items())})
    _save_json_atomic(state.feature_names_history_file,
                      {str(k): v for k, v in sorted(state.feature_names_history.items())})
    print(f"Saved {len(state.feature_names)} feature names to {state.names_file}")
    _persistence_mod._schedule_hf_push(
        [state.names_file,
         state.feature_name_authors_file,
         state.feature_names_history_file], ui)


def _save_auto_interp():
    _save_json_atomic(state.auto_interp_file,
                      {str(k): v for k, v in sorted(state.auto_interp_names.items())})
    _save_json_atomic(state.auto_interp_authors_file,
                      {str(k): v for k, v in sorted(state.auto_interp_authors.items())})
    _save_json_atomic(state.auto_interp_history_file,
                      {str(k): v for k, v in sorted(state.auto_interp_history.items())})
    print(f"Saved {len(state.auto_interp_names)} auto-interp labels to {state.auto_interp_file}")
    _persistence_mod._schedule_hf_push(
        [state.auto_interp_file,
         state.auto_interp_authors_file,
         state.auto_interp_history_file], ui)


def _display_name(feat: int) -> str:
    """Return the label to show in tables: manual label takes priority over auto-interp."""
    m = state.feature_names.get(feat)
    if m:
        return m
    a = state.auto_interp_names.get(feat)
    return f"[auto] {a}" if a else ""


# `_reconstruct_z_from_heatmaps` and `compute_patch_activations` live in
# explorer/activations.py and are imported above.


# ---------- Image helpers ----------
# The pure helpers (_resolve_image_path, _open_image, _load_image_from_ds,
# pil_to_data_url, _pil_to_bokeh_rgba, ALPHA_JET/VIRIDIS, create_alpha_cmap,
# _missing_image_warned) live in explorer/images.py and are imported at the
# top of this file. Only `load_image` stays here because it closes over
# `state.image_paths` (which is still module-level during the refactor).
# `images.IMAGE_DIRS` and `images.THUMB` are configured below from `args`.
_images_mod.IMAGE_DIRS = tuple(d for d in (args.image_dir, args.extra_image_dir) if d)
_images_mod.THUMB = args.thumb_size
THUMB = args.thumb_size  # local alias for legacy in-file references
# `load_image` is imported from explorer.runtime above (it uses runtime.state).


# Heatmap-overlay + hover-thumbnail rendering helpers all live in
# explorer/rendering.py (imported above).


# `_status_html`, `make_image_grid_html`, `make_compare_aggregations_html`
# live in explorer/html_views.py and are imported above.


# `make_cross_sae_comparison_html` lives in explorer/html_views.py.


# ---------- UMAP data source ----------
# live_mask / live_indices / freq / mean_act / log_freq / umap_backup are all
# already set by state.apply(0) above — just build the source from them.
umap_source = ColumnDataSource(data=dict(
    x=state.umap_coords[state.live_mask, 0],
    y=state.umap_coords[state.live_mask, 1],
    feature_idx=state.live_indices.tolist(),
    frequency=state.freq[state.live_mask].tolist(),
    log_freq=state.log_freq[state.live_mask].tolist(),
    mean_act=state.mean_act[state.live_mask].tolist(),
))


# ---------- UMAP figure ----------
color_mapper = linear_cmap(
    field_name='log_freq', palette=Turbo256,
    low=0, high=float(np.nanmax(state.log_freq[state.live_mask])) if state.live_mask.any() else 1,
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

# ---------- Hover tooltip with MEI thumbnails ----------
# An invisible quad covers the entire UMAP. Its single source row carries
# the pre-rendered HTML for the 3 nearest features, recomputed by the
# Python MouseMove handler. HoverTool is bound *only* to this quad so
# exactly ONE tooltip ever fires — no stats-without-images bleed-through
# from other UMAP points, no chase-the-cursor hit-test lag.
UMAP_HOVER_MAX = 3

hover_target_source = ColumnDataSource(data=dict(html=['']))
hover_target_quad = umap_fig.quad(
    left=-1e9, right=1e9, top=1e9, bottom=-1e9,
    source=hover_target_source,
    fill_alpha=0.0, line_alpha=0.0,
    level="underlay",
)
# Exclude the quad from auto-range — only the scatter should drive the view.
umap_fig.x_range.renderers = [umap_scatter]
umap_fig.y_range.renderers = [umap_scatter]

umap_fig.add_tools(HoverTool(
    tooltips="@html{safe}",
    renderers=[hover_target_quad],
    point_policy="follow_mouse",
    attachment="right",
))

# `_umap_mei_cache` lives in explorer/rendering.py and is imported above.
# `ui.umap_last_hover_feats` holds the last rendered tuple.


def _feature_block_html(f):
    urls = _umap_mei_cache.get(f)
    if urls is None:
        urls = _mei_data_urls(f)
        _umap_mei_cache[f] = urls
    freq_v = int(state.freq[f])
    ma_v = float(state.mean_act[f])
    name = state.feature_names.get(f, "") or state.auto_interp_names.get(f, "")
    name_html = (f'<div style=\"font-size:11px;color:#1a6faf\">{name}</div>'
                 if name else '')
    imgs_html = ''.join(
        f'<img src=\"{u}\" width=\"{UMAP_THUMB_PX}\" height=\"{UMAP_THUMB_PX}\"'
        f' style=\"border:1px solid #ccc;border-radius:4px;display:block;flex-shrink:0\"/>'
        for u in urls
    ) or '<div style=\"font-size:10px;color:#999\">no MEIs</div>'
    return (
        f'<div style=\"display:flex;align-items:center;gap:8px;'
        f'padding:4px 6px;border:1px solid #e6e6e6;border-radius:5px;'
        f'background:#fafafa;margin-bottom:4px\">'
        f'<div style=\"flex:0 0 110px;font-size:11px;line-height:1.4\">'
        f'<div><b>feat {f}</b></div>{name_html}'
        f'<div style=\"color:#666\">freq={freq_v}</div>'
        f'<div style=\"color:#666\">mean act={ma_v:.3f}</div>'
        f'</div>'
        f'<div style=\"display:flex;gap:4px;flex-shrink:0\">{imgs_html}</div>'
        f'</div>'
    )


def _on_umap_mousemove(event):
    """Move the hover-target glyph to the cursor and rebuild its HTML for the
    3 nearest features."""
    if event.x is None or event.y is None:
        return

    xs = np.asarray(umap_source.data['x'], dtype=np.float32)
    ys = np.asarray(umap_source.data['y'], dtype=np.float32)
    if xs.size == 0:
        return

    feat_idx_arr = np.asarray(umap_source.data['feature_idx'], dtype=np.int64)
    dx = xs - float(event.x)
    dy = ys - float(event.y)
    d2 = dx * dx + dy * dy

    rng_x = (umap_fig.x_range.end or 0) - (umap_fig.x_range.start or 0)
    rng_y = (umap_fig.y_range.end or 0) - (umap_fig.y_range.start or 0)
    plot_span = max(abs(rng_x), abs(rng_y), 1e-6)
    radius2 = (0.05 * plot_span) ** 2

    in_range = np.where(d2 < radius2)[0]
    nearest = in_range[np.argsort(d2[in_range])][:UMAP_HOVER_MAX]
    feats = tuple(int(feat_idx_arr[i]) for i in nearest)

    if not feats:
        if hover_target_source.data['html'][0]:
            hover_target_source.data = dict(html=[''])
            ui.umap_last_hover_feats = ()
        return

    if feats == ui.umap_last_hover_feats:
        return

    html = '<div style=\"font-family:sans-serif\">' + ''.join(
        _feature_block_html(f) for f in feats
    ) + '</div>'
    ui.umap_last_hover_feats = feats
    hover_target_source.data = dict(html=[html])


umap_fig.on_event(MouseMove, _on_umap_mousemove)


# ---------- Dataset / model selector ----------
dataset_select = Select(
    title="Active model:",
    value="0",
    options=[(str(i), ds['label']) for i, ds in enumerate(_all_datasets)],
    width=250,
)


def _on_dataset_switch(attr, old, new):
    idx = int(new)
    # Capture the previously selected feature so we can re-select it in the
    # new dataset if the index is still in range. Same SAE feature index is
    # not semantically the same feature across SAEs, but for users flipping
    # back and forth between two models on a fixed index this is exactly
    # the comparison flow they want.
    try:
        prev_feat = int(feature_input.value) if feature_input.value else None
    except ValueError:
        prev_feat = None

    _ensure_loaded(idx)
    state.apply(idx)

    # Rebuild UMAP scatter
    umap_source.data = dict(
        x=state.umap_coords[state.live_mask, 0],
        y=state.umap_coords[state.live_mask, 1],
        feature_idx=state.live_indices.tolist(),
        frequency=state.freq[state.live_mask].tolist(),
        log_freq=state.log_freq[state.live_mask].tolist(),
        mean_act=state.mean_act[state.live_mask].tolist(),
    )
    umap_source.selected.indices = []
    _umap_mei_cache.clear()
    ui.umap_last_hover_feats = ()
    hover_target_source.data = dict(html=[''])
    umap_type_select.value = "Activation Pattern"
    umap_fig.title.text = f"UMAP — {_all_datasets[idx]['label']}"

    # Rebuild feature list
    ui.search_filter = None
    _apply_order(_get_sorted_order())

    # Rebuild active-feature pool for random button
    ui.active_feats = [int(i) for i in range(state.d_model)
                       if state.feature_frequency[i].item() > 0]

    # Kick off hover-thumbnail prewarm for the new dataset (cancels any
    # in-flight prewarm for the previous dataset).
    _prewarm_hover_cache_async()

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
    # Rebuild the patch grid to match the new dataset's patch_grid (it may
    # differ from the primary dataset, e.g. 14×14 vs 16×16). Without this
    # the figure stays sized for the original grid and the click→patch
    # math goes wrong.
    _rebuild_patch_grid(state.patch_grid)
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

    # Refresh the view-ranking options to drop Mean / P75 when the new
    # dataset has no corresponding heatmap tensor. If the current selection
    # is no longer offered, fall back to Top.
    new_view_opts = _available_view_options()
    view_select.options = new_view_opts
    if view_select.value not in new_view_opts:
        view_select.value = VIEW_TOP

    # Show/hide CLIP widgets and clear stale results for the new dataset.
    _apply_clip_visibility()
    clip_result_div.text = ""
    clip_result_source.data = dict(
        feature_idx=[], clip_score=[], frequency=[], mean_act=[], name=[])

    # Re-select the previously focused feature when the index is still
    # in range; otherwise clear the detail panels.
    if prev_feat is not None and 0 <= prev_feat < state.d_model:
        _select_feature(prev_feat)
    else:
        feature_input.value = ""
        stats_div.text = "<h3>Select a feature to explore</h3>"
        status_div.text = _status_html('idle',
            'Model switched — select a feature to explore.')
        for div in [top_heatmap_div, mean_heatmap_div, p75_heatmap_div]:
            div.text = ""


dataset_select.on_change('value', _on_dataset_switch)


# ---------- Detail panels ----------
status_div = Div(
    text=_status_html('idle', 'Select a feature on the UMAP or from the list to begin.'),
    width=900,
)
stats_div = Div(text="<h3>Click a feature on the UMAP to explore it</h3>", width=900)
top_heatmap_div = Div(text="", width=900)
mean_heatmap_div = Div(text="", width=900)
p75_heatmap_div = Div(text="", width=900)
compare_agg_div = Div(text="", width=1400)   # side-by-side aggregation comparison

# Name editing widget (defined here so update_feature_display can reference it)
name_input = TextInput(
    title="Feature name (auto-saved):",
    placeholder="Enter a name for this feature...",
    width=420,
)

# Hidden author tag attached to every label saved during this session.
# Defaults to a per-session anonymous handle; the user can overwrite it via
# `identity_input` to attribute their work. The author lives in sibling
# *_authors.json files alongside the labels JSONs and is not surfaced in the
# main UI — pulling them is a separate operator action.
import uuid as _uuid
_DEFAULT_IDENTITY = f"anon-{_uuid.uuid4().hex[:12]}"

# Bold the title of the identity input. Bokeh's TextInput.title is plain
# text; CSS via an InlineStyleSheet is the cleanest way to style just the
# label inside the widget's shadow DOM.
from bokeh.models import InlineStyleSheet
_BOLD_TITLE_CSS = InlineStyleSheet(css="""
:host label { font-weight: bold; }
""")

identity_input = TextInput(
    title="Your Name (attached to new manual labels):",
    placeholder=_DEFAULT_IDENTITY,
    value=_DEFAULT_IDENTITY,
    width=260,
    stylesheets=[_BOLD_TITLE_CSS],
)


def _current_manual_author() -> str:
    """Resolve the author to attribute to the next manual label save.

    Falls back to the per-session default if the user has cleared the input.
    """
    val = (identity_input.value or "").strip()
    return val or _DEFAULT_IDENTITY


# Auto-interp labels always get this fixed attribution — the model is the
# author of the text, regardless of which operator triggered the call.
_AUTO_INTERP_AUTHOR = "gemini-2.5-flash"


# Gemini auto-interp button — widgets + worker thread + click handler
# all live in explorer/gemini.py. The bootstrap supplies the
# post-label hook that mutates state (auto_interp_names / authors /
# history) and refreshes the table + detail panel.
from explorer import gemini as _gemini

_gemini_api_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")


def _on_gemini_label_applied(feat, label):
    """Doc-thread callback fired by gemini.py after a successful API call.

    Mutates state, persists JSON, refreshes downstream UI. The Gemini
    module re-enables its button + sets its status div separately.
    """
    old_label = state.auto_interp_names.get(feat)
    if old_label is not None and old_label != label:
        _archive_label(state.auto_interp_history, feat, old_label,
                       state.auto_interp_authors.get(feat))
    state.auto_interp_names[feat] = label
    state.auto_interp_authors[feat] = _AUTO_INTERP_AUTHOR
    _save_auto_interp()
    _update_table_names()
    try:
        update_feature_display(feat)
    except Exception:
        pass


def _current_selected_feat():
    """Late-bound: returns the current ``feature_input.value`` as an int,
    or None if not parseable / not yet constructed. ``feature_input`` is
    a module-level global created later in this file; Python late-binds
    free variables in closures, so this works at click time."""
    try:
        return int(feature_input.value)
    except (ValueError, AttributeError, NameError):
        return None


gemini_btn, gemini_status_div, _on_gemini_click = _gemini.build(
    get_current_feat=_current_selected_feat,
    on_label_applied=_on_gemini_label_applied,
    api_key=_gemini_api_key,
)
_runtime.gemini_click = _on_gemini_click  # exposed so update_feature_display
                                          # can fire auto-Gemini without a
                                          # circular import on _gemini

# Zoom slider — crops the main MEI around the peak patch.
# Defaults to the full grid (no crop = full MEI).
_zoom_max = max(2, int(state.heatmap_patch_grid) if state.heatmap_patch_grid else 16)
zoom_slider = Slider(
    title="MEI zoom window",
    value=_zoom_max, start=1, end=_zoom_max, step=1, width=260,
)

# Heatmap opacity slider — controls alpha of the overlay in render_heatmap_overlay
heatmap_alpha_slider = Slider(
    title="Heatmap opacity", value=1.0, start=0.0, end=1.0, step=0.05, width=220,
)

# View selector: which image ranking to show in the detail panel.
# Mean / p75 options are dropped when their corresponding heatmap tensor is
# missing — without it those views show the same plain images as Top, which
# is misleading.
VIEW_TOP     = "Top (max activation)"
VIEW_MEAN    = "Mean activation"
VIEW_P75     = "75th percentile"
VIEW_COMPARE = "Compare aggregations"


def _available_view_options():
    """Return the list of view options valid for the active dataset."""
    opts = [VIEW_TOP]
    if state.mean_heatmaps is not None:
        opts.append(VIEW_MEAN)
    if state.p75_heatmaps is not None:
        opts.append(VIEW_P75)
    opts.append(VIEW_COMPARE)
    return opts


view_select = Select(
    title="Image ranking:",
    value=VIEW_TOP,
    options=_available_view_options(),
    width=220,
)

N_DISPLAY = 9


def update_feature_display(feature_idx):
    feat = int(feature_idx)
    ui.render_token += 1
    my_token = ui.render_token
    gemini_status_div.text = ""

    freq_val = state.feature_frequency[feat].item()
    mean_val = state.feature_mean_act[feat].item()
    p75_val = state.feature_p75_val[feat].item()
    dead = "DEAD FEATURE" if freq_val == 0 else ""

    feat_name = state.feature_names.get(feat, "")
    auto_name = state.auto_interp_names.get(feat, "")
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
        for div in [top_heatmap_div, mean_heatmap_div, p75_heatmap_div,
                    compare_agg_div]:
            div.text = ""
        return

    status_div.text = _status_html(
        'loading', f'&#x23F3; Rendering heatmaps for feature {feat}...')

    # Snapshot slider values once on the doc thread; the worker threads must
    # not touch Bokeh widgets (Bokeh state isn't thread-safe).
    alpha_v = heatmap_alpha_slider.value
    zoom_v  = int(zoom_slider.value)

    def _render_one(img_idx_tensor, act_tensor, ranking_idx, heatmap_tensor=None):
        """Render one (img, caption) pair. Pure function — safe to call from
        any thread because it doesn't touch Bokeh widgets or `state` mutators.
        Returns None if the slot has no image (img_i < 0)."""
        img_i = img_idx_tensor[feat, ranking_idx].item()
        if img_i < 0:
            return None
        try:
            if heatmap_tensor is not None and state.heatmap_patch_grid > 1:
                hmap = heatmap_tensor[feat, ranking_idx].float().numpy()
                hmap = hmap.reshape(state.heatmap_patch_grid, state.heatmap_patch_grid)
            else:
                hmap = None

            if hmap is None:
                plain = load_image(img_i).resize((THUMB, THUMB), Image.BILINEAR)
                act_val = float(act_tensor[feat, ranking_idx].item())
                return (plain, f"act={act_val:.4f}  img {img_i}")
            overlay = render_heatmap_overlay(
                img_i, hmap, size=THUMB, alpha=alpha_v, zoom_patches=zoom_v,
            )
            return (overlay, f"img {img_i}")
        except Exception as e:
            ph = Image.new("RGB", (THUMB, THUMB), "gray")
            return (ph, f"Error: {e}")

    def _submit_ranking(idx_t, act_t, hm_t, *, skip_zero_act=False):
        """Submit up to N_DISPLAY render jobs to the shared executor without
        waiting. Returns the list of futures in display order."""
        futures = []
        for j in range(min(N_DISPLAY, idx_t.shape[1])):
            if idx_t[feat, j].item() < 0:
                # No more valid slots for this feature.
                break
            if skip_zero_act and act_t[feat, j].item() == 0:
                continue
            futures.append(_render_executor.submit(_render_one, idx_t, act_t, j, hm_t))
        return futures

    def _await(futures):
        """Collect futures, dropping early-out None results."""
        return [r for r in (f.result() for f in futures) if r is not None]

    def _render():
        # Bail out if the user has already clicked a different feature.
        if ui.render_token != my_token:
            return

        # Submit all 27 (3 × N_DISPLAY) jobs first so the executor can run
        # them in parallel; only then await. Wall time goes from 27×t to
        # roughly ceil(27/workers)×t.
        top_futs  = _submit_ranking(state.top_img_idx,  state.top_img_act,  state.top_heatmaps)
        mean_futs = _submit_ranking(state.mean_img_idx, state.mean_img_act, state.mean_heatmaps)
        p75_futs  = _submit_ranking(state.p75_img_idx,  state.p75_img_act,  state.p75_heatmaps,
                                    skip_zero_act=True)

        heatmap_infos = _await(top_futs)
        mean_hm_infos = _await(mean_futs)
        p75_hm_infos  = _await(p75_futs)

        # User clicked a different feature while we were rendering — drop
        # the stale results.
        if ui.render_token != my_token:
            return

        top_heatmap_div.text = make_image_grid_html(
            heatmap_infos, f"Top by Max Activation — Heatmap overlay (feature {feat})")
        mean_heatmap_div.text = make_image_grid_html(
            mean_hm_infos, f"Top by Mean Activation — Heatmap overlay (feature {feat})")
        p75_heatmap_div.text = make_image_grid_html(
            p75_hm_infos, f"75th Percentile — Heatmap overlay (feature {feat})")

        # Side-by-side aggregation comparison (paper-ready screenshot view)
        compare_agg_div.text = make_compare_aggregations_html(
            heatmap_infos, mean_hm_infos, p75_hm_infos, feat,
            model_label=state.ds['label'])

        status_div.text = _status_html('ok', f'&#x2713; Feature {feat} ready.')
        _update_view_visibility()

        # Auto-Gemini: when --auto-gemini is on (default) and the user
        # has provided a Google API key, fire an auto-interp call for any
        # selected feature that has neither a manual name nor an existing
        # auto-interp label. Skipped while another Gemini call is in flight
        # (button disabled). Easy to disable session-wide via
        # `--no-auto-gemini` on the bokeh-serve command line.
        if (args.auto_gemini
                and _gemini_api_key
                and not gemini_btn.disabled
                and feat not in state.auto_interp_names
                and feat not in state.feature_names):
            _on_gemini_click()

    curdoc().add_next_tick_callback(_render)


# ---------- View visibility ----------
def _update_view_visibility():
    v = view_select.value
    top_heatmap_div.visible  = (v == VIEW_TOP)
    mean_heatmap_div.visible = (v == VIEW_MEAN)
    p75_heatmap_div.visible  = (v == VIEW_P75)
    compare_agg_div.visible  = (v == VIEW_COMPARE)

view_select.on_change('value', lambda attr, old, new: _update_view_visibility())
_update_view_visibility()  # set initial state


def _rerender_current_feature(attr, old, new):
    """Re-render the current feature on slider release.

    Bound to `value_throttled` (not `value`) so dragging the zoom or opacity
    slider only triggers a single re-render of all 27 heatmaps when the user
    releases the handle, instead of one render per drag tick.
    """
    try:
        feat = int(feature_input.value)
    except ValueError:
        return
    if 0 <= feat < state.d_model:
        update_feature_display(feat)

zoom_slider.on_change('value_throttled', _rerender_current_feature)
heatmap_alpha_slider.on_change('value_throttled', _rerender_current_feature)


# ---------- Callbacks ----------
def _select_feature(feat):
    """Common entry point for selecting a feature from any UI surface:
    sets the input box, renders the detail panels, syncs the UMAP highlight,
    and points the Cross-SAE Compare 'SAE A' inputs at the current
    feature/dataset. Safe to call from a UMAP-driven callback — re-selecting
    the same index in the source is a no-op."""
    feat = int(feat)
    feature_input.value = str(feat)
    update_feature_display(feat)
    feat_list = umap_source.data['feature_idx']
    if feat in feat_list:
        umap_source.selected.indices = [feat_list.index(feat)]
    # Auto-fill the Cross-SAE Compare 'A' side. SAE B stays sticky so the
    # user can keep comparing against a fixed reference dataset.
    cmp_feat_a.value = str(feat)
    cmp_ds_a.value = str(state.idx)


# Publish to runtime so panels (e.g. clip_search, patch_explorer)
# can drive selection without circular imports.
_runtime.select_feature = _select_feature


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
            x=state.umap_backup['act_x'],
            y=state.umap_backup['act_y'],
            feature_idx=state.umap_backup['act_feat'],
            frequency=state.freq[state.live_mask].tolist(),
            log_freq=state.log_freq[state.live_mask].tolist(),
            mean_act=state.mean_act[state.live_mask].tolist(),
        )
        umap_fig.title.text = "UMAP of SAE Features (by activation pattern)"
    else:
        dict_freq = state.freq[state.dict_live_mask]
        dict_log_freq = state.log_freq[state.dict_live_mask]
        dict_mean_act = state.mean_act[state.dict_live_mask]
        umap_source.data = dict(
            x=state.umap_backup['dict_x'],
            y=state.umap_backup['dict_y'],
            feature_idx=state.umap_backup['dict_feat'],
            frequency=dict_freq.tolist(),
            log_freq=dict_log_freq.tolist(),
            mean_act=dict_mean_act.tolist(),
        )
        umap_fig.title.text = "UMAP of SAE Features (by dictionary geometry)"
    _umap_mei_cache.clear()
    ui.umap_last_hover_feats = ()
    hover_target_source.data = dict(html=[''])

umap_type_select.on_change('value', on_umap_type_change)


# Direct feature input
feature_input = TextInput(title="Feature Index:", value="", width=120)
go_button = Button(label="Go", width=60)
random_btn = Button(label="Random", width=70)
random_unlabeled_btn = Button(label="Random Unlabeled", width=130)

def on_go_click():
    try:
        feat = int(feature_input.value)
    except ValueError:
        stats_div.text = "<h3>Please enter a valid integer</h3>"
        return
    if not (0 <= feat < state.d_model):
        stats_div.text = f"<h3>Feature {feat} out of range (0-{state.d_model-1})</h3>"
        return
    _select_feature(feat)

go_button.on_click(on_go_click)

ui.active_feats = [int(i) for i in range(state.d_model)
                   if state.feature_frequency[i].item() > 0]

# Kick off the initial hover-thumbnail prewarm in a background thread.
_prewarm_hover_cache_async()


def _on_random():
    import random
    if not ui.active_feats:
        return
    _select_feature(random.choice(ui.active_feats))

random_btn.on_click(_on_random)


def _on_random_unlabeled():
    """Pick uniformly from active features that have neither a manual name
    nor a Gemini auto-interp label — useful for chewing through the
    review backlog."""
    import random
    pool = [f for f in ui.active_feats
            if f not in state.feature_names and f not in state.auto_interp_names]
    if not pool:
        status_div.text = _status_html(
            'ok', 'All active features are already labeled.')
        return
    _select_feature(random.choice(pool))

random_unlabeled_btn.on_click(_on_random_unlabeled)


# ---------- Sorted feature list + name search ----------
# Widgets, callbacks, and the get_sorted_order/apply_order/update_table_names
# helpers all live in explorer/panels/feature_list.py. The bootstrap holds
# onto the helpers because _on_dataset_switch / on_name_change /
# Gemini's _apply_label all need to refresh the table after mutations.
from explorer.panels import feature_list as _feature_list_mod
_feature_list = _feature_list_mod.build(display_name=_display_name)
feature_list_source = _feature_list['feature_list_source']
feature_table       = _feature_list['feature_table']
_get_sorted_order   = _feature_list['get_sorted_order']
_apply_order        = _feature_list['apply_order']
_update_table_names = _feature_list['update_table_names']


# ---------- Auto-save name on typing ----------
def on_name_change(attr, old, new):
    try:
        feat = int(feature_input.value)
    except ValueError:
        return
    name = new.strip()
    # Selecting a feature programmatically rewrites name_input.value, which
    # fires this handler. Bail if the field already matches what we have on
    # disk for this feature — otherwise every click would write JSON and
    # schedule an HF push.
    if name == state.feature_names.get(feat, ""):
        return
    # Archive any existing entry before overwriting/deleting so the history
    # file retains every label this feature has ever carried.
    old_label = state.feature_names.get(feat)
    if old_label is not None:
        _archive_label(state.feature_names_history,
                       feat, old_label,
                       state.feature_name_authors.get(feat))
    if name:
        state.feature_names[feat] = name
        state.feature_name_authors[feat] = _current_manual_author()
    elif feat in state.feature_names:
        del state.feature_names[feat]
        state.feature_name_authors.pop(feat, None)
    _save_names()
    _update_table_names()

name_input.on_change('value', on_name_change)


# Gemini auto-interp lives entirely in explorer/gemini.py. Its widgets +
# click handler are constructed up near the rest of the detail-panel
# widgets via `_gemini.build(...)`.


# Name-search widgets live in explorer/panels/feature_list.py (built above).


# Summary panel lives in explorer/panels/summary.py.
from explorer.panels import summary as _summary_mod
_summary = _summary_mod.build()
summary_div = _summary['summary_div']
_make_summary_html = _summary_mod._make_summary_html  # re-exported for legacy refs


# ---------- Patch Explorer ----------
# Widgets + handlers + paint-on-drag CustomJS all live in
# explorer/panels/patch_explorer.py. The bootstrap holds onto the
# refresh hook (``rebuild_grid``) so the dataset-switch path can resize
# the patch grid when the new dataset has a different patch_grid.
from explorer.panels import patch_explorer as _patch_explorer_mod
_patch_explorer = _patch_explorer_mod.build()
patch_fig          = _patch_explorer['patch_fig']
patch_info_div     = _patch_explorer['patch_info_div']
patch_feat_table   = _patch_explorer['patch_feat_table']
_rebuild_patch_grid = _patch_explorer['rebuild_grid']


# ---------- CLIP Text Search ----------
# Widgets + handlers live in explorer/panels/clip_search.py. The bootstrap
# only needs the result data source for the dataset-switch refresh path.
from explorer.panels import clip_search as _clip_search_mod
_clip_search = _clip_search_mod.build(display_name=_display_name)
clip_search_panel    = _clip_search['layout']
clip_result_source   = _clip_search['clip_result_source']
clip_result_div      = _clip_search['clip_result_div']
_apply_clip_visibility = _clip_search['apply_visibility']


# ---------- Layout ----------
controls = row(umap_type_select, feature_input, go_button, random_btn, random_unlabeled_btn)

name_panel = column(
    name_input,
    row(gemini_btn, gemini_status_div),
)

feature_list_panel = _feature_list['layout']


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


summary_section = _make_collapsible("SAE Summary",      summary_div)
patch_section   = _make_collapsible("Patch Explorer",   _patch_explorer['layout'])
clip_section    = _make_collapsible("CLIP Text Search", clip_search_panel)

left_panel = column(
    row(dataset_select, identity_input),
    controls,
    umap_fig,
    feature_list_panel,
)

middle_panel = column(
    status_div,
    stats_div,
    name_panel,
    row(view_select, zoom_slider, heatmap_alpha_slider),
    compare_agg_div,
    top_heatmap_div,
    mean_heatmap_div,
    p75_heatmap_div,
)

# --- Cross-SAE comparison section ---
# Widgets + handlers live in explorer/panels/cross_sae.py. The bootstrap
# only needs `cmp_ds_a` and `cmp_feat_a` — `_select_feature` auto-fills
# them when a feature is selected.
from explorer.panels import cross_sae as _cross_sae_mod
_cross_sae = _cross_sae_mod.build(datasets=_all_datasets,
                                  ensure_loaded=_ensure_loaded)
cmp_ds_a   = _cross_sae['cmp_ds_a']
cmp_feat_a = _cross_sae['cmp_feat_a']
cmp_section = _make_collapsible("Cross-SAE Comparison", _cross_sae['layout'])

right_panel = column(summary_section, patch_section, clip_section, cmp_section)

layout = row(left_panel, middle_panel, right_panel)
curdoc().add_root(layout)
curdoc().title = "SAE Feature Explorer"

print("Explorer app ready!")
