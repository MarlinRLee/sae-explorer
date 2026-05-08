"""Patch Explorer panel.

Click (or drag-paint) patches on a single image to find which SAE
features fire on that region. Activations come from the per-dataset
``patch_acts`` sidecar when available; otherwise the explorer falls back
to heatmap reconstruction.

The panel handles:
  - the patch-grid figure + paint-on-drag CustomJS
  - the load/clear buttons + image index input
  - the resulting top-features table

External hooks:
  - ``runtime.select_feature(feat)``  — clicked when the user picks a
    row from the top-features table.
  - ``runtime.state``  — read for the active dataset's images, dim, etc.
"""

from typing import Callable, Optional

import numpy as np
from bokeh.events import MouseMove
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, CustomJS, DataTable,
                          Div, NumberFormatter, TableColumn, TextInput)
from bokeh.plotting import figure

from .. import runtime
from ..images import _pil_to_bokeh_rgba
from ..activations import compute_patch_activations


_PATCH_FIG_PX = 400


def _patch_grid_arrays(pg: int):
    """Return ``(rows, cols, x, y)`` arrays for the pg×pg patch grid.

    Pure helper; testable without Bokeh.
    """
    rows = [r for r in range(pg) for _ in range(pg)]
    cols = [c for _ in range(pg) for c in range(pg)]
    xs = [c + 0.5 for c in cols]
    ys = [pg - r - 0.5 for r in rows]
    return rows, cols, xs, ys


def build(on_feature_pick: Optional[Callable[[int], None]] = None):
    """Construct the patch-explorer panel.

    Returns a dict with the panel layout, individual widgets the
    bootstrap might want to refer to (e.g. for visibility toggles), and
    an ``on_dataset_changed(idx)`` hook the bootstrap calls when the
    active dataset changes (so the patch grid resizes to the new
    patch_grid).
    """
    pg = runtime.state.patch_grid

    rows, cols, xs, ys = _patch_grid_arrays(pg)
    patch_grid_source = ColumnDataSource(data=dict(x=xs, y=ys, row=rows, col=cols))
    patch_bg_source = ColumnDataSource(data=dict(
        image=[], x=[0], y=[0], dw=[pg], dh=[pg],
    ))

    patch_fig = figure(
        width=_PATCH_FIG_PX, height=_PATCH_FIG_PX,
        x_range=(0, pg), y_range=(0, pg),
        tools=["tap", "reset"],
        title="Click or drag to paint patch selection",
        toolbar_location="above",
        visible=False,
    )

    # Paint-on-drag selection: any patch the mouse passes over while the
    # button is held gets added to the selection. We track button state
    # with a document-level mousedown/mouseup listener (set up lazily on
    # first move).
    paint_js = CustomJS(args=dict(source=patch_grid_source, pg=pg), code="""
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
    patch_fig.js_on_event(MouseMove, paint_js)
    patch_fig.image_rgba(source=patch_bg_source,
                         image='image', x='x', y='y', dw='dw', dh='dh')
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
        width=310, height=350, index_position=None, sortable=True, visible=False,
    )
    patch_info_div = Div(
        text="<i>Load an image, then click patches to find top features.</i>",
        width=310,
    )

    def rebuild_grid(new_pg: int) -> None:
        """Resize the figure + grid source to a new patch_grid size.

        Called on dataset switch when the new dataset's patch grid differs
        from the one the figure was built with. Without this, the
        click→patch index math (which uses pg) silently mis-targets cells.
        """
        rows, cols, xs, ys = _patch_grid_arrays(new_pg)
        patch_grid_source.data = dict(x=xs, y=ys, row=rows, col=cols)
        patch_grid_source.selected.indices = []
        patch_bg_source.data = dict(image=[], x=[0], y=[0], dw=[new_pg], dh=[new_pg])
        patch_fig.x_range.start, patch_fig.x_range.end = 0, new_pg
        patch_fig.y_range.start, patch_fig.y_range.end = 0, new_pg
        runtime.ui.current_patch_img = None
        runtime.ui.current_patch_z = None
        patch_feat_source.data = dict(feature_idx=[], patch_act=[], frequency=[], mean_act=[])
        patch_feat_table.visible = False
        patch_info_div.text = (
            "<i>Load an image, then click patches to find top features.</i>")
        # Keep the JS paint handler in sync with the new grid size
        paint_js.args['pg'] = new_pg

    def _do_load_image():
        state = runtime.state
        try:
            img_idx = int(patch_img_input.value)
        except ValueError:
            patch_info_div.text = "<b style='color:red'>Invalid image index</b>"
            return
        if not (0 <= img_idx < state.n_images):
            patch_info_div.text = (
                f"<b style='color:red'>Index out of range "
                f"(0-{state.n_images - 1})</b>")
            return

        runtime.ui.current_patch_img = img_idx
        try:
            pil = runtime.load_image(img_idx)
            bokeh_arr = _pil_to_bokeh_rgba(pil, _PATCH_FIG_PX)
            patch_bg_source.data = dict(
                image=[bokeh_arr], x=[0], y=[0],
                dw=[state.patch_grid], dh=[state.patch_grid],
            )
        except Exception as e:
            patch_info_div.text = f"<b style='color:red'>Error loading image: {e}</b>"
            return

        patch_info_div.text = "<i>Loading patch activations...</i>"
        try:
            z_np = compute_patch_activations(img_idx)
            runtime.ui.current_patch_z = z_np
        except Exception as e:
            patch_info_div.text = f"<b style='color:red'>Error: {e}</b>"
            runtime.ui.current_patch_z = None
            return

        patch_fig.visible = True
        patch_grid_source.selected.indices = []
        patch_feat_source.data = dict(feature_idx=[], patch_act=[], frequency=[], mean_act=[])

        if z_np is None:
            patch_feat_table.visible = False
            patch_info_div.text = (
                f"<b style='color:#888'>Image {img_idx} has no pre-computed patch "
                f"activations and no GPU runner is available. Run "
                f"precompute_heatmaps.py with --save-patch-acts to enable "
                f"GPU-free exploration for all images.</b>")
            return

        patch_feat_table.visible = True
        ds = state.ds
        pa = ds.get('patch_acts')
        if pa is not None and img_idx in pa['img_to_row']:
            source_label = "patch_acts (complete)"
        else:
            source_label = "heatmap reconstruction (partial)"
        patch_info_div.text = (
            f"Image {img_idx} loaded ({source_label}). "
            f"Drag to select a region, or click individual patches.")

    def _do_clear():
        patch_grid_source.selected.indices = []
        patch_feat_source.data = dict(feature_idx=[], patch_act=[], frequency=[], mean_act=[])
        patch_info_div.text = "<i>Selection cleared.</i>"

    def _top_features_for(patch_indices, top_n: int = 20):
        """Sum SAE activations over selected patches; return top features.

        Pure-ish: reads ``runtime.ui.current_patch_z`` and
        ``runtime.state.feature_frequency`` / ``feature_mean_act`` only.
        """
        z_np = runtime.ui.current_patch_z
        if z_np is None:
            return [], [], [], []
        state = runtime.state
        z_selected = z_np[patch_indices]
        feat_sums = z_selected.sum(axis=0)
        top_feats = np.argsort(-feat_sums)[:top_n]
        top_feats = top_feats[feat_sums[top_feats] > 0]
        feats = top_feats.tolist()
        acts = feat_sums[top_feats].tolist()
        freqs = [int(state.feature_frequency[f].item()) for f in feats]
        means = [float(state.feature_mean_act[f].item()) for f in feats]
        return feats, acts, freqs, means

    def _on_patch_select(attr, old, new):
        if runtime.ui.current_patch_img is None:
            return
        if not new:
            patch_feat_source.data = dict(feature_idx=[], patch_act=[], frequency=[], mean_act=[])
            patch_info_div.text = "<i>Selection cleared.</i>"
            return
        rows = [patch_grid_source.data['row'][i] for i in new]
        cols = [patch_grid_source.data['col'][i] for i in new]
        cur_pg = runtime.state.patch_grid
        patch_indices = [r * cur_pg + c for r, c in zip(rows, cols)]
        feats, acts, freqs, means = _top_features_for(patch_indices)
        patch_feat_source.data = dict(
            feature_idx=feats, patch_act=acts, frequency=freqs, mean_act=means)
        patch_info_div.text = (
            f"{len(new)} patch(es) selected -> {len(feats)} feature(s) found. "
            f"Click a row below to explore the feature.")

    def _on_feat_row_select(attr, old, new):
        if not new:
            return
        cb = on_feature_pick or runtime.select_feature
        if cb is not None:
            cb(patch_feat_source.data['feature_idx'][new[0]])

    load_patch_btn.on_click(_do_load_image)
    clear_patch_btn.on_click(_do_clear)
    patch_grid_source.selected.on_change('indices', _on_patch_select)
    patch_feat_source.selected.on_change('indices', _on_feat_row_select)

    layout = column(
        row(patch_img_input, load_patch_btn, clear_patch_btn),
        patch_fig,
        patch_info_div,
        patch_feat_table,
    )

    def on_dataset_changed(idx: int) -> None:
        rebuild_grid(runtime.state.patch_grid)

    return {
        'layout':              layout,
        'patch_fig':           patch_fig,
        'patch_info_div':      patch_info_div,
        'patch_feat_table':    patch_feat_table,
        'rebuild_grid':        rebuild_grid,
        'on_dataset_changed':  on_dataset_changed,
    }
