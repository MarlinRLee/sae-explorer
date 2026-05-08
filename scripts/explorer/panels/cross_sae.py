"""Cross-SAE comparison panel.

Side-by-side 2x2 image grids for two (dataset, feature) pairs. The
comparison datasets are loaded lazily on first use via
``runtime.ensure_loaded``. The "SAE A" inputs are auto-filled by
``selection.select_feature`` (via the bootstrap) so the user can compare
the currently-selected feature without re-typing.

Tests can call ``build(datasets=[fake_ds_a, fake_ds_b])`` and inspect the
returned widgets without driving Bokeh's event loop.
"""

from typing import Callable, Optional

from bokeh.layouts import column, row
from bokeh.models import Button, Div, Select, Slider, TextInput

from .. import runtime
from ..html_views import make_cross_sae_comparison_html


def build(datasets: list,
          ensure_loaded: Optional[Callable[[int], None]] = None,
          initial_idx_a: int = 0):
    """Construct the cross-SAE compare panel.

    Args:
        datasets: the lazy-loaded dataset list (typically ``runtime.datasets``).
        ensure_loaded: callable that lazily-loads dataset[idx] on first
            access. Defaults to ``runtime.ensure_loaded`` if None.
        initial_idx_a: dataset index to default 'SAE A' to.

    Returns a dict:
        ``layout``       -- the inner column (wrap in a collapsible later)
        ``cmp_ds_a``     -- exposed so select_feature can auto-fill it
        ``cmp_feat_a``   -- exposed so select_feature can auto-fill it
        ``on_dataset_changed`` -- hook to refresh dropdown options when a
            new dataset gets lazy-loaded
    """
    options = [(str(i), ds['label']) for i, ds in enumerate(datasets)]
    n = len(datasets)

    cmp_ds_a = Select(title="SAE A:", value=str(initial_idx_a), options=options)
    cmp_feat_a = TextInput(title="Feature (SAE A):", value="0", width=100)
    cmp_ds_b = Select(title="SAE B:", value=str(min(1, n - 1)), options=options)
    cmp_feat_b = TextInput(title="Feature (SAE B):", value="0", width=100)
    cmp_alpha_slider = Slider(
        title="Heatmap opacity", value=1.0, start=0.0, end=1.0, step=0.05, width=220,
    )
    cmp_btn = Button(label="Generate Comparison", button_type="primary", width=200)
    cmp_output_div = Div(text="", width=400)

    def _generate(force: bool = False):
        """Render the side-by-side comparison.

        ``force=True`` from the Generate button. Slider on_change passes
        ``force=False`` so dragging opacity doesn't render a comparison
        the user hasn't asked for yet — it only re-renders an existing one.
        """
        if not force and not cmp_output_div.text:
            return
        try:
            idx_a = int(cmp_ds_a.value)
            idx_b = int(cmp_ds_b.value)
            fa = int(cmp_feat_a.value)
            fb = int(cmp_feat_b.value)
            loader = ensure_loaded or runtime.ensure_loaded
            if loader is not None:
                loader(idx_a)
                loader(idx_b)
            cmp_output_div.text = make_cross_sae_comparison_html(
                datasets[idx_a], fa, datasets[idx_b], fb,
                alpha=cmp_alpha_slider.value,
            )
        except Exception as e:
            cmp_output_div.text = f'<p style="color:red">Error: {e}</p>'

    cmp_btn.on_click(lambda: _generate(force=True))
    cmp_alpha_slider.on_change('value', lambda attr, old, new: _generate())

    layout = column(
        row(cmp_ds_a, cmp_feat_a),
        row(cmp_ds_b, cmp_feat_b),
        row(cmp_alpha_slider, cmp_btn),
        cmp_output_div,
    )

    def on_dataset_changed(idx: int) -> None:
        """No-op for now — the dropdown ``options`` already cover every
        dataset. Hook reserved for later if labels change."""
        return

    return {
        'layout':              layout,
        'cmp_ds_a':            cmp_ds_a,
        'cmp_feat_a':          cmp_feat_a,
        'cmp_ds_b':            cmp_ds_b,
        'cmp_feat_b':          cmp_feat_b,
        'cmp_btn':             cmp_btn,
        'cmp_alpha_slider':    cmp_alpha_slider,
        'cmp_output_div':      cmp_output_div,
        'on_dataset_changed':  on_dataset_changed,
    }
