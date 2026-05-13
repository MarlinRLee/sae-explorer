"""CLIP free-text search panel.

Always built — when the active dataset has no CLIP data, the search
widgets are hidden and an "unavailable" banner takes their place. Drives
``runtime.select_feature`` when the user clicks a result row.

Search resolution order:
  1. Exact vocab match (case-insensitive) → use the precomputed
     ``clip_text_scores`` column directly.
  2. Free-text query with a CLIP model available → lazy-load the model
     via :mod:`explorer.clip_loader` and dot with
     ``clip_feature_embeds``.
  3. Otherwise → show an error suggesting the user try a vocab term.

Tests can call ``build()`` and inspect the returned widgets without
driving any actual CLIP load.
"""

from typing import Callable, Optional

import torch
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, DataTable, Div,
                          NumberFormatter, TableColumn, TextInput)

from .. import runtime
from ..clip_loader import get_clip


def build(display_name: Callable[[int], str],
          on_feature_pick: Optional[Callable[[int], None]] = None):
    """Construct the CLIP search panel.

    Args:
        display_name: callable returning the manual-or-auto-interp name
            for a feature id. Bootstrap supplies its ``_display_name``.
        on_feature_pick: callable invoked when the user clicks a result
            row. Defaults to ``runtime.select_feature``.
    """
    clip_query_input = TextInput(
        title="Search features by text (CLIP):",
        placeholder="e.g. 'dog', 'red stripes', 'water'...",
        width=280,
    )
    clip_search_btn = Button(label="Search", width=70, button_type="primary")
    clip_result_div = Div(text="", width=360)
    clip_top_k_input = TextInput(title="Top-K results:", value="20", width=70)

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
        width=470, height=300, index_position=None, sortable=True,
    )

    clip_unavailable_div = Div(
        text="<i style='color:#aaa'>CLIP text search unavailable for this dataset - "
             "run <code>scripts/add_clip_embeddings.py</code> to enable.</i>",
        width=470,
    )

    def _do_search():
        state = runtime.state
        if not state.has_clip:
            clip_result_div.text = (
                "<i style='color:#aaa'>Active dataset has no CLIP embeddings.</i>"
            )
            return

        query = clip_query_input.value.strip()
        if not query:
            clip_result_div.text = "<i>Enter a text query above.</i>"
            return

        try:
            top_k = max(1, int(clip_top_k_input.value))
        except ValueError:
            top_k = 20

        vocab_lower = [v.lower() for v in (state.clip_vocab or [])]
        if state.clip_vocab and query.lower() in vocab_lower:
            col = vocab_lower.index(query.lower())
            scores_vec = state.clip_scores_f32[:, col]
        elif state.clip_embeds is not None:
            clip_result_div.text = "<i>Encoding query with CLIP...</i>"
            try:
                # Imported here so tests don't need transformers on path.
                import sys, os
                _src = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
                if _src not in sys.path:
                    sys.path.insert(0, _src)
                from clip_utils import compute_text_embeddings
                clip_m, clip_p, clip_dev = get_clip()
                q_embed = compute_text_embeddings([query], clip_m, clip_p, clip_dev)
                scores_vec = (state.clip_embeds.float() @ q_embed.T).squeeze(-1)
            except Exception as exc:
                clip_result_div.text = f"<span style='color:#c00'>CLIP error: {exc}</span>"
                return
        else:
            clip_result_div.text = (
                f"<span style='color:#c00'>Query not in vocab and no feature embeddings "
                f"available. Try one of: {', '.join((state.clip_vocab or [])[:8])}...</span>"
            )
            return

        top_indices = torch.topk(scores_vec, k=min(top_k, len(scores_vec))).indices.tolist()
        # Clear prior selection before reassigning data — otherwise the
        # DataTable can fail to repaint its cells (Bokeh 3.x quirk: stale
        # selection state keeps the table visually frozen even though
        # .data is updated and row clicks resolve to the new feature_idx).
        clip_result_source.selected.indices = []
        clip_result_source.data = dict(
            feature_idx=top_indices,
            clip_score=[float(scores_vec[i]) for i in top_indices],
            frequency=[int(state.feature_frequency[i].item()) for i in top_indices],
            mean_act=[float(state.feature_mean_act[i].item()) for i in top_indices],
            name=[display_name(int(i)) for i in top_indices],
        )
        clip_result_div.text = (
            f'<span style="color:#1a6faf"><b>{len(top_indices)}</b> features for '
            f'&ldquo;{query}&rdquo;</span>'
        )

    clip_search_btn.on_click(_do_search)

    def _on_result_select(attr, old, new):
        if not new:
            return
        cb = on_feature_pick or runtime.select_feature
        if cb is not None:
            cb(clip_result_source.data['feature_idx'][new[0]])

    clip_result_source.selected.on_change('indices', _on_result_select)

    def apply_visibility() -> None:
        """Show/hide widgets based on whether the active dataset has CLIP data."""
        available = bool(runtime.state.has_clip)
        clip_unavailable_div.visible = not available
        for w in (clip_query_input, clip_top_k_input, clip_search_btn,
                  clip_result_div, clip_result_table):
            w.visible = available

    layout = column(
        clip_unavailable_div,
        row(clip_query_input, clip_top_k_input, clip_search_btn),
        clip_result_div,
        clip_result_table,
    )
    apply_visibility()  # initial state

    def on_dataset_changed(idx: int) -> None:
        apply_visibility()
        clip_result_div.text = ""
        clip_result_source.data = dict(
            feature_idx=[], clip_score=[], frequency=[], mean_act=[], name=[])

    return {
        'layout':              layout,
        'clip_query_input':    clip_query_input,
        'clip_search_btn':     clip_search_btn,
        'clip_result_div':     clip_result_div,
        'clip_top_k_input':    clip_top_k_input,
        'clip_result_source':  clip_result_source,
        'clip_result_table':   clip_result_table,
        'apply_visibility':    apply_visibility,
        'do_search':           _do_search,
        'on_dataset_changed':  on_dataset_changed,
    }
