"""Sortable feature table + name search panel.

Displays every feature in the active dataset, sorted by frequency
(descending). The user can:
  - click a row to select that feature (drives ``runtime.select_feature``)
  - search by manual or Gemini-auto-interp label substring
  - clear the search to restore the full list

The bootstrap also calls back into this module via
``runtime.apply_order`` / ``runtime.get_sorted_order`` /
``runtime.update_table_names`` to refresh the table after dataset
switches or label edits.
"""

from typing import Callable, Optional

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (Button, ColumnDataSource, DataTable, Div,
                          NumberFormatter, TableColumn, TextInput)

from .. import runtime


def build(display_name: Callable[[int], str],
          on_feature_pick: Optional[Callable[[int], None]] = None):
    """Construct the feature table + search row.

    Args:
        display_name: callable returning the manual-or-auto-interp name
            for a feature id (the bootstrap's ``_display_name``).
        on_feature_pick: callable invoked when the user clicks a row.
            Defaults to ``runtime.select_feature``.

    Returns a dict with the panel layout, the data source, and the
    refresh hooks (``apply_order`` / ``get_sorted_order`` /
    ``update_table_names``) the bootstrap publishes to ``runtime``.
    """
    state = runtime.state
    init_order = np.argsort(-state.freq)

    feature_list_source = ColumnDataSource(data=dict(
        feature_idx=init_order.tolist(),
        frequency=state.freq[init_order].tolist(),
        mean_act=state.mean_act[init_order].tolist(),
        p75_val=state.p75[init_order].tolist(),
        name=[display_name(int(i)) for i in init_order],
    ))

    feature_table = DataTable(
        source=feature_list_source,
        columns=[
            TableColumn(field="feature_idx", title="Feature", width=60),
            TableColumn(field="frequency",   title="Freq", width=70,
                        formatter=NumberFormatter(format="0,0")),
            TableColumn(field="mean_act",    title="Mean Act", width=80,
                        formatter=NumberFormatter(format="0.0000")),
            TableColumn(field="p75_val",     title="P75", width=70,
                        formatter=NumberFormatter(format="0.0000")),
            TableColumn(field="name",        title="Name", width=200),
        ],
        width=500, height=500, sortable=True, index_position=None,
    )

    def get_sorted_order():
        """Return the freq-desc index order, filtered by
        ``runtime.ui.search_filter`` (a set of feat ids) when one is set."""
        s = runtime.state
        order = np.argsort(-s.freq)
        f = runtime.ui.search_filter
        if f is not None:
            order = order[np.isin(order, list(f))]
        return order

    def apply_order(order) -> None:
        s = runtime.state
        feature_list_source.data = dict(
            feature_idx=order.tolist(),
            frequency=s.freq[order].tolist(),
            mean_act=s.mean_act[order].tolist(),
            p75_val=s.p75[order].tolist(),
            name=[display_name(int(i)) for i in order],
        )

    def update_table_names() -> None:
        """Refresh the name column without changing the row order."""
        apply_order(np.asarray(feature_list_source.data['feature_idx']))

    def _on_table_select(attr, old, new):
        if not new:
            return
        cb = on_feature_pick or runtime.select_feature
        if cb is not None:
            cb(feature_list_source.data['feature_idx'][new[0]])

    feature_list_source.selected.on_change('indices', _on_table_select)

    # Search row
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
            runtime.ui.search_filter = None
            search_result_div.text = ""
            apply_order(get_sorted_order())
            return
        # Search manual labels AND Gemini auto-interp labels -- auto-interp
        # is by far the larger pool on a typical dataset.
        s = runtime.state
        matches = {i for i, name in s.feature_names.items() if query in name.lower()}
        matches |= {i for i, name in s.auto_interp_names.items() if query in name.lower()}
        runtime.ui.search_filter = matches
        apply_order(get_sorted_order())
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
        runtime.ui.search_filter = None
        search_result_div.text = ""
        apply_order(get_sorted_order())

    search_btn.on_click(_do_search)
    clear_search_btn.on_click(_do_clear_search)

    search_panel = column(
        row(search_input, search_btn, clear_search_btn),
        search_result_div,
    )
    layout = column(search_panel, feature_table)

    def on_dataset_changed(idx: int) -> None:
        runtime.ui.search_filter = None
        apply_order(get_sorted_order())

    return {
        'layout':                layout,
        'feature_list_source':   feature_list_source,
        'feature_table':         feature_table,
        'search_input':          search_input,
        'search_result_div':     search_result_div,
        'get_sorted_order':      get_sorted_order,
        'apply_order':           apply_order,
        'update_table_names':    update_table_names,
        'on_dataset_changed':    on_dataset_changed,
    }
