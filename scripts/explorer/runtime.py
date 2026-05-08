"""Runtime-mutable references shared across the explorer package.

This is a lightweight stand-in for a full ``Context`` dataclass. Bokeh's
single-file demo idiom means the entry script (``scripts/explorer_app.py``)
is the natural place to construct the active ``_State`` / ``_UI``
instances; package modules that need to read them late-bind through
this module.

Usage:

    # In the bootstrap, after argparse + state construction:
    from explorer import runtime
    runtime.args = args
    runtime.state = state
    runtime.ui = ui
    runtime.datasets = _all_datasets

    # In a package module that needs the active state:
    from explorer import runtime
    def render(...):
        s = runtime.state
        ...

The slots are ``None`` until the bootstrap populates them, so
package-level code must not access them at import time. Read them inside
function bodies that run after the bootstrap has finished.

Per-session caveat: today's Bokeh-serve idiom re-runs ``explorer_app.py``
once per browser session in a fresh module namespace, but the imported
``explorer.*`` modules persist across sessions in the same process. That
means ``runtime.state`` is *shared* across sessions — the last session
to construct ``state`` wins. This matches the pre-refactor monolith's
behaviour for any *module-level* state: the user-visible widgets are
per-session (Bokeh creates them fresh per ``add_root``) but the lazy-load
caches (``_all_datasets``, ``_umap_mei_cache``, the inference cache on
each dataset dict) are shared. Caches sharing across sessions is benign;
the main session-isolation concern is the ``feature_names`` dict, which
is mutated by user input. A future Context refactor would move that
into per-session storage.
"""

# Populated by scripts/explorer_app.py after argparse, state, and dataset
# loading have completed. Modules that read these MUST do so inside
# function bodies (not at import time) — they are ``None`` until the
# bootstrap finishes.
args = None      # argparse.Namespace
state = None     # explorer.state._State
ui = None        # explorer.state._UI
datasets = None  # list of dataset dicts (was _all_datasets in the monolith)


# ---------- Late-bound cross-module callables ----------
# Panel modules call these instead of importing from the bootstrap, which
# would be a hard cycle. The bootstrap assigns them in dependency order
# *after* the relevant function/widget exists. Each is None until then;
# panel callbacks must defer access to call time, never import time.

ensure_loaded = None         # Callable[[int], None]
                             # Lazy-load the dataset at the given index.

select_feature = None        # Callable[[int], None]
                             # Hub for "user wants to look at this feature."
                             # Sets feature_input, drives update_feature_display,
                             # syncs UMAP highlight, auto-fills cross-SAE A.

update_feature_display = None  # Callable[[int], None]
                               # Re-renders the detail panel for a feature.
                               # Distinct from select_feature: this one only
                               # re-renders, doesn't move the rest of the UI.

gemini_click = None          # Callable[[], None]
                             # Fires a Gemini auto-interp call for the
                             # currently-selected feature. Used by both the
                             # explicit "Label with Gemini" button and the
                             # auto-Gemini-on-visit flow.

apply_order = None           # Callable[[ndarray], None]
                             # Re-populate the sortable feature table with
                             # the given index order. Used by search /
                             # name-update / dataset-switch flows.

get_sorted_order = None      # Callable[[], ndarray]
                             # Return the current freq-desc index order
                             # (filtered by ui.search_filter).

update_table_names = None    # Callable[[], None]
                             # Refresh the name column on the feature table
                             # without changing the row order.


def load_image(img_idx: int):
    """Load an image by global index using the active dataset's ``image_paths``.

    Defined here (rather than in :mod:`explorer.images`) because it
    depends on the runtime-mutable ``state``. Imports ``_open_image``
    lazily to avoid a hard cycle at module-load time.
    """
    from . import images as _images
    return _images._open_image(state.image_paths[img_idx])
