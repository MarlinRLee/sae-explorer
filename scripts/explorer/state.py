"""Per-session state objects + CLI argument plumbing for the explorer.

Two classes live here:

- ``_UI`` — ephemeral session UI state (render token, search filter, lazy
  CLIP handle, debounce timeout handles, etc.). Replaces the
  list-as-mutable-cell pattern (``_var = [None]``) that was scattered across
  the original monolith.

- ``_State`` — owns the active dataset's tensors and derived numpy arrays.
  ``apply(idx)`` rebinds every field for the dataset at index ``idx`` of
  the dataset list provided at construction.

Plus argparse helpers so tests / alternate launchers can build a fake
``args`` namespace without invoking ``sys.argv``.
"""

import argparse
import os

import numpy as np


# ---------- Argument parsing ----------
def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Factored out so tests can introspect or
    extend the schema without invoking sys.argv."""
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=str, required=True,
                   help="Path to a configs/models.yaml describing the demo's "
                        "models. Exactly one entry must set 'primary: true'.")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Local directory holding the registry's .pt files "
                        "(downloaded by scripts/bootstrap_demo.py).")
    p.add_argument("--image-dir", type=str, required=True,
                   help="Primary image directory used during precompute")
    p.add_argument("--extra-image-dir", type=str, default=None,
                   help="Additional image directory used during precompute")
    p.add_argument("--thumb-size", type=int, default=256)
    p.add_argument("--inference-cache-size", type=int, default=64,
                   help="Number of images to keep in the patch-activations LRU cache")
    p.add_argument("--names-file", type=str, default=None,
                   help="Path to JSON file for saving feature names "
                        "(default: <primary-data>_feature_names.json)")
    p.add_argument("--clip-model", type=str, default="openai/clip-vit-large-patch14",
                   help="HuggingFace CLIP model ID for free-text search "
                        "(only loaded on first out-of-vocab query)")
    p.add_argument("--google-api-key", type=str, default=None,
                   help="Google API key for Gemini auto-interp button "
                        "(default: GOOGLE_API_KEY env var)")
    p.add_argument("--hover-prewarm-count", type=int, default=2000,
                   help="On dataset load, background-prewarm UMAP hover thumbnails "
                        "for the top-K most-frequent live features so the first "
                        "hover doesn't stall. 0 disables prewarm.")
    p.add_argument("--auto-gemini", action=argparse.BooleanOptionalAction, default=False,
                   help="When on, automatically run the Gemini auto-interp on "
                        "every feature selected that lacks both a manual name "
                        "and an auto-interp label. Off by default — pass "
                        "--auto-gemini to enable. Requires GOOGLE_API_KEY (or "
                        "--google-api-key) to actually fire.")
    return p


def _parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI args. ``argv=None`` defers to sys.argv; tests can pass a list."""
    return _build_parser().parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    """Soft-validate args: hard-fail on missing registry, warn on the rest.

    The registry walker validates each compare entry's .pt existence as it
    lazy-loads, so we only check the registry file itself here."""
    if not os.path.exists(args.registry):
        raise SystemExit(f"--registry file not found: {args.registry}")
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"--data-dir is not a directory: {args.data_dir}")
    for bd in filter(None, [args.image_dir, args.extra_image_dir]):
        if not os.path.isdir(bd):
            print(f"Warning: image dir does not exist: {bd}")


# ---------- UI session state ----------
class _UI:
    """Ephemeral UI session state.

    Replaces the list-as-mutable-cell pattern (``_var = [None]``) used in
    the original single-file demo with regular attributes that nested
    callbacks can mutate cleanly, without needing ``global`` or ``_var[0]``
    indirection. One instance per Bokeh session, one place to look for
    "what is the UI doing right now".
    """
    render_token: int = 0              # incremented on each new feature selection
    current_patch_img = None           # image idx loaded into the patch explorer
    current_patch_z = None             # cached (n_patches, d_model) for that image
    search_filter = None               # set of feat indices matching the name search
    hf_push_pending = None             # active HF-push debounce timeout handle
    clip_handle = None                 # lazy CLIP (model, processor, device) tuple
    umap_last_hover_feats: tuple = ()  # last feats rendered in the hover preview
    active_feats = ()                  # feats with frequency > 0 (Random button)
    prewarm_token: int = 0             # bumped on each dataset switch; the
                                       # background hover-cache prewarm thread
                                       # captures the value and aborts when it
                                       # changes, preventing stale entries.
    suppress_name_save: bool = False   # set True while update_feature_display
                                       # programmatically rewrites name_input.value


# ---------- Active dataset state ----------
class _State:
    """Owns the active dataset's tensors and derived numpy arrays.

    Instantiate with the list of dataset dicts produced by
    ``_load_dataset_dict``. Call ``apply(idx)`` to rebind every field to
    that index. Lazy compares are loaded by the caller before ``apply``.
    """

    # ds key -> attribute name. Key absent in ds -> KeyError on apply.
    _REQUIRED = {
        'image_paths':       'image_paths',
        'd_model':           'd_model',
        'n_images':          'n_images',
        'patch_grid':        'patch_grid',
        'image_size':        'image_size',
        'top_img_idx':       'top_img_idx',
        'top_img_act':       'top_img_act',
        'mean_img_idx':      'mean_img_idx',
        'mean_img_act':      'mean_img_act',
        'p75_img_idx':       'p75_img_idx',
        'p75_img_act':       'p75_img_act',
        'feature_frequency': 'feature_frequency',
        'feature_mean_act':  'feature_mean_act',
        'feature_p75_val':   'feature_p75_val',
        'umap_coords':       'umap_coords',
        'dict_umap_coords':  'dict_umap_coords',
        'feature_names':              'feature_names',
        'auto_interp_names':          'auto_interp_names',
        'feature_name_authors':       'feature_name_authors',
        'auto_interp_authors':        'auto_interp_authors',
        'feature_names_history':      'feature_names_history',
        'auto_interp_history':        'auto_interp_history',
        'names_file':                 'names_file',
        'auto_interp_file':           'auto_interp_file',
        'feature_name_authors_file':  'feature_name_authors_file',
        'auto_interp_authors_file':   'auto_interp_authors_file',
        'feature_names_history_file': 'feature_names_history_file',
        'auto_interp_history_file':   'auto_interp_history_file',
    }
    # Optional ds keys (None when missing). Heatmaps and CLIP are absent on
    # some precompute outputs (e.g. CLS-token SAEs, no-clip sidecar).
    _OPTIONAL = {
        'top_heatmaps':      'top_heatmaps',
        'mean_heatmaps':     'mean_heatmaps',
        'p75_heatmaps':      'p75_heatmaps',
        'clip_scores':       'clip_scores',
        'clip_vocab':        'clip_vocab',
        'clip_embeds':       'clip_embeds',
        'clip_scores_f32':   'clip_scores_f32',
    }

    def __init__(self, datasets: list):
        self._datasets = datasets

    def apply(self, idx: int) -> None:
        ds = self._datasets[idx]
        self.idx = idx
        for ds_key, attr in self._REQUIRED.items():
            setattr(self, attr, ds[ds_key])
        for ds_key, attr in self._OPTIONAL.items():
            setattr(self, attr, ds.get(ds_key))
        self.heatmap_patch_grid = ds.get('heatmap_patch_grid', self.patch_grid)
        self.has_clip = (self.clip_scores is not None
                         and self.clip_vocab is not None)

        # Derived arrays used by UMAP, feature list, and callbacks
        self.freq     = self.feature_frequency.numpy()
        self.mean_act = self.feature_mean_act.numpy()
        self.log_freq = np.log10(self.freq + 1)
        self.p75      = self.feature_p75_val.numpy()
        self.live_mask         = ~np.isnan(self.umap_coords[:, 0])
        self.live_indices      = np.where(self.live_mask)[0]
        self.dict_live_mask    = ~np.isnan(self.dict_umap_coords[:, 0])
        self.dict_live_indices = np.where(self.dict_live_mask)[0]
        self.umap_backup = dict(
            act_x=self.umap_coords[self.live_mask, 0].tolist(),
            act_y=self.umap_coords[self.live_mask, 1].tolist(),
            act_feat=self.live_indices.tolist(),
            dict_x=self.dict_umap_coords[self.dict_live_mask, 0].tolist(),
            dict_y=self.dict_umap_coords[self.dict_live_mask, 1].tolist(),
            dict_feat=self.dict_live_indices.tolist(),
        )

    @property
    def ds(self) -> dict:
        return self._datasets[self.idx]

    @property
    def compare_datasets(self) -> list:
        return [d for i, d in enumerate(self._datasets) if i != self.idx]
