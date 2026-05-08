"""On-disk + HuggingFace persistence helpers for feature-name JSON files.

``_save_json_atomic`` is a generic atomic write. ``_schedule_hf_push``
debounces uploads so a burst of edits collapses into a single Hub commit.

The thin per-state wrappers (``_save_names`` / ``_save_auto_interp``) live
in the bootstrap module — they read/mutate the active ``_State`` and
``_UI`` instances. This module stays pure-ish: ``_schedule_hf_push`` takes
the ``ui`` object explicitly so it can null/restore ``ui.hf_push_pending``
without reaching for module globals.
"""

import json
import os
import threading
from datetime import datetime, timezone

from bokeh.io import curdoc


def _now_iso() -> str:
    """UTC ISO-8601 timestamp at second resolution. Used as the ``ts`` field
    in archived label entries."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _archive_label(history: dict, feat: int, label, author) -> None:
    """Append an overwritten/deleted ``(label, author)`` pair into the
    per-feature history list. ``history`` is the dict on a ``_State``
    instance (e.g. ``state.feature_names_history``); it's mutated in place.
    """
    history.setdefault(feat, []).append({
        'label':  label,
        'author': author,
        'ts':     _now_iso(),
    })


def _save_json_atomic(path: str, payload: dict) -> None:
    """Write JSON to a sibling temp file then atomic-rename. Guards against
    corrupting the existing file if the process is killed mid-write."""
    tmp = path + ".tmp"
    with open(tmp, 'w') as _f:
        json.dump(payload, _f, indent=2)
    os.replace(tmp, path)


def _schedule_hf_push(file_paths, ui) -> None:
    """Debounce HF dataset upload: wait 2 s after the last save, then push
    in a background thread. No-op if HF_TOKEN / HF_DATASET_REPO are unset
    (i.e. running locally).

    ``file_paths`` accepts a single path string or an iterable of paths;
    all are uploaded in one debounced batch so that authors-files and
    label-files saved together don't cancel each other through the
    single-slot ``ui.hf_push_pending`` debounce.

    ``ui`` is a session ``_UI`` instance whose ``hf_push_pending``
    attribute is used to coalesce concurrent saves.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    file_paths = [p for p in file_paths if p]
    if not file_paths:
        return

    hf_token = os.environ.get("HF_TOKEN")
    hf_repo  = os.environ.get("HF_DATASET_REPO")
    if not (hf_token and hf_repo):
        return

    # Cancel any already-pending push for this session.
    if ui.hf_push_pending is not None:
        try:
            curdoc().remove_timeout_callback(ui.hf_push_pending)
        except Exception:
            pass

    def _push_thread():
        from huggingface_hub import upload_file
        for fp in file_paths:
            try:
                upload_file(
                    path_or_fileobj=fp,
                    path_in_repo=os.path.basename(fp),
                    repo_id=hf_repo,
                    repo_type="dataset",
                    token=hf_token,
                    commit_message="Update feature labels",
                )
                print(f"  Pushed {os.path.basename(fp)} to HF dataset {hf_repo}")
            except Exception as e:
                print(f"  Warning: HF push failed for {os.path.basename(fp)}: {e}")

    def _fire():
        ui.hf_push_pending = None
        threading.Thread(target=_push_thread, daemon=True).start()

    ui.hf_push_pending = curdoc().add_timeout_callback(_fire, 2000)
