"""On-disk + HuggingFace persistence helpers for feature-name JSON files.

``_save_json_atomic`` is a generic atomic write. ``_schedule_hf_push``
debounces uploads so a burst of edits collapses into a single Hub commit;
before each upload it union-merges the local file with the remote copy
so concurrent sessions don't clobber each other's edits.
``_maybe_pull_and_merge_local`` is the symmetric boot-side hook used by
``loaders.py`` to seed the local sidecar from the canonical remote copy
(HF Space containers have ephemeral filesystems, so without this a cold
start would only see the bundled labels).

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


def _is_history_path(path: str) -> bool:
    return path.endswith('_history.json')


def _merge_label_dicts(a: dict, b: dict, *, is_history: bool, winner: str) -> dict:
    """Union-merge two sidecar dicts loaded from JSON.

    For label / author files (``is_history=False``) values are scalar; on
    key conflict, ``winner`` ('a' or 'b') selects which side keeps the
    value. For history files (``is_history=True``) values are lists of
    ``{label, author, ts}`` records; lists from both sides are
    concatenated and deduped by ``(label, author, ts)``.
    """
    if is_history:
        out = {}
        for k in set(a) | set(b):
            seen = set()
            merged = []
            for entry in (a.get(k) or []) + (b.get(k) or []):
                if not isinstance(entry, dict):
                    continue
                sig = (entry.get('label'), entry.get('author'), entry.get('ts'))
                if sig in seen:
                    continue
                seen.add(sig)
                merged.append(entry)
            out[k] = merged
        return out
    if winner == 'a':
        out = dict(b)
        out.update(a)
    else:
        out = dict(a)
        out.update(b)
    return out


def _fetch_remote_dict(filename, repo, token):
    """Download a JSON sidecar from the HF dataset and return the parsed
    dict, or ``None`` if the file doesn't exist or the fetch fails. Uses
    the HF cache, so repeat calls are cheap when the remote hasn't
    changed."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    try:
        local = hf_hub_download(
            repo_id=repo, filename=filename, repo_type="dataset", token=token,
        )
        with open(local) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    except Exception as e:
        print(f"  Warning: HF fetch failed for {filename}: {e}")
        return None


def _maybe_pull_and_merge_local(path: str) -> None:
    """Pull-on-boot: fetch the remote copy of this sidecar from the HF
    dataset and union-merge it into the local file (remote wins on
    conflict), then write the merged result back to ``path``.

    HF Space containers have ephemeral filesystems, so a cold start sees
    only the bundled (often stale or empty) sidecar — without this hook,
    the first save in a fresh session would clobber the canonical remote
    copy with that bundled snapshot. No-op if ``HF_TOKEN`` /
    ``HF_DATASET_REPO`` are unset or the remote file is absent.
    """
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo  = os.environ.get("HF_DATASET_REPO")
    if not (hf_token and hf_repo):
        return
    remote = _fetch_remote_dict(os.path.basename(path), hf_repo, hf_token)
    if remote is None:
        return
    local = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                local = raw
        except Exception:
            local = {}
    merged = _merge_label_dicts(
        remote, local, is_history=_is_history_path(path), winner='a',
    )
    _save_json_atomic(path, merged)
    if len(merged) != len(local):
        print(f"  HF pull {os.path.basename(path)}: "
              f"remote={len(remote)} local={len(local)} -> {len(merged)}")


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
                # Union-merge against the canonical remote copy before
                # uploading so a stale in-session state (e.g. on a Space
                # cold start, or alongside a concurrent editor) cannot
                # drop keys that exist on the remote. Local wins on
                # conflict because the user just edited those entries.
                remote = _fetch_remote_dict(
                    os.path.basename(fp), hf_repo, hf_token,
                )
                if remote is not None:
                    try:
                        with open(fp) as _lf:
                            local = json.load(_lf)
                    except Exception:
                        local = {}
                    if not isinstance(local, dict):
                        local = {}
                    merged = _merge_label_dicts(
                        remote, local,
                        is_history=_is_history_path(fp), winner='b',
                    )
                    _save_json_atomic(fp, merged)
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
