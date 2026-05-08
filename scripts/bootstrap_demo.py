"""Bootstrap the SAE Feature Explorer demo from a YAML registry.

Single shared entry point used by both ``scripts/run_explorer_local.sh``
and ``hf_space/entrypoint.sh``: reads ``configs/models.yaml``, downloads
the .pt sidecars, heatmap sidecars, and SAE checkpoints from the HF
dataset repo into ``--target``, downloads + extracts the thumbnails
tarball into ``--image-dir``, and (with ``--launch``) execs the bokeh
server pointed at the registry.

Idempotent: files that already exist are skipped.

Examples:

    # Local
    python scripts/bootstrap_demo.py \\
        --registry configs/models.yaml \\
        --target ./local_data --image-dir ./local_images --launch

    # HF Space (entrypoint.sh)
    python scripts/bootstrap_demo.py \\
        --registry /app/configs/models.yaml \\
        --target /app/data --image-dir /app/images \\
        --launch --port 7860 --app /app/scripts/explorer_app.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile

from huggingface_hub import hf_hub_download

# Make the sibling explorer/ package importable when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from explorer.registry import Registry, load_registry


def _hf_download(repo_id: str, filename: str, local_dir: str,
                 token: str | None) -> str:
    """Download ``filename`` from ``repo_id`` into ``local_dir`` if missing.

    Returns the local path. Skip-and-print when the file is already
    present so repeat runs are fast and quiet.
    """
    dest = os.path.join(local_dir, filename)
    if os.path.exists(dest):
        print(f"  [skip] {filename}")
        return dest
    print(f"  [pull] {filename}")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=local_dir,
        token=token,
    )


def fetch_models(reg: Registry, target: str, token: str | None) -> None:
    """Download .pt + _heatmaps.pt + .pth for every entry in the registry."""
    if not reg.hf_data_repo:
        # Synthetic / offline registry — files are produced locally; nothing
        # to fetch. Trust ``target`` is already populated.
        print("[models] no hf_data_repo set — skipping download")
        return
    os.makedirs(target, exist_ok=True)
    for entry in reg.all_entries:
        print(f"[{entry.label}]")
        _hf_download(reg.hf_data_repo, entry.data_file, target, token)
        heatmap_file = entry.data_file.replace('.pt', '_heatmaps.pt')
        try:
            _hf_download(reg.hf_data_repo, heatmap_file, target, token)
        except Exception as e:
            # Heatmaps are optional — CLS-token SAEs and minimal demos lack them.
            print(f"  [warn] no heatmap sidecar for {entry.label}: {e}")
        try:
            _hf_download(reg.hf_data_repo, entry.sae_file, target, token)
        except Exception as e:
            # SAE weights are optional for the explorer (precomputed sidecars
            # carry everything the UI needs); only required if you want the
            # download link in the summary panel to work.
            print(f"  [warn] no SAE checkpoint for {entry.label}: {e}")


def fetch_images(reg: Registry, image_dir: str, token: str | None) -> None:
    """Download + extract the thumbnails tarball into ``image_dir``."""
    if not reg.hf_images_repo or not reg.images_tarball:
        print("[images] no hf_images_repo set — skipping download")
        return
    os.makedirs(image_dir, exist_ok=True)
    if os.listdir(image_dir):
        print(f"[images] {image_dir} already populated, skipping download")
        return
    print(f"[images] downloading {reg.images_tarball} from {reg.hf_images_repo}")
    tar_path = hf_hub_download(
        repo_id=reg.hf_images_repo,
        filename=reg.images_tarball,
        repo_type="dataset",
        local_dir=os.path.dirname(image_dir.rstrip('/')) or '.',
        token=token,
    )
    print(f"[images] extracting {tar_path} -> {image_dir}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=os.path.dirname(image_dir.rstrip('/')) or '.',
                       filter='data')
    # Tarball top-level is conventionally `hf_images/`; flatten into image_dir
    # if that's where it landed.
    extracted_root = os.path.join(os.path.dirname(image_dir.rstrip('/')) or '.',
                                  'hf_images')
    if os.path.isdir(extracted_root) and os.path.abspath(extracted_root) != os.path.abspath(image_dir):
        for name in os.listdir(extracted_root):
            os.rename(os.path.join(extracted_root, name),
                      os.path.join(image_dir, name))
        os.rmdir(extracted_root)
    n = len(os.listdir(image_dir))
    print(f"[images] extracted {n} files into {image_dir}")


def launch_bokeh(registry_path: str, target: str, image_dir: str,
                 port: int, app: str) -> None:
    """Exec ``bokeh serve`` pointed at the registry. Replaces this process."""
    cmd = [
        'bokeh', 'serve', app,
        '--port', str(port),
        '--allow-websocket-origin=*',
        '--session-token-expiration', '86400',
        '--args',
        '--registry',  registry_path,
        '--data-dir',  target,
        '--image-dir', image_dir,
    ]
    print(f"[launch] {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--registry',  required=True,
                   help="Path to configs/models.yaml")
    p.add_argument('--target',    required=True,
                   help="Local directory for .pt / .pth files")
    p.add_argument('--image-dir', required=True,
                   help="Local directory for extracted thumbnails")
    p.add_argument('--launch', action='store_true',
                   help="After download, exec bokeh serve")
    p.add_argument('--port', type=int, default=5006)
    p.add_argument('--app', default=os.path.join(_HERE, 'explorer_app.py'),
                   help="Path to scripts/explorer_app.py")
    p.add_argument('--skip-images', action='store_true',
                   help="Skip thumbnail download (e.g. when --image-dir is "
                        "already populated by an outside process).")
    args = p.parse_args()

    reg = load_registry(args.registry)
    token = os.environ.get('HF_TOKEN') or None

    fetch_models(reg, args.target, token)
    if not args.skip_images:
        fetch_images(reg, args.image_dir, token)

    if args.launch:
        launch_bokeh(args.registry, args.target, args.image_dir, args.port, args.app)


if __name__ == '__main__':
    main()
