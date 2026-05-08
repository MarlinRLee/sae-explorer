"""Model registry — the YAML-defined list of demos the explorer serves.

One entry per (backbone, layer, token_type, SAE shape) combination. Exactly
one entry has ``primary: true``; it is loaded eagerly when the app starts.
The rest are lazy compares, loaded on first selection from the dropdown.

The same registry is consumed by:

- ``scripts/explorer_app.py`` (via ``--registry``) to build the dataset list
  and the model dropdown.
- ``scripts/bootstrap_demo.py`` to drive HF Hub downloads for both the
  local launcher and the HF Space entrypoint.

To add a model, append one block to ``configs/models.yaml`` — no other
file changes are required for the dropdown to pick it up. See the top
of that YAML for the schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass(frozen=True)
class ModelEntry:
    """One row of the registry."""
    id:         str
    label:      str
    data_file:  str
    sae_file:   str
    backbone:   str
    layer:      int
    token_type: str
    primary:    bool = False


@dataclass(frozen=True)
class Registry:
    """Parsed ``configs/models.yaml``.

    ``primary`` is the eagerly-loaded model; ``compares`` is everything
    else, in YAML order. ``hf_data_repo`` / ``hf_images_repo`` /
    ``images_tarball`` are pulled from the ``defaults:`` block.
    """
    primary:        ModelEntry
    compares:       List[ModelEntry]
    hf_data_repo:   str
    hf_images_repo: str
    images_tarball: str

    @property
    def all_entries(self) -> List[ModelEntry]:
        return [self.primary, *self.compares]

    def data_url(self, entry: ModelEntry) -> Optional[str]:
        if not self.hf_data_repo or not entry.data_file:
            return None
        return f"https://huggingface.co/datasets/{self.hf_data_repo}/resolve/main/{entry.data_file}"

    def sae_url(self, entry: ModelEntry) -> Optional[str]:
        if not self.hf_data_repo or not entry.sae_file:
            return None
        return f"https://huggingface.co/datasets/{self.hf_data_repo}/resolve/main/{entry.sae_file}"


def load_registry(path: str) -> Registry:
    """Parse a registry YAML and validate. Raises ``ValueError`` on malformed input."""
    with open(path) as f:
        doc = yaml.safe_load(f)

    defaults = doc.get('defaults') or {}
    raw_models = doc.get('models') or []
    if not raw_models:
        raise ValueError(f"{path}: 'models' list is empty")

    entries = [ModelEntry(
        id         = m['id'],
        label      = m['label'],
        data_file  = m['data_file'],
        sae_file   = m['sae_file'],
        backbone   = m['backbone'],
        layer      = int(m['layer']),
        token_type = m['token_type'],
        primary    = bool(m.get('primary', False)),
    ) for m in raw_models]

    primaries = [e for e in entries if e.primary]
    if len(primaries) != 1:
        raise ValueError(
            f"{path}: expected exactly one model with 'primary: true', got {len(primaries)}"
        )
    primary = primaries[0]
    compares = [e for e in entries if not e.primary]

    return Registry(
        primary        = primary,
        compares       = compares,
        hf_data_repo   = defaults.get('hf_data_repo',   ''),
        hf_images_repo = defaults.get('hf_images_repo', ''),
        images_tarball = defaults.get('images_tarball', 'hf_images.tar.gz'),
    )
