"""
Shared utilities for precompute_explorer_data.py and precompute_heatmaps.py.
"""

import glob
import os
import re

import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------- Constants ----------

RESOLUTION = 16
INPUT_SIZE  = RESOLUTION * 16   # 256 — DINOv3 default input size

_IMG_EXTS = ['*.jpg', '*.jpeg', '*.png', '*.JPEG']


# ---------- Image discovery ----------

def glob_images(root, recursive=False):
    paths = []
    for ext in _IMG_EXTS:
        pattern = os.path.join(root, '**', ext) if recursive else os.path.join(root, ext)
        paths += glob.glob(pattern, recursive=recursive)
    return paths


def resolve_path(stored_path, base_dirs):
    """Resolve a stored image path, falling back to base_dirs if the original is stale."""
    if os.path.exists(stored_path):
        return stored_path
    name = os.path.basename(stored_path)
    for base in base_dirs:
        if base:
            candidate = os.path.join(base, name)
            if os.path.exists(candidate):
                return candidate
    return None


# ---------- Dataset ----------

class ImageFolder(Dataset):
    """Load images from a flat directory or class-subdirectory layout."""

    def __init__(self, roots, recursive=False, transform_fn=None):
        if isinstance(roots, str):
            roots = [roots]
        self.paths = sorted(p for root in roots for p in glob_images(root, recursive))
        if not self.paths:
            raise FileNotFoundError(f"No images found in {roots}")
        self._transform_fn = transform_fn

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self._transform_fn(img), self.paths[idx]


# ---------- Token extraction ----------

def extract_tokens(hidden, backbone_name, token_type, n_reg):
    """
    Pull the relevant token subset from a backbone output.

    hidden : (bs, all_tokens, d_hidden)
    returns: (bs, n_tokens, d_hidden)

    Token layout:
      DINOv3 — [CLS, reg_0 … reg_{n_reg-1}, patch_0 … patch_N]
      CLIP   — [CLS, patch_0 … patch_N]  (no register tokens)
    """
    if token_type == 'cls':
        return hidden[:, 0:1, :]
    if token_type == 'all':
        return hidden
    # spatial (default)
    if backbone_name == 'clip' or backbone_name.startswith('dinov2'):
        return hidden[:, 1:, :]     # no register tokens
    return hidden[:, 1 + n_reg:, :]


# ---------- SAE loading ----------

def parse_top_k_from_path(sae_path):
    """Parse the SAE top-k value from a filename like 'sae_1_..._k160_...'."""
    m = re.search(r'_k(\d+)[_.]', os.path.basename(sae_path))
    if m is None:
        raise ValueError(
            f"Could not parse top-k from SAE filename '{sae_path}'. "
            f"Expected a pattern like '_k160_' in the filename."
        )
    return int(m.group(1))


def load_sae(sae_path, d_hidden, d_model, top_k, device):
    """Load and return a TopKSAE in eval mode."""
    from overcomplete.sae import TopKSAE
    sae = TopKSAE(input_shape=d_hidden, nb_concepts=d_model, top_k=top_k, device=device)
    sae.load_state_dict(torch.load(sae_path, map_location=device, weights_only=True))
    sae.eval()
    return sae
