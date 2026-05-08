"""Lazy CLIP model loader for free-text feature search.

The CLIP weights are heavy and not needed for most explorer interactions
(precomputed CLIP scores cover the in-vocab queries). This module loads
the model exactly once, on the first out-of-vocab query, and caches the
result on ``runtime.ui.clip_handle`` so subsequent queries return
instantly.

The actual model loading is delegated to :func:`clip_utils.load_clip` so
this module stays free of transformers-specific code paths.
"""

import torch

from . import runtime


def get_clip():
    """Load CLIP once (using ``runtime.args.clip_model`` for the model id)
    and return the cached ``(model, processor, device)`` tuple.

    Subsequent calls return the cached tuple unchanged. Picks ``cuda:0``
    when available, otherwise falls back to CPU.
    """
    if runtime.ui.clip_handle is None:
        # Imported lazily so importing this module doesn't drag in the
        # transformers stack at package load.
        import sys, os
        _src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from clip_utils import load_clip
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[CLIP] Loading {runtime.args.clip_model} on {dev} (first free-text query)...")
        m, p = load_clip(dev, model_name=runtime.args.clip_model)
        runtime.ui.clip_handle = (m, p, dev)
        print("[CLIP] Ready.")
    return runtime.ui.clip_handle
