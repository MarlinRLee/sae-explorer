"""Gemini auto-interp: API wrapper + Bokeh button + worker thread.

The module owns:
  - the ``Label with Gemini`` button + its status ``Div``
  - the click handler that gathers MEI image paths/heatmaps for the
    currently-selected feature and spawns a worker thread
  - the worker thread that calls ``google.genai`` and posts the result
    back onto the Bokeh document via ``add_next_tick_callback``

It does **not** own the post-label state mutation (writing to
``state.auto_interp_names``, persisting JSON files, refreshing tables /
detail panels). Those are passed in via the ``on_label_applied`` callback
so this module stays decoupled from the bootstrap's persistence layer.

The ``build()`` factory returns the widgets and the click handler so the
bootstrap can:
  - place the widgets in the layout, and
  - publish the click handler to ``runtime.gemini_click`` for the
    auto-Gemini-on-visit flow.

Tests can call ``build()`` with a stub ``feature_input``, a stub
``on_label_applied``, and ``api_key=None`` to verify the disabled-state
defaults without invoking the network.
"""

import base64
import io
import os
import threading
from typing import Callable, Optional

from PIL import Image
from bokeh.io import curdoc
from bokeh.models import Button, Div

from . import runtime
from .images import _resolve_image_path


# ---------- Tunables ----------
DEFAULT_MODEL = "gemini-2.5-flash"   # also serves as the auto-interp author tag
N_GEMINI_IMAGES = 6                  # max top-MEIs to send per call
HM_ALPHA = 0.25                      # heatmap overlay opacity (currently unused
                                     # — Gemini sees the plain image, not the
                                     # blended overlay; reserved for the
                                     # future heatmap-aware prompt path)

SYSTEM_PROMPT = (
    "You are labeling features of a Sparse Autoencoder (SAE) trained on a "
    "vision transformer. Each SAE feature is a sparse direction in activation "
    "space that fires strongly on certain visual patterns."
)
USER_PROMPT = (
    "The images below are the top maximally-activating images for one SAE feature. "
    "In 2-5 words, give a precise label for the visual concept this feature detects. "
    "Be specific - prefer 'dog snout close-up' over 'dog', or 'brick wall texture' "
    "over 'texture'. "
    "Reply with ONLY the label, no explanation, no punctuation at the end."
)


def _label_thread(feat: int,
                  mei_items,
                  doc,
                  api_key: str,
                  model: str,
                  on_label_applied: Callable[[int, str], None],
                  btn: Button,
                  status_div: Div) -> None:
    """Worker entry point: encode top images, call Gemini, post result back
    onto the Bokeh document.

    ``mei_items`` is a list of ``(stored_image_path, heatmap_or_None)``.
    Heatmaps are reserved for a future overlay-prompt path; for now only
    the resolved image is sent.
    """
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        parts = []
        for path, _heatmap in mei_items[:N_GEMINI_IMAGES]:
            resolved = _resolve_image_path(path)
            if resolved is None:
                continue
            try:
                img = Image.open(resolved).convert("RGB").resize((224, 224), Image.BILINEAR)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
            except Exception:
                continue

        if not parts:
            def _no_images():
                btn.disabled = False
                status_div.text = "<span style='color:#c00'>No images could be loaded.</span>"
            doc.add_next_tick_callback(_no_images)
            return

        parts.append(types.Part.from_text(text=USER_PROMPT))
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        label = response.text.strip().strip(".,;:\"'")

        def _apply(feat=feat, label=label):
            try:
                on_label_applied(feat, label)
            except Exception as e:
                print(f"  [Gemini] post-label hook for feat {feat} raised: {e}")
            btn.disabled = False
            status_div.text = (
                f"<span style='color:#1a6faf'><b>Labeled:</b> {label}</span>"
            )
            print(f"  [Gemini] feat {feat}: {label}")

        doc.add_next_tick_callback(_apply)

    except Exception as e:
        err = str(e)

        def _show_err(err=err):
            btn.disabled = False
            status_div.text = f"<span style='color:#c00'>Error: {err[:120]}</span>"
            print(f"  [Gemini] feat {feat} error: {err}")

        doc.add_next_tick_callback(_show_err)


def build(get_current_feat: Callable[[], Optional[int]],
          on_label_applied: Callable[[int, str], None],
          api_key: Optional[str] = None,
          model: str = DEFAULT_MODEL):
    """Build the Gemini button + status div and return ``(btn, status_div, on_click)``.

    ``get_current_feat()``  zero-arg callable returning the
        currently-selected feat id (int) or None. The bootstrap typically
        wraps a ``TextInput.value`` read in a try/except, so the gemini
        module stays decoupled from the specific widget.
    ``on_label_applied(feat, label)``  called from the doc thread once
        Gemini returns a label, BEFORE the button is re-enabled. The
        bootstrap uses this to mutate state (auto_interp_names /
        auto_interp_authors / history), persist JSON, and refresh the
        detail panel.
    ``api_key``  if None, defaults to ``runtime.args.google_api_key`` /
        ``GOOGLE_API_KEY`` env var. The button is disabled (and a "no
        key set" banner is shown) when no key is found.
    ``model``  the Gemini model identifier; default ``gemini-2.5-flash``.
    """
    if api_key is None:
        api_key = (getattr(runtime.args, 'google_api_key', None)
                   or os.environ.get("GOOGLE_API_KEY"))

    btn = Button(
        label="Label with Gemini",
        width=140,
        button_type="warning",
        disabled=(api_key is None),
    )
    status_div = Div(
        text=("<i style='color:#aaa'>No GOOGLE_API_KEY set</i>"
              if api_key is None else ""),
        width=300,
    )

    def on_click():
        feat = get_current_feat()
        if feat is None:
            status_div.text = "<span style='color:#c00'>Select a feature first.</span>"
            return

        state = runtime.state
        if state.feature_frequency[feat].item() == 0:
            status_div.text = "<span style='color:#c00'>Dead feature - no images.</span>"
            return

        n_top_stored = state.top_img_idx.shape[1]
        mei_items = []
        for j in range(n_top_stored):
            idx = state.top_img_idx[feat, j].item()
            if idx >= 0:
                hm = None
                if state.top_heatmaps is not None:
                    hm = state.top_heatmaps[feat, j].float().numpy().reshape(
                        state.heatmap_patch_grid, state.heatmap_patch_grid)
                mei_items.append((state.image_paths[idx], hm))

        if not mei_items:
            status_div.text = "<span style='color:#c00'>No MEI paths found.</span>"
            return

        btn.disabled = True
        status_div.text = "<i style='color:#888'>Calling Gemini...</i>"

        doc = curdoc()
        threading.Thread(
            target=_label_thread,
            args=(feat, mei_items, doc, api_key, model,
                  on_label_applied, btn, status_div),
            daemon=True,
        ).start()

    if api_key:
        btn.on_click(on_click)

    return btn, status_div, on_click
