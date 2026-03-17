"""
Batch auto-interp for SAE features using the Gemini vision API.

Sends the top-N maximally-activating images (MEIs) for each feature to
Gemini and writes short natural-language labels to a sidecar JSON file:

    <data>_auto_interp.json

This file is SEPARATE from the manual label file (<data>_feature_names.json)
so that auto-interp never overwrites anything the user has typed by hand.
The explorer app shows the auto-interp label only when no manual label exists.

Requirements
------------
    pip install google-genai pillow

    export GOOGLE_API_KEY="your-key-here"

Usage
-----
    # Label all live features (can be slow for d=32K; use --end-feat to limit)
    python auto_interp_vlm.py --data ../explorer_data_d32000_k160_val.pt

    # Restartable: skip features already labeled, process a range
    python auto_interp_vlm.py --data ../explorer_data_d32000_k160_val.pt \\
        --start-feat 0 --end-feat 2000 --skip-labeled

    # Use more images per feature for better accuracy
    python auto_interp_vlm.py --data ../explorer_data_d32000_k160_val.pt \\
        --n-images 8 --model gemini-2.0-flash

Output
------
    <data>_auto_interp.json  — dict mapping str(feature_idx) → label string
"""

import argparse
import io
import json
import os
import sys
import time

import torch
from PIL import Image

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are labeling features of a Sparse Autoencoder (SAE) trained on a "
    "vision transformer. Each SAE feature is a sparse direction in activation "
    "space that fires strongly on certain visual patterns."
)

USER_PROMPT = (
    "The images below are the top maximally-activating images for one SAE feature. "
    "In 2–5 words, give a precise label for the visual concept this feature detects. "
    "Be specific — prefer 'dog snout close-up' over 'dog', or 'brick wall texture' "
    "over 'texture'. "
    "Reply with ONLY the label, no explanation, no punctuation at the end."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(path: str, size: int = 224) -> bytes:
    """Resize an image and return raw JPEG bytes."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _resolve_path(stored_path: str, image_dir: str | None, extra_image_dir: str | None) -> str | None:
    """Resolve an image path the same way the explorer app does."""
    if os.path.isabs(stored_path) and os.path.exists(stored_path):
        return stored_path
    basename = os.path.basename(stored_path)
    for base in filter(None, [image_dir, extra_image_dir]):
        candidate = os.path.join(base, basename)
        if os.path.exists(candidate):
            return candidate
    # Last resort: stored path as-is
    if os.path.exists(stored_path):
        return stored_path
    return None


def label_feature(
    client: genai.Client,
    model_name: str,
    image_paths: list[str],
    n_images: int,
    image_dir: str | None,
    extra_image_dir: str | None,
    img_size: int = 224,
) -> str | None:
    """
    Send up to n_images MEIs to Gemini and return a short label string,
    or None if no images could be loaded.
    """
    parts = []
    loaded = 0
    for p in image_paths[:n_images]:
        resolved = _resolve_path(p, image_dir, extra_image_dir)
        if resolved is None:
            continue
        try:
            raw = _encode_image(resolved, size=img_size)
            parts.append(types.Part.from_bytes(data=raw, mime_type="image/jpeg"))
            loaded += 1
        except Exception:
            continue

    if loaded == 0:
        return None

    parts.append(types.Part.from_text(text=USER_PROMPT))

    response = client.models.generate_content(
        model=model_name,
        contents=parts,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    label = response.text.strip().strip(".,;:\"'")
    return label if label else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch auto-interp labels for SAE features via Gemini")
    parser.add_argument("--data", required=True,
                        help="Path to explorer_data.pt")
    parser.add_argument("--output", default=None,
                        help="Output JSON path "
                             "(default: <data>_auto_interp.json)")
    parser.add_argument("--image-dir", default=None,
                        help="Primary image directory (same as explorer --image-dir)")
    parser.add_argument("--extra-image-dir", default=None,
                        help="Extra image directory (same as explorer --extra-image-dir)")
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--n-images", type=int, default=6,
                        help="MEIs to send per feature (default: 6)")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Resize images to this square size before sending (default: 224)")
    parser.add_argument("--start-feat", type=int, default=0,
                        help="First feature index to process (default: 0)")
    parser.add_argument("--end-feat", type=int, default=None,
                        help="Last feature index (exclusive); default: all features")
    parser.add_argument("--skip-labeled", action="store_true",
                        help="Skip features that already have an auto-interp label "
                             "(makes the run restartable)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between API calls (default: 0.5)")
    parser.add_argument("--save-interval", type=int, default=25,
                        help="Save JSON every N features processed (default: 25)")
    args = parser.parse_args()

    # --- Gemini setup ---
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    print(f"Using Gemini model: {args.model}")

    # --- Load explorer data ---
    print(f"Loading {args.data} ...")
    d = torch.load(args.data, map_location="cpu", weights_only=False)
    image_paths  = d["image_paths"]
    d_model      = d["d_model"]
    top_img_idx  = d["top_img_idx"]      # (d_model, n_top)
    freq         = d["feature_frequency"] # (d_model,)
    n_top_stored = top_img_idx.shape[1]
    print(f"  d_model={d_model}, n_top_stored={n_top_stored}")

    # --- Output file ---
    output_path = args.output or (os.path.splitext(args.data)[0] + "_auto_interp.json")
    existing: dict[int, str] = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = {int(k): v for k, v in json.load(f).items()}
        print(f"Loaded {len(existing)} existing auto-interp labels from {output_path}")

    # --- Feature range ---
    end_feat = args.end_feat if args.end_feat is not None else d_model
    feat_range = range(args.start_feat, end_feat)
    print(f"Processing features {args.start_feat} – {end_feat - 1} "
          f"({end_feat - args.start_feat} total)")

    # --- Process ---
    n_labeled = 0
    n_skipped = 0
    n_dead    = 0
    n_failed  = 0

    for feat in feat_range:
        # Skip dead features
        if freq[feat].item() == 0:
            n_dead += 1
            continue

        # Skip already-labeled if requested
        if args.skip_labeled and feat in existing:
            n_skipped += 1
            continue

        # Collect MEI paths
        mei_paths = []
        for j in range(n_top_stored):
            idx = top_img_idx[feat, j].item()
            if idx >= 0:
                mei_paths.append(image_paths[idx])

        if not mei_paths:
            n_dead += 1
            continue

        try:
            label = label_feature(
                client, args.model, mei_paths, args.n_images,
                args.image_dir, args.extra_image_dir, args.img_size,
            )
            if label:
                existing[feat] = label
                n_labeled += 1
                print(f"  feat {feat:6d}: {label}")
            else:
                n_failed += 1
                print(f"  feat {feat:6d}: (no images loaded)")
        except Exception as e:
            n_failed += 1
            err_str = str(e)
            print(f"  feat {feat:6d}: ERROR — {e}")
            # Daily quota exhausted — no point continuing until tomorrow
            if "PerDay" in err_str or "per_day" in err_str.lower():
                print("\nDaily quota exhausted. Saving progress and exiting.")
                break
            # Model not available — no point retrying on any feature
            if "NOT_FOUND" in err_str or "no longer available" in err_str:
                print("\nModel not found/unavailable. Check model name and exit.")
                break

        # Periodic save
        processed = n_labeled + n_failed
        if processed > 0 and processed % args.save_interval == 0:
            with open(output_path, "w") as f:
                json.dump({str(k): v for k, v in sorted(existing.items())}, f, indent=2)
            print(f"  [checkpoint] saved {len(existing)} labels to {output_path}")

        if args.sleep:
            time.sleep(args.sleep)

    # --- Final save ---
    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(existing.items())}, f, indent=2)

    print(f"\nDone.")
    print(f"  Labeled:  {n_labeled}")
    print(f"  Skipped (already labeled): {n_skipped}")
    print(f"  Dead/no images: {n_dead}")
    print(f"  Failed:   {n_failed}")
    print(f"  Total in file: {len(existing)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
