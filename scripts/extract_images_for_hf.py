#!/usr/bin/env python3
"""
Extract and thumbnail all images referenced by explorer_data*.pt files.
Run on MSI where the source images live.

Usage:
    python extract_images_for_hf.py [--out-dir /tmp/hf_images] [--size 256]

Output directory can then be uploaded to a private HF dataset repo:
    huggingface-cli upload YOUR_USERNAME/sae-explorer-images /tmp/hf_images . --repo-type dataset
"""

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

IMAGE_DIRS = [
    "/scratch.global/lee02328/val",            # ImageNet val (has class subdirs)
    "/scratch.global/lee02328/coco/val2017",  # COCO val2017 (flat)
]

BASE_DIR = Path("/users/9/lee02328/Ada_Comp/SAE_IN_COCO/smart_init_stability_SAE")


def build_lookup(image_dirs: list[str]) -> dict[str, Path]:
    """Recursively scan all image dirs and return basename → full path mapping."""
    print("Scanning image directories...")
    lookup: dict[str, Path] = {}
    for base in image_dirs:
        p = Path(base)
        if not p.exists():
            print(f"  WARNING: {base} not found, skipping")
            continue
        count = 0
        for img in p.rglob("*"):
            if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".JPEG"}:
                lookup[img.name] = img
                count += 1
        print(f"  {base}: {count} images indexed")
    print(f"  Total indexed: {len(lookup)}")
    return lookup


def main():
    parser = argparse.ArgumentParser(description="Extract thumbnails for HF Space deployment")
    parser.add_argument("--out-dir", default="/tmp/hf_images",
                        help="Directory to write thumbnails into")
    parser.add_argument("--size", type=int, default=256,
                        help="Thumbnail max side length in pixels (default: 256)")
    parser.add_argument("--quality", type=int, default=85,
                        help="JPEG quality (default: 85)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect unique image basenames from all explorer_data*.pt files
    # Skip heatmap sidecars — those don't contain image_paths
    pt_files = sorted(f for f in BASE_DIR.glob("explorer_data*.pt")
                      if "_heatmaps" not in f.name and "_patch_acts" not in f.name)
    if not pt_files:
        print(f"No explorer_data*.pt files found in {BASE_DIR}")
        return

    print(f"Found {len(pt_files)} data .pt file(s):")
    for f in pt_files:
        print(f"  {f.name}")

    all_basenames: set[str] = set()
    for pt in pt_files:
        d = torch.load(pt, map_location="cpu", weights_only=False)
        paths = d.get("image_paths", [])
        for p in paths:
            all_basenames.add(os.path.basename(p))
        print(f"  {pt.name}: {len(paths)} entries  ({len(all_basenames)} unique basenames so far)")

    print(f"\nTotal unique images to extract: {len(all_basenames)}")

    # Build recursive lookup to handle ImageNet's class subdirectories
    lookup = build_lookup(IMAGE_DIRS)

    found, skipped, missing = 0, 0, []
    for i, fname in enumerate(sorted(all_basenames)):
        dest = out / fname
        if dest.exists():
            skipped += 1
            found += 1
            continue

        src = lookup.get(fname)
        if src is not None:
            img = Image.open(src).convert("RGB")
            img.thumbnail((args.size, args.size), Image.LANCZOS)
            img.save(dest, "JPEG", quality=args.quality)
            found += 1
        else:
            missing.append(fname)

        if (i + 1) % 1000 == 0:
            pct = (i + 1) / len(all_basenames) * 100
            print(f"  [{pct:5.1f}%] {i+1}/{len(all_basenames)} processed — "
                  f"{found} saved, {len(missing)} missing")

    print(f"\nDone.")
    print(f"  Thumbnails saved: {found}  ({skipped} already existed, skipped)")
    print(f"  Missing images:   {len(missing)}")
    if missing:
        print(f"  Sample missing:   {missing[:5]}{'...' if len(missing) > 5 else ''}")

    total_mb = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"  Total size:       {total_mb:.1f} MB")
    print(f"\nNext step — upload to HF:")
    print(f"  huggingface-cli upload YOUR_USERNAME/sae-explorer-images {out} . --repo-type dataset")


if __name__ == "__main__":
    main()
