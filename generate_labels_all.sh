#!/usr/bin/env bash
set -euo pipefail

# List of datasets
DATASETS=("banana" "bean" "blackgram" "coconut" "corn" "guava" "papaya" "rice" "sunflower")

for ds in "${DATASETS[@]}"; do
    IMG_DIR="datasets/$ds/images"
    OUT_FILE="datasets/$ds/labels.csv"

    echo "Generating labels for $ds ..."

    # Create header
    echo "image_path,label" > "$OUT_FILE"
# scripts/generate_labels_all.py
import os
import csv
import json
import glob
from pathlib import Path
from typing import List, Tuple

DATASETS = ["banana", "bean", "blackgram", "coconut", "corn", "guava", "papaya", "rice", "sunflower"]

REPO_ROOT = Path(__file__).resolve().parents[1]
DS_ROOT   = REPO_ROOT / "datasets"
PROJ_ROOT = REPO_ROOT / "project"

def list_classes(root: Path) -> List[str]:
    """Return sorted class-folder names under a given root (one level)."""
    if not root.exists():
        return []
    classes = [p.name for p in root.iterdir() if p.is_dir()]
    return sorted(classes)

def collect_images_from_classdirs(root: Path, classes: List[str]) -> List[Tuple[Path, str]]:
    """Collect image paths under root/<class>/* for given classes; return (path, class_name)."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    items = []
    for c in classes:
        cdir = root / c
        if not cdir.is_dir():
            continue
        for ext in exts:
            for p in cdir.glob(ext):
                items.append((p, c))
    return items

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def make_labels_for_dataset(ds: str):
    ds_dir   = DS_ROOT / ds
    out_csv  = ds_dir / "labels.csv"
    out_map  = ds_dir / "class_map.json"

    # Strategy 1: datasets/<ds>/images/<class>/*
    img_root = ds_dir / "images"
    classes_img = list_classes(img_root)

    # Strategy 2: project/<ds>_split/{train,val}/<class>/*
    split_root_train = PROJ_ROOT / f"{ds}_split" / "train"
    split_root_val   = PROJ_ROOT / f"{ds}_split" / "val"
    classes_split = sorted(set(list_classes(split_root_train) + list_classes(split_root_val)))

    # Decide source
    src_type = None
    items: List[Tuple[Path, str]] = []

    if classes_img:
        src_type = "datasets/images"
        items = collect_images_from_classdirs(img_root, classes_img)
        classes = classes_img
    elif classes_split:
        src_type = "project/split"
        items = collect_images_from_classdirs(split_root_train, classes_split) + \
                collect_images_from_classdirs(split_root_val,   classes_split)
        classes = classes_split
    else:
        print(f"[{ds}] ❌ No images found in expected locations.")
        return

    if not items:
        print(f"[{ds}] ❌ Found class folders but no images.")
        return

    # Build class -> int mapping (stable: sorted class names)
    class_to_idx = {c:i for i, c in enumerate(sorted(classes))}

    ensure_dir(ds_dir)

    # Write CSV: path,label_int (NO header)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        for p, cname in items:
            if src_type == "datasets/images":
                # write path relative to datasets/<ds> (e.g., images/class/img.jpg)
                rel = p.relative_to(ds_dir)
                w.writerow([str(rel), class_to_idx[cname]])
            else:
                # write absolute path for project/split sources
                w.writerow([str(p.resolve()), class_to_idx[cname]])

    # Save mapping for reference
    with out_map.open("w") as f:
        json.dump({"classes": sorted(classes), "class_to_idx": class_to_idx}, f, indent=2)

    print(f"[{ds}] ✅ Source: {src_type} | images: {len(items)} | classes: {len(classes)}")
    print(f"     -> {out_csv}")
    print(f"     -> {out_map}")

def main():
    print("Generating labels.csv for datasets:", ", ".join(DATASETS))
    for ds in DATASETS:
        make_labels_for_dataset(ds)

if __name__ == "__main__":
    main()

    # Loop through class subfolders
    find "$IMG_DIR" -type f \( -iname '*.jpg' -o -iname '*.png' \) | while read -r file; do
        # Get relative path from dataset root
        rel_path="${file#datasets/$ds/}"
        # Extract label name (parent folder of image)
        label=$(basename "$(dirname "$file")")
        echo "$rel_path,$label" >> "$OUT_FILE"
    done

    echo "Saved -> $OUT_FILE"


echo "✅ Done generating labels for all datasets."
