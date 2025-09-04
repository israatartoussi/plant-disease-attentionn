import argparse, hashlib, os, shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = {"train","valid","validation","val","test"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def detect_structure(src: Path):
    # split-level if it has standard split dirs; otherwise assume class-level
    split_dirs = [d for d in src.iterdir() if d.is_dir() and d.name.lower() in SPLIT_NAMES]
    if split_dirs:
        return "split", split_dirs
    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    return "imagefolder", class_dirs

def merge_dataset(src: Path, dst: Path, copy: bool = False):
    kind, nodes = detect_structure(src)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0

    if kind == "split":
        # inside each split, expect class subdirs
        for split in nodes:
            for cls in sorted([d for d in split.iterdir() if d.is_dir()]):
                out_cls = dst / cls.name
                out_cls.mkdir(exist_ok=True)
                for img in cls.rglob("*"):
                    if img.is_file() and is_image(img):
                        # stable unique name
                        h = hashlib.md5(str(img.relative_to(src)).encode()).hexdigest()[:8]
                        new_name = f"{split.name}_{img.stem}_{h}{img.suffix.lower()}"
                        target = out_cls / new_name
                        if copy:
                            shutil.copy2(img, target)
                        else:
                            if not target.exists():
                                os.symlink(img.resolve(), target)
                        n += 1
    else:  # imagefolder already
        for cls in sorted(nodes):
            out_cls = dst / cls.name
            out_cls.mkdir(exist_ok=True)
            for img in cls.rglob("*"):
                if img.is_file() and is_image(img):
                    target = out_cls / img.name
                    if copy:
                        shutil.copy2(img, target)
                    else:
                        if not target.exists():
                            os.symlink(img.resolve(), target)
                    n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source dataset root")
    ap.add_argument("--dst", required=True, help="destination ImageFolder root")
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlinks")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    total = merge_dataset(src, dst, copy=args.copy)
    print(f"Merged {total} images -> {dst}")

if __name__ == "__main__":
    main()
