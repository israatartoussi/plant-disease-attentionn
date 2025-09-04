#!/usr/bin/env bash
# ===== env_probe.sh =====
# Run from your repo root. It will create diag_out/report.txt with everything needed.

set -euo pipefail

OUT_DIR="diag_out"
OUT_FILE="${OUT_DIR}/report.txt"
mkdir -p "$OUT_DIR"

log() { echo -e "$@" | tee -a "$OUT_FILE"; }
hdr() { echo -e "\n===== $@ =====" | tee -a "$OUT_FILE"; }

# 0) Repo basics
hdr "BASIC INFO (time, cwd, git)"
date | tee -a "$OUT_FILE"
echo "CWD: $(pwd)" | tee -a "$OUT_FILE"
if command -v git >/dev/null 2>&1; then
  echo "Git present: YES" | tee -a "$OUT_FILE"
  (git rev-parse --is-inside-work-tree >/dev/null 2>&1 && \
   { echo "Git repo: YES"; \
     echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"; \
     echo "Git remotes:"; git remote -v; \
     echo "Git status (short):"; git status -sb; } \
  ) 2>/dev/null | tee -a "$OUT_FILE" || echo "Git repo: NO" | tee -a "$OUT_FILE"
else
  echo "Git present: NO" | tee -a "$OUT_FILE"
fi

# 1) System & Python
hdr "SYSTEM"
uname -a | tee -a "$OUT_FILE"
echo "Shell: $SHELL" | tee -a "$OUT_FILE"

hdr "PYTHON & PACKAGES"
if command -v python >/dev/null 2>&1; then
  echo "Python: $(python --version 2>&1)" | tee -a "$OUT_FILE"
else
  echo "Python not found on PATH!" | tee -a "$OUT_FILE"
fi

# 2) GPU/CUDA/Torch details (via Python)
python - <<'PY' 2>&1 | tee -a "diag_out/report.txt"
import sys, json, torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}:", torch.cuda.get_device_name(i))
    print("cuDNN enabled:", torch.backends.cudnn.enabled)
else:
    print("No CUDA devices detected.")
try:
    import torchvision
    print("Torchvision version:", torchvision.__version__)
except Exception as e:
    print("Torchvision import error:", e)
PY

# 3) nvidia-smi if available
hdr "NVIDIA-SMI"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee -a "$OUT_FILE"
else
  echo "nvidia-smi not found." | tee -a "$OUT_FILE"
fi

# 4) Directory layout
hdr "DIRECTORY TREE (depth 3)"
if command -v tree >/dev/null 2>&1; then
  tree -L 3 -a . | tee -a "$OUT_FILE"
else
  echo "(tree not installed; using find)" | tee -a "$OUT_FILE"
  find . -maxdepth 3 -print | sed 's#^\./##' | tee -a "$OUT_FILE"
fi

# 5) Datasets summary (expects datasets/<name>/ with images/ and optional labels.csv)
hdr "DATASETS SUMMARY"
DATASETS_DIR="datasets"
if [ -d "$DATASETS_DIR" ]; then
  for ds in "$DATASETS_DIR"/*; do
    [ -d "$ds" ] || continue
    ds_name="$(basename "$ds")"
    echo "--- DATASET: ${ds_name} ---" | tee -a "$OUT_FILE"
    # Count images
    IMG_COUNT=$(find "$ds" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l | tr -d ' ')
    echo "Images found: ${IMG_COUNT}" | tee -a "$OUT_FILE"
    # labels.csv
    if [ -f "$ds/labels.csv" ]; then
      echo "labels.csv: PRESENT" | tee -a "$OUT_FILE"
      echo "labels.csv head (first 5 lines):" | tee -a "$OUT_FILE"
      head -n 5 "$ds/labels.csv" | tee -a "$OUT_FILE"
      # Try to extract unique labels via Python
      python - <<PY 2>&1 | tee -a "diag_out/report.txt"
import csv, os, sys
ds_path = r"""$ds"""
csv_path = os.path.join(ds_path, "labels.csv")
try:
    labels = []
    with open(csv_path, newline="") as f:
        for r in csv.reader(f):
            if not r: continue
            # expects: path,label  (adjust if your format differs)
            if len(r) >= 2:
                labels.append(r[1].strip())
    uniq = sorted(set(labels))
    print("Unique labels (count={}):".format(len(uniq)), uniq)
except Exception as e:
    print("Could not parse labels.csv:", e)
PY
    else
      echo "labels.csv: MISSING" | tee -a "$OUT_FILE"
      # Try class folders heuristic if present: datasets/<ds>/images/<class>/*
      if [ -d "$ds/images" ]; then
        echo "Inferring classes from subfolders under images/ ..." | tee -a "$OUT_FILE"
        find "$ds/images" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort | tee -a "$OUT_FILE" || true
      fi
    fi
  done
else
  echo "No 'datasets/' directory found at $(pwd)" | tee -a "$OUT_FILE"
fi

# 6) Check expected project files
hdr "EXPECTED PROJECT FILES"
for f in \
  "src/data.py" \
  "src/train_loop.py" \
  "src/metrics.py" \
  "src/models/build.py" \
  "scripts/train_kfold.py" \
  "src/models/mobilevitv2.py" \
  "src/models/cbam.py" \
  "src/models/c2psa.py"
do
  if [ -e "$f" ]; then
    echo "[OK] $f" | tee -a "$OUT_FILE"
  else
    echo "[MISSING] $f" | tee -a "$OUT_FILE"
  fi
done

# 7) Python environment freeze (optional, saved separately)
hdr "PIP FREEZE (saved to diag_out/pip_freeze.txt)"
if command -v python >/dev/null 2>&1; then
  python -m pip freeze > "${OUT_DIR}/pip_freeze.txt" 2>/dev/null || true
  echo "Saved pip freeze to ${OUT_DIR}/pip_freeze.txt" | tee -a "$OUT_FILE"
else
  echo "Python not found; skipping pip freeze." | tee -a "$OUT_FILE"
fi

# 8) Disk space (to ensure room for checkpoints/plots)
hdr "DISK SPACE"
df -h | tee -a "$OUT_FILE"

echo -e "\nDONE. Full report at: ${OUT_FILE}"
