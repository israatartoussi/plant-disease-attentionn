# Lightweight Attention Models for Plant Disease Classification

MobileViT-v2 baseline + attention (CBAM, BAM, SAM, C2PSA) on 9 datasets with 5-fold CV (200 epochs), plus reporting tools.

## Contents
- Training: `cv_core.py`, `mobilevitv2_*_cv.py`
- Reporting: `make_tables.py`, `make_pdf_matplotlib.py`
- Results (kept): `runs/report/` → `results_report.pdf`, `metrics_tables.md`, `combined_summary.csv`

## How to reproduce (example)
```bash
python mobilevitv2_baseline_cv.py --data datasets_merged/banana --epochs 200 --out runs/banana
python make_tables.py
python make_pdf_matplotlib.py


# 3) اكتب `.gitignore` نظيف (المرّة الماضية تقطّع السطر)
```bash
cat > .gitignore << 'EOF'
# Datasets
datasets*/
datasets_merged*/

# Training runs (keep only reports)
runs/*
!runs/report/**
!runs/report/results_report.pdf
!runs/report/metrics_tables.md
!runs/report/combined_summary.csv

# Checkpoints & large artifacts
*.pth
*.pt
*.ckpt

# Caches & temp
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
*.log
