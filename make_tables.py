# make_tables.py
import json
from pathlib import Path

DATASETS = ["banana", "bean", "blackgram", "coconut", "corn", "guava", "papaya", "rice", "sunflower"]
VARIANTS = ["baseline", "cbam", "sam", "bam", "c2psa"]
NAMES = {"baseline": "Baseline", "cbam": "CBAM", "sam": "SAM", "bam": "BAM", "c2psa": "C2PSA"}

METRICS = [
    ("accuracy",  "accuracy_mean",  "accuracy_std"),
    ("f1",        "f1_mean",        "f1_std"),
    ("precision", "precision_mean", "precision_std"),
    ("recall",    "recall_mean",    "recall_std"),
]

def read_summary(dataset: str, variant: str):
    f = Path(f"runs/{dataset}/{variant}/summary_{variant}.json")
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None

def fmt_cell(mean, std):
    if mean is None or std is None:
        return "–"
    return f"{mean*100:.2f}% ± {std*100:.2f}"

def bold(s: str) -> str:
    return f"**{s}**"

def table_for_dataset(dataset: str) -> str:

    rows = {}
    for v in VARIANTS:
        summ = read_summary(dataset, v)
        if summ is None:
            rows[v] = {mkey: None for _, mkey, _ in METRICS}
            rows[v].update({skey: None for _, _, skey in METRICS})
        else:
            rows[v] = summ

    best_idx = {}
    for col_name, mean_key, _ in METRICS:
        best_val = None
        for v in VARIANTS:
            val = rows[v].get(mean_key) if rows[v] is not None else None
            if val is None:
                continue
            if (best_val is None) or (val > best_val):
                best_val = val
        best_idx[col_name] = best_val


    md = []
    md.append(f"### {dataset}")
    md.append("")
    md.append("| Variant | Accuracy | F1 | Precision | Recall |")
    md.append("|:--|:--:|:--:|:--:|:--:|")


    for v in VARIANTS:
        cells = [NAMES[v]]
        for col_name, mean_key, std_key in METRICS:
            mean = rows[v].get(mean_key)
            std  = rows[v].get(std_key)
            cell = fmt_cell(mean, std)
            if (mean is not None) and (best_idx[col_name] is not None) and abs(mean - best_idx[col_name]) < 1e-12:
                cell = bold(cell)
            cells.append(cell)
        md.append("| " + " | ".join(cells) + " |")

    md.append("")  
    return "\n".join(md)

def main():
    out_dir = Path("runs/report")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "metrics_tables.md"

    parts = []
    parts.append("# Results (5-fold CV, 200 epochs with augmentation)")
    parts.append("")
    for ds in DATASETS:
        parts.append(table_for_dataset(ds))

    out_md.write_text("\n".join(parts))
    print(f"Wrote {out_md}")

if __name__ == "__main__":
    main()
