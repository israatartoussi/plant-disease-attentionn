# make_pdf_matplotlib.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DATASETS = ["banana","bean","blackgram","coconut","corn","guava","papaya","rice","sunflower"]
VARIANTS = [
    ("baseline","Baseline"),
    ("cbam","Baseline+CBAM"),
    ("sam","Baseline+SAM"),
    ("bam","Baseline+BAM"),
    ("c2psa","Baseline+C2PSA"),
]

BASE = Path("runs")
REPORT_DIR = BASE/"report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def read_summary(ds, var):
    p = BASE/ds/var/f"summary_{var}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def pct(x):
    try:
        return f"{100.0*float(x):.2f}"
    except Exception:
        return "N/A"

csv_lines = ["dataset,method,accuracy,f1,precision,recall"]
for ds in DATASETS:
    for var,label in VARIANTS:
        s = read_summary(ds, var)
        if s is None:
            csv_lines.append(f"{ds},{label},N/A,N/A,N/A,N/A")
        else:
            csv_lines.append(",".join([
                ds, label,
                pct(s["accuracy_mean"]), pct(s["f1_mean"]),
                pct(s["precision_mean"]), pct(s["recall_mean"])
            ]))
(REPORT_DIR/"combined_summary.csv").write_text("\n".join(csv_lines))

pdf_path = REPORT_DIR/"results_report.pdf"
with PdfPages(pdf_path) as pdf:
    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 عمودي تقريبا
        ax.axis('off')
        ax.set_title(f"{ds} — 5-Fold CV (200 epochs, with augmentation)", fontsize=16, pad=20)

        header = ["Method", "Accuracy (%)", "F1 (%)", "Precision (%)", "Recall (%)"]
        rows = []
        for var,label in VARIANTS:
            s = read_summary(ds, var)
            if s is None:
                rows.append([label, "N/A","N/A","N/A","N/A"])
            else:
                rows.append([
                    label,
                    pct(s["accuracy_mean"]),
                    pct(s["f1_mean"]),
                    pct(s["precision_mean"]),
                    pct(s["recall_mean"]),
                ])

        table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)

        ax.text(0.5, 0.06,
                "Means/stds computed on held-out TEST folds. N/A = run missing.",
                ha='center', va='center', fontsize=9)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

print("Wrote:", pdf_path)
print("Also wrote:", REPORT_DIR/"combined_summary.csv")
