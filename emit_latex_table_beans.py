import os, json
from glob import glob

METRICS_DIR = "cv5_bean_out/metrics"

# map filenames to pretty row labels (order matters)
ROW_ORDER = [
    ("MobileViTv2",               "MobileViTv2 (baseline)"),
    ("MobileViTv2_CBAM",          "+ CBAM"),
    ("MobileViTv2_BAM",           "+ BAM"),
    ("MobileViTv2_SAM",           "+ SAM"),
    ("MobileViTv2_C2PSA",         "+ C2PSA"),
]

def fmt_pct(mu, sd):
    return f"{mu*100:.2f} ± {sd*100:.2f}"

def fmt_dec(mu):
    return f"{mu:.3f}"

def load_one(name):
    path = os.path.join(METRICS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

rows = []
for key, label in ROW_ORDER:
    d = load_one(key)
    if d is None:
        continue
    acc = fmt_pct(d["accuracy_mean"], d["accuracy_std"])
    prec = fmt_dec(d["precision_mean"])
    rec  = fmt_dec(d["recall_mean"])
    f1   = fmt_dec(d["f1_mean"])
    rows.append((label, acc, prec, rec, f1))

latex = []
latex.append("\\begin{table}[H]")
latex.append("\\centering")
latex.append("\\caption{Classification results on the Bean Disease (Uganda) dataset (5-fold CV).}")
latex.append("\\label{tab:bean_results}")
latex.append("\\begin{tabular}{lcccc}")
latex.append("\\toprule")
latex.append("\\textbf{Model} & \\textbf{Accuracy (\\%)} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\")
latex.append("\\midrule")
for r in rows:
    latex.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\")
latex.append("\\bottomrule")
latex.append("\\end{tabular}")
latex.append("\\end{table}")

print("\n".join(latex))
