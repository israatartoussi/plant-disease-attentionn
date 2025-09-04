import os, numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

# ---------- High-DPI helpers ----------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_confusion_matrix_highdpi(y_true, y_pred, class_names, out_path, dpi=600):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)  # default colormap
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def to_tsne_2d(X, perplexity=30):
    if X.ndim == 2 and X.shape[1] == 2:
        return X
    return TSNE(n_components=2, perplexity=perplexity, learning_rate="auto",
                init="pca", random_state=42).fit_transform(X)

def save_tsne_side_by_side(X_base, X_attn, y, out_path, title_left, title_right, dpi=600):
    Zb = to_tsne_2d(X_base)
    Za = to_tsne_2d(X_attn)

    pad = 5.0
    def lims(Z):
        x0, y0 = Z.min(axis=0); x1, y1 = Z.max(axis=0)
        return (x0 - pad, x1 + pad, y0 - pad, y1 + pad)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(Zb[:, 0], Zb[:, 1], c=y, s=14)
    ax1.set_title(title_left, fontsize=14)
    x0,x1,y0,y1 = lims(Zb); ax1.set_xlim(x0,x1); ax1.set_ylim(y0,y1)
    ax1.set_xlabel("t-SNE 1", fontsize=12); ax1.set_ylabel("t-SNE 2", fontsize=12)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(Za[:, 0], Za[:, 1], c=y, s=14)
    ax2.set_title(title_right, fontsize=14)
    x0,x1,y0,y1 = lims(Za); ax2.set_xlim(x0,x1); ax2.set_ylim(y0,y1)
    ax2.set_xlabel("t-SNE 1", fontsize=12); ax2.set_ylabel("t-SNE 2", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def compose_gradcam_top_bottom(img_top_path, img_bottom_path, out_path, dpi=600,
                               title_top="Baseline", title_bottom="Attention"):
    if not (os.path.exists(img_top_path) and os.path.exists(img_bottom_path)):
        print(f"[WARN] Missing Grad-CAMs: {img_top_path} | {img_bottom_path}")
        return
    top = plt.imread(img_top_path)
    bot = plt.imread(img_bottom_path)

    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(2,1,1); ax1.imshow(top); ax1.axis("off"); ax1.set_title(title_top, fontsize=14)
    ax2 = fig.add_subplot(2,1,2); ax2.imshow(bot); ax2.axis("off"); ax2.set_title(title_bottom, fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- Batch config per dataset ----------

# عدّلي class_names وأفضل attention variant لكل داتاست حسب نتائجك
CONFIGS = [
    # name, classes, best_attn_key, human label for figure
    ("corn",       ["Healthy","Rust","Gray Leaf Spot","Blight"],       "sam",   "MobileViTv2 + SAM"),
    ("bean",       ["Healthy","Angular Leaf Spot","Rust"],             "c2psa", "MobileViTv2 + C2PSA"),
    ("guava",      ["Healthy","Canker","Rust"],                        "c2psa", "MobileViTv2 + C2PSA"),
    ("papaya",     ["Healthy","Black Spot","Powdery Mildew"],          "c2psa", "MobileViTv2 + C2PSA"),
    ("blackgram",  ["Healthy","Yellow Mosaic","Leaf Spot"],            "c2psa", "MobileViTv2 + C2PSA"),
    ("banana",     ["Cordana","Healthy","Pestalotiopsis","Sigatoka"],  "bam",   "MobileViTv2 + BAM"),
    ("coconut",    ["Healthy","Bud Rot","Leaf Spot"],                  "c2psa", "MobileViTv2 + C2PSA"),
    ("rice",       ["Bacterial Blight","Brown Spot","Leaf Smut"],      "c2psa", "MobileViTv2 + C2PSA"),
    ("sunflower",  ["Healthy","Rust","Blight"],                        "bam",   "MobileViTv2 + BAM"),
]

def run_all(root_data="data", root_gradcam="gradcam", out_root="figures"):
    ensure_dir(out_root)
    for name, class_names, best_key, best_label in CONFIGS:
        print(f"\n=== {name.upper()} ===")
        ds_in = os.path.join(root_data, name)
        ds_out = os.path.join(out_root, name)
        ensure_dir(ds_out)

        # --- Confusion matrix ---
        y_true_p = os.path.join(ds_in, "y_true.npy")
        y_pred_p = os.path.join(ds_in, f"y_pred_{best_key}.npy")
        if os.path.exists(y_true_p) and os.path.exists(y_pred_p):
            y_true = np.load(y_true_p); y_pred = np.load(y_pred_p)
            save_confusion_matrix_highdpi(
                y_true, y_pred, class_names,
                os.path.join(ds_out, f"{name}_confusion_{best_key}.png"),
                dpi=600
            )
        else:
            print(f"[WARN] Skip confusion: missing {y_true_p} or {y_pred_p}")

        # --- t-SNE ---
        f_base = os.path.join(ds_in, "features_baseline.npy")
        f_attn = os.path.join(ds_in, f"features_{best_key}.npy")
        labels_p = os.path.join(ds_in, "labels.npy")
        if all(os.path.exists(p) for p in [f_base, f_attn, labels_p]):
            Xb = np.load(f_base); Xa = np.load(f_attn); y = np.load(labels_p)
            save_tsne_side_by_side(
                Xb, Xa, y,
                os.path.join(ds_out, f"{name}_tsne_baseline_vs_{best_key}.png"),
                title_left="MobileViTv2",
                title_right=best_label,
                dpi=600
            )
        else:
            print(f"[WARN] Skip t-SNE: need {f_base}, {f_attn}, {labels_p}")

        # --- Grad-CAM compose ---
        g_top = os.path.join(root_gradcam, f"{name}_baseline.png")
        g_bot = os.path.join(root_gradcam, f"{name}_{best_key}.png")
        compose_gradcam_top_bottom(
            g_top, g_bot,
            os.path.join(ds_out, f"{name}_gradcam_baseline_vs_{best_key}.png"),
            dpi=600,
            title_top="MobileViTv2",
            title_bottom=best_label
        )

if __name__ == "__main__":
    run_all()
