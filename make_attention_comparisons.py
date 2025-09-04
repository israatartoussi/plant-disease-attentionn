import argparse
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
matplotlib.use("Agg")

# ---------- model factories ----------
try:
    from models.mobilevitv2_baseline import get_mobilevitv2_base
    from models.mobilevitv2_cbam import get_mobilevitv2_cbam
    from models.mobilevitv2_bam import  get_mobilevitv2_bam
    from models.mobilevitv2_sam import  get_mobilevitv2_sam
    from models.mobilevitv2_c2psa import get_mobilevitv2_c2psa
except ImportError:
    from mobilevitv2_baseline import get_mobilevitv2_base
    from mobilevitv2_cbam import get_mobilevitv2_cbam
    from mobilevitv2_bam import  get_mobilevitv2_bam
    from mobilevitv2_sam import  get_mobilevitv2_sam
    from mobilevitv2_c2psa import get_mobilevitv2_c2psa

VARIANTS = {
    "baseline": get_mobilevitv2_base,
    "cbam":     get_mobilevitv2_cbam,
    "bam":      get_mobilevitv2_bam,
    "sam":      get_mobilevitv2_sam,
    "c2psa":    get_mobilevitv2_c2psa,
}
ATTN_VARIANTS = ["cbam", "bam", "sam", "c2psa"]

# ---------- utils ----------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model(variant: str, num_classes: int) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return VARIANTS[variant](num_classes=num_classes)

def get_val_transforms(img_size: int):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    tfm_no_norm = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        # intentionally no ToTensor/Normalize -> keeps PIL.Image for clean overlay
    ])
    tfm = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tfm, tfm_no_norm

def find_best_attention_variant(runs_root: Path, dataset: str) -> str:
    best, best_f1 = None, -1.0
    for v in ATTN_VARIANTS:
        p = runs_root / dataset / v / f"summary_{v}.json"
        if not p.exists():
            continue
        try:
            j = json.loads(p.read_text())
            f1 = float(j.get("f1_mean", -1.0))
            if f1 > best_f1:
                best_f1, best = f1, v
        except Exception:
            pass
    if best is None:
        raise RuntimeError(f"No attention summaries found under {runs_root/dataset}")
    return best

def find_best_fold_dir(variant_dir: Path) -> Path:
    best_dir, best_f1 = None, -1.0
    for k in range(1, 6):
        fd = variant_dir / f"fold{k}"
        tj = fd / "test_metrics.json"
        if tj.exists():
            try:
                j = json.loads(tj.read_text())
                f1 = float(j.get("test_f1", -1.0))
                if f1 > best_f1:
                    best_f1, best_dir = f1, fd
            except Exception:
                pass
    if best_dir is None:
        fd = variant_dir / "fold1"
        if not fd.exists():
            raise RuntimeError(f"No fold dirs in {variant_dir}")
        best_dir = fd
    return best_dir

def load_model_with_best_ckpt(variant: str, runs_root: Path, dataset: str,
                              n_classes: int, device: torch.device) -> Tuple[nn.Module, str]:
    m = build_model(variant, n_classes).to(device)
    var_dir = runs_root / dataset / variant
    fold_dir = find_best_fold_dir(var_dir)
    ckpt = fold_dir / "best.pth"
    if not ckpt.exists():
        raise RuntimeError(f"Missing checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state["state_dict"])
    m.eval()
    return m, fold_dir.name

@torch.no_grad()
def predict_softmax(model: nn.Module, x: torch.Tensor):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(prob.argmax())
    conf = float(prob.max())
    return prob, pred, conf

def try_forward_features(model: nn.Module, x: torch.Tensor):
    # 1) direct forward_features
    if hasattr(model, "forward_features"):
        try:
            return model.forward_features(x)
        except Exception:
            pass
    # 2) model.backbone.forward_features / forward
    if hasattr(model, "backbone"):
        bb = model.backbone
        if hasattr(bb, "forward_features"):
            try:
                return bb.forward_features(x)
            except Exception:
                pass
        if hasattr(bb, "forward"):
            try:
                return bb.forward(x)
            except Exception:
                pass
    return None

def _find_classifier_module(model: nn.Module) -> Optional[nn.Module]:
    last_linear = None
    last_conv1x1 = None
    named = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
            named = (name, m)
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
            last_conv1x1 = m
            named = (name, m)
        if any(k in name.lower() for k in ["head", "classifier", "classif", "fc"]):
            named = (name, m)
    if last_linear is not None:
        return last_linear
    if last_conv1x1 is not None:
        return last_conv1x1
    if named is not None:
        return named[1]
    return None

def get_features_from_model(model: nn.Module, x: torch.Tensor):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    # try feature forward paths
    feat = try_forward_features(model, x)
    if feat is None:
        # hook inputs to classifier-like module as penultimate features
        target = _find_classifier_module(model)
        if target is None:
            raise RuntimeError("Could not locate a classifier module to hook.")
        captured = []
        def pre_hook(_m, inp):
            captured.append(inp[0].detach())
        h = target.register_forward_pre_hook(pre_hook)
        _ = model(x)
        h.remove()
        if not captured:
            raise RuntimeError("Classifier hook did not capture features.")
        feat = captured[0]
    # pool/flatten
    if feat.ndim == 4:
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    else:
        feat = feat.flatten(1)
    return feat.detach().cpu().numpy()

def find_last_conv2d(module: nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def gradcam_on_tensor(model: nn.Module, x: torch.Tensor, device: torch.device,
                      target_class: Optional[int] = None):
    model.eval()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.to(device)
    target_layer = find_last_conv2d(model)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer for Grad-CAM.")
    acts, grads = [], []
    def fwd_hook(_m, _i, o): acts.append(o)
    def bwd_hook(_m, gi, go): grads.append(go[0])
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    logits = model(x)
    if target_class is None:
        target_class = int(logits.argmax(1).item())
    score = logits[:, target_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=True)
    A = acts[-1]   # [B,C,H,W]
    G = grads[-1]  # [B,C,H,W]
    w = G.mean(dim=(2, 3), keepdim=True)     # [B,C,1,1]
    cam = (w * A).sum(dim=1, keepdim=True)   # [B,1,H,W]
    cam = torch.relu(cam)
    cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                        mode="bilinear", align_corners=False)[0, 0]
    cam -= cam.min()
    cam /= (cam.max() + 1e-6)
    h1.remove(); h2.remove()
    return cam.detach().cpu().numpy()

def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.40):
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cmap = matplotlib.colormaps.get_cmap("jet")
    heat_rgb = (np.array(cmap(heat))[:, :, :3] * 255).astype(np.uint8)
    base = np.array(pil_img, dtype=np.float32)
    out = (1 - alpha) * base + alpha * heat_rgb
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def collect_tsne_features(model, ds, tfm, max_items, device):
    idxs = list(range(len(ds))); random.shuffle(idxs); idxs = idxs[:max_items]
    feats, labels = [], []
    for i in idxs:
        p, y = ds.samples[i]
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = get_features_from_model(model, x)
        feats.append(f[0]); labels.append(y)
    return np.stack(feats, 0), np.array(labels, dtype=int)

def plot_tsne_side_by_side(X1, y1, X2, y2, classes, best_name, out_png, out_pdf):
    ts1 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    ts2 = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42)
    Z1 = ts1.fit_transform(X1); Z2 = ts2.fit_transform(X2)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle(f"t-SNE — Baseline (MobileViT-v2) vs Best Attention ({best_name.upper()})",
                 fontsize=16, fontweight="bold", y=0.98)
    for ax, Z, y, title in [
        (axs[0], Z1, y1, "MobileViT-v2 (Baseline)"),
        (axs[1], Z2, y2, f"Best Attention ({best_name.upper()})"),
    ]:
        for ci, cname in enumerate(classes):
            m = (y == ci)
            if not np.any(m): continue
            ax.scatter(Z[m, 0], Z[m, 1], s=14, label=cname, alpha=0.8)
        ax.set_title(title); ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")
        ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    plt.close(fig)

# ---------- confusion matrices ----------
def get_test_indices_for_fold(labels, seed, fold_name):
    from sklearn.model_selection import StratifiedKFold
    k = int(fold_name.replace("fold", ""))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    i = 1
    for _, test_idx in skf.split(np.zeros(len(labels)), labels):
        if i == k:
            return np.array(test_idx, dtype=int)
        i += 1
    raise RuntimeError("Fold index out of range")

@torch.no_grad()
def infer_preds(model, paths, tfm, device):
    y_pred = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        logits = model(x)
        y_pred.append(int(torch.argmax(logits, dim=1).item()))
    return y_pred

def plot_confmat(cm, classes, title, out_png, out_pdf, normalize=False):
    if normalize:
        with np.errstate(invalid="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(classes)), labels=classes)
    fmt = ".2f" if normalize else "d"
    thr = cm.max()/2. if cm.size else 0.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black",
                    fontsize=8)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label"); ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    plt.close(fig)

def make_confmats_for_variant(variant, fold_name, model, ds, tfm, classes, seed, out_dir):
    labels = [y for _, y in ds.samples]
    test_idx = get_test_indices_for_fold(labels, seed, fold_name)
    test_paths = [Path(ds.samples[i][0]) for i in test_idx]
    y_true = [labels[i] for i in test_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred = infer_preds(model, test_paths, tfm, device)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    rep = classification_report(y_true, y_pred, target_names=classes, digits=4)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / f"confusion_{variant}.csv", cm.astype(int), fmt="%d", delimiter=",")
    (out_dir / f"classification_report_{variant}.txt").write_text(rep)

    plot_confmat(cm, classes, f"Confusion Matrix — {variant.capitalize()}",
                 out_dir / f"confusion_{variant}.png",
                 out_dir / f"confusion_{variant}.pdf", normalize=False)
    plot_confmat(cm, classes, f"Confusion Matrix (Normalized) — {variant.capitalize()}",
                 out_dir / f"confusion_{variant}_normalized.png",
                 out_dir / f"confusion_{variant}_normalized.pdf", normalize=True)
    return cm

def plot_confmat_side_by_side(cm_a, cm_b, classes, best_name, out_png, out_pdf):
    fig, axs = plt.subplots(1, 2, figsize=(9.8, 4.2))
    for ax, cm, title in [
        (axs[0], cm_a, "MobileViT-v2 (Baseline)"),
        (axs[1], cm_b, f"Best Attention ({best_name.upper()})"),
    ]:
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(classes)), labels=classes)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                        fontsize=8, color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.suptitle(f"Confusion Matrices — Baseline vs Best Attention ({best_name.upper()})",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    cb = fig.colorbar(axs[0].images[0], ax=axs, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel("Count", rotation=-90, va="bottom")
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    plt.close(fig)

# ---------- Grad-CAM pairs ----------
def make_gradcam_pairs(b_model, a_model, files, classes, tfm, tfm_no_norm, out_dir, device, n_samples=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "gradcam_pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for i, fp in enumerate(files[:n_samples], start=1):
        img_raw = Image.open(fp).convert("RGB")
        img_for_overlay = tfm_no_norm(img_raw)  # still PIL.Image
        x = tfm(img_raw).unsqueeze(0).to(device)

        _, b_pred, _ = predict_softmax(b_model, x)
        hm_b = gradcam_on_tensor(b_model, x, device, target_class=b_pred)
        ov_b = overlay_heatmap_on_image(img_for_overlay, hm_b, alpha=0.40)

        _, a_pred, _ = predict_softmax(a_model, x)
        hm_a = gradcam_on_tensor(a_model, x, device, target_class=a_pred)
        ov_a = overlay_heatmap_on_image(img_for_overlay, hm_a, alpha=0.40)

        fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4))
        fig.suptitle("Grad-CAM — MobileViT-v2 vs Best Attention", fontsize=16, fontweight="bold", y=0.98)
        axs[0].imshow(ov_b); axs[0].set_title("MobileViT-v2 (Baseline)"); axs[0].axis("off")
        axs[1].imshow(ov_a); axs[1].set_title("Best Attention"); axs[1].axis("off")
        fig.tight_layout()
        p_png = pairs_dir / f"pair_{i:02d}.png"
        p_pdf = pairs_dir / f"pair_{i:02d}.pdf"
        fig.savefig(p_png, dpi=400, bbox_inches="tight")
        fig.savefig(p_pdf, dpi=400, bbox_inches="tight")
        plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--tsne-max", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs_root = Path(args.runs_root)
    out_dir   = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ds = datasets.ImageFolder(args.data, transform=None)
    classes = ds.classes
    n_classes = len(classes)

    best_attn = find_best_attention_variant(runs_root, args.dataset)

    b_model, b_fold = load_model_with_best_ckpt("baseline", runs_root, args.dataset, n_classes, device)
    a_model, a_fold = load_model_with_best_ckpt(best_attn, runs_root, args.dataset, n_classes, device)

    tfm, tfm_no = get_val_transforms(args.img_size)

    # pick samples for Grad-CAM
    idxs = list(range(len(ds.samples))); random.shuffle(idxs)
    files_for_cam = [Path(ds.samples[i][0]) for i in idxs[:max(3, args.n_samples)]]
    make_gradcam_pairs(b_model, a_model, files_for_cam, classes, tfm, tfm_no, out_dir, device, n_samples=args.n_samples)

    # t-SNE
    Xb, yb = collect_tsne_features(b_model, ds, tfm, max_items=args.tsne_max, device=device)
    Xa, ya = collect_tsne_features(a_model, ds, tfm, max_items=args.tsne_max, device=device)
    plot_tsne_side_by_side(
        Xb, yb, Xa, ya, classes, best_attn,
        out_png=out_dir / "tsne_baseline_vs_best.png",
        out_pdf=out_dir / "tsne_baseline_vs_best.pdf",
    )

    # confusion matrices (same test fold IDs used during CV)
    cm_dir = out_dir / "confusion_matrices"
    cm_b = make_confmats_for_variant("baseline", b_fold, b_model, ds, tfm, classes, args.seed, cm_dir)
    cm_a = make_confmats_for_variant(best_attn, a_fold, a_model, ds, tfm, classes, args.seed, cm_dir)
    plot_confmat_side_by_side(
        cm_b, cm_a, classes, best_attn,
        out_png=cm_dir / "confusion_side_by_side.png",
        out_pdf=cm_dir / "confusion_side_by_side.pdf",
    )

    print(f"Done. Outputs in: {out_dir}  (best attention = {best_attn}, folds: baseline={b_fold}, best={a_fold})")

if __name__ == "__main__":
    main()
