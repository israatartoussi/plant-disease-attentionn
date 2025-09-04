
import argparse, json, os, random, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# نستفيد من نفس الدوال المستعملة بالتدريب
from cv_core import build_model, get_transforms

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def softmax_np(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-9, None)

def plot_cm(cm, classes, out_png, title="Confusion Matrix", normalize=False):
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
    fig, ax = plt.subplots(figsize=(5,4), dpi=160)
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, ylabel='True', xlabel='Pred')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(j, i, format(val, fmt), ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)
    ax.set_title(title if not normalize else title + " (normalized)")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def tensor_to_pil(img_tensor):
    # بدون normalize بالتدريب، فـ ToPILImage كافي
    from torchvision.transforms.functional import to_pil_image
    return to_pil_image(img_tensor.cpu().detach().clamp(0,1))

def get_last_conv_module(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last

def gradcam_single(model, img_tensor, target_class, device):
    """
    محاولة Grad-CAM عامة على آخر Conv2d؛ إذا فشلت، منرجّع None
    """
    model.eval()
    conv = get_last_conv_module(model)
    if conv is None:
        return None  # ما في Convs
    fmap = []
    grads = []

    def fwd_hook(m, inp, out):
        fmap.append(out.detach())

    def bwd_hook(m, gin, gout):
        grads.append(gout[0].detach())

    h1 = conv.register_forward_hook(fwd_hook)
    h2 = conv.register_full_backward_hook(bwd_hook)

    try:
        img_tensor = img_tensor.to(device, non_blocking=True).unsqueeze(0)
        img_tensor.requires_grad_(True)
        logits = model(img_tensor)
        if isinstance(logits, (list, tuple)): logits = logits[0]
        score = logits[0, int(target_class)]
        model.zero_grad(set_to_none=True)
        score.backward()

        # تجميع الأوزان
        g = grads[-1]        # [B,C,H,W]
        a = fmap[-1]         # [B,C,H,W]
        weights = g.mean(dim=(2,3), keepdim=True)  # GAP على H,W => [B,C,1,1]
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))  # [B,1,H,W]
        cam = F.interpolate(cam, size=img_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0,0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()
    except Exception:
        return None
    finally:
        h1.remove(); h2.remove()

def overlay_cam_on_pil(pil_img, cam_np, alpha=0.35):
    """ overlay باستخدام matplotlib colormap ثم PIL alpha blend """
    from matplotlib import cm
    cmap = cm.get_cmap('jet')
    heat = cmap(cam_np)[:, :, :3]  # RGB
    heat = (heat * 255).astype(np.uint8)
    from PIL import Image
    heat_pil = Image.fromarray(heat).resize(pil_img.size, resample=Image.BILINEAR)
    return Image.blend(pil_img.convert("RGB"), heat_pil.convert("RGB"), alpha)

def run_for_fold(k, args, classes, labels, full_val_tfms):
    seed = args.seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # أعِد نفس تقسيم الـ 5-fold المستعمل بالتدريب (نفس seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(labels)), labels))
    _, test_idx = folds[k-1]

    fold_dir = Path(args.out)/args.variant/f"fold{k}"
    ckpt_path = fold_dir/"best.pth"
    if not ckpt_path.exists():
        print(f"[fold{k}] ⚠️ Skip: {ckpt_path} غير موجود.")
        return

    # داتا لوادر
    test_ds = Subset(datasets.ImageFolder(args.data, transform=full_val_tfms), test_idx.tolist())
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # نموذج
    model = build_model(args.variant, num_classes=len(classes)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    model.eval()

    print(f"[fold{k}] Generating artifacts…")

    # ====== تجميع التنبؤات/اللوجِتس ======
    all_logits = []
    all_preds  = []
    all_tgts   = []
    all_paths  = []
    with torch.no_grad():
        for x, t in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if isinstance(logits, (list, tuple)): logits = logits[0]
            all_logits.append(logits.cpu().numpy())
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_tgts.append(t.cpu().numpy())
    if len(all_logits)==0:
        print(f"[fold{k}] ⚠️ لا عينات؟")
        return
    logits_np = np.concatenate(all_logits, axis=0)
    preds_np  = np.concatenate(all_preds, axis=0)
    tgts_np   = np.concatenate(all_tgts, axis=0)

    # نستخرج المسارات بترتيب test_loader (Subset يسحب indices بالترتيب)
    base_ds = datasets.ImageFolder(args.data, transform=None)
    test_paths = [ base_ds.samples[i][0] for i in test_idx.tolist() ]

    # ====== CM / Report / CSV ======
    cm = confusion_matrix(tgts_np, preds_np, labels=list(range(len(classes))))
    np.savetxt(fold_dir/"confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")
    plot_cm(cm, classes, fold_dir/"confusion_matrix.png", title="Confusion Matrix", normalize=False)
    plot_cm(cm, classes, fold_dir/"confusion_matrix_normalized.png", title="Confusion Matrix", normalize=True)

    rep = classification_report(tgts_np, preds_np, target_names=classes, digits=4)
    (fold_dir/"classification_report.txt").write_text(rep)

    probs = softmax_np(logits_np)
    conf = probs.max(axis=1)

    import csv
    with open(fold_dir/"test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","true_idx","true_name","pred_idx","pred_name","correct","confidence"])
        for p, t, y, c in zip(test_paths, tgts_np, preds_np, conf):
            w.writerow([p, int(t), classes[int(t)], int(y), classes[int(y)], bool(t==y), float(c)])

    # ====== t-SNE على اللوجِتس ======
    try:
        n_samples = logits_np.shape[0]
        perplexity = min(30, max(5, n_samples//5))
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto",
                    random_state=seed, perplexity=perplexity)
        emb = tsne.fit_transform(logits_np)
        fig, ax = plt.subplots(figsize=(5,4), dpi=160)
        scatter = ax.scatter(emb[:,0], emb[:,1], c=tgts_np, s=8, alpha=0.8, cmap='tab10')
        legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Class")
        ax.add_artist(legend1)
        ax.set_title("t-SNE (logits)")
        fig.tight_layout()
        fig.savefig(fold_dir/"tsne.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        (fold_dir/"tsne_ERROR.txt").write_text(str(e))

    # ====== Grad-CAM لعدد n-gradcam ======
    try:
        n = args.n_gradcam
        if n > 0:
            # فضّل العينات الغلط، وإذا قليلة كمّل عشوائي صح
            mis_idx = np.where(tgts_np != preds_np)[0].tolist()
            cor_idx = np.where(tgts_np == preds_np)[0].tolist()
            random.shuffle(mis_idx); random.shuffle(cor_idx)
            chosen = (mis_idx[:n] + cor_idx[:max(0, n-len(mis_idx))])[:n]

            grad_dir = fold_dir / "gradcam"
            ensure_dir(grad_dir)

            # لازم ناخد الصور الأصلية بنفس ترانسفورم الفاليد
            _, val_tfms = get_transforms(args.img_size)
            val_base = datasets.ImageFolder(args.data, transform=val_tfms)
            for j, global_i in enumerate(chosen, 1):
                idx_in_base = test_idx.tolist()[global_i]
                pil_img, true_t = val_base[idx_in_base]
                # حوّل لتنسور CHW
                if not torch.is_tensor(pil_img):
                    # الحالة الافتراضية: ImageFolder مع transform بيرجع Tensor
                    from torchvision.transforms.functional import to_tensor
                    x = to_tensor(pil_img)
                else:
                    x = pil_img
                cam = gradcam_single(model, x, int(preds_np[global_i]), device)
                if cam is None:
                    # fallback: vanilla saliency
                    x1 = x.to(device).unsqueeze(0).requires_grad_(True)
                    out = model(x1)
                    if isinstance(out, (list, tuple)): out = out[0]
                    cls = int(preds_np[global_i])
                    score = out[0, cls]
                    model.zero_grad(set_to_none=True)
                    score.backward()
                    sal = x1.grad.abs().max(dim=1)[0].detach().cpu().numpy()[0]
                    sal -= sal.min(); sal /= (sal.max()+1e-8)
                    cam = sal
                # حفظ صورة overlay
                pil_orig = tensor_to_pil(x)
                over = overlay_cam_on_pil(pil_orig, cam)
                out_name = f"idx{global_i:04d}_true-{classes[int(tgts_np[global_i])]}_pred-{classes[int(preds_np[global_i])]}_conf-{conf[global_i]:.2f}.png"
                over.save(grad_dir/out_name)
    except Exception as e:
        (fold_dir/"gradcam_NOTE.txt").write_text(
            "Grad-CAM fallback or error:\n"+str(e)
        )

    print(f"[fold{k}] Done -> {fold_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="ImageFolder root")
    p.add_argument("--variant", required=True, choices=["baseline","cbam","sam","bam","c2psa"])
    p.add_argument("--out", required=True, help="runs/<dataset>")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-gradcam", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)

    # داتا كاملة (لفهرسة المسارات + لائحة اللابلز)
    full_ds = datasets.ImageFolder(args.data, transform=None)
    labels = [y for _, y in full_ds.samples]
    classes = full_ds.classes

    # نفس ترانسفورم الفاليد
    _, val_tfms = get_transforms(args.img_size)

    # نفّذ لكل فولد 1..5
    for k in range(1, 6):
        run_for_fold(k, args, classes, labels, val_tfms)

if __name__ == "__main__":
    main()

