# cv_core.py
import json, random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----- import model factories (package or script mode) -----
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

try:
    from tqdm import tqdm  # optional
except Exception:
    def tqdm(x, **k): return x

@dataclass
class TrainConfig:
    data: str
    out: str
    variant: str
    img_size: int = 256
    batch_size: int = 32
    epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 0.05
    seed: int = 42
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 1.0
    val_pct: float = 0.1  # inner validation percentage from the 4 training folds

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms(img_size: int):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize(int(img_size * 1.1)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms

    

def class_weights_from_subset(subset: Subset, num_classes: int) -> torch.Tensor:
    _, targets_all = zip(*subset.dataset.samples)
    subset_targets = [targets_all[i] for i in subset.indices]
    counts = np.bincount(subset_targets, minlength=num_classes)
    counts = np.maximum(counts, 1)
    w = 1.0 / counts
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)

def train_one_epoch(model, loader, device, criterion, optimizer, scaler, cfg: TrainConfig):
    model.train()
    running = 0.0
    for images, targets in tqdm(loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if cfg.amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        running += loss.item() * images.size(0)
    return running / max(1, len(loader.dataset))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_tgts = [], []
    for images, tgts in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(1).cpu().numpy().tolist()
        all_preds += preds
        all_tgts  += tgts.numpy().tolist()
    acc = accuracy_score(all_tgts, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_tgts, all_preds, average='weighted', zero_division=0)
    return acc, prec, rec, f1

def build_model(variant: str, num_classes: int) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant} (choose from {list(VARIANTS)})")
    return VARIANTS[variant](num_classes=num_classes)

def run_cv(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tfms, val_tfms = get_transforms(cfg.img_size)
    full_ds = datasets.ImageFolder(cfg.data, transform=None)
    labels = [y for _, y in full_ds.samples]
    n_classes = len(full_ds.classes)

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
    per_fold = []

    for fold, (train_outer_idx, test_idx) in enumerate(outer.split(np.zeros(len(labels)), labels), 1):
        y_outer = np.array(labels)[train_outer_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_pct, random_state=cfg.seed + fold)
        inner_train_rel, inner_val_rel = next(sss.split(np.zeros(len(train_outer_idx)), y_outer))
        train_idx = np.array(train_outer_idx)[inner_train_rel].tolist()
        val_idx   = np.array(train_outer_idx)[inner_val_rel].tolist()

        train_subset = Subset(datasets.ImageFolder(cfg.data, transform=train_tfms), train_idx)
        val_subset   = Subset(datasets.ImageFolder(cfg.data, transform=val_tfms),   val_idx)
        test_subset  = Subset(datasets.ImageFolder(cfg.data, transform=val_tfms),   test_idx)

        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=True)
        test_loader  = DataLoader(test_subset, batch_size=cfg.batch_size, shuffle=False,
                                  num_workers=cfg.num_workers, pin_memory=True)
                # --- resume support: skip finished folds, or finalize from best.pth ---
        ckpt_dir = out_dir / cfg.variant / f"fold{fold}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt_dir / "best.pth"
        done_json = ckpt_dir / "test_metrics.json"

        
        if done_json.exists():
            try:
                per_fold.append(json.loads(done_json.read_text()))
                print(f"[resume] Skip fold{fold}: found {done_json}")
                continue
            except Exception:
                pass  

        # إذا في best.pth بس ما في test_metrics، قيّم واكتبها وكمّل
        if best_ckpt.exists() and not done_json.exists():
            model = build_model(cfg.variant, n_classes).to(device)
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["state_dict"])
            acc_t, prec_t, rec_t, f1_t = evaluate(model, test_loader, device)
            fold_row = {
                "fold": fold,
                "test_accuracy": float(acc_t),
                "test_precision": float(prec_t),
                "test_recall": float(rec_t),
                "test_f1": float(f1_t),
            }
            per_fold.append(fold_row)
            (ckpt_dir / "test_metrics.json").write_text(json.dumps(fold_row, indent=2))
            print(f"[resume] Finalized fold{fold} from existing best.pth -> {done_json}")
            continue
        # --- end resume support ---


        model = build_model(cfg.variant, n_classes).to(device)

        cw = class_weights_from_subset(train_subset, n_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

        best_f1 = -1.0
        ckpt_dir = out_dir / cfg.variant / f"fold{fold}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt_dir / "best.pth"

        for _ in range(cfg.epochs):
            _ = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, cfg)
            _, _, _, f1_v = evaluate(model, val_loader, device)
            scheduler.step()
            if f1_v > best_f1:
                best_f1 = f1_v
                torch.save({
                    "state_dict": model.state_dict(),
                    "variant": cfg.variant,
                    "classes": full_ds.classes,
                }, best_ckpt)

        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        acc_t, prec_t, rec_t, f1_t = evaluate(model, test_loader, device)

        fold_row = {
            "fold": fold,
            "test_accuracy": float(acc_t),
            "test_precision": float(prec_t),
            "test_recall": float(rec_t),
            "test_f1": float(f1_t),
        }
        per_fold.append(fold_row)
        (ckpt_dir / "test_metrics.json").write_text(json.dumps(fold_row, indent=2))

    # aggregate
    import pandas as pd, csv
    df = pd.DataFrame(per_fold)
    summary = {
        "variant": cfg.variant,
        "epochs": cfg.epochs,
        "img_size": cfg.img_size,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "val_pct": cfg.val_pct,
        "accuracy_mean": float(df["test_accuracy"].mean()),
        "precision_mean": float(df["test_precision"].mean()),
        "recall_mean": float(df["test_recall"].mean()),
        "f1_mean": float(df["test_f1"].mean()),
        "accuracy_std": float(df["test_accuracy"].std(ddof=0)),
        "precision_std": float(df["test_precision"].std(ddof=0)),
        "recall_std": float(df["test_recall"].std(ddof=0)),
        "f1_std": float(df["test_f1"].std(ddof=0)),
        "n_folds": len(df),
        "note": "Means/stds computed on held-out TEST folds.",
    }
    var_dir = out_dir / cfg.variant
    var_dir.mkdir(parents=True, exist_ok=True)
    (var_dir / f"summary_{cfg.variant}.json").write_text(json.dumps(summary, indent=2))

    csv_path = var_dir / f"metrics_{cfg.variant}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "test_accuracy", "test_precision", "test_recall", "test_f1"])
        writer.writeheader()
        for r in per_fold: writer.writerow(r)

    return summary
