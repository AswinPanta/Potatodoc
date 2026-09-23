"""
Potato Disease Detection — Stable Training on PlantVillage
==========================================================
Adapted from potato_doc_production_v2_fixed.py for small dataset (2,152 images).

Key adaptations:
  1. K-fold cross-validation (5-fold) for reliable metrics
  2. WeightedRandomSampler for severe class imbalance (152 healthy vs 1000 disease)
  3. Heavy augmentation to compensate for small data
  4. Lighter models to prevent overfitting
  5. Two-phase training with conservative fine-tuning
  6. Stability assessment (variance across folds)

Dataset: PlantVillage potato subset
  Potato___Early_blight/   (1,000 images)
  Potato___Late_blight/    (1,000 images)
  Potato___healthy/        (152 images)
"""

import os
import json
import time
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
    balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
NUM_CLASSES = 3
N_FOLDS = 5
EPOCHS_HEAD = 15
EPOCHS_FINETUNE = 40
PATIENCE = 10
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
WEIGHT_DECAY = 0.05
DROPPATH = 0.1
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4

# ============================================================
# File handling code: Read / create directory
# (independent of platform, created ahead, where Python code is)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent  # where this Python code is
# Dataset lives in sibling ../dataset (resolved relative to code, no hardcode)
DATA_DIR = (BASE_DIR.parent / "dataset" / "PlantVillage").resolve()
RESULTS_DIR = BASE_DIR / "results_plantvillage"


def ensure_dir(path: Path) -> Path:
    """Generate directory independent of platform, created ahead."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# Create necessary directory ahead (before any core ML code runs)
ensure_dir(RESULTS_DIR)

# ============================================================
# Your Core Machine learning Codes
# ============================================================

CLASS_DIRS = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]
CLASS_NAMES = ["Early Bllight", "Late Blight", "Healthy"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Lighter models for small dataset
MODELS_CONFIG = [
    {"name": "efficientnet_b0", "timm_name": "efficientnet_b0", "img_size": 224, "batch_size": 32},
    {"name": "convnext_tiny", "timm_name": "convnext_tiny.fb_in22k", "img_size": 224, "batch_size": 32},
    {"name": "mobilenetv2", "timm_name": "mobilenetv2_100", "img_size": 224, "batch_size": 32},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


device = get_device()
set_seed(SEED)

# ============================================================
# DATA LOADING
# ============================================================
print("Scanning PlantVillage dataset...")
records = []
for cls_idx, cls_dir in enumerate(CLASS_DIRS):
    cls_path = DATA_DIR / cls_dir
    for f in sorted(cls_path.glob("*.*")):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            records.append({"path": str(f), "class_idx": cls_idx, "class_name": CLASS_NAMES[cls_idx]})
df = pd.DataFrame(records)
print(f"Total images: {len(df)}")

dist = df["class_name"].value_counts()
for cls, cnt in dist.items():
    print(f"  {cls:15s}: {cnt:6d} ({cnt/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {dist.max()/dist.min():.1f}:1")

# ============================================================
# AUGMENTATION
# ============================================================
def get_transforms(img_size, is_train=True, strong=False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if HAS_ALB:
        if is_train:
            tfs = [
                A.RandomResizedCrop((img_size, img_size), scale=(0.4 if strong else 0.6, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
            ]
            if strong:
                tfs += [
                    A.OneOf([
                        A.RandomBrightnessContrast(0.3, 0.3, p=1),
                        A.HueSaturationValue(20, 30, 20, p=1),
                        A.CLAHE(clip_limit=2, p=1),
                    ], p=0.8),
                    A.OneOf([
                        A.GaussianBlur(3, p=1),
                        A.GaussNoise(10, 50, p=1),
                        A.MotionBlur(3, p=1),
                    ], p=0.3),
                    A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, p=0.3),
                ]
            else:
                tfs += [A.ColorJitter(0.3, 0.3, 0.3, 0.1, p=0.5)]
            tfs += [A.Normalize(IMAGENET_MEAN, IMAGENET_STD), ToTensorV2()]
            return A.Compose(tfs)
        return A.Compose([
            A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
            A.CenterCrop(img_size, img_size),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])
    if is_train:
        tfs = [
            transforms.RandomResizedCrop(img_size, scale=(0.4 if strong else 0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
        ]
        if strong:
            tfs += [transforms.RandAugment(num_ops=3, magnitude=9), transforms.RandomGrayscale(p=0.1)]
        else:
            tfs += [transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)]
        tfs += [transforms.ToTensor(), norm]
        return transforms.Compose(tfs)
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ])

# ============================================================
# DATASET & SAMPLER
# ============================================================
class PotatoDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            if self.transform:
                if HAS_ALB and not isinstance(self.transform, transforms.Compose):
                    img = self.transform(image=np.array(img))["image"]
                else:
                    img = self.transform(img)
            return img, self.labels[idx]
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))


def make_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)


def make_loaders(Xtr, ytr, Xva, yva, img_size, batch_size, strong=False):
    tr_ds = PotatoDataset(Xtr, ytr, get_transforms(img_size, True, strong))
    va_ds = PotatoDataset(Xva, yva, get_transforms(img_size, False))
    sampler = make_weighted_sampler(ytr)
    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(tr_ds, batch_size, sampler=sampler, **kw),
        DataLoader(va_ds, batch_size * 2, shuffle=False, **kw),
    )

# ============================================================
# MODEL
# ============================================================
def build_model(timm_name, num_classes=NUM_CLASSES, drop_path=0.0):
    return timm.create_model(timm_name, pretrained=True, num_classes=num_classes, drop_path_rate=drop_path)


def cpu_sd(sd):
    return {k: v.detach().cpu() for k, v in sd.items()}


def save_ckpt(path, payload):
    ensure_dir(Path(path).parent)  # create necessary directory ahead
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

# ============================================================
# TRAINING LOOP
# ============================================================
def train_one_epoch(model, loader, crit, opt, scaler, mix_fn, device):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yin = y
        if mix_fn is not None:
            if x.size(0) % 2 != 0:
                x = x[:-1]
                y = y[:-1]
            if x.size(0) >= 2:
                x, yin = mix_fn(x, y)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type if device.type == "cuda" else "cpu"):
            out = model(x)
            loss = crit(out, yin)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tl += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return tl / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, L, tl, tot = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        o = model(x)
        tl += ce(o, y).item() * x.size(0)
        tot += x.size(0)
        P.append(o.argmax(1).cpu().numpy())
        L.append(y.cpu().numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    return {
        "loss": tl / tot,
        "accuracy": float((P == L).mean()),
        "macro_f1": float(f1_score(L, P, average="macro")),
        "predictions": P,
        "labels": L,
    }


def train_fold(fold, model_cfg, Xtr, ytr, Xva, yva, device):
    ensure_dir(RESULTS_DIR)  # create necessary directory ahead
    name = model_cfg["name"]
    ckpt_path = RESULTS_DIR / f"{name}_fold{fold}_best.pth"

    if ckpt_path.exists():
        print(f"  [SKIP] {name} fold {fold} checkpoint exists")
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    set_seed(SEED + fold)
    img_size = model_cfg["img_size"]
    bs = model_cfg["batch_size"]

    tr_loader, va_loader = make_loaders(Xtr, ytr, Xva, yva, img_size, bs, strong=False)
    model = build_model(model_cfg["timm_name"], drop_path=DROPPATH).to(device)

    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=np.array(ytr))
    cw = torch.tensor(cw, dtype=torch.float32).to(device)

    mix_fn = Mixup(
        mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA, prob=0.8,
        switch_prob=0.5, label_smoothing=LABEL_SMOOTH, num_classes=NUM_CLASSES
    )

    # Phase 1: head-only
    for p in model.parameters():
        p.requires_grad = False
    head = model.classifier if hasattr(model, "classifier") else model.head
    for p in head.parameters():
        p.requires_grad = True

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = optim.AdamW(head_params, lr=1e-3, weight_decay=0.01)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS_HEAD)
    scaler1 = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")

    print(f"\n--- {name} Fold {fold} Phase 1: head-only ({EPOCHS_HEAD} epochs) ---")
    best_vf1 = 0
    best_state_p1 = None

    for ep in range(1, EPOCHS_HEAD + 1):
        tl, ta = train_one_epoch(model, tr_loader, nn.CrossEntropyLoss(weight=cw), opt1, scaler1, None, device)
        vr = evaluate(model, va_loader, device)
        sched1.step()
        tag = "*" if vr["macro_f1"] > best_vf1 else ""
        if vr["macro_f1"] > best_vf1:
            best_vf1 = vr["macro_f1"]
            best_state_p1 = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  ep {ep:2d} | train_loss={tl:.4f} acc={ta:.4f} | val_f1={vr['macro_f1']:.4f}{tag}")

    model.load_state_dict(best_state_p1)

    # Phase 2: full fine-tune
    for p in model.parameters():
        p.requires_grad = True

    head_names = ["classifier", "head"]
    head_ids = set()
    for n, p in model.named_parameters():
        if any(hn in n for hn in head_names):
            head_ids.add(id(p))

    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    head_params = [p for p in model.parameters() if id(p) in head_ids]

    opt2 = optim.AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS_FINETUNE)
    scaler2 = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")
    crit_soft = SoftTargetCrossEntropy()

    print(f"--- {name} Fold {fold} Phase 2: fine-tune (max {EPOCHS_FINETUNE} epochs, patience={PATIENCE}) ---")
    best_vf1_p2 = 0
    best_state_p2 = None
    wait = 0

    for ep in range(1, EPOCHS_FINETUNE + 1):
        tl, ta = train_one_epoch(model, tr_loader, crit_soft, opt2, scaler2, mix_fn, device)
        vr = evaluate(model, va_loader, device)
        sched2.step()

        tag = "*" if vr["macro_f1"] > best_vf1_p2 else ""
        if vr["macro_f1"] > best_vf1_p2:
            best_vf1_p2 = vr["macro_f1"]
            best_state_p2 = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print(f"  ep {ep:2d} | train_loss={tl:.4f} acc={ta:.4f} | val_f1={vr['macro_f1']:.4f}{tag}")

        if wait >= PATIENCE:
            print(f"  Early stopping at epoch {ep}")
            break

    model.load_state_dict(best_state_p2)
    save_ckpt(ckpt_path, {
        "state_dict": cpu_sd(model.state_dict()),
        "timm_name": model_cfg["timm_name"],
        "img_size": model_cfg["img_size"],
        "class_names": CLASS_NAMES,
        "val_f1": best_vf1_p2,
        "fold": fold,
    })
    print(f"  Saved {ckpt_path.name} (val_f1={best_vf1_p2:.4f})")
    del model
    torch.cuda.empty_cache()
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)

# ============================================================
# K-FOLD CROSS-VALIDATION
# ============================================================
print("\n" + "=" * 60)
print("K-FOLD CROSS-VALIDATION (5-Fold)")
print("=" * 60)

X_all = df["path"].tolist()
y_all = df["class_idx"].tolist()

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_fold_results = {}

for model_cfg in MODELS_CONFIG:
    model_name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")
        Xtr = [X_all[i] for i in train_idx]
        ytr = [y_all[i] for i in train_idx]
        Xva = [X_all[i] for i in val_idx]
        yva = [y_all[i] for i in val_idx]

        print(f"  Train: {len(Xtr)} (class dist: {Counter(ytr)})")
        print(f"  Val:   {len(Xva)} (class dist: {Counter(yva)})")

        ckpt = train_fold(fold, model_cfg, Xtr, ytr, Xva, yva, device)
        fold_results.append(ckpt["val_f1"])

    mean_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    all_fold_results[model_name] = {
        "fold_f1s": fold_results,
        "mean_f1": float(mean_f1),
        "std_f1": float(std_f1),
    }
    print(f"\n{model_name} Results: {mean_f1:.4f} +/- {std_f1:.4f}")
    for i, f1 in enumerate(fold_results):
        print(f"  Fold {i+1}: {f1:.4f}")

# ============================================================
# SELECT BEST MODEL
# ============================================================
best_model_name = max(all_fold_results.keys(), key=lambda n: all_fold_results[n]["mean_f1"])
best_cfg = next(c for c in MODELS_CONFIG if c["name"] == best_model_name)
print(f"\nBest model: {best_model_name} (mean F1: {all_fold_results[best_model_name]['mean_f1']:.4f})")

# ============================================================
# FINAL 80/20 EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("FINAL TRAINING ON 80/20 SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

final_ckpt = train_fold(0, best_cfg, X_train, y_train, X_test, y_test, device)

# ============================================================
# CONFUSION MATRIX & REPORT
# ============================================================
print("\n" + "=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

model = build_model(best_cfg["timm_name"])
model.load_state_dict(final_ckpt["state_dict"])
model.eval().to(device)

tf = get_transforms(best_cfg["img_size"], False)
test_ds = PotatoDataset(X_test, y_test, tf)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

result = evaluate(model, test_loader, device)
ba = balanced_accuracy_score(y_test, result["predictions"])

print(f"\n--- {best_model_name} Test Results ---")
print(f"  Accuracy:       {result['accuracy']:.4f}")
print(f"  Macro F1:       {result['macro_f1']:.4f}")
print(f"  Balanced Acc:   {ba:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, result["predictions"], target_names=CLASS_NAMES, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, result["predictions"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[0].set_title(f"{best_model_name} - Confusion Matrix")
axes[0].set_ylabel("True")
axes[0].set_xlabel("Predicted")

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1].set_title(f"{best_model_name} - Normalized Confusion Matrix")
axes[1].set_ylabel("True")
axes[1].set_xlabel("Predicted")

plt.tight_layout()
ensure_dir(RESULTS_DIR)  # create necessary directory ahead
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()

print("\n=== Per-Class Performance ===")
for i, cls in enumerate(CLASS_NAMES):
    cls_mask = np.array(y_test) == i
    cls_correct = result["predictions"][cls_mask] == i
    print(f"  {cls:15s}: {cls_correct.mean():.4f} ({cls_correct.sum()}/{cls_mask.sum()})")

# ============================================================
# STABILITY ASSESSMENT
# ============================================================
print("\n" + "=" * 60)
print("STABILITY ASSESSMENT")
print("=" * 60)

for name, res in all_fold_results.items():
    m, s = res["mean_f1"], res["std_f1"]
    cv = s / m if m > 0 else float("inf")
    if cv < 0.01 and m > 0.95:
        verdict = "STABLE"
    elif cv < 0.02 and m > 0.90:
        verdict = "MODERATELY STABLE"
    else:
        verdict = "UNSTABLE"
    print(f"  {name:20s}: F1={m:.4f}+/-{s:.4f}  CV={cv:.4f}  [{verdict}]")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    "model": best_model_name,
    "timm_name": best_cfg["timm_name"],
    "dataset": "PlantVillage",
    "total_images": len(df),
    "class_distribution": {CLASS_NAMES[i]: int(dist.get(CLASS_NAMES[i], 0)) for i in range(NUM_CLASSES)},
    "k_fold_results": all_fold_results,
    "final_test": {
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "balanced_accuracy": ba,
    },
    "config": {
        "n_folds": N_FOLDS,
        "label_smooth": LABEL_SMOOTH,
        "mixup_alpha": MIXUP_ALPHA,
        "cutmix_alpha": CUTMIX_ALPHA,
        "weight_decay": WEIGHT_DECAY,
        "droppath": DROPPATH,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "epochs_head": EPOCHS_HEAD,
        "epochs_finetune": EPOCHS_FINETUNE,
        "patience": PATIENCE,
    }
}

# Result -> always export to file (trained model + metrics)
ensure_dir(RESULTS_DIR)
with open(RESULTS_DIR / "plantvillage_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best model: {best_model_name}")
print(f"K-Fold mean F1: {all_fold_results[best_model_name]['mean_f1']:.4f} +/- {all_fold_results[best_model_name]['std_f1']:.4f}")
print(f"Final test F1: {result['macro_f1']:.4f}")
print(f"Results saved to: {RESULTS_DIR}")
