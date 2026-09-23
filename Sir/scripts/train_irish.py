"""
Irish main-model trainer — ConvNeXt-Tiny on IrishPotato37G.
Reads dataset config from config/datasets.yaml, registers result in config/models.yaml.

Usage:
    python scripts/train_irish.py [--seed 42]

Prerequisites:
    1. IrishPotato37G download complete (results/EXTRACT_DONE.flag in dataset/)
    2. python scripts/prepare_data.py --dataset irish

Checkpoints → checkpoints/irish/
Metrics     → results/irish/
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

try:
    import yaml
except ImportError:
    print("Missing pyyaml. Run: pip install pyyaml")
    sys.exit(1)

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALB = True
except ImportError:
    HAS_ALB = False

BASE_DIR = Path(__file__).resolve().parent.parent  # where this Python code is
DATASETS_CFG = BASE_DIR / "config" / "datasets.yaml"
MODELS_CFG = BASE_DIR / "config" / "models.yaml"


# ============================================================
# File handling code: Read / create directory
# (independent of platform, created ahead, where Python code is)
# ============================================================
def ensure_dir(path: Path) -> Path:
    """Generate directory independent of platform, created ahead."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_repo_path(p: str) -> Path:
    """Resolve config path relative to code location (platform-independent)."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (BASE_DIR / pp).resolve()

# --- training knobs (from train_v3.py balanced recipe) ---
SEED = 42
MODEL_ID = "convnext_irish"
TIMM_NAME = "convnext_tiny.fb_in22k"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 50
PATIENCE = 10
LABEL_SMOOTH = 0.1
MIXUP_PROB = 0.7
MIXUP_A, CUTMIX_A = 0.2, 1.0
WD = 0.05
DROPPATH = 0.1
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def get_transforms(img_size, is_train=True, strong=False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if HAS_ALB:
        if is_train:
            tfs = [
                A.RandomResizedCrop((img_size, img_size), scale=(0.5 if strong else 0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
            ]
            if strong:
                tfs += [
                    A.OneOf([
                        A.RandomBrightnessContrast(0.3, 0.3, p=1),
                        A.HueSaturationValue(20, 30, 20, p=1),
                    ], p=0.8),
                    A.OneOf([A.GaussianBlur(3, p=1), A.GaussNoise(10, 50, p=1)], p=0.3),
                    A.Rotate(limit=15, p=0.5),
                ]
            else:
                tfs += [A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5), A.Rotate(limit=15, p=0.5)]
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
            transforms.RandomResizedCrop(img_size, scale=(0.5 if strong else 0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
        ]
        if strong:
            tfs += [transforms.RandAugment(num_ops=2, magnitude=9), transforms.RandomGrayscale(p=0.1)]
        else:
            tfs += [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomRotation(15)]
        tfs += [transforms.ToTensor(), norm]
        return transforms.Compose(tfs)
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), norm,
    ])


class IrishDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform

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


def make_loaders(Xtr, ytr, Xva, yva, strong=False):
    tr_ds = IrishDataset(Xtr, ytr, get_transforms(IMG_SIZE, True, strong))
    va_ds = IrishDataset(Xva, yva, get_transforms(IMG_SIZE, False))
    class_counts = np.bincount(ytr)
    sample_weights = (1.0 / class_counts)[ytr]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(ytr), replacement=True)
    kw = dict(num_workers=0, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(tr_ds, BATCH_SIZE, sampler=sampler, **kw),
        DataLoader(va_ds, BATCH_SIZE * 2, shuffle=False, **kw),
    )


def train_one_epoch(model, loader, crit, opt, scaler, mix_fn, device):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yin = y
        if mix_fn is not None:
            if x.size(0) % 2 != 0:
                x, y = x[:-1], y[:-1]
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
    return {"loss": tl / tot, "accuracy": float((P == L).mean()),
            "macro_f1": float(f1_score(L, P, average="macro")),
            "predictions": P, "labels": L}


def build_model():
    return timm.create_model(TIMM_NAME, pretrained=True, num_classes=3, drop_path_rate=DROPPATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset config + split cache
    dcfg = yaml.safe_load(open(DATASETS_CFG))["irish"]
    cache_path = resolve_repo_path(dcfg["split_cache"])  # platform-independent
    if not cache_path.exists():
        print(f"ERROR: split cache missing: {cache_path}")
        print("Run: python scripts/prepare_data.py --dataset irish")
        sys.exit(1)
    d = json.load(open(cache_path))
    Xtr, ytr, Xva, yva, Xte, yte = (d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                                    d["X_test"], d["y_test"])
    print(f"Split: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")
    class_names = dcfg["class_names"]

    ckpt_dir = BASE_DIR / "checkpoints" / "irish"
    ensure_dir(ckpt_dir)  # create necessary directory ahead
    ckpt_path = ckpt_dir / "convnext_tiny_v3_best.pth"
    results_dir = BASE_DIR / "results" / "irish"
    ensure_dir(results_dir)  # create necessary directory ahead

    if ckpt_path.exists():
        print(f"Checkpoint exists: {ckpt_path}, skipping training")
    else:
        tr_loader, va_loader = make_loaders(Xtr, ytr, Xva, yva)
        model = build_model().to(device)

        cw = compute_class_weight("balanced", classes=np.arange(3), y=np.array(ytr))
        cw = torch.tensor(cw, dtype=torch.float32).to(device)
        mix_fn = Mixup(mixup_alpha=MIXUP_A, cutmix_alpha=CUTMIX_A, prob=MIXUP_PROB,
                       switch_prob=0.5, label_smoothing=LABEL_SMOOTH, num_classes=3)

        # Phase 1: head-only
        for p in model.parameters():
            p.requires_grad = False
        head = model.head if hasattr(model, "head") else model.classifier
        for p in head.parameters():
            p.requires_grad = True
        opt1 = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                           lr=1e-3, weight_decay=0.01)
        sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=EPOCHS_HEAD)
        scaler1 = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")

        print(f"\nPhase 1: head-only ({EPOCHS_HEAD} epochs)")
        best_f1, best_state = 0, None
        for ep in range(1, EPOCHS_HEAD + 1):
            tl, ta = train_one_epoch(model, tr_loader, nn.CrossEntropyLoss(weight=cw),
                                     opt1, scaler1, None, device)
            vr = evaluate(model, va_loader, device)
            sched1.step()
            tag = "*" if vr["macro_f1"] > best_f1 else ""
            if vr["macro_f1"] > best_f1:
                best_f1 = vr["macro_f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  ep {ep:2d} | loss={tl:.4f} acc={ta:.4f} | val_f1={vr['macro_f1']:.4f}{tag}")
        model.load_state_dict(best_state)

        # Phase 2: full fine-tune
        for p in model.parameters():
            p.requires_grad = True
        head_ids = {id(p) for n, p in model.named_parameters() if "head" in n or "classifier" in n}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        head_params = [p for p in model.parameters() if id(p) in head_ids]
        opt2 = optim.AdamW([{"params": backbone_params, "lr": BACKBONE_LR},
                            {"params": head_params, "lr": HEAD_LR}], weight_decay=WD)
        sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS_FINETUNE)
        scaler2 = torch.amp.GradScaler(device.type if device.type == "cuda" else "cpu")

        print(f"\nPhase 2: fine-tune (max {EPOCHS_FINETUNE}, patience={PATIENCE})")
        best_f1, best_state, wait = 0, None, 0
        for ep in range(1, EPOCHS_FINETUNE + 1):
            tl, ta = train_one_epoch(model, tr_loader, SoftTargetCrossEntropy(),
                                     opt2, scaler2, mix_fn, device)
            vr = evaluate(model, va_loader, device)
            sched2.step()
            tag = "*" if vr["macro_f1"] > best_f1 else ""
            if vr["macro_f1"] > best_f1:
                best_f1 = vr["macro_f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            print(f"  ep {ep:2d} | loss={tl:.4f} acc={ta:.4f} | val_f1={vr['macro_f1']:.4f}{tag}")
            if wait >= PATIENCE:
                print(f"  Early stopping at epoch {ep}")
                break

        model.load_state_dict(best_state)
        ensure_dir(ckpt_path.parent)  # create necessary directory ahead
        # Result -> always export to file (trained model)
        tmp = str(ckpt_path) + ".tmp"
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "timm_name": TIMM_NAME, "img_size": IMG_SIZE,
                    "class_names": class_names, "val_f1": best_f1}, tmp)
        os.replace(tmp, ckpt_path)
        print(f"Saved {ckpt_path} (val_f1={best_f1:.4f})")
        del model
        torch.cuda.empty_cache()

    # Final test evaluation
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    te_ds = IrishDataset(Xte, yte, get_transforms(IMG_SIZE, False))
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=0)
    r = evaluate(model, te_loader, device)
    print(f"\nTest: acc={r['accuracy']:.4f} macro_f1={r['macro_f1']:.4f}")
    print(classification_report(yte, r["predictions"], target_names=class_names, digits=4))

    # Result -> always export to file (metrics)
    ensure_dir(results_dir)
    json.dump({"model": MODEL_ID, "test_accuracy": r["accuracy"],
               "test_macro_f1": r["macro_f1"], "val_f1": ckpt["val_f1"]},
              open(results_dir / "irish_results.json", "w"), indent=2)

    # Enable model in registry
    mcfg = yaml.safe_load(open(MODELS_CFG))
    mcfg[MODEL_ID]["enabled"] = True
    yaml.safe_dump(mcfg, open(MODELS_CFG, "w"), sort_keys=False)
    print(f"Enabled {MODEL_ID} in config/models.yaml")
    print("DONE")


if __name__ == "__main__":
    main()
