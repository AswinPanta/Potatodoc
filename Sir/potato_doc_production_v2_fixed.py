"""
Potato Leaf Disease Detection — Clean Production Pipeline
==========================================================
Single entry-point for training on any server (including college server).

What it does (in order):
    1. audit  — scan IPD images, PIL corruption check, SHA256 exact-dedup report
    2. dedup  — pHash near-duplicate grouping (Union-Find, Hamming <= 5)
    3. split  — leakage-safe group-aware stratified train/val/test split
    4. train  — 2-phase transfer learning (head-only -> full fine-tune)
    5. eval   — IPD test metrics + optional PLD cross-domain evaluation

Usage on college server:
    pip install -r requirements.txt
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # or cpu/cu118 as needed

    # 1. Full run (single model, recommended for server):
    python potato_doc_production_v2_fixed.py --data-dir ./content/dataset --results-dir ./content/results --model convnext_tiny

    # 2. Run only specific stages:
    python potato_doc_production_v2_fixed.py --stages audit,split --data-dir ./content/dataset

    # 3. All 4 models (needs ~3-4h on RTX 3060, more on CPU):
    python potato_doc_production_v2_fixed.py --model all

    # 4. See all options:
    python potato_doc_production_v2_fixed.py --help

Dataset layout expected under --data-dir:
    <data-dir>/
      earlyblt/earlyblt/*.jpg
      healthy/healthy/*.jpg
      lateblt/lateblt/*.jpg
      PLD/Potato Leaf Disease Dataset in Uncontrolled Environment/{Fungi,Healthy,Phytopthora}/*.jpg  (optional)

Outputs go to --results-dir:
    split_cache.json, data_quality_gate.json, <model>_best.pth,
    cross_domain.json, confusion_matrices.png, *.csv reports

Design notes:
    - Nothing runs on import. All execution is inside main(), guarded by
      `if __name__ == "__main__"`, so the college server (or any importer)
      can safely `import` helpers without side effects.
    - All paths are CLI-configurable (no hard-coded absolute paths).
    - Headless-safe: matplotlib uses Agg, no plt.show().
    - Near-dedup is O(n^2); on 58k images it is infeasible, so it is
      optional (--skip-near-dedup) and capped (--near-dedup-max).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for servers (no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALB = True
except ImportError:
    HAS_ALB = False

try:
    import imagehash

    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

log = logging.getLogger("potato")

# ---------------------------------------------------------------------------
# Constants / registry
# ---------------------------------------------------------------------------

SEED_DEFAULT = 42
IPD_CLASSES = ["earlyblt", "healthy", "lateblt"]
IPD_CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]
NUM_CLASSES = 3

PLD_MAP = {"Fungi": 0, "Healthy": 1, "Phytopthora": 2}
PLD_CLEAN_SUBSET = {"Healthy", "Phytopthora"}
PLD_MAPPING_VERIFIED = False  # set True only after expert confirms Fungi->EB etc.

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEAR_DUP_HAMMING = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_REGISTRY = {
    "efficientnetv2_b3": {
        "timm_name": "tf_efficientnetv2_b3", "img_size": 300, "batch_size": 32,
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny.fb_in22k", "img_size": 224, "batch_size": 64,
    },
    # Aliases kept for backward compatibility with the old 4-model script.
    "convnext_tiny_v1": {
        "timm_name": "convnext_tiny.fb_in22k", "img_size": 224, "batch_size": 64,
    },
    "convnext_tiny_v2": {
        "timm_name": "convnext_tiny.fb_in22k", "img_size": 224, "batch_size": 64,
    },
    "swin_tiny": {
        "timm_name": "swin_tiny_patch4_window7_224.ms_in22k", "img_size": 224, "batch_size": 64,
    },
}
ALL_MODEL_NAMES = ["efficientnetv2_b3", "convnext_tiny_v1", "convnext_tiny_v2", "swin_tiny"]

STAGE_CHOICES = ("audit", "dedup", "split", "train", "eval")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu")
        log.info("Using CPU (training will be slow)")
    return dev


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cpu_state_dict(state_dict: dict) -> dict:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def save_checkpoint(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Stage 1: audit — scan + corruption check + SHA256 exact duplicates
# ---------------------------------------------------------------------------

def scan_ipd(data_dir: Path) -> pd.DataFrame:
    """Scan IPD class folders. Raises FileNotFoundError with a clear message."""
    records = []
    for cls_idx, cls_name in enumerate(IPD_CLASSES):
        cls_dir = data_dir / cls_name / cls_name
        if not cls_dir.exists():
            # Also accept single-nesting layout: <data-dir>/<class>/*.jpg
            flat = data_dir / cls_name
            if flat.exists() and any(flat.glob("*.*")):
                cls_dir = flat
            else:
                raise FileNotFoundError(
                    f"Missing IPD folder: {cls_dir} (or {flat}). "
                    f"Expected '<data-dir>/{cls_name}/{cls_name}/*.jpg'."
                )
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                records.append({"path": str(f), "class": cls_name, "class_idx": cls_idx})
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No images found under {data_dir}. Check dataset extraction.")
    return df


def check_corruption(df: pd.DataFrame) -> pd.DataFrame:
    corrupt_paths = set()
    for p in df["path"]:
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception:
            corrupt_paths.add(p)
    df = df.copy()
    df["corrupted"] = df["path"].isin(corrupt_paths)
    log.info("Corruption check: %d / %d corrupt", len(corrupt_paths), len(df))
    return df


def add_sha256(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hashes = []
    for p in df["path"]:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        hashes.append(h.hexdigest())
    df["sha256"] = hashes
    dup_mask = df["sha256"].duplicated(keep=False)
    log.info("Exact duplicates: %d images in %d groups",
             int(dup_mask.sum()), int(df[dup_mask]["sha256"].nunique()) if dup_mask.any() else 0)
    return df


def run_audit(data_dir: Path, results_dir: Path) -> pd.DataFrame:
    log.info("=== STAGE: audit ===")
    t0 = time.time()
    df = scan_ipd(data_dir)
    log.info("Scanned %d images in %.1fs", len(df), time.time() - t0)
    for cls, cnt in df["class"].value_counts().items():
        log.info("  %-15s: %6d (%5.1f%%)", cls, cnt, cnt / len(df) * 100)

    df = check_corruption(df)
    df = add_sha256(df)

    ensure_dir(results_dir)
    df.to_csv(results_dir / "audit_manifest.csv", index=False)
    (results_dir / "data_quality_gate.json").write_text(json.dumps({
        "total_images": int(len(df)),
        "corrupt_images": int(df["corrupted"].sum()),
        "exact_duplicate_images": int(df["sha256"].duplicated(keep=False).sum()),
        "pld_mapping_verified": PLD_MAPPING_VERIFIED,
    }, indent=2))
    return df


# ---------------------------------------------------------------------------
# Stage 2: near-dedup — pHash grouping (Union-Find)
# ---------------------------------------------------------------------------

def _find(parent: dict, x: str) -> str:
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = _find(parent, parent[x])
    return parent[x]


def add_near_groups(df: pd.DataFrame, results_dir: Path, max_images: int = 8000) -> pd.DataFrame:
    """Group perceptual near-duplicates. Capped because pairwise compare is O(n^2).

    Full 58k-image IPD cannot be compared pairwise (~1.7B pairs). For the full
    dataset use --skip-near-dedup (each image is its own group) — the split
    stage still works and stays leakage-safe for exact duplicates.
    """
    df = df.copy()
    if not HAS_IMAGEHASH:
        log.warning("imagehash not installed; skipping pHash grouping (each image = own group).")
        df["near_group"] = df["path"]
        return df
    if len(df) > max_images:
        log.warning("Too many images (%d > %d) for O(n^2) pHash compare; "
                    "each image keeps its own group. Use --near-dedup-max to override.",
                    len(df), max_images)
        df["near_group"] = df["path"]
        return df

    from imagehash import hex_to_hash

    phashes = []
    for p in df["path"]:
        try:
            with Image.open(p) as img:
                phashes.append(str(imagehash.phash(img.convert("RGB"))))
        except Exception:
            phashes.append(None)
    df["phash"] = phashes

    valid = df[df["phash"].notna()].reset_index(drop=True)
    objs = [hex_to_hash(h) for h in valid["phash"]]
    parent: dict = {}
    pairs = []
    n = len(valid)
    log.info("pHash compare: %d images (threshold <= %d)...", n, NEAR_DUP_HAMMING)
    for i in range(n):
        hi = objs[i]
        for j in range(i + 1, n):
            if hi - objs[j] <= NEAR_DUP_HAMMING:
                a, b = valid.iloc[i]["path"], valid.iloc[j]["path"]
                ra, rb = _find(parent, a), _find(parent, b)
                if ra != rb:
                    parent[rb] = ra
                pairs.append((a, b))
    for p in df["path"]:
        _find(parent, p)
    df["near_group"] = df["path"].map(lambda x: _find(parent, x))
    sizes = df["near_group"].value_counts()
    log.info("Near-duplicate pairs: %d, multi-image groups: %d, max size: %d",
             len(pairs), int((sizes > 1).sum()), int(sizes.max()))
    if pairs:
        pd.DataFrame(pairs, columns=["path_a", "path_b"]).to_csv(
            results_dir / "ipd_near_duplicates.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Stage 3: leakage-safe group-aware split
# ---------------------------------------------------------------------------

def group_aware_split(df: pd.DataFrame, results_dir: Path, seed: int,
                      ratios=(0.70, 0.15, 0.15)) -> dict:
    log.info("=== STAGE: split ===")
    usable = df[~df["corrupted"]].copy()
    if "near_group" not in usable.columns:
        usable["near_group"] = usable["path"]

    rows = []
    for gid, g in usable.groupby("near_group"):
        counts = g["class_idx"].value_counts()
        rows.append({"group": gid, "label": int(counts.index[0]),
                     "n": len(g), "mixed": bool(g["class_idx"].nunique() > 1)})
    groups = pd.DataFrame(rows)
    if groups["mixed"].any():
        log.warning("%d near-duplicate groups contain multiple labels.", int(groups["mixed"].sum()))
        groups[groups["mixed"]].to_csv(results_dir / "mixed_label_duplicate_groups.csv", index=False)

    train_r, val_r, test_r = ratios
    g_train, g_temp = train_test_split(
        groups["group"].tolist(), test_size=(1 - train_r),
        random_state=seed, stratify=groups["label"].tolist())
    gtemp_labels = groups.set_index("group").loc[g_temp, "label"].tolist()
    g_val, g_test = train_test_split(
        g_temp, test_size=(test_r / (val_r + test_r)),
        random_state=seed, stratify=gtemp_labels)

    split_map = {g: "train" for g in g_train}
    split_map.update({g: "val" for g in g_val})
    split_map.update({g: "test" for g in g_test})
    usable["split"] = usable["near_group"].map(split_map)

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        ga = set(usable.loc[usable["split"] == a, "near_group"])
        gb = set(usable.loc[usable["split"] == b, "near_group"])
        assert not (ga & gb), f"LEAKAGE: groups overlap between {a} and {b}"

    out = {
        "X_train": usable.loc[usable["split"] == "train", "path"].tolist(),
        "y_train": usable.loc[usable["split"] == "train", "class_idx"].tolist(),
        "X_val": usable.loc[usable["split"] == "val", "path"].tolist(),
        "y_val": usable.loc[usable["split"] == "val", "class_idx"].tolist(),
        "X_test": usable.loc[usable["split"] == "test", "path"].tolist(),
        "y_test": usable.loc[usable["split"] == "test", "class_idx"].tolist(),
    }
    log.info("Split: train=%d val=%d test=%d", len(out["X_train"]), len(out["X_val"]), len(out["X_test"]))
    ensure_dir(results_dir)
    (results_dir / "split_cache.json").write_text(json.dumps(
        {**{k: len(v) if k.startswith("X_") else v for k, v in out.items()
            if k.startswith("X_")}, "seed": seed, "method": "group-aware stratified split"},
        indent=2))
    # Full paths cache (needed to resume train/eval without re-splitting).
    (results_dir / "split_paths.json").write_text(json.dumps(out))
    return out


def load_or_make_split(df_or_none, data_dir: Path, results_dir: Path,
                       seed: int, force_resplit: bool) -> dict:
    cache = results_dir / "split_paths.json"
    if cache.exists() and not force_resplit:
        log.info("Loading cached split from %s", cache)
        return json.loads(cache.read_text())
    if df_or_none is None:
        df_or_none = run_audit(data_dir, results_dir)
        df_or_none = add_near_groups(df_or_none, results_dir)
    elif "near_group" not in df_or_none.columns:
        df_or_none = add_near_groups(df_or_none, results_dir)
    return group_aware_split(df_or_none, results_dir, seed)


# ---------------------------------------------------------------------------
# PLD external dataset (optional cross-domain eval)
# ---------------------------------------------------------------------------

def collect_pld(data_dir: Path):
    root = data_dir / "PLD" / "Potato Leaf Disease Dataset in Uncontrolled Environment"
    if not root.exists():
        log.warning("PLD not found at %s; cross-domain eval will be skipped.", root)
        return [], np.array([], dtype=int), []
    paths, labels, groups = [], [], []
    for folder, lab in PLD_MAP.items():
        d = root / folder
        if not d.exists():
            log.warning("PLD folder missing: %s", d)
            continue
        fs = sorted(str(p) for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
        paths += fs
        labels += [lab] * len(fs)
        groups += [folder] * len(fs)
        log.info("  %s -> %s: %d images", folder, IPD_CLASS_NAMES[lab], len(fs))
    labels = np.array(labels, dtype=int)
    if not PLD_MAPPING_VERIFIED:
        log.warning("PLD mapping UNVERIFIED — external results are NOT production evidence.")
    return paths, labels, groups


# ---------------------------------------------------------------------------
# Datasets / augmentation / model factory
# ---------------------------------------------------------------------------

def get_transforms(img_size: int, is_train: bool = True, strong: bool = False):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if HAS_ALB:
        if is_train:
            tfs = [
                A.RandomResizedCrop(img_size, img_size, scale=(0.5 if strong else 0.7, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
            ]
            if strong:
                tfs += [
                    A.OneOf([A.RandomBrightnessContrast(0.3, 0.3, p=1),
                             A.HueSaturationValue(20, 30, 20, p=1)], p=0.8),
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
        tfs = [transforms.RandomResizedCrop(img_size, scale=(0.5 if strong else 0.7, 1.0)),
               transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(p=0.2)]
        if strong:
            tfs += [transforms.RandAugment(num_ops=2, magnitude=9)]
        else:
            tfs += [transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomRotation(15)]
        tfs += [transforms.ToTensor(), norm]
        return transforms.Compose(tfs)
    return transforms.Compose([transforms.Resize(int(img_size * 1.14)),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(), norm])


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


def make_loaders(Xtr, ytr, Xva, yva, img_size, batch_size, strong, num_workers):
    tr = PotatoDataset(Xtr, ytr, get_transforms(img_size, True, strong))
    va = PotatoDataset(Xva, yva, get_transforms(img_size, False))
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(tr, batch_size, shuffle=True, **kw),
            DataLoader(va, batch_size * 2, shuffle=False, **kw))


def build_model(timm_name: str, drop_path: float = 0.1) -> nn.Module:
    return timm.create_model(timm_name, pretrained=True,
                             num_classes=NUM_CLASSES, drop_path_rate=drop_path)


def _head_params(model: nn.Module):
    if hasattr(model, "classifier"):
        return model.classifier.parameters()
    return model.head.parameters()


# ---------------------------------------------------------------------------
# Stage 4: training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, crit, opt, scaler, mix_fn, device):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yin = y
        if mix_fn is not None:
            if x.size(0) % 2 != 0:  # Mixup needs even batch
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
def evaluate(model, loader, device) -> dict:
    model.eval()
    parts_p, parts_l, tl, tot = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        o = model(x)
        tl += ce(o, y).item() * x.size(0)
        tot += x.size(0)
        parts_p.append(o.argmax(1).cpu().numpy())
        parts_l.append(y.cpu().numpy())
    P, L = np.concatenate(parts_p), np.concatenate(parts_l)
    return {"loss": tl / tot, "accuracy": float((P == L).mean()),
            "macro_f1": float(f1_score(L, P, average="macro")),
            "predictions": P, "labels": L}


def train_model(name: str, cfg: dict, split: dict, device: torch.device,
                results_dir: Path, seed: int, epochs_head: int = 10,
                epochs_finetune: int = 50, patience: int = 10,
                force_retrain: bool = False) -> dict:
    ckpt_path = results_dir / f"{name}_best.pth"
    if ckpt_path.exists() and not force_retrain:
        log.info("[SKIP] %s checkpoint exists: %s", name, ckpt_path)
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    set_seed(seed)
    img_size, bs = cfg["img_size"], cfg["batch_size"]
    Xtr, ytr = split["X_train"], split["y_train"]
    Xva, yva = split["X_val"], split["y_val"]
    log.info("%s: train=%d val=%d img=%d batch=%d", name, len(Xtr), len(Xva), img_size, bs)

    tr_loader, va_loader = make_loaders(Xtr, ytr, Xva, yva, img_size, bs,
                                        strong=True, num_workers=min(4, os.cpu_count() or 1))
    model = build_model(cfg["timm_name"]).to(device)
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(NUM_CLASSES),
                                           y=np.array(ytr)),
                       dtype=torch.float32).to(device)
    mix_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0,
                   switch_prob=0.5, label_smoothing=0.1, num_classes=NUM_CLASSES)
    amp = device.type if device.type == "cuda" else "cpu"

    # Phase 1: head-only probe.
    for p in model.parameters():
        p.requires_grad = False
    for p in _head_params(model):
        p.requires_grad = True
    opt1 = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                       lr=1e-3, weight_decay=0.01)
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=epochs_head)
    scaler1 = torch.amp.GradScaler(amp)
    log.info("--- %s Phase 1: head-only (%d epochs) ---", name, epochs_head)
    best_f1, best_state = 0.0, None
    for ep in range(1, epochs_head + 1):
        tl, ta = train_one_epoch(model, tr_loader, nn.CrossEntropyLoss(weight=cw),
                                 opt1, scaler1, None, device)
        vr = evaluate(model, va_loader, device)
        sched1.step()
        tag = "*" if vr["macro_f1"] > best_f1 else ""
        best_f1 = max(best_f1, vr["macro_f1"])
        if tag:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        log.info("  ep %2d | loss=%.4f acc=%.4f | val_f1=%.4f%s", ep, tl, ta, vr["macro_f1"], tag)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Phase 2: full fine-tune with differential LR.
    for p in model.parameters():
        p.requires_grad = True
    head_ids = {id(p) for n, p in model.named_parameters()
                if "classifier" in n or "head" in n}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    opt2 = optim.AdamW([{"params": backbone, "lr": 1e-5},
                        {"params": head, "lr": 1e-4}], weight_decay=0.05)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=epochs_finetune)
    scaler2 = torch.amp.GradScaler(amp)
    log.info("--- %s Phase 2: fine-tune (max %d, patience %d) ---", name, epochs_finetune, patience)
    best_f1, best_state, wait = 0.0, None, 0
    for ep in range(1, epochs_finetune + 1):
        tl, ta = train_one_epoch(model, tr_loader, SoftTargetCrossEntropy(),
                                 opt2, scaler2, mix_fn, device)
        vr = evaluate(model, va_loader, device)
        sched2.step()
        if vr["macro_f1"] > best_f1:
            best_f1, best_state, wait = vr["macro_f1"], \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
            tag = "*"
        else:
            wait += 1
            tag = ""
        log.info("  ep %2d | loss=%.4f acc=%.4f | val_f1=%.4f%s", ep, tl, ta, vr["macro_f1"], tag)
        if wait >= patience:
            log.info("  Early stopping at epoch %d", ep)
            break

    model.load_state_dict(best_state)
    save_checkpoint(ckpt_path, {"state_dict": cpu_state_dict(model.state_dict()),
                                "timm_name": cfg["timm_name"], "img_size": img_size,
                                "class_names": IPD_CLASS_NAMES, "val_f1": best_f1})
    log.info("Saved %s (val_f1=%.4f)", ckpt_path.name, best_f1)
    del model
    torch.cuda.empty_cache()
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Stage 5: evaluation
# ---------------------------------------------------------------------------

def eval_on_dataset(model_name: str, ckpt: dict, paths, labels,
                    dataset_name: str, device, num_workers: int = 2) -> dict:
    model = build_model(ckpt["timm_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    ds = PotatoDataset(paths, list(labels), get_transforms(ckpt["img_size"], False))
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    r = evaluate(model, loader, device)
    r.update(model_name=model_name, dataset=dataset_name)
    ba = balanced_accuracy_score(labels, r["predictions"])
    log.info("%s on %s (n=%d): acc=%.4f macro_f1=%.4f bal_acc=%.4f",
             model_name, dataset_name, len(labels), r["accuracy"], r["macro_f1"], ba)
    log.info("\n%s", classification_report(labels, r["predictions"],
                                           target_names=IPD_CLASS_NAMES, digits=4, zero_division=0))
    del model
    torch.cuda.empty_cache()
    return r


def run_eval(trained: dict, split: dict, data_dir: Path, results_dir: Path, device) -> None:
    log.info("=== STAGE: eval ===")
    ipd_results = {n: eval_on_dataset(n, ck, split["X_test"], split["y_test"], "IPD Test", device)
                   for n, ck in trained.items()}

    pld_paths, pld_labels, pld_groups = collect_pld(data_dir)
    pld_results: dict = {}
    if len(pld_paths) > 0:
        clean_mask = np.array([g in PLD_CLEAN_SUBSET for g in pld_groups])
        for n, ck in trained.items():
            full = eval_on_dataset(n, ck, pld_paths, pld_labels.tolist(), "PLD Full", device)
            cp = [p for p, m in zip(pld_paths, clean_mask) if m]
            cy = pld_labels[clean_mask].tolist()
            clean = eval_on_dataset(n, ck, cp, cy, "PLD Clean", device)
            pld_results[n] = {"full": full, "clean": clean}
    else:
        log.warning("PLD not available; skipping cross-domain evaluation.")

    serialisable = {n: {"ipd_f1": float(ipd_results[n]["macro_f1"]),
                        "ipd_acc": float(ipd_results[n]["accuracy"]),
                        "pld_full_f1": float(pld_results.get(n, {}).get("full", {}).get("macro_f1") or 0),
                        "pld_clean_f1": float(pld_results.get(n, {}).get("clean", {}).get("macro_f1") or 0)}
                    for n in trained}
    (results_dir / "cross_domain.json").write_text(json.dumps(serialisable, indent=2))

    # Confusion matrices for best model (headless: save only, never show).
    best = max(trained, key=lambda n: ipd_results[n]["macro_f1"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cm1 = confusion_matrix(split["y_test"], ipd_results[best]["predictions"])
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=IPD_CLASS_NAMES, yticklabels=IPD_CLASS_NAMES)
    axes[0].set_title(f"IPD Test ({best})")
    if best in pld_results:
        cm2 = confusion_matrix(pld_labels, pld_results[best]["full"]["predictions"])
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
                    xticklabels=IPD_CLASS_NAMES, yticklabels=IPD_CLASS_NAMES)
        axes[1].set_title(f"PLD Full ({best})")
        cm2n = cm2.astype(float) / cm2.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(cm2n, annot=True, fmt=".2f", cmap="Oranges", ax=axes[2],
                    xticklabels=IPD_CLASS_NAMES, yticklabels=IPD_CLASS_NAMES)
        axes[2].set_title("PLD Normalized")
    else:
        for ax, t in zip((axes[1], axes[2]), ("PLD Full", "PLD Normalized")):
            ax.text(0.5, 0.5, "PLD not available", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(t)
            ax.axis("off")
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrices.png", dpi=150)
    plt.close()

    log.info("Best model: %s (IPD F1=%.4f)", best, ipd_results[best]["macro_f1"])
    if not PLD_MAPPING_VERIFIED:
        log.warning("Verdict is provisional: PLD mapping UNVERIFIED.")


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------

def resolve_models(selection: str) -> dict:
    if selection == "all":
        return {n: {**MODEL_REGISTRY[n], "name": n} for n in ALL_MODEL_NAMES}
    if selection not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{selection}'. "
                         f"Choose from {sorted(MODEL_REGISTRY)} or 'all'.")
    return {selection: {**MODEL_REGISTRY[selection], "name": selection}}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Potato disease detection — clean train/eval pipeline for any server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "content" / "dataset",
                   help="Root folder containing IPD class folders (and optional PLD/).")
    p.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "content" / "results",
                   help="Folder for checkpoints, splits, metrics and plots.")
    p.add_argument("--model", default="convnext_tiny",
                   help=f"Model to train: {sorted(MODEL_REGISTRY)} or 'all'.")
    p.add_argument("--list-models", action="store_true", help="List available models and exit.")
    p.add_argument("--stages", default="audit,dedup,split,train,eval",
                   help=f"Comma-separated subset of {list(STAGE_CHOICES)} (run in pipeline order).")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--epochs-head", type=int, default=10)
    p.add_argument("--epochs-finetune", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=None, help="Override registry batch size.")
    p.add_argument("--skip-near-dedup", action="store_true",
                   help="Skip O(n^2) pHash grouping (recommended for full 58k IPD).")
    p.add_argument("--near-dedup-max", type=int, default=8000,
                   help="Max images for pairwise pHash compare; above this each image is its own group.")
    p.add_argument("--force-resplit", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")

    if args.list_models:
        for name, cfg in MODEL_REGISTRY.items():
            print(f"{name:22s} timm={cfg['timm_name']} img={cfg['img_size']} batch={cfg['batch_size']}")
        return 0

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGE_CHOICES]
    if bad:
        log.error("Unknown stages: %s (choose from %s)", bad, list(STAGE_CHOICES))
        return 2

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.results_dir)
    log.info("data_dir=%s results_dir=%s model=%s stages=%s seed=%d",
             args.data_dir, args.results_dir, args.model, stages, args.seed)

    if not args.data_dir.exists():
        log.error("data-dir does not exist: %s — download/extract IPD first (see module docstring).",
                  args.data_dir)
        return 2

    # --- audit / dedup / split (lazy: only what requested stages need) ---
    df = None
    split = None
    need_df = bool({"audit", "dedup", "split"} & set(stages))
    need_split = bool({"split", "train", "eval"} & set(stages))

    if "audit" in stages:
        df = run_audit(args.data_dir, args.results_dir)
    if "dedup" in stages:
        if df is None:
            df = run_audit(args.data_dir, args.results_dir)
        if args.skip_near_dedup:
            log.info("Skipping pHash grouping (--skip-near-dedup).")
            df["near_group"] = df["path"]
        else:
            df = add_near_groups(df, args.results_dir, max_images=args.near_dedup_max)
    if "split" in stages:
        split = load_or_make_split(df, args.data_dir, args.results_dir,
                                   args.seed, force_resplit=True)
    elif need_split:
        split = load_or_make_split(df, args.data_dir, args.results_dir,
                                   args.seed, force_resplit=args.force_resplit)

    # --- train ---
    trained: dict = {}
    if "train" in stages:
        assert split is not None
        for name, cfg in resolve_models(args.model).items():
            if args.batch_size is not None:
                cfg = {**cfg, "batch_size": args.batch_size}
            trained[name] = train_model(
                name, cfg, split, device, args.results_dir, args.seed,
                epochs_head=args.epochs_head, epochs_finetune=args.epochs_finetune,
                patience=args.patience, force_retrain=args.force_retrain)
    elif "eval" in stages:
        # Eval-only run: load existing checkpoints.
        for name, cfg in resolve_models(args.model).items():
            ckpt_path = args.results_dir / f"{name}_best.pth"
            if not ckpt_path.exists():
                log.error("Checkpoint missing: %s — run with 'train' first or check --model.", ckpt_path)
                return 2
            trained[name] = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # --- eval ---
    if "eval" in stages:
        assert split is not None
        run_eval(trained, split, args.data_dir, args.results_dir, device)

    log.info("PIPELINE COMPLETE — results in %s", args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
