import os, sys, json, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import timm

# ============================================================
# CONFIG
# ============================================================
SEED = 42
DATA_DIR = Path(r"C:\Users\shadb\Downloads\dataset")
CLASSES = ["earlyblt", "healthy", "lateblt"]
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]
NUM_CLASSES = len(CLASSES)
OUTPUT_DIR = DATA_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_TO_TRAIN = [
    {"name": "efficientnetv2_b3", "timm_name": "tf_efficientnetv2_b3", "img_size": 300, "batch_size": 32},
    {"name": "convnext_tiny", "timm_name": "convnext_tiny.fb_in22k", "img_size": 224, "batch_size": 64},
    {"name": "swin_tiny", "timm_name": "swin_tiny_patch4_window7_224.ms_in22k", "img_size": 224, "batch_size": 64},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("Using CPU")
    return dev


# ============================================================
# DATASET
# ============================================================
class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.file_paths[idx]).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))


def prepare_dataset():
    cache = OUTPUT_DIR / "split_cache.json"
    if cache.exists():
        d = json.load(open(cache))
        X_train, y_train = d["X_train"], d["y_train"]
        X_val, y_val = d["X_val"], d["y_val"]
        X_test, y_test = d["X_test"], d["y_test"]
        print(f"Loaded cached split: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
        return X_train, y_train, X_val, y_val, X_test, y_test

    print("Preparing dataset...")
    all_paths, all_labels = [], []
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls_name / cls_name
        for f in cls_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                all_paths.append(str(f))
                all_labels.append(cls_idx)

    print("Scanning image integrity...")
    t0 = time.time()
    good_paths, good_labels, n_bad = [], [], 0
    for p, l in zip(all_paths, all_labels):
        try:
            with Image.open(p) as im:
                im.verify()
            good_paths.append(p)
            good_labels.append(l)
        except Exception:
            n_bad += 1
    print(f"Integrity scan done in {time.time()-t0:.0f}s - removed {n_bad} corrupt files")
    all_paths, all_labels = good_paths, good_labels

    print(f"Total images: {len(all_paths)}")
    for i, c in enumerate(CLASSES):
        print(f"  {c}: {all_labels.count(i)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths, all_labels, test_size=0.3, random_state=SEED, stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    json.dump({
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }, open(cache, "w"))
    print("Split cached to", cache)
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_transforms(img_size, is_train=True):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        norm,
    ])


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, img_size, batch_size):
    train_ds = ImageDataset(X_train, y_train, get_transforms(img_size, True))
    val_ds = ImageDataset(X_val, y_val, get_transforms(img_size, False))
    test_ds = ImageDataset(X_test, y_test, get_transforms(img_size, False))

    nw = min(6, os.cpu_count() or 1)
    common = dict(num_workers=nw, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              persistent_workers=nw > 0, prefetch_factor=3 if nw > 0 else None, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ============================================================
# MODEL
# ============================================================
def build_model(timm_name, num_classes, device):
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Built {timm_name} - {n_params:.1f}M params")
    return model


def cpu_sd(sd):
    return {k: v.detach().cpu() for k, v in sd.items()}


# ============================================================
# TRAINING
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, total = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(outputs.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    acc = (all_preds == all_labels).mean()
    return {
        "loss": total_loss / total,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


def save_last(path, payload):
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_model(model_name, timm_name, img_size, batch_size,
                X_train, y_train, X_val, y_val, X_test, y_test, device):
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    set_seed(SEED)

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, img_size, batch_size
    )
    model = build_model(timm_name, NUM_CLASSES, device)

    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (NUM_CLASSES * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler("cuda")
    last_path = OUTPUT_DIR / f"{model_name}_last.pth"

    payload = None
    if last_path.exists():
        try:
            payload = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            print(f"  [RESUME] from stage {payload['stage']} epoch {payload['epoch']} "
                  f"(best val F1 {payload['best_f1']:.4f})")
        except Exception as e:
            print(f"  [RESUME] failed ({e}) - starting fresh")
            payload = None

    best_f1 = payload["best_f1"] if payload else -1.0
    best_state = payload["best_state"] if payload else None
    stage = payload["stage"] if payload else 1
    epoch = payload["epoch"] + 1 if payload else 1
    need_opt_load = payload is not None

    while stage <= 2:
        max_ep = 10 if stage == 1 else 30
        if stage == 1:
            for p in model.parameters():
                p.requires_grad = False
            head = getattr(model, "classifier", None) or getattr(model, "head", None) or getattr(model, "fc", None)
            if head is None:
                lins = [m for m in model.modules() if isinstance(m, nn.Linear)]
                head = lins[-1]
            tparams = list(head.parameters())
            for p in tparams:
                p.requires_grad = True
            opt = optim.AdamW(tparams, lr=1e-3, weight_decay=0.01)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
            print(f"  Stage 1: head-only ({sum(p.numel() for p in tparams):,} params)")
        else:
            for p in model.parameters():
                p.requires_grad = True
            hk = ("head", "classifier", "fc")
            hp = [p for n, p in model.named_parameters() if any(k in n for k in hk)]
            bp = [p for n, p in model.named_parameters() if not any(k in n for k in hk)]
            opt = optim.AdamW([{"params": hp, "lr": 1e-4}, {"params": bp, "lr": 1e-5}], weight_decay=0.01)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
            print("  Stage 2: full fine-tune")

        if need_opt_load and payload is not None and payload.get("stage") == stage:
            opt.load_state_dict(payload["opt"])
            sch.load_state_dict(payload["sch"])
            scaler.load_state_dict(payload["scaler"])
        need_opt_load = False

        patience = 0
        while epoch <= max_ep:
            t0 = time.time()
            tl, ta = train_one_epoch(model, train_loader, criterion, opt, scaler, device)
            v = evaluate(model, val_loader, device)
            sch.step()
            improved = v["macro_f1"] > best_f1
            if improved:
                best_f1 = float(v["macro_f1"])
                best_state = cpu_sd(model.state_dict())
                patience = 0
            else:
                patience += 1

            save_last(last_path, {
                "stage": stage, "epoch": epoch,
                "model": cpu_sd(model.state_dict()),
                "opt": opt.state_dict(), "sch": sch.state_dict(),
                "scaler": scaler.state_dict(),
                "best_f1": best_f1, "best_state": best_state,
            })

            dt = time.time() - t0
            mark = " *" if improved else ""
            print(f"    Epoch {epoch:02d}/{max_ep} | "
                  f"Train Loss {tl:.4f} Acc {ta:.4f} | "
                  f"Val Loss {v['loss']:.4f} Acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f}{mark} | "
                  f"{dt:.0f}s", flush=True)
            if patience >= 7:
                print(f"    Early stop stage {stage}")
                break
            epoch += 1

        stage += 1
        epoch = 1
        payload = None

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"  Best val macro-F1: {best_f1:.4f}")

    ckpt_path = OUTPUT_DIR / f"{model_name}_best.pth"
    torch.save({"model": model_name, "state_dict": cpu_sd(model.state_dict()),
                "img_size": img_size, "val_f1": best_f1, "timm_name": timm_name}, ckpt_path)
    if last_path.exists():
        last_path.unlink()
    print(f"  Saved: {ckpt_path}")

    test = evaluate(model, test_loader, device)
    print(f"\n  TEST RESULTS - {model_name}:")
    print(f"  Accuracy:  {test['accuracy']:.4f}")
    print(f"  Macro-F1:  {test['macro_f1']:.4f}")
    print(classification_report(test["labels"], test["predictions"],
                                target_names=CLASS_NAMES, digits=4))

    del model
    torch.cuda.empty_cache()
    return {
        "model_name": model_name,
        "ckpt_path": str(ckpt_path),
        "val_macro_f1": best_f1,
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "img_size": img_size,
        "timm_name": timm_name,
    }


# ============================================================
# ENSEMBLE
# ============================================================
def ensemble_evaluate(ckpt_paths, X_test, y_test, device):
    print(f"\n{'='*60}")
    print("ENSEMBLE EVALUATION")
    print(f"{'='*60}")

    all_probs = []
    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ckpt["timm_name"], NUM_CLASSES, device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ds = ImageDataset(X_test, y_test, get_transforms(ckpt["img_size"], False))
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

        probs_list = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                probs_list.append(torch.softmax(outputs.float(), dim=1).cpu().numpy())
        all_probs.append(np.concatenate(probs_list))
        del model
        torch.cuda.empty_cache()

    avg_probs = np.mean(all_probs, axis=0)
    preds = avg_probs.argmax(axis=1)
    y_arr = np.array(y_test)

    print(f"\nENSEMBLE ({len(ckpt_paths)} models) - TEST RESULTS:")
    print(f"Accuracy:  {(preds == y_arr).mean():.4f}")
    print(f"Macro-F1:  {f1_score(y_arr, preds, average='macro'):.4f}")
    print(classification_report(y_arr, preds, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_arr, preds))
    return preds, avg_probs


# ============================================================
# MAIN
# ============================================================
def eval_saved(ckpt_path, X_test, y_test, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["timm_name"], NUM_CLASSES, device)
    model.load_state_dict(ckpt["state_dict"])
    ds = ImageDataset(X_test, y_test, get_transforms(ckpt["img_size"], False))
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    test = evaluate(model, loader, device)
    del model
    torch.cuda.empty_cache()
    return {
        "model_name": ckpt["model"],
        "ckpt_path": ckpt_path,
        "val_macro_f1": ckpt["val_f1"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "img_size": ckpt["img_size"],
        "timm_name": ckpt["timm_name"],
    }


def main():
    flag = OUTPUT_DIR / "ALL_DONE.flag"
    if flag.exists():
        print("All phases already complete - nothing to do.")
        return

    set_seed(SEED)
    device = get_device()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset()

    results = []
    for cfg in MODELS_TO_TRAIN:
        ckpt_path = OUTPUT_DIR / f"{cfg['name']}_best.pth"
        if ckpt_path.exists():
            print(f"\n[RESUME] Checkpoint found for {cfg['name']} - skipping training, re-evaluating")
            res = eval_saved(str(ckpt_path), X_test, y_test, device)
            print(f"  {cfg['name']}: val-F1={res['val_macro_f1']:.4f} test-F1={res['test_macro_f1']:.4f}")
            results.append(res)
            continue
        res = train_model(
            cfg["name"], cfg["timm_name"], cfg["img_size"], cfg["batch_size"],
            X_train, y_train, X_val, y_val, X_test, y_test, device,
        )
        results.append(res)
        with open(OUTPUT_DIR / "ipd_results.json", "w") as f:
            json.dump([
                {"model": r["model_name"], "val_macro_f1": round(r["val_macro_f1"], 4),
                 "test_accuracy": round(r["test_accuracy"], 4),
                 "test_macro_f1": round(r["test_macro_f1"], 4)}
                for r in results
            ], f, indent=2)

    print(f"\n{'='*60}")
    print("PHASE 2: MODEL SELECTION (by val macro-F1)")
    print(f"{'='*60}")
    results_sorted = sorted(results, key=lambda x: x["val_macro_f1"], reverse=True)
    for r in results_sorted:
        print(f"  {r['model_name']:25s} val-F1={r['val_macro_f1']:.4f}  test-F1={r['test_macro_f1']:.4f}")

    ckpt_paths = [r["ckpt_path"] for r in results_sorted]
    ensemble_evaluate(ckpt_paths, X_test, y_test, device)

    flag.touch()
    print("\nALL PHASES COMPLETE")


if __name__ == "__main__":
    main()
