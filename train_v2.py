import os, json, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, f1_score

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# ============================================================
# CONFIG - regularized retrain of ConvNeXt-Tiny (v2)
# ============================================================
SEED = 42
DATA_DIR = Path(r"C:\Users\shadb\Downloads\dataset")
CLASSES = ["earlyblt", "healthy", "lateblt"]
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]
NUM_CLASSES = 3
OUTPUT_DIR = DATA_DIR / "results"
MODEL_NAME = "convnext_tiny_v2"
TIMM_NAME = "convnext_tiny.fb_in22k"
IMG_SIZE = 224
BATCH_SIZE = 64

# Anti-overfitting hyperparameters
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
WEIGHT_DECAY = 0.05
CROP_SCALE = (0.5, 1.0)
DROPPATH = 0.1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.transform(img), self.labels[i]
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))


def load_split():
    d = json.load(open(OUTPUT_DIR / "split_cache.json"))
    print(f"Split loaded: train {len(d['X_train'])} val {len(d['X_val'])} test {len(d['X_test'])}")
    return d


def get_transforms(is_train=True):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        norm,
    ])


def cpu_sd(sd):
    return {k: v.detach().cpu() for k, v in sd.items()}


def save_last(path, payload):
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_one_epoch(model, loader, crit, opt, scaler, mixup_fn, device):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if mixup_fn is not None:
            x_soft, y_soft = mixup_fn(x, y)
        else:
            x_soft, y_soft = x, y
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(x_soft)
            loss = crit(out, y_soft) if y_soft.dtype == torch.float32 else crit(out, y_soft)
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
    preds, labels, probs_all, tloss, tot = [], [], [], 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tloss += ce(out, y).item() * x.size(0)
        tot += x.size(0)
        probs_all.append(torch.softmax(out.float(), 1).cpu().numpy())
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds, labels = np.concatenate(preds), np.concatenate(labels)
    return {"loss": tloss / tot,
            "macro_f1": f1_score(labels, preds, average="macro"),
            "accuracy": (preds == labels).mean(),
            "predictions": preds, "labels": labels}


def main():
    flag = OUTPUT_DIR / "V2_DONE.flag"
    if flag.exists():
        print("v2 training already complete.")
        return

    set_seed(SEED)
    device = torch.device("cuda")
    d = load_split()

    train_ds = ImageDataset(d["X_train"], d["y_train"], get_transforms(True))
    val_ds = ImageDataset(d["X_val"], d["y_val"], get_transforms(False))
    test_ds = ImageDataset(d["X_test"], d["y_test"], get_transforms(False))

    nw = min(6, os.cpu_count() or 1)
    common = dict(num_workers=nw, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              persistent_workers=nw > 0, prefetch_factor=3 if nw > 0 else None, **common)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, **common)

    model = timm.create_model(TIMM_NAME, pretrained=True, num_classes=NUM_CLASSES,
                              drop_path_rate=DROPPATH).to(device)

    mixup_fn = Mixup(mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA,
                     label_smoothing=LABEL_SMOOTH, num_classes=NUM_CLASSES, prob=1.0)
    crit_train = SoftTargetCrossEntropy()

    scaler = torch.amp.GradScaler("cuda")
    last_path = OUTPUT_DIR / f"{MODEL_NAME}_last.pth"

    payload = None
    if last_path.exists():
        try:
            payload = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            print(f"[RESUME] stage {payload['stage']} epoch {payload['epoch']} best {payload['best_f1']:.4f}")
        except Exception as e:
            print(f"[RESUME] failed ({e})")
            payload = None

    best_f1 = payload["best_f1"] if payload else -1.0
    best_state = payload["best_state"] if payload else None
    stage = payload["stage"] if payload else 1
    epoch = payload["epoch"] + 1 if payload else 1
    need_opt = payload is not None

    while stage <= 2:
        max_ep = 10 if stage == 1 else 50
        use_mixup = stage == 2  # head-only stage trains clean for stable probe
        fn_mix = mixup_fn if use_mixup else None

        if stage == 1:
            for p in model.parameters():
                p.requires_grad = False
            head = getattr(model, "head", None)
            for p in head.parameters():
                p.requires_grad = True
            opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
            crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
            print(f"Stage 1: head-only ({sum(p.numel() for p in head.parameters()):,} params)")
        else:
            for p in model.parameters():
                p.requires_grad = True
            hk = ("head", "classifier", "fc")
            hp = [p for n, p in model.named_parameters() if any(k in n for k in hk)]
            bp = [p for n, p in model.named_parameters() if not any(k in n for k in hk)]
            opt = optim.AdamW([{"params": hp, "lr": 1e-4}, {"params": bp, "lr": 1e-5}],
                              weight_decay=WEIGHT_DECAY)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
            crit = crit_train
            print("Stage 2: full fine-tune (MixUp+CutMix active)")

        if need_opt and payload is not None and payload.get("stage") == stage:
            opt.load_state_dict(payload["opt"])
            sch.load_state_dict(payload["sch"])
            scaler.load_state_dict(payload["scaler"])
            sch.T_max = max_ep
        need_opt = False

        patience = 0
        while epoch <= max_ep:
            t0 = time.time()
            tl, ta = train_one_epoch(model, train_loader, crit, opt, scaler, fn_mix, device)
            v = evaluate(model, val_loader, device)
            sch.step()
            improved = v["macro_f1"] > best_f1
            if improved:
                best_f1 = float(v["macro_f1"])
                best_state = cpu_sd(model.state_dict())
                patience = 0
            else:
                patience += 1
            save_last(last_path, {"stage": stage, "epoch": epoch,
                                  "model": cpu_sd(model.state_dict()),
                                  "opt": opt.state_dict(), "sch": sch.state_dict(),
                                  "scaler": scaler.state_dict(),
                                  "best_f1": best_f1, "best_state": best_state})
            mark = " *" if improved else ""
            print(f"  Epoch {epoch:02d}/{max_ep} | Train Loss {tl:.4f} Acc {ta:.4f} | "
                  f"Val Loss {v['loss']:.4f} F1 {v['macro_f1']:.4f}{mark} | {time.time()-t0:.0f}s", flush=True)
            if patience >= 10:
                print(f"Early stop stage {stage}")
                break
            epoch += 1
        stage += 1
        epoch = 1
        payload = None

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    ckpt_path = OUTPUT_DIR / f"{MODEL_NAME}_best.pth"
    torch.save({"model": MODEL_NAME, "state_dict": cpu_sd(model.state_dict()),
                "img_size": IMG_SIZE, "val_f1": best_f1, "timm_name": TIMM_NAME}, ckpt_path)
    if last_path.exists():
        last_path.unlink()
    print(f"Saved: {ckpt_path}")

    t = evaluate(model, test_loader, device)
    print("\nTEST RESULTS - convnext_tiny_v2:")
    print(f"Accuracy: {t['accuracy']:.4f}")
    print(f"Macro-F1: {t['macro_f1']:.4f}")
    print(classification_report(t["labels"], t["predictions"],
                                target_names=CLASS_NAMES, digits=4))
    flag.touch()
    print("V2 COMPLETE")


if __name__ == "__main__":
    main()
