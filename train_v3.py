import os, json, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

# ============================================================
# IrishPotato37G trainer - balanced anti-overfit AND anti-underfit recipe
# ============================================================
SEED = 42
DATA_DIR = Path(r"C:\Users\shadb\Downloads\dataset")
BIG = DATA_DIR / "IrishPotato37G"
CLASSES = ["earlyblt", "healthy", "lateblt"]
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]
NUM_CLASSES = 3
RESULTS = DATA_DIR / "results"
MODEL_NAME = "convnext_tiny_v3"
TIMM_NAME = "convnext_tiny.fb_in22k"
IMG_SIZE = 224
BATCH_SIZE = 64

# --- balance knobs ---
LABEL_SMOOTH = 0.1      # anti-overfit: stops overconfident memorization
MIXUP_PROB = 0.7        # <1.0 leaves clean batches -> anti-underfit headroom
MIXUP_A, CUTMIX_A = 0.2, 1.0
WD = 0.05               # anti-overfit weight decay
DROPPATH = 0.1          # anti-overfit stochastic depth
CROP_SCALE = (0.5, 1.0)
MAX_EP = {1: 10, 2: 50}
PATIENCE = 10           # long patience = anti-underfit (won't quit too early)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class ImageDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.tf(img), self.labels[i]
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))


def prepare():
    cache = RESULTS / "split_cache_v3.json"
    if cache.exists():
        d = json.load(open(cache))
        print(f"Split loaded: {len(d['X_train'])}/{len(d['X_val'])}/{len(d['X_test'])}")
        return d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]

    paths, labels = [], []
    for ci, c in enumerate(CLASSES):
        for f in sorted((BIG / c).iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths.append(str(f)); labels.append(ci)
    print(f"Big dataset images: {len(paths)} ({[labels.count(i) for i in range(3)]})")
    Xtr, Xt, ytr, yt = train_test_split(paths, labels, test_size=0.3, random_state=SEED, stratify=labels)
    Xva, Xte, yva, yte = train_test_split(Xt, yt, test_size=0.5, random_state=SEED, stratify=yt)
    json.dump({"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva,
               "X_test": Xte, "y_test": yte}, open(cache, "w"))
    print(f"Train {len(Xtr)} | Val {len(Xva)} | Test {len(Xte)}")
    return Xtr, ytr, Xva, yva, Xte, yte


def get_tf(train=True):
    n = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=CROP_SCALE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(), n])
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)), transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), n])


def cpu_sd(sd):
    return {k: v.detach().cpu() for k, v in sd.items()}


def save_last(p, payload):
    t = str(p) + ".tmp"; torch.save(payload, t); os.replace(t, p)


def train_epoch(model, loader, crit, opt, scaler, mix, device):
    model.train(); tl, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        xin = x
        ysoft = None
        if mix is not None:
            xin, ysoft = mix(x, y)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out = model(xin)
            loss = crit(out, ysoft) if ysoft is not None else crit(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        tl += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()   # accuracy vs CLEAN labels
        total += x.size(0)
    return tl / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); P, L, tl, tot = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        o = model(x)
        tl += ce(o, y).item() * x.size(0); tot += x.size(0)
        P.append(o.argmax(1).cpu().numpy()); L.append(y.cpu().numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    return {"loss": tl / tot, "macro_f1": float(f1_score(L, P, average="macro")),
            "accuracy": float((P == L).mean()), "predictions": P, "labels": L}


def main():
    flag = RESULTS / "V3_DONE.flag"
    if flag.exists():
        print("v3 already complete."); return

    set_seed(SEED)
    device = torch.device("cuda")
    Xtr, ytr, Xva, yva, Xte, yte = prepare()

    nw = min(6, os.cpu_count() or 1)
    common = dict(num_workers=nw, pin_memory=True)
    tr = DataLoader(ImageDataset(Xtr, ytr, get_tf(True)), batch_size=BATCH_SIZE, shuffle=True,
                    persistent_workers=nw > 0, prefetch_factor=3 if nw > 0 else None, **common)
    va = DataLoader(ImageDataset(Xva, yva, get_tf(False)), batch_size=BATCH_SIZE * 2, shuffle=False, **common)
    te = DataLoader(ImageDataset(Xte, yte, get_tf(False)), batch_size=BATCH_SIZE * 2, shuffle=False, **common)

    model = timm.create_model(TIMM_NAME, pretrained=True, num_classes=NUM_CLASSES,
                              drop_path_rate=DROPPATH).to(device)

    mixup = Mixup(mixup_alpha=MIXUP_A, cutmix_alpha=CUTMIX_A, label_smoothing=LABEL_SMOOTH,
                  num_classes=NUM_CLASSES, prob=MIXUP_PROB)
    soft_ce = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler("cuda")
    last_path = RESULTS / f"{MODEL_NAME}_last.pth"

    payload = None
    if last_path.exists():
        try:
            payload = torch.load(last_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            print(f"[RESUME] stage {payload['stage']} ep {payload['epoch']} best {payload['best_f1']:.4f}")
        except Exception as e:
            print(f"[RESUME] failed ({e})"); payload = None

    best_f1 = payload["best_f1"] if payload else -1.0
    best_state = payload["best_state"] if payload else None
    stage = payload["stage"] if payload else 1
    epoch = payload["epoch"] + 1 if payload else 1
    need_opt = payload is not None

    while stage <= 2:
        max_ep = MAX_EP[stage]
        if stage == 1:
            for p in model.parameters(): p.requires_grad = False
            head = model.head
            for p in head.parameters(): p.requires_grad = True
            opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=WD)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
            crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
            fn_mix = None
            print(f"Stage 1: head probe ({sum(p.numel() for p in head.parameters()):,} params)")
        else:
            for p in model.parameters(): p.requires_grad = True
            hk = ("head", "classifier", "fc")
            hp = [p for n_, p in model.named_parameters() if any(k in n_ for k in hk)]
            bp = [p for n_, p in model.named_parameters() if not any(k in n_ for k in hk)]
            opt = optim.AdamW([{"params": hp, "lr": 1e-4}, {"params": bp, "lr": 1e-5}], weight_decay=WD)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
            crit = soft_ce
            fn_mix = mixup
            print("Stage 2: full fine-tune (MixUp/CutMix p=%.2f)" % MIXUP_PROB)

        if need_opt and payload is not None and payload.get("stage") == stage:
            opt.load_state_dict(payload["opt"]); sch.load_state_dict(payload["sch"])
            scaler.load_state_dict(payload["scaler"]); sch.T_max = max_ep
        need_opt = False

        pat = 0
        while epoch <= max_ep:
            t0 = time.time()
            tl, ta = train_epoch(model, tr, crit, opt, scaler, fn_mix, device)
            v = evaluate(model, va, device)
            sch.step()
            imp = v["macro_f1"] > best_f1
            if imp:
                best_f1 = v["macro_f1"]; best_state = cpu_sd(model.state_dict()); pat = 0
            else:
                pat += 1
            save_last(last_path, {"stage": stage, "epoch": epoch,
                                  "model": cpu_sd(model.state_dict()),
                                  "opt": opt.state_dict(), "sch": sch.state_dict(),
                                  "scaler": scaler.state_dict(),
                                  "best_f1": best_f1, "best_state": best_state})
            gap = ta - v["accuracy"]
            print(f"  Epoch {epoch:02d}/{max_ep} | Train Loss {tl:.4f} Acc(clean) {ta:.4f} | "
                  f"Val Loss {v['loss']:.4f} Acc {v['accuracy']:.4f} F1 {v['macro_f1']:.4f} | "
                  f"gap {gap:+.4f} | {time.time()-t0:.0f}s{' *' if imp else ''}", flush=True)
            if pat >= PATIENCE:
                print(f"Early stop stage {stage}"); break
            epoch += 1
        stage += 1; epoch = 1; payload = None

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    ckpt = RESULTS / f"{MODEL_NAME}_best.pth"
    torch.save({"model": MODEL_NAME, "state_dict": cpu_sd(model.state_dict()),
                "img_size": IMG_SIZE, "val_f1": best_f1, "timm_name": TIMM_NAME}, ckpt)
    if last_path.exists(): last_path.unlink()
    print(f"Saved: {ckpt}")

    t = evaluate(model, te, device)
    print("\nTEST RESULTS - convnext_tiny_v3 (IrishPotato37G):")
    print(f"Accuracy: {t['accuracy']:.4f}")
    print(f"Macro-F1: {t['macro_f1']:.4f}")
    print(classification_report(t["labels"], t["predictions"], target_names=CLASS_NAMES, digits=4))
    flag.touch()
    print("V3 COMPLETE")


if __name__ == "__main__":
    main()
