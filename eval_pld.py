import json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, f1_score, confusion_matrix

SEED = 42
DATA_DIR = Path(r"C:\Users\shadb\Downloads\dataset")
RESULTS = DATA_DIR / "results"
OUT = RESULTS / "pld_results.json"

# IPD label order: 0=Early Blight, 1=Healthy, 2=Late Blight
IPD_NAMES = ["Early Blight", "Healthy", "Late Blight"]

# PLD folder -> IPD label mapping (cross-dataset protocol)
PLD_MAP = {
    "Fungi": 0,        # early-blight-type fungal lesions (Alternaria)
    "Healthy": 1,
    "Phytopthora": 2,  # late blight (Phytophthora infestans)
}
CLEAN_SUBSET = {"Healthy", "Phytopthora"}  # unambiguous mapping only


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = paths, labels, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs_all = []
    for x, _ in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda"):
            out = model(x)
        probs_all.append(torch.softmax(out.float(), dim=1).cpu().numpy())
    return np.concatenate(probs_all)


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def collect_pld():
    root = DATA_DIR / "PLD" / "Potato Leaf Disease Dataset in Uncontrolled Environment"
    paths, labels, groups = [], [], []
    exts = (".jpg", ".jpeg", ".png")
    for folder, lab in PLD_MAP.items():
        d = root / folder
        fs = sorted(str(p) for p in d.iterdir() if p.suffix.lower() in exts)
        paths += fs
        labels += [lab] * len(fs)
        groups += [folder] * len(fs)
    return paths, labels, groups


def report(name, y, pred, probs):
    acc = (pred == y).mean()
    f1m = f1_score(y, pred, average="macro")
    print(f"\n--- {name} ---")
    print(f"n={len(y)}  Accuracy={acc:.4f}  Macro-F1={f1m:.4f}")
    print(classification_report(y, pred, target_names=IPD_NAMES, digits=4, zero_division=0))
    print("Confusion Matrix (rows=true EB/H/LB):")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))
    return {"name": name, "n": int(len(y)), "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1m), 4),
            "per_class_f1": [round(float(v), 4) for v in f1_score(y, pred, average=None, zero_division=0)]}


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpts = sorted(RESULTS.glob("*_best.pth"))
    assert len(ckpts) >= 1, "No trained model checkpoints found"
    print(f"Found {len(ckpts)} checkpoints: {[c.stem for c in ckpts]}")

    paths, labels, groups = collect_pld()
    print(f"PLD mapped images: {len(paths)} "
          f"(EB/Fungi={groups.count('Fungi')}, H={groups.count('Healthy')}, "
          f"LB/Phyto={groups.count('Phytopthora')})")

    clean_mask = np.array([g in CLEAN_SUBSET for g in groups])
    y_full = np.array(labels)
    y_clean = y_full[clean_mask]

    per_model_probs = {}
    summary = {"models": {}, "ensemble": {}}

    for c in ckpts:
        ck = torch.load(c, map_location=device, weights_only=False)
        import timm
        model = timm.create_model(ck["timm_name"], pretrained=False, num_classes=3)
        model.load_state_dict(ck["state_dict"])
        model = model.to(device)

        ds = ImageDataset(paths, labels, get_eval_transform(ck["img_size"]))
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        probs = predict(model, loader, device)
        per_model_probs[ck["model"]] = probs
        pred = probs.argmax(1)

        summary["models"][ck["model"]] = {
            "val_macro_f1_ipd": round(float(ck["val_f1"]), 4),
            "full": report(f"{ck['model']} | PLD-full(mapped)", y_full, pred, probs),
            "clean": report(f"{ck['model']} | PLD-clean(H+LB)", y_clean, pred[clean_mask], None),
        }
        del model
        torch.cuda.empty_cache()

    # Soft-voting ensemble over ALL available models (best selection done on IPD val)
    names = list(per_model_probs.keys())
    avg = np.mean([per_model_probs[n] for n in names], axis=0)
    epred = avg.argmax(1)
    summary["ensemble"]["members"] = names
    summary["ensemble"]["full"] = report(f"Ensemble({len(names)}) | PLD-full(mapped)", y_full, epred, avg)
    summary["ensemble"]["clean"] = report(f"Ensemble({len(names)}) | PLD-clean(H+LB)", y_clean, epred[clean_mask], None)

    json.dump(summary, open(OUT, "w"), indent=2)
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
