import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix

import timm

SEED = 42
DATA_DIR = Path(r"C:\Users\shadb\Downloads\dataset")
RESULTS = DATA_DIR / "results"
IPD_NAMES = ["Early Blight", "Healthy", "Late Blight"]
PLD_MAP = {"Fungi": 0, "Healthy": 1, "Phytopthora": 2}
CLEAN = {"Healthy", "Phytopthora"}


def get_tf(size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DS(torch.utils.data.Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import random
        from PIL import Image
        try:
            return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))


@torch.no_grad()
def probs_of(model, loader, device):
    out_all = []
    for x, _ in loader:
        x = x.to(device)
        with torch.amp.autocast("cuda"):
            o = model(x)
        out_all.append(torch.softmax(o.float(), 1).cpu().numpy())
    return np.concatenate(out_all)


def collect_ipd_test():
    d = json.load(open(RESULTS / "split_cache.json"))
    return d["X_test"], np.array(d["y_test"])


def collect_pld():
    root = DATA_DIR / "PLD" / "Potato Leaf Disease Dataset in Uncontrolled Environment"
    paths, labels, groups = [], [], []
    for folder, lab in PLD_MAP.items():
        fs = sorted(str(p) for p in (root / folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        paths += fs
        labels += [lab] * len(fs)
        groups += [folder] * len(fs)
    return paths, np.array(labels), groups


def summarize(tag, y, pred):
    acc = (pred == y).mean()
    f1m = f1_score(y, pred, average="macro")
    per = f1_score(y, pred, average=None, zero_division=0)
    print(f"{tag:34s} Acc={acc:.4f}  MacroF1={f1m:.4f}  per-class={'/'.join(f'{v:.3f}' for v in per)}")
    return {"acc": round(float(acc), 4), "macro_f1": round(float(f1m), 4),
            "per_class_f1": [round(float(v), 4) for v in per]}


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    ckpts = {
        "convnext_tiny_v1": RESULTS / "convnext_tiny_best.pth",
        "convnext_tiny_v2": RESULTS / "convnext_tiny_v2_best.pth",
    }

    ipd_paths, ipd_y = collect_ipd_test()
    pld_paths, pld_y, groups = collect_pld()
    clean_mask = np.array([g in CLEAN for g in groups])

    summary = {}
    all_probs_ipd, all_probs_pld = [], []
    for name, cp in ckpts.items():
        ck = torch.load(cp, map_location=device, weights_only=False)
        model = timm.create_model(ck["timm_name"], pretrained=False, num_classes=3)
        model.load_state_dict(ck["state_dict"])
        model.eval().to(device)

        li = DataLoader(DS(ipd_paths, ipd_y, get_tf(ck["img_size"])), batch_size=128,
                        shuffle=False, num_workers=4, pin_memory=True)
        lp = DataLoader(DS(pld_paths, pld_y, get_tf(ck["img_size"])), batch_size=64,
                        shuffle=False, num_workers=4, pin_memory=True)

        pi = probs_of(model, li, device)
        pp = probs_of(model, lp, device)
        all_probs_ipd.append(pi)
        all_probs_pld.append(pp)

        print(f"\n=== {name} (val F1 {ck['val_f1']:.4f}) ===")
        s = {"ipd_val_f1": round(float(ck['val_f1']), 4)}
        s["ipd_test"] = summarize("IPD test", ipd_y, pi.argmax(1))
        s["pld_full"] = summarize("PLD full(mapped)", pld_y, pp.argmax(1))
        pc = pp[clean_mask]
        yc = pld_y[clean_mask]
        s["pld_clean"] = summarize("PLD clean(H+LB)", yc, pc.argmax(1))
        summary[name] = s
        del model
        torch.cuda.empty_cache()

    print("\n=== Ensemble(v1+v2) ===")
    ei = np.mean(all_probs_ipd, axis=0).argmax(1)
    ep = np.mean(all_probs_pld, axis=0).argmax(1)
    epc = ep[clean_mask]
    ens = {}
    ens["ipd_test"] = summarize("Ensemble IPD test", ipd_y, ei)
    ens["pld_full"] = summarize("Ensemble PLD full", pld_y, ep)
    ens["pld_clean"] = summarize("Ensemble PLD clean", pld_y[clean_mask], epc)
    summary["ensemble_v1v2"] = ens

    json.dump(summary, open(RESULTS / "v2_comparison.json", "w"), indent=2)
    print("\nSaved:", RESULTS / "v2_comparison.json")


if __name__ == "__main__":
    main()
