import sys
import csv
from pathlib import Path

import torch
import timm
from torchvision import transforms
from PIL import Image

DEPLOY = Path(__file__).parent
BUNDLE = DEPLOY / "potato_bundle.pth"
CLASS_NAMES = ["Early Blight", "Healthy", "Late Blight"]
IMG_EXT = (".jpg", ".jpeg", ".png")


def load_bundle():
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for m in bundle["models"]:
        net = timm.create_model(m["timm_name"], pretrained=False, num_classes=bundle["num_classes"])
        net.load_state_dict(m["state_dict"])
        net.eval().to(device)
        n = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        s = m["img_size"]
        tf = transforms.Compose([
            transforms.Resize(int(s * 1.14)), transforms.CenterCrop(s),
            transforms.ToTensor(), n])
        models.append({"key": m["key"], "net": net, "tf": tf, "f1": m.get("val_f1")})
    print(f"Loaded {len(models)} models on {device}")
    return models, device


@torch.no_grad()
def predict_image(path, models, device):
    img = Image.open(path).convert("RGB")
    probs = {}
    for m in models:
        x = m["tf"](img).unsqueeze(0).to(device)
        p = torch.softmax(m["net"](x), dim=1)[0]
        probs[m["key"]] = p.cpu()
    ensemble = sum(probs.values()) / len(probs)
    return ensemble.argmax().item(), ensemble, probs


def collect(folder):
    return sorted([p for p in Path(folder).rglob("*")
                   if p.is_file() and p.suffix.lower() in IMG_EXT])


FOLDER_TO_IDX = {"earlyblight": 0, "healthy": 1, "lateblight": 2}


def run_folder(folder, out_csv=None):
    models, device = load_bundle()
    files = collect(folder)
    rows, correct_by_key = [], {m["key"]: 0 for m in models}
    ens_correct, n = 0, 0
    for f in files:
        key = f.parent.name.lower().replace(" ", "")
        ti = FOLDER_TO_IDX.get(key)
        pi, pe, pm = predict_image(f, models, device)
        if ti is not None:
            n += 1
            ens_correct += int(pi == ti)
            for k, p in pm.items():
                correct_by_key[k] += int(p.argmax().item() == ti)
        rows.append({"image": str(f), "true": CLASS_NAMES[ti] if ti is not None else "",
                     "pred_ensemble": CLASS_NAMES[pi],
                     **{k: CLASS_NAMES[p.argmax().item()] for k, p in pm.items()},
                     "conf_ensemble": round(float(pe[pi]), 4)})
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    if n:
        print(f"\n=== {folder} ({n} labeled) ===")
        print(f"ENSEMBLE({len(models)}): accuracy {ens_correct/n:.3f}")
        for k in correct_by_key:
            print(f"  {k:20s}: accuracy {correct_by_key[k]/n:.3f}")
    return rows


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEPLOY / "test_images_100"
    csv_out = sys.argv[2] if len(sys.argv) > 2 else str(DEPLOY / "predictions.csv")
    rows = run_folder(target, csv_out)
    print(f"\nSaved CSV: {csv_out} ({len(rows)} rows)")
    for r in rows[:10]:
        print(f"{Path(r['image']).name[:40]:42s} -> {r['pred_ensemble']:12s} conf={r['conf_ensemble']}")
