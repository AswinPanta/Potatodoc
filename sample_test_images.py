import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE = Path(r"C:\Users\shadb\Downloads\dataset")
DEPLOY = BASE / "deployment"
IMG_EXT = (".jpg", ".jpeg", ".png")

# PLD real-world mapping: Fungi->EarlyBlight, Healthy->Healthy, Phytopthora->LateBlight
MAP = {"Fungi": "EarlyBlight", "Healthy": "Healthy", "Phytopthora": "LateBlight"}
PLD = BASE / "PLD" / "Potato Leaf Disease Dataset in Uncontrolled Environment"

paths, labels = [], []
for src, dst in MAP.items():
    for f in (PLD / src).iterdir():
        if f.suffix.lower() in IMG_EXT:
            paths.append(str(f)); labels.append(dst)

tr, te = train_test_split(list(range(len(paths))), test_size=100,
                          random_state=42, stratify=labels)
out = DEPLOY / "test_images_100"
if out.exists():
    shutil.rmtree(out)
for i in te:
    d = out / labels[i]; d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths[i], d / Path(paths[i]).name)

counts = {}
for i in te:
    counts[labels[i]] = counts.get(labels[i], 0) + 1
print(f"PLD real-world sample: {len(te)} images -> {counts}")

# IPD held-out reference (same distribution as training)
cache = json.load(open(BASE / "results" / "split_cache.json"))
Xt, yt = cache["X_test"], cache["y_test"]
NAME = {0: "EarlyBlight", 1: "Healthy", 2: "LateBlight"}
idx = list(range(len(Xt)))
_, ri = train_test_split(idx, test_size=100, random_state=7, stratify=yt)
out2 = DEPLOY / "ipd_reference_100"
if out2.exists():
    shutil.rmtree(out2)
c2 = {}
for i in ri:
    d = out2 / NAME[yt[i]]; d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Xt[i], d / Path(Xt[i]).name)
    c2[NAME[yt[i]]] = c2.get(NAME[yt[i]], 0) + 1
print(f"IPD reference sample: {len(ri)} images -> {c2}")
