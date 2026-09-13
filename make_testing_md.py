import csv, json
from pathlib import Path
from datetime import date

DEPLOY = Path(r"C:\Users\shadb\Downloads\dataset\deployment")
rows = list(csv.DictReader(open(DEPLOY / "predictions_real100.csv", encoding="utf-8")))
KEYS = ["efficientnet_b3_v1", "convnext_tiny_v1", "swin_tiny_v1", "convnext_tiny_v2"]
errors = json.load(open(Path(r"C:\Users\shadb\AppData\Local\Temp\opencode") / "errors.json"))

per = {}
for r in rows:
    d = per.setdefault(r["true"], [0, 0]); d[1] += 1; d[0] += int(r["pred_ensemble"] == r["true"])

L = []
L.append("# TESTING REPORT - Potato Leaf Disease Classifier")
L.append(f"\n**Date:** {date.today().isoformat()}  ")
L.append("**Model file (deliverable .pth):** `C:\\Users\\shadb\\Downloads\\dataset\\deployment\\potato_bundle.pth` (385 MB)  ")
L.append("**Prediction method:** soft-vote average of 4 models (EfficientNetV2-B3 + ConvNeXt-Tiny v1 + Swin-Tiny v1 + ConvNeXt-Tiny v2)\n")

L.append("## 1. Real-world test: 100 unseen field images (PLD)")
L.append("\n| Model | Accuracy /100 |\n|---|---|")
for name, acc in [("Ensemble (4-model)", "42%"), ("Swin-Tiny v1", "42%"), ("ConvNeXt-Tiny v1", "40%"),
                  ("ConvNeXX-Tiny v2".replace("XX", "x"), "39%"), ("EfficientNet-B3 v1", "37%")]:
    L.append(f"| {name} | {acc} |")
L.append("\n### Per-class results (ensemble)\n\n| True class | Correct | Accuracy |\n|---|---|---|")
order = ["Late Blight", "Healthy", "Early Blight"]
for t in order:
    c, n = per[t]
    L.append(f"| {t} | {c}/{n} | {c/n*100:.0f}% |")

L.append("\n## 2. Reference test: 100 held-out IPD lab images")
L.append("\n| Ensemble & every single model | Accuracy |\n|---|---|\n| All | **100/100 (100%)** |")

L.append("\n## 3. Images that made errors (real-world test)")
L.append(f"\n**Total errors: {len(errors)} / 100**\n")
all_agree = sum(1 for e in errors if e[4] == ". . . .")
L.append(f"- {all_agree}/{len(errors)} errors: ALL 4 models were wrong together (hard domain-shift cases)")
L.append("- Main pattern: Early Blight predicted as Late Blight (44 cases), Healthy predicted as Late Blight (9 cases), Early->Healthy (5 cases)\n")
L.append("| # | Image filename | True label | Predicted as | Confidence | 4 models correct? (E/C/S/V2) |")
L.append("|---|---|---|---|---|---|")
for i, (t, p, name, conf, votes) in enumerate(errors, 1):
    L.append(f"| {i} | `{name}` | {t} | **{p}** | {conf} | {'ALL WRONG' if votes == '. . . . ' else votes.replace('.','wrong').replace('Y','right')} |")

L.append("\n## 4. Conclusion")
L.append("""
- Same-distribution (lab) images: perfect accuracy.
- Real-world field photos: Late Blight detection excellent (100%), but Early vs Late Blight confusion under uncontrolled conditions drags overall to ~42%.
- Remedy in progress: v3 training on the 37 GB expanded Irish Potato dataset (downloading) to add diversity and close this gap.
""")

out = DEPLOY / "testing.md"
out.write_text("\n".join(L), encoding="utf-8")
print(f"WRITTEN: {out} ({out.stat().st_size} bytes, {len(errors)} errors listed)")
