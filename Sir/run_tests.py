import glob
import json
from pathlib import Path
import requests

# ============================================================
# File handling code: Read / create directory
# (independent of platform, created ahead, where Python code is)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent  # where this Python code is


def ensure_dir(path: Path) -> Path:
    """Generate directory independent of platform, created ahead."""
    path.mkdir(parents=True, exist_ok=True)
    return path


BASE = "http://localhost:8000"
# Dataset lives in sibling ../dataset (resolved relative to code, no hardcode)
PV = BASE_DIR.parent / "dataset" / "PlantVillage"

tests = [
    ("Early Blight", sorted(glob.glob(str(PV / "Potato___Early_blight" / "*.JPG")))[:3]),
    ("Late Blight", sorted(glob.glob(str(PV / "Potato___Late_blight" / "*.JPG")))[:3]),
    ("Healthy", sorted(glob.glob(str(PV / "Potato___healthy" / "*.JPG")))[:3]),
]

print("=== PREDICT TESTS (convnext_plantvillage) ===")
results = []
for true_label, paths in tests:
    for p in paths:
        with open(p, "rb") as f:
            r = requests.post(
                f"{BASE}/predict?model_id=convnext_plantvillage",
                files={"file": f}, timeout=60).json()
        ok = "OK" if r.get("class") == true_label else "MISS"
        results.append((true_label, r.get("class"), r.get("confidence"), ok))
        print(f"{true_label:15s} -> {r.get('class'):15s} {r.get('confidence', 0):.4f} [{ok}]")

correct = sum(1 for _, _, _, ok in results if ok == "OK")
print(f"\nAccuracy: {correct}/{len(results)} = {correct/len(results):.1%}")

print("\n=== DEFAULT MODEL (no model_id) ===")
with open(tests[0][1][0], "rb") as f:
    r = requests.post(f"{BASE}/predict", files={"file": f}, timeout=60).json()
print(f"class={r.get('class')} model_id={r.get('model_id')} conf={r.get('confidence')}")

print("\n=== GRADCAM TEST ===")
with open(tests[0][1][0], "rb") as f:
    r = requests.post(f"{BASE}/gradcam?model_id=convnext_plantvillage",
                      files={"file": f}, timeout=120).json()
if "heatmap" in r:
    print(f"gradcam OK: predicted={r.get('predicted_class')} "
          f"heatmap_len={len(r['heatmap'])} conf={r.get('confidence', 0):.4f}")
else:
    print(f"gradcam FAIL: {r}")

# Result -> always export to file; create directory ahead
ensure_dir(BASE_DIR)
with open(BASE_DIR / "test_run_results.json", "w") as f:
    json.dump({"predict_tests": [
        {"true": t, "pred": p, "conf": c, "ok": o} for t, p, c, o in results],
        "accuracy": correct / len(results)}, f, indent=2)
print("\nSaved test_run_results.json")
