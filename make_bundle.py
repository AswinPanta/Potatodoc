import torch
from pathlib import Path

BASE = Path(r"C:\Users\shadb\Downloads\dataset")
R = BASE / "results"
DEPLOY = BASE / "deployment"
DEPLOY.mkdir(exist_ok=True)

FILES = {
    "efficientnetv2_b3_best.pth": "efficientnet_b3_v1",
    "convnext_tiny_best.pth": "convnext_tiny_v1",
    "swin_tiny_best.pth": "swin_tiny_v1",
    "convnext_tiny_v2_best.pth": "convnext_tiny_v2",
}

bundle = {"format": "potato-leaf-disease-bundle-v1", "num_classes": 3,
          "class_names": ["Early Blight", "Healthy", "Late Blight"], "models": []}

for fname, key in FILES.items():
    ck = torch.load(R / fname, map_location="cpu", weights_only=False)
    sd = {k: v.detach().cpu() for k, v in ck["state_dict"].items()}
    bundle["models"].append({
        "key": key,
        "timm_name": ck["timm_name"],
        "img_size": int(ck.get("img_size", 224)),
        "val_f1": float(ck.get("val_f1", -1)),
        "state_dict": sd,
    })
    print(f"packed {key}: timm={ck['timm_name']} img={ck.get('img_size',224)} f1={ck.get('val_f1')}")

out = DEPLOY / "potato_bundle.pth"
torch.save(bundle, out)
print(f"\nBUNDLE SAVED: {out} ({out.stat().st_size/1e6:.0f} MB)")
