"""
Potato Disease Detection — Registry-Driven Multi-Model Server
==============================================================
Loads model + server config from config/*.yaml.
Serves all enabled models; mobile app discovers via GET /models.

Data flow:
    config/datasets.yaml  → where raw data lives (train/prepare only)
    config/models.yaml    → model_id → checkpoint + preprocessing
    config/server.yaml    → host, port, default model

Run:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import time
from pathlib import Path

import numpy as np
import torch
import timm
import yaml
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from torchvision import transforms

# ============================================================
# LOAD CONFIGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_CFG = yaml.safe_load(open(BASE_DIR / "config" / "models.yaml"))
SERVER_CFG = yaml.safe_load(open(BASE_DIR / "config" / "server.yaml"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_MODEL = SERVER_CFG.get("default_model", "convnext_plantvillage")

# ============================================================
# APP + MODEL REGISTRY
# ============================================================
app = FastAPI(title="Potato Disease Detection API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

REGISTRY = {}


def make_preprocess(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


for model_id, m in MODELS_CFG.items():
    if m.get("enabled", True) is False:
        print(f"[SKIP] {model_id} (disabled in models.yaml)")
        continue
    ckpt_path = BASE_DIR / m["checkpoint"]
    if not ckpt_path.exists():
        print(f"[SKIP] {model_id} (checkpoint missing: {ckpt_path})")
        continue
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = timm.create_model(ckpt.get("timm_name", m["timm_name"]),
                              pretrained=False, num_classes=len(m["class_names"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    REGISTRY[model_id] = {
        "model": model,
        "preprocess": make_preprocess(m["img_size"]),
        "class_names": m["class_names"],
        "display_name": m.get("display_name", model_id),
        "dataset": m.get("dataset", "unknown"),
        "val_f1": ckpt.get("val_f1"),
    }
    print(f"[OK] {model_id}: {m['timm_name']} val_f1={ckpt.get('val_f1', '?')}")

if not REGISTRY:
    print("WARNING: no models loaded! Check config/models.yaml checkpoints.")


def get_entry(model_id: str):
    return REGISTRY.get(model_id) or REGISTRY.get(DEFAULT_MODEL)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/ping")
def ping():
    return "ok"


@app.get("/models")
def list_models():
    return {
        "models": list(REGISTRY.keys()),
        "modelNames": {mid: e["display_name"] for mid, e in REGISTRY.items()},
        "default": DEFAULT_MODEL if DEFAULT_MODEL in REGISTRY else next(iter(REGISTRY), None),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = Query(default=None),
):
    entry = get_entry(model_id or DEFAULT_MODEL)
    if entry is None:
        return JSONResponse(status_code=503, content={"error": "No models loaded"})
    try:
        t0 = time.time()
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = entry["preprocess"](img).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast(device.type if device.type == "cuda" else "cpu"):
                probs = torch.softmax(entry["model"](tensor), dim=1).cpu().numpy()[0]
        names = entry["class_names"]
        pred_idx = int(np.argmax(probs))
        elapsed = time.time() - t0
        result = {
            "class": names[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": {names[i]: round(float(probs[i]), 4) for i in range(len(names))},
            "model": entry["display_name"],
            "model_id": model_id or DEFAULT_MODEL,
            "inference_time_ms": round(elapsed * 1000, 1),
        }
        print(f"[{result['model_id']}] {result['class']} ({result['confidence']:.2%}) in {elapsed*1000:.0f}ms")
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    model_id: str = Query(default=None),
):
    entry = get_entry(model_id or DEFAULT_MODEL)
    if entry is None:
        return JSONResponse(status_code=503, content={"error": "No models loaded"})
    try:
        import torch.nn.functional as F
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        img_size = entry["preprocess"].transforms[1].size
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]

        tensor = entry["preprocess"](img).unsqueeze(0).to(device)
        tensor.requires_grad_(True)
        entry["model"].zero_grad()
        output = entry["model"](tensor)
        pred_class = output.argmax(dim=1)
        output[0, pred_class].backward()

        attn = tensor.grad.data.abs().mean(dim=1, keepdim=True).cpu()
        attn = F.interpolate(attn, size=(img_size, img_size), mode="bilinear", align_corners=False)
        attn = attn.squeeze().numpy()
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img.resize((img_size, img_size)))
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(img.resize((img_size, img_size)))
        axes[1].imshow(cm.jet(attn), alpha=0.5)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "heatmap": f"data:image/png;base64,{heatmap_b64}",
            "predicted_class": entry["class_names"][pred_class.item()],
            "confidence": float(torch.softmax(output, dim=1).max()),
            "model_id": model_id or DEFAULT_MODEL,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_CFG.get("host", "0.0.0.0"),
                port=SERVER_CFG.get("port", 8000))
