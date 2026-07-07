from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

MODELS = {}
MODEL_LOAD_ERRORS = {}

for mid, path in [
    ("cnn-baseline", BASE_DIR / "../saved_models/1"),
    ("transfer-learning", BASE_DIR / "../saved_models/2"),
    ("mobilenetv2", BASE_DIR / "../saved_models/3"),
]:
    try:
        MODELS[mid] = tf.keras.models.load_model(path)
    except Exception as exc:
        MODEL_LOAD_ERRORS[mid] = str(exc)

# Map model IDs to display names
MODEL_NAMES = {
  "cnn-baseline": "CNN Baseline",
  "transfer-learning": "Transfer Learning",
  "mobilenetv2": "MobileNetV2",
}

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB").resize((256, 256))
    return np.array(image)

def preprocess_for_model(image: np.ndarray, model_id: str) -> np.ndarray:
    if model_id == "mobilenetv2":
        return preprocess_input(image)
    # Baseline and transfer-learning models already contain an internal
    # Rescaling(1./255) layer, so they expect raw [0, 255] inputs.
    return image

@app.get("/ping")
def ping():
  return "Hello, I am alive"

@app.get("/models")
def get_models():
  return {"models": list(MODELS.keys()), "modelNames": MODEL_NAMES}

@app.get("/health")
def health():
    if len(MODELS) == 3:
        return {"status": "success", "models_loaded": list(MODELS.keys())}
    return {
        "status": "failure",
        "models_loaded": list(MODELS.keys()),
        "errors": MODEL_LOAD_ERRORS,
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = "ensemble"
):
    try:
        image = read_file_as_image(await file.read())
    except Exception:
        return {"error": "Invalid image file"}

    img_batch = np.expand_dims(image, 0)

    if model_id == "ensemble":
        model_ids = list(MODELS.keys())
        all_predictions = []
        for mid in model_ids:
            processed = preprocess_for_model(img_batch, mid)
            all_predictions.append(MODELS[mid].predict(processed)[0])
        avg_predictions = np.mean(all_predictions, axis=0)
        predicted_class = CLASS_NAMES[np.argmax(avg_predictions)]
        confidence = float(np.max(avg_predictions))
        individual = [
            {
                "model": MODEL_NAMES[model_ids[idx]],
                "class": CLASS_NAMES[np.argmax(pred)],
                "confidence": float(np.max(pred))
            }
            for idx, pred in enumerate(all_predictions)
        ]
        return {
            'class': predicted_class,
            'confidence': confidence,
            'individual': individual
        }
    else:
        if model_id not in MODELS:
            return {"error": "Model not found"}
        processed = preprocess_for_model(img_batch, model_id)
        model = MODELS[model_id]
        predictions = model.predict(processed)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            'class': predicted_class,
            'confidence': confidence
        }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
