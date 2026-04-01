
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://192.168.100.115:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load all models
MODELS = {
  "cnn-baseline": tf.keras.models.load_model("../saved_models/1"),
  "transfer-learning": tf.keras.models.load_model("../saved_models/2"),
  "mobilenetv2": tf.keras.models.load_model("../saved_models/3"),
}

# Map model IDs to display names
MODEL_NAMES = {
  "cnn-baseline": "CNN Baseline",
  "transfer-learning": "Transfer Learning",
  "mobilenetv2": "MobileNetV2",
}

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
  return "Hello, I am alive"

@app.get("/models")
async def get_models():
  return {"models": list(MODELS.keys()), "modelNames": MODEL_NAMES}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_id: str = "ensemble"
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    if model_id == "ensemble":
        # Ensemble: average predictions from all models
        all_predictions = []
        model_ids = list(MODELS.keys())
        for model in MODELS.values():
            all_predictions.append(model.predict(img_batch)[0])
        avg_predictions = np.mean(all_predictions, axis=0)
        predicted_class = CLASS_NAMES[np.argmax(avg_predictions)]
        confidence = float(np.max(avg_predictions))
        # Also return individual model predictions for transparency
        individual = []
        for idx, pred in enumerate(all_predictions):
            model_id = model_ids[idx]
            individual.append({
                "model": MODEL_NAMES[model_id],
                "class": CLASS_NAMES[np.argmax(pred)],
                "confidence": float(np.max(pred))
            })
        return {
            'class': predicted_class,
            'confidence': confidence,
            'individual': individual
        }
    else:
        # Use selected model
        if model_id not in MODELS:
            return {"error": "Model not found"}
        model = MODELS[model_id]
        predictions = model.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            'class': predicted_class,
            'confidence': confidence
        }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
