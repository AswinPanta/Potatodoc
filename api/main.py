from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import io
import base64
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

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

# Find saved_models directory (Docker path vs local dev path)
model_base = BASE_DIR / "../saved_models"
if not model_base.exists():
    model_base = BASE_DIR / "saved_models"
    logger.info(f"Using model path: {model_base}")

MODEL_PATHS = {
    "cnn-baseline": model_base / "1",
    "transfer-learning": model_base / "2",
    "mobilenetv2": model_base / "3",
}

for mid, path in MODEL_PATHS.items():
    try:
        if path.exists():
            MODELS[mid] = tf.keras.models.load_model(str(path))
            logger.info(f"Loaded model {mid} from {path}")
        else:
            MODEL_LOAD_ERRORS[mid] = f"Path does not exist: {path}"
    except Exception as exc:
        MODEL_LOAD_ERRORS[mid] = str(exc)
        logger.warning(f"Failed to load model {mid}: {exc}")

# Map model IDs to display names
MODEL_NAMES = {
  "cnn-baseline": "CNN Baseline",
  "transfer-learning": "Transfer Learning",
  "mobilenetv2": "MobileNetV2",
}

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
UNKNOWN_CLASS = "Unknown"
UNKNOWN_THRESHOLD = 0.70  # Max confidence below this -> "Unknown Image"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB").resize((256, 256))
    return np.array(image)

def preprocess_for_model(image: np.ndarray, model_id: str) -> np.ndarray:
    if model_id == "mobilenetv2":
        return preprocess_input(image)
    # Baseline and transfer-learning models already contain an internal
    # Rescaling(1./255) layer, so they expect raw [0, 255] inputs.
    return image

# ---------- Grad-CAM Helpers ----------

def find_last_conv_layer(model):
    """Find the last Conv2D layer name in a Keras model, recursing into sub-models."""
    last_conv = [None]

    def _search(layer):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv[0] = layer.name
        elif hasattr(layer, 'layers'):
            for sub in layer.layers:
                _search(sub)

    for layer in model.layers:
        _search(layer)

    return last_conv[0]


def compute_gradcam(model, img_array, model_id):
    """
    Compute Grad-CAM heatmap for a given model and input image.
    Returns a heatmap array of shape (H, W) in range [0, 1].
    """
    last_conv_name = find_last_conv_layer(model)
    if last_conv_name is None:
        raise ValueError(f"Could not find a Conv2D layer in model '{model_id}'")

    # Build a model that outputs both conv activations and final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    # Convert to tensor and ensure float32
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # Weight feature maps by importance
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(
        tf.multiply(pooled_grads, conv_outputs), axis=-1
    )

    # ReLU + normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())

    return heatmap.numpy()


def generate_heatmap_overlay(heatmap, original_image):
    """
    Generate a base64-encoded PNG of the heatmap overlaid on the original image.
    original_image: numpy array of shape (256, 256, 3), range [0, 255]
    """
    # Resize heatmap to match original dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (256, 256)
    ).numpy().squeeze()

    # Apply jet colormap
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # RGBA -> RGB
    heatmap_colored = np.uint8(255 * heatmap_colored)

    # Blend with original image using PIL
    original_pil = Image.fromarray(original_image)
    heatmap_pil = Image.fromarray(heatmap_colored)
    blended = Image.blend(original_pil, heatmap_pil, alpha=0.4)

    # Convert to base64
    buffered = io.BytesIO()
    blended.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{img_base64}"


def get_all_heatmaps(image):
    """Compute Grad-CAM heatmaps for all models."""
    heatmaps = {}
    for mid in MODELS.keys():
        try:
            processed = preprocess_for_model(np.expand_dims(image, 0), mid)
            heatmap_raw = compute_gradcam(MODELS[mid], processed, mid)
            overlay = generate_heatmap_overlay(heatmap_raw, image)
            heatmaps[MODEL_NAMES[mid]] = overlay
        except Exception as exc:
            heatmaps[MODEL_NAMES[mid]] = None
    return heatmaps


# ---------- Unknown Image Detection ----------

def compute_entropy(probabilities):
    """Compute Shannon entropy of a probability distribution."""
    probs = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

def is_unknown_image(predictions, threshold=UNKNOWN_THRESHOLD):
    """
    Check whether the model predictions indicate an unknown/non-potato-leaf image.
    Uses two signals:
      1. Max confidence below threshold → low certainty overall
      2. Normalized entropy above threshold → flat/uncertain distribution
    """
    max_prob = float(np.max(predictions))

    # Normalized entropy: 1.0 = uniform (max uncertainty), 0.0 = one class certain
    n_classes = len(predictions)
    max_entropy = np.log(n_classes)
    actual_entropy = compute_entropy(predictions)
    norm_entropy = actual_entropy / max_entropy if max_entropy > 0 else 0.0

    # High entropy + low max confidence -> likely not a potato leaf
    is_low_conf = max_prob < threshold
    is_high_entropy = norm_entropy > 0.85

    is_unknown = is_low_conf or is_high_entropy

    return is_unknown, max_prob, norm_entropy


# ---------- API Routes ----------

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

        # Check for unknown image
        unknown, max_conf, norm_entropy = is_unknown_image(avg_predictions)
        if unknown:
            return {
                'class': UNKNOWN_CLASS,
                'confidence': max_conf,
                'is_unknown': True,
                'entropy': float(norm_entropy),
                'message': 'This does not appear to be a potato leaf. Please upload a clear photo of a potato leaf.',
                'individual': [
                    {
                        "model": MODEL_NAMES[model_ids[idx]],
                        "class": CLASS_NAMES[np.argmax(pred)],
                        "confidence": float(np.max(pred))
                    }
                    for idx, pred in enumerate(all_predictions)
                ]
            }

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
            'probabilities': {
                CLASS_NAMES[i]: float(avg_predictions[i])
                for i in range(len(CLASS_NAMES))
            },
            'individual': individual,
            'is_unknown': False
        }
    else:
        if model_id not in MODELS:
            return {"error": "Model not found"}
        processed = preprocess_for_model(img_batch, model_id)
        model = MODELS[model_id]
        predictions = model.predict(processed)

        unknown, max_conf, norm_entropy = is_unknown_image(predictions[0])
        if unknown:
            return {
                'class': UNKNOWN_CLASS,
                'confidence': max_conf,
                'is_unknown': True,
                'entropy': float(norm_entropy),
                'message': 'This does not appear to be a potato leaf. Please upload a clear photo of a potato leaf.'
            }

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        return {
            'class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                CLASS_NAMES[i]: float(predictions[0][i])
                for i in range(len(CLASS_NAMES))
            },
            'is_unknown': False
        }


@app.post("/gradcam")
async def gradcam(
    file: UploadFile = File(...),
    model_id: str = "ensemble"
):
    """
    Compute Grad-CAM heatmap overlay for the uploaded image.
    Returns base64-encoded heatmap overlay image(s).
    """
    try:
        image = read_file_as_image(await file.read())
    except Exception:
        return {"error": "Invalid image file"}

    try:
        if model_id == "ensemble":
            heatmaps = get_all_heatmaps(image)
            return {
                "model_id": model_id,
                "heatmaps": heatmaps
            }
        else:
            if model_id not in MODELS:
                return {"error": "Model not found"}
            processed = preprocess_for_model(np.expand_dims(image, 0), model_id)
            heatmap_raw = compute_gradcam(MODELS[model_id], processed, model_id)
            overlay = generate_heatmap_overlay(heatmap_raw, image)
            return {
                "model_id": model_id,
                "heatmap": overlay
            }
    except Exception as exc:
        return {"error": f"Grad-CAM computation failed: {str(exc)}"}


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
