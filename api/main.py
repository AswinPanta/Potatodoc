from fastapi import FastAPI, File, UploadFile, Request, HTTPException
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
import time
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------- Constants ----------

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_PREFIXES = (b"\xff\xd8\xff", b"\x89PNG")  # JPEG, PNG

# ---------- Rate Limiting ----------

RATE_LIMIT_WINDOW = 60       # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # max requests per window per IP

# In-memory request tracker: {client_ip: deque([timestamp, ...])}
_request_tracker = defaultdict(lambda: deque(maxlen=RATE_LIMIT_MAX_REQUESTS))


def _prune_tracker():
    """Remove expired entries from all tracked IPs."""
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW
    stale_ips = []
    for ip, timestamps in list(_request_tracker.items()):
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        if not timestamps:
            stale_ips.append(ip)
    for ip in stale_ips:
        del _request_tracker[ip]


def check_rate_limit(request: Request):
    """
    Check whether the client has exceeded the rate limit.
    Raises HTTPException (429) if over the limit.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW

    # Prune old entries for this IP only (fast path)
    timestamps = _request_tracker[client_ip]
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        retry_after = int(timestamps[0] + RATE_LIMIT_WINDOW - now)
        logger.warning(f"Rate limit hit for {client_ip} — retry after {retry_after}s")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    timestamps.append(now)

    # Occasionally prune the full tracker (every 100 requests)
    if len(_request_tracker) > 100 and len(timestamps) == 1:
        _prune_tracker()

# ---------- File Validation ----------

def validate_image(data: bytes) -> bytes:
    """Check size + magic bytes before processing. Returns the data if valid."""
    if len(data) > MAX_UPLOAD_SIZE:
        raise ValueError(f"File too large ({len(data)} bytes, max {MAX_UPLOAD_SIZE})")
    is_valid_mime = any(data.startswith(sig) for sig in ALLOWED_MIME_PREFIXES)
    is_valid_mime = is_valid_mime or (data[:4] == b"RIFF" and b"WEBP" in data[:12])
    if not is_valid_mime:
        raise ValueError("Unsupported format — use JPEG, PNG, or WebP")
    return data


# ---------- App Setup ----------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://AswinPanta-potatodoc.hf.space",  # HF Space (production)
        "http://localhost:3000",                    # React dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# ---------- Request Logging Middleware ----------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = int((time.time() - start) * 1000)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response


# ---------- Security Headers Middleware ----------

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return response


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
UNKNOWN_THRESHOLD = 0.85    # Max confidence below this -> "Unknown Image"
ENTROPY_THRESHOLD = 0.80     # Norm. entropy above this -> "Unknown Image"
TEMPERATURE_SCALE = 1.5      # Softmax temperature for OOD calibration

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

CONV_LAYER_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.SeparableConv2D,
)


def find_last_conv_layer(model):
    """
    Find the last convolutional layer name in a Keras model,
    recursing into sub-models.  Supports Conv2D, DepthwiseConv2D
    (used by MobileNetV2), and SeparableConv2D.
    """
    last_conv = [None]
    last_depth = [-1]  # track depth to get truly last layer

    def _search(layer, depth=0):
        if isinstance(layer, CONV_LAYER_TYPES):
            if depth >= last_depth[0]:
                last_conv[0] = layer.name
                last_depth[0] = depth
        elif hasattr(layer, 'layers'):
            for sub in layer.layers:
                _search(sub, depth + 1)

    for layer in model.layers:
        _search(layer, 0)

    return last_conv[0]


def compute_gradcam(model, img_array, model_id):
    """
    Compute Grad-CAM heatmap for a given model and input image.
    Returns a dict with:
      - 'heatmap': raw heatmap array of shape (H, W) in range [0, 1]
      - 'overlay': base64 PNG overlay of the heatmap on the original image
      - 'heatmap_raw': base64 PNG of just the heatmap (jet colormap)
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


def array_to_base64(arr: np.ndarray) -> str:
    """Convert a numpy image array (H, W, 3) to a base64 data URI."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def generate_heatmap_overlay(heatmap, original_image):
    """
    Generate two base64-encoded PNGs:
      - 'overlay':  heatmap blended over the original image
      - 'raw':      heatmap alone (jet colormap, no background)
    Also returns the raw heatmap array resized to 256×256.
    """
    # Resize heatmap to match original dimensions
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (256, 256)
    ).numpy().squeeze()
    heatmap_norm = np.clip(heatmap_resized, 0, 1)

    # Apply gamma to emphasise high-attention areas
    heatmap_gamma = np.power(heatmap_norm, 0.7)

    # Jet colormap
    colored = cm.jet(heatmap_gamma)[:, :, :3]  # (H, W, 3) in [0, 1]
    colored_255 = np.uint8(255 * colored)

    # --- Raw heatmap (just the colormap, white background) ---
    # Place the heatmap on a solid white background so it's visible alone
    white_bg = np.full((256, 256, 3), 240, dtype=np.uint8)
    alpha_mask = np.clip(heatmap_norm * 255, 0, 255).astype(np.uint8)
    # Use the heatmap value as alpha for blending with white background
    blended_raw = np.uint8(
        white_bg * (1 - heatmap_norm[:, :, np.newaxis])
        + colored_255 * heatmap_norm[:, :, np.newaxis]
    )
    raw_b64 = array_to_base64(blended_raw)

    # --- Overlay on original image ---
    # Multi-layer blending for a vibrant, interpretable overlay
    # Low-attention areas show the original image clearly
    # High-attention areas show strong red/orange overlay
    original_float = original_image.astype(np.float32)

    # Blend: overlay at 40% opacity everywhere (gives context)
    blended = np.uint8(
        original_float * 0.60 + colored_255 * 0.40
    )

    # Strong highlight for high-attention regions (> 50%)
    mask = heatmap_norm > 0.50
    mask_3d = np.stack([mask] * 3, axis=-1)

    highlighted = np.uint8(
        original_float * 0.20 + colored_255 * 0.80
    )

    final = np.where(mask_3d, highlighted, blended)
    overlay_b64 = array_to_base64(final)

    return {
        'overlay': overlay_b64,
        'raw': raw_b64,
    }


def get_all_heatmaps(image):
    """Compute Grad-CAM heatmaps for all models."""
    heatmaps = {}
    for mid in MODELS.keys():
        try:
            processed = preprocess_for_model(np.expand_dims(image, 0), mid)
            heatmap_raw = compute_gradcam(MODELS[mid], processed, mid)
            result = generate_heatmap_overlay(heatmap_raw, image)
            heatmaps[MODEL_NAMES[mid]] = {
                'overlay': result['overlay'],
                'raw': result['raw'],
            }
        except Exception as exc:
            logger.warning(f"Grad-CAM failed for {mid}: {exc}")
            heatmaps[MODEL_NAMES[mid]] = None
    return heatmaps


# ---------- Unknown Image Detection ----------

def compute_entropy(probabilities):
    """Compute Shannon entropy of a probability distribution."""
    probs = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def temperature_scale(probabilities, temperature=TEMPERATURE_SCALE):
    """
    Apply temperature scaling to soften a probability distribution.

    Given softmax outputs p_i = exp(l_i)/sum(exp(l_j)), scaling by T divides
    the logits: p'_i = exp(l_i/T)/sum(exp(l_j/T)).  Since we only have the
    post-softmax probabilities (not the raw logits), we use the identity:
      p'_i = p_i^(1/T) / sum(p_j^(1/T))

    T > 1 softens the distribution (more uniform), making OOD overconfidence
    easier to detect.  T = 1 is a no-op.
    """
    adjusted = np.power(np.clip(probabilities, 1e-10, 1.0), 1.0 / temperature)
    return adjusted / (np.sum(adjusted) + 1e-10)


def is_likely_plant_leaf(image: np.ndarray) -> bool:
    """
    Two-layer pre-check before model inference.
    
    Layer 1 — Color: Potato leaves are predominantly green.
      Rejects non-green images (animals, buildings, people) immediately.
    
    Layer 2 — Texture: Organic leaves have irregular, detailed textures.
      Rejects uniform/synthetic surfaces (walls, plastic) even if green.
    
    Layer 3 (downstream in is_unknown_image): Temperature-scaled confidence
      and entropy threshold to catch overconfident OOD predictions.
    """
    # ---- Layer 1: Color (green ratio) ----
    total = np.sum(image, axis=2, keepdims=True).astype(np.float32) + 1e-10
    green_ratio = image[:, :, 1].astype(np.float32) / total[:, :, 0]
    avg_green = float(np.mean(green_ratio))
    green_dominant = float(np.mean(
        (image[:, :, 1].astype(np.float32) > image[:, :, 0].astype(np.float32)) &
        (image[:, :, 1].astype(np.float32) > image[:, :, 2].astype(np.float32))
    ))
    passes_color = avg_green > 0.30 and green_dominant > 0.20
    if not passes_color:
        return False

    # ---- Layer 2: Texture (organic vs. synthetic) ----
    # Convert to grayscale and compute intensity variation.
    # Potato leaves have high intensity variation (veins, spots, edges).
    # Synthetic green objects (walls, plastic) have low variation.
    gray = np.mean(image.astype(np.float32), axis=2)
    # Gradient-based edge magnitude
    gy, gx = np.gradient(gray)
    edge_mag = np.sqrt(gx**2 + gy**2)
    
    # Texture features
    intensity_std = float(np.std(gray))          # overall tonal variation
    texture_complexity = float(np.std(edge_mag)) # edge detail variation
    
    # Thresholds (empirically tuned):
    # - intensity_std < 1.5 → very uniform surface (painted wall, plastic)
    # - texture_complexity < 0.8 → smooth, little detail
    passes_texture = intensity_std > 1.5 and texture_complexity > 0.8

    return passes_texture


def is_unknown_image(predictions, threshold=UNKNOWN_THRESHOLD):
    """
    Check whether the model predictions indicate an unknown/non-potato-leaf image.

    Uses two signals internally (on temperature-scaled probabilities):
      1. Max confidence below threshold → low certainty overall
      2. Normalized entropy above threshold → flat/uncertain distribution

    Returns the *original* max confidence for display, but makes the
    determination on calibrated probabilities to catch overconfident OOD images.
    """
    # Apply temperature scaling to calibrate probabilities
    # This softens overconfident predictions, making OOD detection more reliable
    calibrated = temperature_scale(predictions)

    max_prob_cal = float(np.max(calibrated))

    # Normalized entropy: 1.0 = uniform (max uncertainty), 0.0 = one class certain
    n_classes = len(predictions)
    max_entropy = np.log(n_classes)
    actual_entropy = compute_entropy(calibrated)
    norm_entropy = actual_entropy / max_entropy if max_entropy > 0 else 0.0

    is_low_conf = max_prob_cal < threshold
    is_high_entropy = norm_entropy > ENTROPY_THRESHOLD

    is_unknown = is_low_conf or is_high_entropy

    # Return original (uncalibrated) max probability for display purposes
    original_max_prob = float(np.max(predictions))
    return is_unknown, original_max_prob, norm_entropy


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
    request: Request,
    file: UploadFile = File(...),
    model_id: str = "ensemble"
):
    check_rate_limit(request)
    try:
        raw = await file.read()
        validate_image(raw)
        image = read_file_as_image(raw)
    except ValueError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": "Invalid image file"}

    # ---------- Pre-check: color-based OOD detection ----------
    # Potato leaves are green — non-green images (animals, buildings, etc.)
    # can be rejected immediately without running the model.
    if not is_likely_plant_leaf(image):
        return {
            'class': UNKNOWN_CLASS,
            'confidence': 0.0,
            'is_unknown': True,
            'entropy': 1.0,
            'message': 'This does not appear to be a potato leaf. Please upload a clear photo of a potato leaf.',
        }

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
    request: Request,
    file: UploadFile = File(...),
    model_id: str = "ensemble"
):
    """
    Compute Grad-CAM heatmap overlay for the uploaded image.
    Returns base64-encoded heatmap images: 'overlay' (on the original image)
    and 'raw' (heatmap alone).  For ensemble mode, returns per-model results.
    """
    check_rate_limit(request)
    try:
        raw = await file.read()
        validate_image(raw)
        image = read_file_as_image(raw)
    except ValueError as e:
        return {"error": str(e)}
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
            heatmap_array = compute_gradcam(MODELS[model_id], processed, model_id)
            result = generate_heatmap_overlay(heatmap_array, image)
            return {
                "model_id": model_id,
                "model_name": MODEL_NAMES[model_id],
                "overlay": result['overlay'],
                "raw": result['raw'],
            }
    except Exception:
        logger.exception("Grad-CAM computation failed")
        return {"error": "Grad-CAM computation failed — see server logs"}


# ---------- Graceful Shutdown ----------

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down — clearing model references")
    MODELS.clear()
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
