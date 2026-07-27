from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf

# Limit TF thread pools to avoid CPU over-subscription with multiple workers.
# With 2 uvicorn workers and 4 CPU cores, each worker should use ~2 threads.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import io
import base64
import asyncio
import logging
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# ---------- Constants ----------

MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB — must match frontend maxFileSize
ALLOWED_MIME_PREFIXES = (b"\xff\xd8\xff", b"\x89PNG")  # JPEG, PNG

# Rate limiting — per-process. With 2 uvicorn workers, aggregate is 2x this value.
# Adjusted so aggregate ~30 req/min.
RATE_LIMIT_PER_WORKER = 15
RATE_LIMIT_WINDOW = 60

# In-memory request tracker: {client_ip: deque([timestamp, ...])}
_request_tracker = defaultdict(lambda: deque(maxlen=RATE_LIMIT_PER_WORKER))

# Cache for last conv layer names (computed once per model)
_conv_layer_cache = {}

_executor = ThreadPoolExecutor(max_workers=4)


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
    Note: rate limiter is per-process. With 2 uvicorn workers the
    effective aggregate limit is 2x RATE_LIMIT_PER_WORKER.
    """
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW

    # Prune old entries for this IP only (fast path)
    timestamps = _request_tracker[client_ip]
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_PER_WORKER:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: models are loaded at module level (before this runs)
    yield
    # Shutdown: clean up model references and TF session
    logger.info("Shutting down — clearing model references")
    MODELS.clear()
    tf.keras.backend.clear_session()
    _executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# ---------- Download Models from Google Drive if needed ----------
_download_thread = None
try:
    try:
        from backend.model_downloader import download_models, verify_models, get_download_status
    except ImportError:
        from model_downloader import download_models, verify_models, get_download_status
    project_base = BASE_DIR.parent
    
    # Check if models exist locally
    models_dir = project_base / "saved_models"
    verification = verify_models(project_base)
    
    if not all(verification.values()):
        logger.info("Some models missing — starting background download from Google Drive...")
        import threading
        def _bg_download():
            try:
                download_results = download_models(project_base)
                logger.info(f"Download results: {download_results}")
            except Exception as e:
                logger.error(f"Background download failed: {e}")
        _download_thread = threading.Thread(target=_bg_download, daemon=True)
        _download_thread.start()
    else:
        logger.info("All models found locally — skipping download")
        from model_downloader import _update_status
        _update_status(
            state="complete",
            overall_progress=100,
            progress=100,
            message="All models ready!",
            models={k: "done" for k in verification},
        )
except ImportError:
    logger.warning("model_downloader not available — using local saved_models")
except Exception as e:
    logger.warning(f"Model download setup failed: {e} — will try local saved_models")

# Find saved_models directory (Docker path vs local dev path)
model_base = BASE_DIR / "../saved_models"
if not model_base.exists():
    model_base = BASE_DIR / "saved_models"
    logger.info(f"Using model path: {model_base}")

IMAGE_SIZE = 256
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
N_CLASSES = 3

MODEL_NAMES = {
  "cnn-baseline": "CNN Baseline",
  "transfer-learning": "Transfer Learning",
  "mobilenetv2": "MobileNetV2",
}


def _build_transfer_learning_model():
    """Rebuild ResNet50V2 using input_tensor for connected Grad-CAM graph."""
    from tensorflow.keras.applications import ResNet50V2 as _ResNet50V2
    import h5py as _h5py

    inp = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer_1")
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inp)
    x = tf.keras.layers.RandomRotation(0.2)(x)

    base = _ResNet50V2(input_tensor=x, include_top=False,
                       weights="imagenet", pooling="avg")
    base.trainable = False

    x = tf.keras.layers.Dropout(0.3, name="dropout")(base.output)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense")(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="dense_1")(x)
    m = tf.keras.models.Model(inputs=inp, outputs=out)

    h5_path = str(model_base / "2" / "model.h5")
    with _h5py.File(h5_path, "r") as f:
        wg = f["model_weights"]
        base_vars = {v.name.replace(":0", ""): v for v in base.variables}
        resnet_grp = wg["resnet50v2"]
        for h5_name in resnet_grp.keys():
            grp = resnet_grp[h5_name]
            if isinstance(grp, _h5py.Group):
                for sub in ("kernel", "bias", "gamma", "beta", "moving_mean", "moving_variance"):
                    if sub in grp:
                        full = f"{h5_name}/{sub}"
                        if full in base_vars:
                            base_vars[full].assign(np.array(grp[sub]))
        for v in m.variables:
            vn = v.name.replace(":0", "")
            if vn == "dense/kernel":
                v.assign(np.array(wg["dense"]["dense"]["kernel"]))
            elif vn == "dense/bias":
                v.assign(np.array(wg["dense"]["dense"]["bias"]))
            elif vn == "dense_1/kernel":
                v.assign(np.array(wg["dense_1"]["dense_1"]["kernel"]))
            elif vn == "dense_1/bias":
                v.assign(np.array(wg["dense_1"]["dense_1"]["bias"]))
    return m


def _build_mobilenetv2_model():
    """Rebuild MobileNetV2 using input_tensor for connected Grad-CAM graph."""
    from tensorflow.keras.applications import MobileNetV2 as _MobileNetV2

    loaded_sm = tf.saved_model.load(str(model_base / "3"))
    saved_vars = {v.name.replace(":0", ""): v.numpy() for v in loaded_sm.variables}

    inp = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inp)
    x = tf.keras.layers.RandomRotation(0.2)(x)

    base = _MobileNetV2(input_tensor=x, include_top=False,
                        weights="imagenet", pooling="avg")
    base.trainable = False

    x = tf.keras.layers.Dropout(0.3, name="dropout")(base.output)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense")(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="dense_1")(x)
    m = tf.keras.models.Model(inputs=inp, outputs=out)
    _ = m(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32), training=False)

    for v in m.variables:
        vn = v.name.replace(":0", "")
        if vn in saved_vars:
            v.assign(saved_vars[vn])
    return m


# --- Load CNN Baseline (SavedModel format, works directly) ---
try:
    cnn_path = model_base / "1"
    if cnn_path.exists():
        MODELS["cnn-baseline"] = tf.keras.models.load_model(str(cnn_path))
        logger.info("Loaded model cnn-baseline")
    else:
        MODEL_LOAD_ERRORS["cnn-baseline"] = f"Path does not exist: {cnn_path}"
except Exception as exc:
    MODEL_LOAD_ERRORS["cnn-baseline"] = str(exc)
    logger.warning(f"Failed to load cnn-baseline: {exc}")

# --- Load Transfer Learning (rebuild from h5 weights) ---
try:
    tl_path = model_base / "2"
    if tl_path.exists():
        MODELS["transfer-learning"] = _build_transfer_learning_model()
        logger.info("Loaded model transfer-learning (rebuilt from h5)")
    else:
        MODEL_LOAD_ERRORS["transfer-learning"] = f"Path does not exist: {tl_path}"
except Exception as exc:
    MODEL_LOAD_ERRORS["transfer-learning"] = str(exc)
    logger.warning(f"Failed to load transfer-learning: {exc}")

# --- Load MobileNetV2 (rebuild from SavedModel variables) ---
try:
    mv_path = model_base / "3"
    if mv_path.exists():
        MODELS["mobilenetv2"] = _build_mobilenetv2_model()
        logger.info("Loaded model mobilenetv2 (rebuilt from SavedModel vars)")
    else:
        MODEL_LOAD_ERRORS["mobilenetv2"] = f"Path does not exist: {mv_path}"
except Exception as exc:
    MODEL_LOAD_ERRORS["mobilenetv2"] = str(exc)
    logger.warning(f"Failed to load mobilenetv2: {exc}")

# Warm up TF and cache Grad-CAM models after all models are loaded
if MODELS:
    _warm_up_models()
    _build_grad_models()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
UNKNOWN_CLASS = "Unknown"
UNKNOWN_THRESHOLD = 0.85    # Max confidence below this -> "Unknown Image"
ENTROPY_THRESHOLD = 0.80     # Norm. entropy above this -> "Unknown Image"
TEMPERATURE_SCALE = 1.5      # Softmax temperature for OOD calibration

# ---------- Grad-CAM Sub-Model Cache ----------
# Built once at startup to avoid rebuilding tf.keras.Model on every request.
_grad_models = {}


def _build_grad_models():
    """Pre-build Grad-CAM sub-models for each loaded model."""
    for mid, model in MODELS.items():
        try:
            last_conv_name, _ = find_last_conv_layer(model, mid)
            if last_conv_name is None:
                logger.warning(f"No Conv2D layer found for {mid} — Grad-CAM disabled")
                continue
            actual_layer = _find_layer_in_model(model, last_conv_name)
            if actual_layer is None:
                logger.warning(f"Layer '{last_conv_name}' not found in {mid} — Grad-CAM disabled")
                continue
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[actual_layer.output, model.output]
            )
            _grad_models[mid] = grad_model
            logger.info(f"Cached Grad-CAM model for {mid}")
        except Exception as exc:
            logger.warning(f"Failed to build Grad-CAM model for {mid}: {exc}")


def _warm_up_models():
    """Run dummy inferences to warm up TF graph execution."""
    dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    for mid, model in MODELS.items():
        try:
            processed = preprocess_for_model(dummy, mid)
            _ = model(processed, training=False)
            logger.info(f"Warm-up passed for {mid}")
        except Exception as exc:
            logger.warning(f"Warm-up failed for {mid}: {exc}")

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(io.BytesIO(data)).convert("RGB").resize((256, 256), Image.BILINEAR)
    return np.array(image)

def preprocess_for_model(image: np.ndarray, model_id: str) -> np.ndarray:
    if model_id == "mobilenetv2":
        return preprocess_input(image)
    if model_id == "transfer-learning":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input as rl_preprocess
        return rl_preprocess(image)
    # CNN baseline has an internal Rescaling(1./255) layer
    return image


# ---------- Grad-CAM Helpers ----------

CONV_LAYER_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.SeparableConv2D,
)


def find_last_conv_layer(model, model_id=None):
    """
    Find the last convolutional layer in a Keras model, recursing into
    sub-models.  Returns (layer_name, containing_layer) where
    containing_layer is the direct parent layer that owns the conv layer.
    Results are cached by model_id.
    """
    if model_id and model_id in _conv_layer_cache:
        return _conv_layer_cache[model_id]

    best = [None, None]  # [layer_name, containing_layer]
    best_depth = [-1]

    def _search(layer, depth=0, containing=None):
        if isinstance(layer, CONV_LAYER_TYPES):
            if depth >= best_depth[0]:
                best[0] = layer.name
                best[1] = containing or layer
                best_depth[0] = depth
        elif hasattr(layer, 'layers'):
            for sub in layer.layers:
                _search(sub, depth + 1, containing=layer)

    for layer in model.layers:
        _search(layer, 0, containing=None)

    result = (best[0], best[1])
    if model_id:
        _conv_layer_cache[model_id] = result
    return result


def _find_layer_in_model(model, layer_name):
    """Recursively find a layer by name within the model's graph."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if hasattr(layer, 'layers'):
            result = _find_layer_in_model(layer, layer_name)
            if result is not None:
                return result
    return None


def compute_gradcam(model, img_array, model_id):
    img_tensor = tf.cast(img_array, tf.float32)

    if model_id in _grad_models:
        grad_model = _grad_models[model_id]
    else:
        last_conv_name, _ = find_last_conv_layer(model, model_id)
        if last_conv_name is None:
            raise ValueError(f"Could not find a Conv2D layer in model '{model_id}'")
        actual_layer = _find_layer_in_model(model, last_conv_name)
        if actual_layer is None:
            raise ValueError(f"Layer '{last_conv_name}' not found in model graph")
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[actual_layer.output, model.output]
        )

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def array_to_base64(arr: np.ndarray) -> str:
    """Convert a numpy image array (H, W, 3) to a base64 JPEG data URI."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def generate_heatmap_overlay(heatmap, original_image):
    """
    Generate two base64-encoded JPEGs:
      - 'overlay':  heatmap blended over the original image
      - 'raw':      heatmap alone (jet colormap, no background)
    """
    heatmap_resized = np.array(
        Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8)).resize(
            (256, 256), Image.BILINEAR
        )
    ).astype(np.float32) / 255.0
    heatmap_norm = np.clip(heatmap_resized, 0, 1)
    heatmap_gamma = np.power(heatmap_norm, 0.7)

    colored = cm.jet(heatmap_gamma)[:, :, :3]
    colored_255 = np.uint8(255 * colored)

    # --- Raw heatmap (colormap on white background) ---
    white_bg = np.full((256, 256, 3), 240, dtype=np.uint8)
    blended_raw = np.uint8(
        white_bg * (1 - heatmap_norm[:, :, np.newaxis])
        + colored_255 * heatmap_norm[:, :, np.newaxis]
    )
    raw_b64 = array_to_base64(blended_raw)

    # --- Overlay on original image ---
    original_float = original_image.astype(np.float32)

    blended = np.uint8(original_float * 0.60 + colored_255 * 0.40)
    mask = heatmap_norm > 0.50
    mask_3d = np.stack([mask] * 3, axis=-1)
    highlighted = np.uint8(original_float * 0.20 + colored_255 * 0.80)
    final = np.where(mask_3d, highlighted, blended)
    overlay_b64 = array_to_base64(final)

    return {
        'overlay': overlay_b64,
        'raw': raw_b64,
    }


def _compute_single_heatmap(mid, image):
    """Compute Grad-CAM heatmap for a single model (used by thread pool)."""
    processed = preprocess_for_model(np.expand_dims(image, 0), mid)
    heatmap_raw = compute_gradcam(MODELS[mid], processed, mid)
    result = generate_heatmap_overlay(heatmap_raw, image)
    return mid, {
        'overlay': result['overlay'],
        'raw': result['raw'],
    }


def get_all_heatmaps(image):
    """Compute Grad-CAM heatmaps for all models in parallel."""
    heatmaps = {}
    results = list(_executor.map(
        lambda mid: _compute_single_heatmap(mid, image), MODELS.keys()
    ))
    for mid, result in results:
        heatmaps[MODEL_NAMES[mid]] = result
    return heatmaps


# ---------- Unknown Image Detection ----------

def compute_entropy(probabilities):
    """Compute Shannon entropy of a probability distribution."""
    probs = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def temperature_scale(probabilities, temperature=TEMPERATURE_SCALE):
    adjusted = np.power(np.clip(probabilities, 1e-10, 1.0), 1.0 / temperature)
    return adjusted / (np.sum(adjusted) + 1e-10)


def is_likely_plant_leaf(image: np.ndarray) -> bool:
    """
    Multi-layer pre-check before model inference.

    Layer 1 — Green Color: Potato leaves are predominantly green.
      We check that green is the dominant channel and makes up a
      significant portion of total intensity.
    Layer 2 — Leaf Texture: We measure edge density and local gradient
      patterns to distinguish organic leaf surfaces (veins, serrated
      edges) from uniform synthetic surfaces.
    Layer 3 (downstream in is_unknown_image): Temperature-scaled confidence
      and entropy threshold to catch overconfident OOD predictions.

    Returns True only if BOTH green color AND leaf texture are present.
    """
    img_f = image.astype(np.float32)

    # --- Layer 1: Green Color Check ---
    total = np.sum(img_f, axis=2) + 1e-10
    green_ratio = img_f[:, :, 1] / total
    avg_green = float(np.mean(green_ratio))

    green_dominant = float(np.mean(
        (img_f[:, :, 1] > img_f[:, :, 0]) & (img_f[:, :, 1] > img_f[:, :, 2])
    ))

    green_excess = float(np.mean(img_f[:, :, 1] - np.maximum(img_f[:, :, 0], img_f[:, :, 2])))

    passes_color = avg_green > 0.28 and green_dominant > 0.25 and green_excess > 5.0

    # --- Layer 2: Leaf Texture Check ---
    gray = np.mean(img_f, axis=2)
    gy, gx = np.gradient(gray)
    edge_mag = np.sqrt(gx**2 + gy**2)

    edge_density = float(np.mean(edge_mag > 10.0))
    texture_complexity = float(np.std(edge_mag))
    intensity_std = float(np.std(gray))

    passes_texture = (
        edge_density > 0.05
        and texture_complexity > 12.0
        and intensity_std > 25.0
    )

    return passes_color and passes_texture


def is_unknown_image(predictions, threshold=UNKNOWN_THRESHOLD, individual_preds=None):
    calibrated = temperature_scale(predictions)
    max_prob_cal = float(np.max(calibrated))

    n_classes = len(predictions)
    max_entropy = np.log(n_classes)
    actual_entropy = compute_entropy(calibrated)
    norm_entropy = actual_entropy / max_entropy if max_entropy > 0 else 0.0

    is_low_conf = max_prob_cal < threshold
    is_high_entropy = norm_entropy > ENTROPY_THRESHOLD

    # Model agreement: if 2/3 models agree AND confidence is reasonable, override unknown.
    # But don't override if raw confidence is very low (< 0.50) — that's truly uncertain.
    if individual_preds is not None and len(individual_preds) >= 2:
        majority_class = np.argmax(np.bincount([np.argmax(p) for p in individual_preds]))
        agreement = sum(1 for p in individual_preds if np.argmax(p) == majority_class)
        raw_max = float(np.max(predictions))
        if agreement >= 2 and raw_max > 0.50:
            is_low_conf = False
            is_high_entropy = False

    is_unknown = is_low_conf or is_high_entropy

    original_max_prob = float(np.max(predictions))
    return is_unknown, original_max_prob, norm_entropy


# ---------- Ensemble Prediction Helper ----------

def _predict_single_model(mid, img_batch):
    """Run prediction on a single model (runs in thread pool)."""
    processed = preprocess_for_model(img_batch, mid)
    return MODELS[mid].predict(processed, verbose=0)[0]


def _run_ensemble(img_batch):
    """Run all models in parallel using thread pool and return averaged predictions."""
    model_ids = list(MODELS.keys())
    all_predictions = list(_executor.map(
        lambda mid: _predict_single_model(mid, img_batch), model_ids
    ))
    avg_predictions = np.mean(all_predictions, axis=0)
    return model_ids, all_predictions, avg_predictions


# ---------- API Routes ----------

@app.get("/setup/status")
def setup_status():
    """Return model download progress for the onboarding screen."""
    try:
        return get_download_status()
    except NameError:
        return {"state": "idle", "overall_progress": 0, "models": {}, "message": "Waiting..."}
    except Exception as e:
        return {"state": "error", "error": str(e), "overall_progress": 0}


@app.get("/ping")
def ping():
    return "Hello, I am alive"


@app.get("/models")
def get_models():
    return {"models": list(MODELS.keys()), "modelNames": MODEL_NAMES}


@app.get("/health")
def health():
    n_models = len(MODELS)
    if n_models >= 2:
        return {"status": "success", "models_loaded": list(MODELS.keys()), "total_expected": 3, "total_loaded": n_models}
    return JSONResponse(
        status_code=503,
        content={
            "status": "failure",
            "models_loaded": list(MODELS.keys()),
            "total_expected": 3,
            "total_loaded": n_models,
            "errors": MODEL_LOAD_ERRORS,
        },
    )


def _build_unknown_response(message, max_conf, norm_entropy, model_ids=None, all_predictions=None):
    """Build a consistent unknown response with all expected keys."""
    resp = {
        'class': UNKNOWN_CLASS,
        'confidence': max_conf,
        'is_unknown': True,
        'entropy': float(norm_entropy),
        'probabilities': {},
        'message': message,
    }
    if model_ids and all_predictions:
        resp['individual'] = [
            {
                "model": MODEL_NAMES[model_ids[idx]],
                "class": CLASS_NAMES[np.argmax(pred)],
                "confidence": float(np.max(pred)),
            }
            for idx, pred in enumerate(all_predictions)
        ]
    return resp


def _build_success_response(predicted_class, confidence, probabilities, model_ids=None, all_predictions=None):
    """Build a consistent success response."""
    resp = {
        'class': predicted_class,
        'confidence': confidence,
        'probabilities': {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        },
        'is_unknown': False,
    }
    if model_ids and all_predictions:
        resp['individual'] = [
            {
                "model": MODEL_NAMES[model_ids[idx]],
                "class": CLASS_NAMES[np.argmax(pred)],
                "confidence": float(np.max(pred)),
            }
            for idx, pred in enumerate(all_predictions)
        ]
    return resp


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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Pre-check: color-based OOD detection
    if not is_likely_plant_leaf(image):
        return _build_unknown_response(
            'This does not appear to be a potato leaf. Please upload a clear photo of a potato leaf.',
            0.0, 1.0,
        )

    img_batch = np.expand_dims(image, 0)

    if model_id == "ensemble":
        model_ids, all_predictions, avg_predictions = _run_ensemble(img_batch)
        unknown, max_conf, norm_entropy = is_unknown_image(avg_predictions, individual_preds=all_predictions)
        if unknown:
            return _build_unknown_response(
                'This does not appear to be a potato leaf. Please upload a clear photo of a potato leaf.',
                max_conf, float(norm_entropy),
                model_ids, all_predictions,
            )
        predicted_class = CLASS_NAMES[np.argmax(avg_predictions)]
        return _build_success_response(
            predicted_class, float(np.max(avg_predictions)), avg_predictions,
            model_ids, all_predictions,
        )
    else:
        if model_id not in MODELS:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        result = await asyncio.get_event_loop().run_in_executor(
            _executor, lambda: _predict_single_model(model_id, img_batch)
        )
        predictions = result[np.newaxis, :]

        unknown, max_conf, norm_entropy = is_unknown_image(predictions[0])
        if unknown:
            return _build_unknown_response(
                'This does not appear to be a potato leaf. Please upload a clear photo of a potato leaf.',
                max_conf, float(norm_entropy),
            )
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        return _build_success_response(
            predicted_class, float(np.max(predictions[0])), predictions[0],
        )


@app.post("/gradcam")
async def gradcam(
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
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        if model_id == "ensemble":
            heatmaps = get_all_heatmaps(image)
            return {"model_id": model_id, "heatmaps": heatmaps}
        else:
            if model_id not in MODELS:
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
            processed = preprocess_for_model(np.expand_dims(image, 0), model_id)
            heatmap_array = await asyncio.get_event_loop().run_in_executor(
                _executor, lambda: compute_gradcam(MODELS[model_id], processed, model_id)
            )
            result = generate_heatmap_overlay(heatmap_array, image)
            return {
                "model_id": model_id,
                "model_name": MODEL_NAMES[model_id],
                "overlay": result['overlay'],
                "raw": result['raw'],
            }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Grad-CAM computation failed")
        return {"error": "Grad-CAM computation failed — see server logs"}


# ---------- Frontend Static Files ----------
# Serve the production frontend build. API routes above take priority;
# this catch-all only handles paths that don't match any API route.
frontend_build = BASE_DIR.parent / "frontend" / "build"
if frontend_build.exists() and (frontend_build / "index.html").exists():
    from fastapi.responses import FileResponse

    # Serve static assets (CSS, JS, images)
    static_dir = frontend_build / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for SPA client-side routing."""
        # Try to serve the exact file first (favicon.ico, manifest.json, etc.)
        file_path = frontend_build / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        # Fall back to index.html for SPA routing
        index = frontend_build / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404)

    logger.info(f"Serving frontend from {frontend_build}")
else:
    logger.info("No frontend build found at %s — API-only mode", frontend_build)


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
