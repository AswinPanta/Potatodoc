#!/usr/bin/env python3
"""
Comprehensive model testing script for the Potato Disease Classification backend.
Tests: model loading, predictions, Grad-CAM, and unknown image detection.
"""
import sys
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_BASE = PROJECT_ROOT / "saved_models"
TEST_IMAGES = PROJECT_ROOT / "test_images_from_internet"
PLANT_VILLAGE = PROJECT_ROOT / "PlantVillage"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ============================================================
# Model Loading
# ============================================================

IMAGE_SIZE = 256
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
N_CLASSES = 3


def _build_cnn_baseline():
    return tf.keras.models.load_model(str(MODEL_BASE / "1"))


def _build_transfer_learning():
    import h5py
    from tensorflow.keras.applications import ResNet50V2
    inp = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer_1")
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inp)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    base = ResNet50V2(input_tensor=x, include_top=False, weights="imagenet", pooling="avg")
    base.trainable = False
    x = tf.keras.layers.Dropout(0.3, name="dropout")(base.output)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense")(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="dense_1")(x)
    m = tf.keras.models.Model(inputs=inp, outputs=out)

    with h5py.File(str(MODEL_BASE / "2" / "model.h5"), "r") as f:
        wg = f["model_weights"]
        base_vars = {v.name.replace(":0", ""): v for v in base.variables}
        for h5_name in wg["resnet50v2"].keys():
            grp = wg["resnet50v2"][h5_name]
            if isinstance(grp, h5py.Group):
                for sub in ("kernel", "bias", "gamma", "beta", "moving_mean", "moving_variance"):
                    if sub in grp:
                        full = f"{h5_name}/{sub}"
                        if full in base_vars:
                            base_vars[full].assign(np.array(grp[sub]))
        for v in m.variables:
            vn = v.name.replace(":0", "")
            if vn == "dense/kernel": v.assign(np.array(wg["dense"]["dense"]["kernel"]))
            elif vn == "dense/bias": v.assign(np.array(wg["dense"]["dense"]["bias"]))
            elif vn == "dense_1/kernel": v.assign(np.array(wg["dense_1"]["dense_1"]["kernel"]))
            elif vn == "dense_1/bias": v.assign(np.array(wg["dense_1"]["dense_1"]["bias"]))
    return m


def _build_mobilenetv2():
    from tensorflow.keras.applications import MobileNetV2
    loaded_sm = tf.saved_model.load(str(MODEL_BASE / "3"))
    saved_vars = {v.name.replace(":0", ""): v.numpy() for v in loaded_sm.variables}

    inp = tf.keras.Input(shape=INPUT_SHAPE, name="input_layer")
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inp)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    base = MobileNetV2(input_tensor=x, include_top=False, weights="imagenet", pooling="avg")
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


def load_models():
    builders = {
        "cnn-baseline": _build_cnn_baseline,
        "transfer-learning": _build_transfer_learning,
        "mobilenetv2": _build_mobilenetv2,
    }
    models = {}
    for mid, builder in builders.items():
        print(f"\n--- Loading {mid} ---")
        t0 = time.time()
        try:
            m = builder()
            elapsed = time.time() - t0
            print(f"  OK ({elapsed:.1f}s) — input: {m.input_shape}, output: {m.output_shape}, layers: {len(m.layers)}")
            models[mid] = m
        except Exception as e:
            print(f"  FAILED: {e}")
    return models


# ============================================================
# Prediction helpers
# ============================================================

def preprocess_for_model(image, model_id):
    if model_id == "mobilenetv2":
        return mobilenet_preprocess(image)
    if model_id == "transfer-learning":
        from tensorflow.keras.applications.resnet_v2 import preprocess_input as rl_preprocess
        return rl_preprocess(image)
    return image


def read_and_preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((256, 256))
    return np.array(img)


def predict_single(model, image, model_id):
    processed = preprocess_for_model(np.expand_dims(image, 0), model_id)
    preds = model.predict(processed, verbose=0)[0]
    return preds


def ensemble_predict(models, image):
    all_preds = {}
    for mid, model in models.items():
        all_preds[mid] = predict_single(model, image, mid)
    avg = np.mean(list(all_preds.values()), axis=0)
    return all_preds, avg


# ============================================================
# Grad-CAM
# ============================================================

CONV_LAYER_TYPES = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.DepthwiseConv2D,
    tf.keras.layers.SeparableConv2D,
)

_conv_layer_cache = {}


def find_last_conv_layer(model, model_id=None):
    if model_id and model_id in _conv_layer_cache:
        return _conv_layer_cache[model_id]

    best = [None, None]
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
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if hasattr(layer, 'layers'):
            result = _find_layer_in_model(layer, layer_name)
            if result is not None:
                return result
    return None


def compute_gradcam(model, img_array, model_id):
    last_conv_name, containing = find_last_conv_layer(model, model_id)
    if last_conv_name is None:
        raise ValueError(f"No Conv2D layer found for model '{model_id}'")

    actual_layer = _find_layer_in_model(model, last_conv_name)
    if actual_layer is None:
        raise ValueError(f"Layer '{last_conv_name}' not found in model graph")

    img_tensor = tf.cast(img_array, tf.float32)

    try:
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
    except (tf.errors.InvalidArgumentError, ValueError):
        pass

    sub_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 10:
            sub_model = layer
            break
    if sub_model is None:
        raise ValueError("Cannot find sub-model for Grad-CAM fallback")

    fresh_input = tf.keras.Input(shape=img_tensor.shape[1:])
    x = fresh_input
    for layer in sub_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x, training=False)

    sub_grad_model = tf.keras.models.Model(
        inputs=fresh_input,
        outputs=[sub_model.get_layer(last_conv_name).output, sub_model.output]
    )

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        aug_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Sequential):
                aug_layer = layer
                break
        if aug_layer is not None:
            augmented = aug_layer(img_tensor, training=False)
        else:
            augmented = img_tensor

        conv_outputs, sub_out = sub_grad_model(augmented)
        predictions = model(img_tensor, training=False)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    return heatmap.numpy()


def test_gradcam(model, image, model_id):
    processed = preprocess_for_model(np.expand_dims(image, 0), model_id)
    t0 = time.time()
    try:
        heatmap = compute_gradcam(model, processed, model_id)
        elapsed = time.time() - t0
        conv_layer, containing = find_last_conv_layer(model, model_id)
        containing_name = "self" if containing is model else containing.name if hasattr(containing, 'name') else "?"
        print(f"  Grad-CAM OK ({elapsed:.1f}s) — conv layer: '{conv_layer}' (in {containing_name}), "
              f"heatmap shape: {heatmap.shape}, min={heatmap.min():.4f}, max={heatmap.max():.4f}")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Grad-CAM FAILED ({elapsed:.1f}s): {e}")
        return False


# ============================================================
# Unknown Image Detection
# ============================================================

UNKNOWN_THRESHOLD = 0.85
ENTROPY_THRESHOLD = 0.80
TEMPERATURE_SCALE = 1.5

def compute_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

def temperature_scale(probs, temperature=TEMPERATURE_SCALE):
    adjusted = np.power(np.clip(probs, 1e-10, 1.0), 1.0 / temperature)
    return adjusted / (np.sum(adjusted) + 1e-10)

def is_likely_plant_leaf(image):
    gray = np.mean(image.astype(np.float32), axis=2)
    gy, gx = np.gradient(gray)
    edge_mag = np.sqrt(gx**2 + gy**2)
    intensity_std = float(np.std(gray))
    texture_complexity = float(np.std(edge_mag))
    passes_texture = intensity_std > 1.5 and texture_complexity > 0.8

    total = np.sum(image, axis=2, keepdims=True).astype(np.float32) + 1e-10
    green_ratio = image[:, :, 1].astype(np.float32) / total[:, :, 0]
    avg_green = float(np.mean(green_ratio))
    green_dominant = float(np.mean(
        (image[:, :, 1].astype(np.float32) > image[:, :, 0].astype(np.float32)) &
        (image[:, :, 1].astype(np.float32) > image[:, :, 2].astype(np.float32))
    ))
    passes_color = avg_green > 0.20 and green_dominant > 0.10
    return passes_color or passes_texture

def is_unknown_image(predictions, individual_preds=None):
    calibrated = temperature_scale(predictions)
    max_prob_cal = float(np.max(calibrated))
    n_classes = len(predictions)
    max_entropy = np.log(n_classes)
    actual_entropy = compute_entropy(calibrated)
    norm_entropy = actual_entropy / max_entropy if max_entropy > 0 else 0.0
    is_low_conf = max_prob_cal < UNKNOWN_THRESHOLD
    is_high_entropy = norm_entropy > ENTROPY_THRESHOLD
    if individual_preds is not None and len(individual_preds) >= 2:
        majority_class = np.argmax(np.bincount([np.argmax(p) for p in individual_preds]))
        agreement = sum(1 for p in individual_preds if np.argmax(p) == majority_class)
        if agreement >= 2:
            is_low_conf = False
            is_high_entropy = False
    return is_low_conf or is_high_entropy, float(np.max(predictions)), norm_entropy


# ============================================================
# Main Test Runner
# ============================================================

def main():
    print("=" * 70)
    print("POTATO DISEASE CLASSIFICATION — COMPREHENSIVE MODEL TEST")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Project root: {PROJECT_ROOT}")

    # --- 1. Load models ---
    print("\n" + "=" * 70)
    print("SECTION 1: MODEL LOADING")
    print("=" * 70)
    models = load_models()
    if not models:
        print("\nFATAL: No models loaded. Aborting.")
        sys.exit(1)
    print(f"\nLoaded {len(models)}/3 models: {list(models.keys())}")

    # --- 2. Individual model predictions ---
    print("\n" + "=" * 70)
    print("SECTION 2: MODEL PREDICTIONS (individual images)")
    print("=" * 70)

    test_files = list(TEST_IMAGES.glob("*.jpg"))
    if not test_files:
        print("No test images found!")
        sys.exit(1)

    for tf_path in test_files:
        print(f"\n--- Image: {tf_path.name} ---")
        image = read_and_preprocess(tf_path)
        print(f"  Image shape: {image.shape}")

        for mid, model in models.items():
            t0 = time.time()
            preds = predict_single(model, image, mid)
            elapsed = time.time() - t0
            pred_class = CLASS_NAMES[np.argmax(preds)]
            confidence = float(np.max(preds))
            probs_str = ", ".join(f"{CLASS_NAMES[i]}={preds[i]:.4f}" for i in range(3))
            print(f"  [{mid}] {pred_class} ({confidence:.4f}) [{elapsed:.2f}s] — {probs_str}")

    # --- 3. Ensemble predictions ---
    print("\n" + "=" * 70)
    print("SECTION 3: ENSEMBLE PREDICTIONS")
    print("=" * 70)

    for tf_path in test_files:
        print(f"\n--- Image: {tf_path.name} ---")
        image = read_and_preprocess(tf_path)
        t0 = time.time()
        all_preds, avg = ensemble_predict(models, image)
        elapsed = time.time() - t0
        pred_class = CLASS_NAMES[np.argmax(avg)]
        confidence = float(np.max(avg))
        probs_str = ", ".join(f"{CLASS_NAMES[i]}={avg[i]:.4f}" for i in range(3))
        print(f"  [ensemble] {pred_class} ({confidence:.4f}) [{elapsed:.2f}s] — {probs_str}")
        for mid, pred in all_preds.items():
            print(f"    [{mid}] → {CLASS_NAMES[np.argmax(pred)]} ({float(np.max(pred)):.4f})")

    # --- 4. Grad-CAM ---
    print("\n" + "=" * 70)
    print("SECTION 4: GRAD-CAM HEATMAPS")
    print("=" * 70)

    if not test_files:
        print("No test images!")
    else:
        image = read_and_preprocess(test_files[0])
        print(f"Testing with: {test_files[0].name}")

        for mid, model in models.items():
            print(f"\n--- Grad-CAM: {mid} ---")
            test_gradcam(model, image, mid)

    # --- 5. Unknown / Random Image Detection ---
    print("\n" + "=" * 70)
    print("SECTION 5: UNKNOWN IMAGE DETECTION")
    print("=" * 70)

    # Test with a random non-plant image (create synthetic)
    print("\n--- Test 1: Synthetic non-plant image (uniform gray) ---")
    gray_image = np.full((256, 256, 3), 128, dtype=np.uint8)
    passes_leaf = is_likely_plant_leaf(gray_image)
    print(f"  is_likely_plant_leaf: {passes_leaf}")
    if passes_leaf:
        print("  (passed pre-check — testing model confidence)")
        all_preds, avg = ensemble_predict(models, gray_image)
        unknown, conf, entropy = is_unknown_image(avg, individual_preds=list(all_preds.values()))
        print(f"  Ensemble confidence: {conf:.4f}, entropy: {entropy:.4f}, is_unknown: {unknown}")
    else:
        print("  Correctly rejected by pre-check (not a plant leaf)")

    print("\n--- Test 2: Synthetic non-plant image (red rectangle) ---")
    red_image = np.zeros((256, 256, 3), dtype=np.uint8)
    red_image[:, :, 0] = 200  # red channel
    passes_leaf = is_likely_plant_leaf(red_image)
    print(f"  is_likely_plant_leaf: {passes_leaf}")
    if passes_leaf:
        all_preds, avg = ensemble_predict(models, red_image)
        unknown, conf, entropy = is_unknown_image(avg, individual_preds=list(all_preds.values()))
        print(f"  Ensemble confidence: {conf:.4f}, entropy: {entropy:.4f}, is_unknown: {unknown}")
    else:
        print("  Correctly rejected by pre-check (not a plant leaf)")

    print("\n--- Test 3: Random noise image ---")
    noise_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    passes_leaf = is_likely_plant_leaf(noise_image)
    print(f"  is_likely_plant_leaf: {passes_leaf}")
    if passes_leaf:
        all_preds, avg = ensemble_predict(models, noise_image)
        unknown, conf, entropy = is_unknown_image(avg, individual_preds=list(all_preds.values()))
        print(f"  Ensemble confidence: {conf:.4f}, entropy: {entropy:.4f}, is_unknown: {unknown}")
    else:
        print("  Correctly rejected by pre-check (not a plant leaf)")

    print("\n--- Test 4: Solid green image (mimics wall, should pass color but fail texture) ---")
    green_image = np.zeros((256, 256, 3), dtype=np.uint8)
    green_image[:, :, 1] = 150
    passes_leaf = is_likely_plant_leaf(green_image)
    print(f"  is_likely_plant_leaf: {passes_leaf}")
    if passes_leaf:
        all_preds, avg = ensemble_predict(models, green_image)
        unknown, conf, entropy = is_unknown_image(avg, individual_preds=list(all_preds.values()))
        print(f"  Ensemble confidence: {conf:.4f}, entropy: {entropy:.4f}, is_unknown: {unknown}")
    else:
        print("  Correctly rejected by pre-check (not a plant leaf)")

    # Test with actual diseased leaf images to make sure they pass
    print("\n--- Test 5: Known leaf images (should NOT be unknown) ---")
    for tf_path in test_files:
        image = read_and_preprocess(tf_path)
        passes_leaf = is_likely_plant_leaf(image)
        all_preds, avg = ensemble_predict(models, image)
        unknown, conf, entropy = is_unknown_image(avg, individual_preds=list(all_preds.values()))
        print(f"  {tf_path.name}: passes_leaf={passes_leaf}, confidence={conf:.4f}, "
              f"entropy={entropy:.4f}, is_unknown={unknown}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Models loaded: {len(models)}/3")
    print(f"Models: {list(models.keys())}")
    for mid in models:
        print(f"  {mid}: input={models[mid].input_shape}, output={models[mid].output_shape}")


if __name__ == "__main__":
    main()
