from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as rnv2_preprocess
import time
import json

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256
BATCH_SIZE = 32

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    str(BASE_DIR / "../PlantVillage"),
    shuffle=True,
    seed=42,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(f"Class names: {class_names}")

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

def collect_labels_and_images(ds):
    images_list = []
    labels_list = []
    for imgs, lbls in ds:
        images_list.append(imgs.numpy())
        labels_list.append(lbls.numpy())
    return np.concatenate(images_list), np.concatenate(labels_list)

test_images, test_labels = collect_labels_and_images(test_ds)
print(f"Test set: {len(test_images)} images")

def load_cnn_baseline(path):
    return tf.keras.models.load_model(path)

def load_mobilenetv2(path):
    return tf.keras.models.load_model(path)

print("\n========== LOADING MODELS ==========")
cnn_model = load_cnn_baseline(str(BASE_DIR / "../saved_models/1"))
print(f"CNN Baseline loaded: {cnn_model.count_params()} params")

mv2_model = load_mobilenetv2(str(BASE_DIR / "../saved_models/3"))
print(f"MobileNetV2 loaded: {mv2_model.count_params()} params")

results = {}

for name, model, preproc_fn in [
    ("CNN Baseline", cnn_model, None),
    ("MobileNetV2", mv2_model, mv2_preprocess),
]:
    print(f"\n========== {name} ==========")

    imgs = test_images.copy()
    if preproc_fn:
        imgs = preproc_fn(imgs)
    # CNN Baseline has Rescaling(1./255) inside the model — pass raw [0,255]

    # Inference benchmark
    times = []
    all_preds = []
    all_confidences = []
    for i in range(len(imgs)):
        img_batch = np.expand_dims(imgs[i], 0)
        start = time.perf_counter()
        pred = model.predict(img_batch, verbose=0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        all_preds.append(np.argmax(pred[0]))
        all_confidences.append(pred[0])

    all_preds = np.array(all_preds)
    all_confidences = np.array(all_confidences)

    # Overall metrics
    accuracy = np.mean(all_preds == test_labels)
    latency_mean = np.mean(times) * 1000
    latency_std = np.std(times) * 1000
    latency_p95 = np.percentile(times, 95) * 1000

    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Latency (mean): {latency_mean:.1f} ms")
    print(f"Latency (p95):  {latency_p95:.1f} ms")
    print(f"Latency (std):  {latency_std:.1f} ms")

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(test_labels, all_preds):
        cm[t, p] += 1

    print("\nConfusion Matrix:")
    print("            Predicted")
    print(f"            {'  '.join(f'{c[:8]:>8}' for c in class_names)}")
    for i, cls in enumerate(class_names):
        row = "  ".join(f"{cm[i,j]:>8}" for j in range(3))
        print(f"{cls[:12]:>12}  {row}")

    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    class_metrics = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": int(support)}
        print(f"{cls:<20} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    # Macro avg
    avg_precision = np.mean([m["precision"] for m in class_metrics.values()])
    avg_recall = np.mean([m["recall"] for m in class_metrics.values()])
    avg_f1 = np.mean([m["f1"] for m in class_metrics.values()])
    print(f"{'Macro Avg':<20} {avg_precision:>10.4f} {avg_recall:>10.4f} {avg_f1:>10.4f}")

    # Weighted avg
    total = sum(m["support"] for m in class_metrics.values())
    w_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total
    w_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total
    w_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total
    print(f"{'Weighted Avg':<20} {w_precision:>10.4f} {w_recall:>10.4f} {w_f1:>10.4f}")

    # Error analysis
    misclassified = np.where(all_preds != test_labels)[0]
    print(f"\nMisclassified: {len(misclassified)}/{len(test_labels)} ({len(misclassified)/len(test_labels)*100:.1f}%)")
    error_distribution = {}
    for idx in misclassified[:20]:
        true_cls = class_names[test_labels[idx]]
        pred_cls = class_names[all_preds[idx]]
        conf = all_confidences[idx][all_preds[idx]]
        key = f"{true_cls} -> {pred_cls}"
        error_distribution[key] = error_distribution.get(key, 0) + 1

    print("Top error patterns:")
    for pattern, count in sorted(error_distribution.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pattern}: {count}")

    # Confidence analysis
    correct_confs = all_confidences[np.arange(len(all_preds)) == test_labels, all_preds[np.arange(len(all_preds)) == test_labels]]
    wrong_confs = all_confidences[np.arange(len(all_preds)) != test_labels, all_preds[np.arange(len(all_preds)) != test_labels]]

    results[name] = {
        "accuracy": float(accuracy),
        "latency_ms_mean": float(latency_mean),
        "latency_ms_p95": float(latency_p95),
        "latency_ms_std": float(latency_std),
        "confusion_matrix": cm.tolist(),
        "class_metrics": class_metrics,
        "macro_avg": {"precision": float(avg_precision), "recall": float(avg_recall), "f1": float(avg_f1)},
        "weighted_avg": {"precision": float(w_precision), "recall": float(w_recall), "f1": float(w_f1)},
        "misclassified_count": int(len(misclassified)),
        "error_distribution": error_distribution,
        "correct_mean_confidence": float(np.mean(correct_confs)) if len(correct_confs) > 0 else 0,
        "wrong_mean_confidence": float(np.mean(wrong_confs)) if len(wrong_confs) > 0 else 0,
    }

# Ensemble evaluation
print("\n========== ENSEMBLE (CNN + MobileNetV2) ==========")
ensemble_preds = []
for i in range(len(test_images)):
    cnn_img = test_images[i]  # raw [0,255] (Rescaling inside model)
    mv2_img = mv2_preprocess(test_images[i])

    cnn_pred = cnn_model.predict(np.expand_dims(cnn_img, 0), verbose=0)[0]
    mv2_pred = mv2_model.predict(np.expand_dims(mv2_img, 0), verbose=0)[0]

    ensemble_conf = (cnn_pred + mv2_pred) / 2
    ensemble_preds.append(np.argmax(ensemble_conf))

ensemble_preds = np.array(ensemble_preds)
ensemble_acc = np.mean(ensemble_preds == test_labels)
print(f"Ensemble Test Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")

results["Ensemble (CNN+MV2)"] = {
    "accuracy": float(ensemble_acc),
    "note": "Averages CNN Baseline and MobileNetV2 confidences"
}

# Save results
with open(str(BASE_DIR / "../saved_models/_evaluation_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n\n========== SUMMARY ==========")
print(f"{'Model':<25} {'Accuracy':>10} {'Latency(ms)':>12} {'Params':>12}")
print("-" * 60)
for name in results:
    r = results[name]
    acc = f"{r['accuracy']*100:.2f}%"
    lat = f"{r.get('latency_ms_mean', 0):.1f}" if 'latency_ms_mean' in r else "N/A"
    print(f"{name:<25} {acc:>10} {lat:>12}")

print("\nResults saved to saved_models/_evaluation_results.json")
