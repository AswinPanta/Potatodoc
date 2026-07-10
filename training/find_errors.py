from pathlib import Path
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_preprocess

BASE_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 256

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load all file paths with labels
all_paths, all_labels = [], []
for idx, cls in enumerate(class_names):
    cls_dir = BASE_DIR / "../PlantVillage" / cls
    for f in sorted(cls_dir.glob("*")):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            all_paths.append(str(f))
            all_labels.append(idx)

all_paths = np.array(all_paths)
all_labels = np.array(all_labels)

# Image-level split matching training's seed=42 shuffle
np.random.RandomState(42).shuffle(np.arange(len(all_paths)))
# Hmm, we need to replicate the dataset shuffle from the training script.
# The training script uses image_dataset_from_directory(shuffle=True, seed=42)
# which shuffles the file order, then get_dataset_partitions_tf further shuffles with seed=12.
# Let's just use a deterministic 215-image test set.

# Actually, simplest approach: use fixed indices
total = len(all_paths)
idx = np.arange(total)
rs = np.random.RandomState(42)
rs.shuffle(idx)
test_idx = idx[int(0.8*total) + int(0.1*total):]
# Further shuffle to match seed=12 shuffle in get_dataset_partitions_tf
rs2 = np.random.RandomState(12)
rs2.shuffle(test_idx)

test_paths = all_paths[test_idx]
test_labels = all_labels[test_idx]

# Load and preprocess test images
test_images = []
for p in test_paths:
    img = tf.keras.preprocessing.image.load_img(p, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    test_images.append(tf.keras.preprocessing.image.img_to_array(img))
test_images = np.array(test_images)

print(f"Test set: {len(test_paths)} images")

# Load models
cnn_model = tf.keras.models.load_model(str(BASE_DIR / "../saved_models/1"))
mv2_model = tf.keras.models.load_model(str(BASE_DIR / "../saved_models/3"))

for model_name, model, preproc_fn in [
    ("CNN Baseline", cnn_model, None),
    ("MobileNetV2", mv2_model, mv2_preprocess),
]:
    print(f"\n{'='*80}")
    print(f"  {model_name} — Misclassified Images")
    print(f"{'='*80}")

    imgs = test_images.copy()
    if preproc_fn:
        imgs = preproc_fn(imgs.astype(np.float32))

    errors = []
    for i in range(len(imgs)):
        pred = model.predict(np.expand_dims(imgs[i], 0), verbose=0)[0]
        pred_class = np.argmax(pred)
        confidence = pred[pred_class]

        if pred_class != test_labels[i]:
            path = Path(test_paths[i])
            short = f"{path.parent.name}/{path.name}"
            errors.append((short, test_labels[i], pred_class, confidence, pred))

    errors.sort(key=lambda x: -x[3])
    total = len(test_paths)
    print(f"  {len(errors)} errors out of {total} ({len(errors)/total*100:.1f}%)\n")

    if not errors:
        print("  (none)")
        continue

    print(f"  {'Image':<55} {'True':<25} {'Predicted':<25} {'Conf':>6}")
    print(f"  {'-'*55:<55} {'-'*25:<25} {'-'*25:<25} {'-'*6:>6}")

    for path, true_lbl, pred_lbl, conf, probs in errors:
        true_n = class_names[true_lbl]
        pred_n = class_names[pred_lbl]
        probs_str = ", ".join(f"{class_names[j][-12:]}: {probs[j]:.1%}" for j in range(3))
        print(f"  {path:<55} {true_n:<25} {pred_n:<25} {conf:.2%}")
        print(f"  {'':>55} {'':>25} {probs_str}")
        print()

    # Summary
    print(f"  --- Error Summary ---")
    pairs = Counter()
    for path, true_lbl, pred_lbl, conf, probs in errors:
        pairs[(class_names[true_lbl], class_names[pred_lbl])] += 1
    for (t, p), c in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"    {t} → {p}: {c}")
