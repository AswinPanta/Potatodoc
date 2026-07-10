import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input

BASE_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS_HEAD = 6
EPOCHS_FINE_TUNE = 6
CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
N_CLASSES = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    str(BASE_DIR / "../PlantVillage"),
    shuffle=True,
    seed=42,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

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

def preprocess_fn(image, label):
    return preprocess_input(image), label

train_ds = train_ds.map(preprocess_fn).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess_fn).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess_fn).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

base_model = ResNet50V2(
    input_shape=INPUT_SHAPE,
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=INPUT_SHAPE)
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(N_CLASSES, activation='softmax')(x)
model = models.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

phase1_callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(
        filepath=str(BASE_DIR / "../saved_models/_tl_phase1_best.weights.h5"),
        save_best_only=True, save_weights_only=True, monitor='val_accuracy'
    )
]

print("Phase 1: training head with frozen base model...")
history1 = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS_HEAD,
    callbacks=phase1_callbacks
)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

phase2_callbacks = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint(
        filepath=str(BASE_DIR / "../saved_models/_tl_phase2_best.weights.h5"),
        save_best_only=True, save_weights_only=True, monitor='val_accuracy'
    )
]

print("Phase 2: fine-tuning last 50 layers...")
history2 = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=phase2_callbacks
)

scores = model.evaluate(test_ds)
print(f"\nTest loss: {scores[0]:.4f}")
print(f"Test accuracy: {scores[1]:.4f}")

model.save(str(BASE_DIR / "../saved_models/2"))
print("Transfer Learning model saved to saved_models/2")

with open(str(BASE_DIR / "../saved_models/_tl_results.txt"), "w") as f:
    f.write(f"Test loss: {scores[0]:.4f}\n")
    f.write(f"Test accuracy: {scores[1]:.4f}\n")
    best_val_acc = max(history2.history['val_accuracy'])
    f.write(f"Best validation accuracy (phase 2): {best_val_acc:.4f}\n")
    f.write(f"Phase 1 epochs: {len(history1.history['loss'])}\n")
    f.write(f"Phase 2 epochs: {len(history2.history['loss'])}\n")
