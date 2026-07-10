import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10
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

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

def augment_fn(image, label):
    return data_augmentation(image, training=True), label

train_ds = train_ds.map(augment_fn).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = models.Sequential([
    layers.Input(shape=INPUT_SHAPE),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(N_CLASSES, activation='softmax'),
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

cb = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(
        filepath=str(BASE_DIR / "../saved_models/_cnn_best.weights.h5"),
        save_best_only=True, save_weights_only=True, monitor='val_accuracy'
    )
]

print("Training CNN Baseline...")
history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
    callbacks=cb
)

scores = model.evaluate(test_ds)
print(f"\nTest loss: {scores[0]:.4f}")
print(f"Test accuracy: {scores[1]:.4f}")

# Save as version 1 (overwrite existing)
model.save(str(BASE_DIR / "../saved_models/1"))
print("CNN Baseline saved to saved_models/1")

# Save training results
with open(str(BASE_DIR / "../saved_models/_cnn_results.txt"), "w") as f:
    f.write(f"Test loss: {scores[0]:.4f}\n")
    f.write(f"Test accuracy: {scores[1]:.4f}\n")
    best_val_acc = max(history.history['val_accuracy'])
    f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
    f.write(f"Epochs trained: {len(history.history['loss'])}\n")
