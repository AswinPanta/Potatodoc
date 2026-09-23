# Potato Disease Detection - Complete Beginner's Guide

## What Does This Program Do?

This program trains an **AI model** to look at photos of potato leaves and tell you
if the plant is:
- **Healthy** (no disease)
- **Early Blight** (a common fungal disease)
- **Late Blight** (the disease that caused the Irish Potato Famine)

It uses **2,152 images** from the PlantVillage dataset to learn the difference.

---

## Table of Contents

1. [How a Neural Network Learns](#1-how-a-neural-network-learns)
2. [The Big Picture - Step by Step](#2-the-big-picture---step-by-step)
3. [Configuration Settings](#3-configuration-settings)
4. [Data Loading - Reading the Images](#4-data-loading---reading-the-images)
5. [Data Augmentation - Making More Training Data](#5-data-augmentation---making-more-training-data)
6. [The Dataset Class](#6-the-dataset-class)
7. [Fixing Class Imbalance](#7-fixing-class-imbalance)
8. [The Model - What Actually Does the Thinking](#8-the-model---what-actually-does-the-thinking)
9. [Two-Phase Training Strategy](#9-two-phase-training-strategy)
10. [K-Fold Cross-Validation](#10-k-fold-cross-validation)
11. [The Training Loop Explained](#11-the-training-loop-explained)
12. [Evaluation Metrics - How We Measure Success](#12-evaluation-metrics---how-we-measure-success)
13. [Final Results](#13-final-results)
14. [File Outputs](#14-file-outputs)

---

## 1. How a Neural Network Learns

Think of a neural network like a baby learning to recognize animals.

**Step 1:** You show the baby thousands of pictures of cats and dogs.
**Step 2:** The baby starts guessing. At first it gets them wrong.
**Step 3:** You say "wrong!" or "correct!"
**Step 4:** The baby adjusts its brain to get better.
**Step 5:** After enough practice, the baby can look at a NEW picture and know what it is.

A neural network does the same thing, but with math:
- It looks at a picture
- It makes a guess (e.g., "I think this is 70% Early Blight, 20% Late Blight, 10% Healthy")
- It compares the guess to the correct answer
- It adjusts its internal numbers (called **weights**) to do better next time

The **loss** is a number that tells us how wrong the guess was. Lower = better.
The **goal of training** is to make the loss as low as possible.

---

## 2. The Big Picture - Step by Step

Here is what the program does from start to finish:

```
START
  |
  v
1. Load all 2,152 images and their labels (which disease each one has)
  |
  v
2. Try 3 different AI model architectures:
     - EfficientNet-B0
     - ConvNeXt-Tiny
     - MobileNetV2
  |
  v
3. For EACH model, do 5-Fold Cross-Validation:
     - Split data into 5 equal parts
     - Train on 4 parts, test on the remaining 1 part
     - Repeat 5 times so each part gets a turn as the test set
  |
  v
4. Pick the best model (ConvNeXt-Tiny won with 100% F1 score)
  |
  v
5. Do a final 80/20 train/test split with the best model
     - Train on 80% of images
     - Test on 20% of images (never seen during training)
  |
  v
6. Save the trained model and results
  |
  v
END
```

---

## 3. Configuration Settings

These are the "settings" at the top of the file. Think of them as the recipe:

```python
SEED = 42
```
**What it is:** A random number seed.
**Why:** Computers use "random" numbers during training. Setting a seed means
every time you run the program, you get the **same** "random" results. This makes
the experiment **reproducible** - you get the same answer every time.

```python
NUM_CLASSES = 3
```
**What it is:** We have 3 categories to choose from (Early Blight, Late Blight, Healthy).

```python
N_FOLDS = 5
```
**What it is:** We split our data into 5 parts for cross-validation.

```python
EPOCHS_HEAD = 15
```
**What it is:** In Phase 1 of training, we go through all the images 15 times.
An **epoch** = one complete pass through all training images.

```python
EPOCHS_FINETUNE = 40
```
**What it is:** In Phase 2, we go through all images up to 40 more times
(stops early if no improvement).

```python
PATIENCE = 10
```
**What it is:** If the model doesn't improve for 10 epochs in a row, we stop
early. This prevents wasting time when the model has already learned as much
as it can.

```python
LABEL_SMOOTH = 0.1
```
**What it is:** Instead of telling the model "this is 100% Early Blight",
we say "this is 90% Early Blight, and 10% spread across other classes".
**Why:** This prevents the model from being overconfident and helps it generalize better.

```python
MIXUP_ALPHA = 0.2, CUTMIX_ALPHA = 1.0
```
**What it is:** Settings for data augmentation tricks (explained in section 5).

```python
WEIGHT_DECAY = 0.05
```
**What it is:** A penalty that prevents the model from relying too heavily on
any single feature. It encourages the model to learn simpler, more general patterns.

```python
DROPPATH = 0.1
```
**What it is:** During training, randomly "turn off" 10% of the model's connections.
**Why:** Forces the model to not rely on any single path. Makes it more robust.

```python
BACKBONE_LR = 1e-5, HEAD_LR = 1e-4
```
**What it is:** The **learning rate** - how big of steps the model takes when
adjusting its weights. Small numbers = small steps = slower but more careful learning.
The "backbone" (feature extractor) learns slower than the "head" (classifier).

```python
DATA_DIR = Path(r"C:\Users\shadb\Downloads\dataset\PlantVillage")
```
**What it is:** Where your images are stored on your computer.

```python
RESULTS_DIR = BASE_DIR / "results_plantvillage"
```
**What it is:** Where the program saves trained models and results.

---

## 4. Data Loading - Reading the Images

```python
records = []
for cls_idx, cls_dir in enumerate(CLASS_DIRS):
    cls_path = DATA_DIR / cls_dir
    for f in sorted(cls_path.glob("*.*")):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            records.append({
                "path": str(f),
                "class_idx": cls_idx,
                "class_name": CLASS_NAMES[cls_idx]
            })
df = pd.DataFrame(records)
```

**What this does, step by step:**

1. It goes through each folder: `Potato___Early_blight/`, `Potato___Late_blight/`, `Potato___healthy/`
2. For each folder, it finds every image file (.jpg, .png, etc.)
3. It creates a record for each image like:
   ```
   {
     "path": "C:\...\Potato___Early_blight\img001.jpg",
     "class_idx": 0,        (0=Early Blight, 1=Late Blight, 2=Healthy)
     "class_name": "Early Blight"
   }
   ```
4. It puts all records into a table (called a DataFrame)

After this, the program knows where every image is and what disease it has.

The output showed:
```
Early Bllight  :   1000 (46.5%)
Late Blight    :   1000 (46.5%)
Healthy        :    152 (7.1%)
Imbalance ratio: 6.6:1
```

This means there are **way fewer healthy images** than diseased ones.
This is a problem called **class imbalance** (explained in section 7).

---

## 5. Data Augmentation - Making More Training Data

With only 2,152 images, the model might **overfit** - it memorizes the exact
training images instead of learning general patterns.

**Data augmentation** creates new training images by transforming existing ones:

```python
def get_transforms(img_size, is_train=True, strong=False):
```

### Training Augmentations (what we do to training images):

| Augmentation | What it does | Example |
|---|---|---|
| `RandomResizedCrop` | Crops a random part of the image and resizes it | Focuses on different parts of the leaf |
| `HorizontalFlip` | Flips the image left-right | A leaf pointing left could also point right |
| `VerticalFlip` | Flips the image up-down | Leaves can be photographed from any angle |
| `ShiftScaleRotate` | Moves, zooms, or rotates the image slightly | Camera can be at different positions |
| `ColorJitter` | Changes brightness, contrast, saturation | Different lighting conditions |
| `GaussianBlur` | Blurs the image slightly | Out-of-focus photos |
| `GaussNoise` | Adds random dots (noise) | Low-quality camera photos |
| `CoarseDropout` | Randomly hides rectangular patches | Parts of the leaf might be obscured |

### Validation Augmentations (simpler, for testing):

| Augmentation | What it does |
|---|---|
| `Resize` | Makes the image a standard size |
| `CenterCrop` | Crops the center of the image |
| `Normalize` | Adjusts pixel values to match what the model expects |

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```
**What it is:** These numbers come from ImageNet (a huge dataset of 14 million images).
Since our model was pre-trained on ImageNet, we scale our images the same way.

**Why train vs validation are different:**
- Training: We augment aggressively to create variety (like studying with lots of practice problems)
- Validation: We use clean, consistent images (like taking the actual exam)

---

## 6. The Dataset Class

```python
class PotatoDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths       # list of image file paths
        self.labels = labels     # list of labels (0, 1, or 2)
        self.transform = transform  # augmentation function

    def __len__(self):
        return len(self.paths)   # how many images total

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")  # open image
        if self.transform:
            img = self.transform(img)  # apply augmentation
        return img, self.labels[idx]   # return (image, label)
```

**What it is:** A custom class that tells PyTorch how to access our images.

Think of it like a book index:
- `__len__` tells you how many pages (images) there are
- `__getitem__(3)` gives you page 3 (the 4th image + its label)

The `try/except` block at line 209-210 handles corrupted images:
if an image can't be opened, it randomly picks a different one instead of crashing.

---

## 7. Fixing Class Imbalance

**The problem:** We have 1000 Early Blight, 1000 Late Blight, but only 152 Healthy images.
If we don't fix this, the model will learn "most images are diseased" and just guess
disease every time - it would be right 93% of the time but useless for detecting healthy plants.

**The solution: WeightedRandomSampler**

```python
def make_weighted_sampler(labels):
    class_counts = np.bincount(labels)        # [1000, 1000, 152]
    class_weights = 1.0 / class_counts         # [0.001, 0.001, 0.0066]
    sample_weights = class_weights[labels]     # assign weight to each image
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
```

**How it works in plain English:**
- Healthy images (only 152) get a **higher weight** (0.0066)
- Disease images (1000 each) get a **lower weight** (0.001)
- When sampling for training, healthy images are picked **more often**
- This balances the training so the model sees roughly equal amounts of each class

It's like studying for an exam where 93% of questions are about Topic A and
7% about Topic B. You'd spend extra time on Topic B to make sure you know it well.

---

## 8. The Model - What Actually Does the Thinking

```python
def build_model(timm_name, num_classes=3, drop_path=0.0):
    return timm.create_model(timm_name, pretrained=True, num_classes=3, drop_path_rate=drop_path)
```

**What is `timm`?** It's a library of pre-built AI model architectures.
Instead of building a neural network from scratch, we use models that researchers
already designed and tested.

**What is `pretrained=True`?** The model already "knows" how to recognize shapes,
colors, edges, and textures from being trained on 14 million ImageNet photos.
We then **fine-tune** it to focus specifically on potato diseases.

### The Three Models We Tested:

| Model | What it is | Why we chose it |
|---|---|---|
| **EfficientNet-B0** | A balanced model, good speed and accuracy | Great baseline |
| **ConvNeXt-Tiny** | A modern CNN inspired by transformers | Best performance |
| **MobileNetV2** | A lightweight model for mobile phones | Fast inference |

### How a CNN (Convolutional Neural Network) works:

```
Input Image (224x224 pixels)
  |
  v
[Convolutional Layers] - Detect features like edges, textures, spots
  |
  v
[Deeper Layers] - Combine features into patterns like "brown spots", "yellow halos"
  |
  v
[Even Deeper] - Recognize disease-specific patterns
  |
  v
[Classifier Head] - Makes final decision: Early Blight, Late Blight, or Healthy
  |
  v
Output: [0.95, 0.03, 0.02]  (95% sure it's Early Blight)
```

### Model Anatomy:

```
Model
  |
  +-- Backbone (Feature Extractor)
  |     - Reads the image
  |     - Extracts features (edges, textures, patterns)
  |     - This is the "eyes" of the model
  |
  +-- Head (Classifier)
        - Takes the features
        - Makes the final decision
        - This is the "brain" of the model
```

---

## 9. Two-Phase Training Strategy

Training happens in **two phases** to get the best results:

### Phase 1: Head-Only Training (15 epochs)

```python
# Freeze the backbone - don't change feature extraction
for p in model.parameters():
    p.requires_grad = False

# Only train the classifier head
head = model.classifier if hasattr(model, "classifier") else model.head
for p in head.parameters():
    p.requires_grad = True
```

**Analogy:** Imagine you hired a photographer (the backbone) who already knows
how to take great photos. You're only training the detective (the head) who
looks at the photos and makes decisions.

**Why:** The backbone already knows how to "see" from ImageNet training.
We only need to teach it what potato diseases look like. Training everything
at once with a small dataset could scramble the useful knowledge.

**Learning rate:** `1e-3` (relatively fast - the head is simple, can learn quickly)

### Phase 2: Full Fine-Tuning (up to 40 epochs)

```python
# Unfreeze everything
for p in model.parameters():
    p.requires_grad = True

# Different learning rates for backbone vs head
optimizer = AdamW([
    {"params": backbone_params, "lr": 1e-5},   # very slow - don't ruin what it knows
    {"params": head_params, "lr": 1e-4},        # moderate - keep refining decisions
])
```

**Analogy:** Now we let the photographer adjust slightly to focus specifically
on potato leaves, while the detective continues improving.

**Why different learning rates:**
- **Backbone (1e-5):** Very small steps. It already knows how to see; we don't want to mess that up.
- **Head (1e-4):** Larger steps. It's still learning to classify diseases.

---

## 10. K-Fold Cross-Validation

**The problem:** If we train on one set of images and test on another, our
results might depend on which images ended up where by luck.

**The solution:** K-Fold Cross-Validation - try every possible split.

### How 5-Fold works:

```
Dataset split into 5 equal parts:

Fold 1: [TEST] [Train] [Train] [Train] [Train]
Fold 2: [Train] [TEST] [Train] [Train] [Train]
Fold 3: [Train] [Train] [TEST] [Train] [Train]
Fold 4: [Train] [Train] [Train] [TEST] [Train]
Fold 5: [Train] [Train] [Train] [Train] [TEST]

Each fold: train on 4 parts, test on the remaining 1 part.
Final score = average of all 5 test scores.
```

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

**What is "Stratified"?** It means each fold has the **same proportion** of
each disease class. Without this, one fold might accidentally get mostly
healthy images, giving misleading results.

```python
for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
```

This loops through all 5 folds. For each fold:
1. Split data into train (1,721 images) and validation (431 images)
2. Train the model on the training set
3. Test on the validation set
4. Record the F1 score

The **mean F1 across all 5 folds** tells us how reliable the model is.

---

## 11. The Training Loop Explained

This is the heart of the program - where the model actually learns.

### `train_one_epoch()` function:

```python
def train_one_epoch(model, loader, criterion, optimizer, scaler, mix_fn, device):
    model.train()  # Set model to training mode

    for x, y in loader:          # Go through batches of images
        x, y = x.to(device), y.to(device)  # Move to GPU

        # Optional: Mixup augmentation
        if mix_fn is not None:
            x, yin = mix_fn(x, y)

        # Step 1: Clear old gradients
        optimizer.zero_grad(set_to_none=True)

        # Step 2: Make a prediction
        with torch.amp.autocast(device.type):
            output = model(x)            # Forward pass
            loss = criterion(output, yin) # Calculate how wrong we are

        # Step 3: Learn from the mistake
        scaler.scale(loss).backward()     # Calculate gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
        scaler.step(optimizer)            # Update weights
        scaler.update()

    return avg_loss, accuracy
```

### Step-by-step breakdown:

**1. Forward Pass (`model(x)`)**
- Feed the image through the network
- Get back a prediction: e.g., `[0.7, 0.2, 0.1]`

**2. Calculate Loss (`criterion(output, y)`)**
- Compare prediction to the true label
- Loss = how wrong we were (lower is better)
- Uses **CrossEntropyLoss** (standard for classification)
- Uses **class weights** to give more importance to the underrepresented Healthy class

**3. Backward Pass (`loss.backward()`)**
- Calculate how much each weight in the network contributed to the error
- This is called computing **gradients**

**4. Update Weights (`optimizer.step()`)**
- Adjust each weight in the direction that reduces the loss
- The **learning rate** controls how big the adjustment is

**5. Gradient Scaling (`scaler`)**
- Uses **mixed precision training** (float16 instead of float32)
- Makes training faster on GPU with almost no accuracy loss
- The scaler prevents numerical underflow in float16

**6. Gradient Clipping (`clip_grad_norm_`)**
- Limits gradient size to 1.0
- Prevents a single bad batch from causing huge, destructive weight updates

### What is Mixup?

```python
x, yin = mix_fn(x, y)
```

Mixup creates **blended training examples**:

```
Image A: Early Blight (label: [1, 0, 0])
Image B: Healthy     (label: [0, 0, 1])

Mixup (70% A + 30% B):
  Blended image:    70% of A + 30% of B
  Blended label:    [0.7, 0.0, 0.3]
```

**Why:** The model learns smoother decision boundaries instead of being
overconfident. It's like studying blurry examples that are between two categories.

### What is Early Stopping?

```python
if wait >= PATIENCE:
    print(f"  Early stopping at epoch {ep}")
    break
```

If the validation score doesn't improve for 10 epochs:
- The model has likely learned everything it can
- Stop training to save time
- Use the best version (not the last version)

---

## 12. Evaluation Metrics - How We Measure Success

### Accuracy
```
Accuracy = Correct Predictions / Total Predictions
```
Example: If 430 out of 431 images are correct = 99.54% accuracy

**Problem with accuracy:** If 93% of images are diseased, a model that
always guesses "diseased" gets 93% accuracy but is useless.

### Macro F1 Score (the main metric)

**Precision:** Of all the images the model called "Early Blight",
how many actually were Early Blight?

```
Precision = True Positives / (True Positives + False Positives)
```

**Recall:** Of all the actual Early Blight images,
how many did the model correctly identify?

```
Recall = True Positives / (True Positives + False Negatives)
```

**F1 Score:** The harmonic mean of Precision and Recall

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Macro F1:** Average F1 across all 3 classes.
This is our main metric because it treats all classes equally,
even the small Healthy class.

### Confusion Matrix

A table showing what the model got right and wrong:

```
                    Predicted
                 Early  Late  Healthy
Actual Early  [  200,    0,     0  ]   <- All 200 Early Blight correct
Actual Late   [    0,  198,     2  ]   <- 2 Late Blight confused as Healthy
Actual Healthy[    0,    0,    31  ]   <- All 31 Healthy correct
```

### Balanced Accuracy
Like regular accuracy, but gives equal weight to each class.
More meaningful when classes are imbalanced.

---

## 13. Final Results

Your training produced these results:

### K-Fold Cross-Validation Results:

| Model | Mean F1 | Std Dev | Stability |
|---|---|---|---|
| EfficientNet-B0 | 0.8925 | 0.0171 | UNSTABLE |
| **ConvNeXt-Tiny** | **1.0000** | **0.0000** | **STABLE** |
| MobileNetV2 | 0.9147 | 0.0169 | MODERATELY STABLE |

**ConvNeXt-Tiny** achieved a perfect 1.0 F1 score across all 5 folds
with zero variance - this is exceptional.

### Final Test Results (80/20 split):

| Metric | Score |
|---|---|
| Accuracy | 99.54% |
| Macro F1 | 98.79% |
| Balanced Accuracy | 99.67% |

### Per-Class Performance:

| Disease | Correct | Total | Accuracy |
|---|---|---|---|
| Early Blight | 200 | 200 | 100.0% |
| Late Blight | 198 | 200 | 99.0% |
| Healthy | 31 | 31 | 100.0% |

### What Do These Numbers Mean?

- The model is **extremely reliable** at identifying potato diseases
- It got 429 out of 431 test images correct
- The 2 errors were Late Blight images misclassified as Healthy
- It performs equally well on all three classes despite the class imbalance

---

## 14. File Outputs

After training, these files were saved in `results_plantvillage/`:

```
results_plantvillage/
  |
  +-- convnext_tiny_fold0_best.pth    (trained model - fold 1)
  +-- convnext_tiny_fold1_best.pth    (trained model - fold 2)
  +-- convnext_tiny_fold2_best.pth    (trained model - fold 3)
  +-- convnext_tiny_fold3_best.pth    (trained model - fold 4)
  +-- convnext_tiny_fold4_best.pth    (trained model - fold 5)
  |
  +-- efficientnet_b0_fold*_best.pth  (efficientnet models)
  +-- mobilenetv2_fold*_best.pth      (mobilenet models)
  |
  +-- confusion_matrix.png            (visual evaluation chart)
  +-- plantvillage_results.json       (all metrics and settings)
```

### What is a `.pth` file?

It's a PyTorch checkpoint file containing:
- The model's weights (all the learned numbers)
- The model architecture name
- The image size it expects
- The class names
- The F1 score it achieved

You can load it later to make predictions without retraining:

```python
checkpoint = torch.load("convnext_tiny_fold0_best.pth")
model = timm.create_model(checkpoint["timm_name"], num_classes=3)
model.load_state_dict(checkpoint["state_dict"])
```

### What is `plantvillage_results.json`?

A text file (in JSON format) containing all the numbers from training.
You can open it in any text editor to review the results.

---

## Quick Reference: Key Terms

| Term | Meaning |
|---|---|
| **Epoch** | One complete pass through all training images |
| **Batch** | A small group of images processed at once (32 images) |
| **Loss** | A number measuring how wrong the model's predictions are |
| **Gradient** | Direction and magnitude of weight adjustment needed |
| **Learning Rate** | How big of steps to take when adjusting weights |
| **Overfitting** | Model memorizes training data instead of learning general patterns |
| **Underfitting** | Model hasn't learned enough to make good predictions |
| **Fine-tuning** | Adjusting a pre-trained model for a new task |
| **Backbone** | The feature-extracting part of the model (the "eyes") |
| **Head** | The classification part of the model (the "brain") |
| **Augmentation** | Artificial modifications to create more training data |
| **Cross-Validation** | Testing the model on multiple different train/test splits |
| **F1 Score** | Balance between Precision and Recall (0 = worst, 1 = best) |
| **GPU** | Graphics card - special hardware that speeds up training by 10-50x |
| **Mixed Precision** | Using float16 instead of float32 for faster training |
| **Checkpoint** | A saved snapshot of the model at a specific point in training |
