# Potato Disease Detection - How the Code Works

## What is this project?

This project uses **deep learning** (a type of AI) to detect diseases in potato leaves from photos. It looks at a photo of a potato leaf and tells you if it has **Early Blight**, **Late Blight**, or is **Healthy**.

There is no website or app - this is just the **training and testing code** that builds and evaluates the AI models.

---

## The Two Datasets

The code works with two sets of images:

| Dataset | What it is | Size | Purpose |
|---------|-----------|------|---------|
| **IPD** (Indian Potato Disease) | Lab-quality photos | ~58,709 images | Main training data |
| **PLD** (PlantVillage) | Field-quality photos | ~3,076 images | Testing how well the model works on different quality images |

---

## How the Code Works (Step by Step)

### Step 1: Scan the Images

The code first goes through all the folders and collects every image file (.jpg, .jpeg, .png). For each image, it records:
- Which folder it came from (this tells us the disease class)
- The full file path

It organizes them into three categories: `earlyblt` (Early Blight), `lateblt` (Late Blight), and `healthy`.

---

### Step 2: Check for Bad Images

Every single image is opened and verified using `PIL Image.verify()`. If an image is corrupted or can't be read, it gets flagged and removed from the dataset.

---

### Step 3: Find Duplicate Images

**Exact duplicates** - The code calculates a unique fingerprint (SHA256 hash) for every image. If two images have the same fingerprint, they are identical copies. These duplicates get grouped together.

**Near duplicates** - The code also calculates a "perceptual hash" (pHash) for each image. This is like a fingerprint that catches images that look almost the same (maybe slightly resized or compressed). If two images have a very similar pHash (Hamming distance <= 5), they are flagged as near-duplicates. A **Union-Find** algorithm groups all near-duplicates together so they stay in the same split.

This is important because if nearly identical images end up in both the training set and the test set, the model will look like it performs better than it actually does.

---

### Step 4: Split the Data

The images are split into three groups:
- **Training set (70%)** - Used to teach the model
- **Validation set (15%)** - Used to check progress during training
- **Test set (15%)** - Used for final evaluation

The split is done carefully:
1. It is **stratified** - meaning each split gets the same proportion of each disease class
2. It is **group-aware** - meaning near-duplicate images always stay in the same split. If Image A and Image B are near-duplicates, they cannot end up in different splits

---

### Step 5: Prepare the Images for Training

Before feeding images to the model, they are transformed:

**For training** (to make the model more robust):
- Randomly crop and resize to 224x224 or 300x300 (depending on model)
- Randomly flip horizontally
- Randomly change brightness, contrast, saturation
- Randomly blur, add noise, rotate
- Normalize pixel values

**For validation/testing** (consistent processing):
- Resize slightly larger than needed
- Center crop to the exact size
- Normalize pixel values

The code uses **Albumentations** library for this (or falls back to PyTorch's built-in transforms if Albumentations is not installed).

---

### Step 6: Build the AI Models

The code uses 4 different pre-trained model architectures (all from the `timm` library):

| Model | Parameters | Input Size |
|-------|-----------|------------|
| EfficientNetV2-B3 | 12.8M | 300x300 |
| ConvNeXt-Tiny v1 | 27.8M | 224x224 |
| ConvNeXt-Tiny v2 | 27.8M | 224x224 |
| Swin-Tiny | 27.5M | 224x224 |

All of these models were originally trained on millions of general images (ImageNet). The code **fine-tunes** them specifically for potato disease detection. This is called **transfer learning** - instead of starting from scratch, we start with a model that already knows how to see patterns in images, and teach it our specific task.

Each model has been modified to output exactly 3 values (one for each disease class).

---

### Step 7: Train the Models (2-Phase Training)

Training happens in two phases:

#### Phase 1: Train Only the Head (10 epochs)

- The main body of the model (the "backbone") is **frozen** - it doesn't change
- Only the final classification layer (the "head") is trained
- This is like teaching the model's "opinion layer" while keeping its "eyes" fixed
- Uses AdamW optimizer with learning rate 0.001
- Loss function: CrossEntropyLoss with class weights (to handle imbalanced data)

#### Phase 2: Fine-Tune Everything (up to 50 epochs)

- Now the entire model is **unfrozen** and can be updated
- Uses **differential learning rates** - the backbone learns slowly (lr=0.00001) and the head learns faster (lr=0.0001)
- Uses advanced techniques:
  - **MixUp**: Blends two training images and their labels together
  - **CutMix**: Cuts a piece from one image and pastes it onto another
  - **Label Smoothing**: Makes the model less overconfident
  - **AMP (Automatic Mixed Precision)**: Uses half-precision math for speed
  - **Gradient Clipping**: Prevents the model from learning too aggressively
- **Early stopping**: If the model doesn't improve for 10 epochs, training stops
- The best model from each phase is saved

This whole process is repeated for all 4 model architectures.

---

### Step 8: Evaluate the Models

After training, each model is tested:

**On IPD test set (same dataset quality as training):**
- Each model predicts the class for every test image
- Metrics computed: Accuracy, F1-score, Precision, Recall per class
- A confusion matrix is plotted showing what the model gets right and wrong

**On PLD dataset (different image quality - field photos):**
- Same metrics are computed
- This tests if the model can work on images it was NOT trained on
- **Domain gap** is calculated: IPD F1-score minus PLD F1-score

---

### Step 9: Error Analysis

The best model is analyzed more deeply on the PLD dataset:
- Which classes does it confuse the most?
- Does it make confident wrong predictions?
- Does it make uncertain correct predictions?
- Visual examples of mistakes are saved as images

---

### Step 10: Final Verdict

The code prints a summary:
- All models score ~99% accuracy on IPD (lab-quality images)
- All models score ~30-50% on PLD (field-quality images)
- The **domain gap is about 50 points**
- **Conclusion: NOT PRODUCTION READY** - the models cannot reliably work on real-world field photos

---

## Key Output Files

After running, the code produces:

| File | What it contains |
|------|-----------------|
| `{model}_best.pth` | Trained model weights (4 files) |
| `split_cache.json` | How the data was split |
| `data_quality_gate.json` | How many bad/duplicate images were found |
| `ipd_near_duplicates.csv` | List of near-duplicate image pairs |
| `cross_domain.json` | IPD vs PLD performance comparison |
| `confusion_matrices.png` | Visual confusion matrix plots |
| `error_high_conf_errors.png` | Images the model got wrong confidently |
| `error_low_conf_correct.png` | Images the model got right but was unsure |

---

## Technologies Used

| Technology | What it's used for |
|-----------|-------------------|
| Python | Programming language |
| PyTorch | Deep learning framework |
| timm | Pre-trained model library |
| Albumentations | Image augmentation |
| scikit-learn | Data splitting and metrics |
| imagehash | Near-duplicate detection |
| PIL/Pillow | Image loading and verification |
| Matplotlib/Seaborn | Plotting results |
| CUDA | GPU acceleration |

---

## How to Run

```bash
# Install dependencies
python setup.py

# Run the main pipeline (IPD + PLD)
python potato_doc_production_v2_fixed.py

# OR run the PlantVillage-only pipeline (smaller dataset)
python train_plantvillage.py
```

**Note:** Requires a CUDA-capable GPU with at least 8GB VRAM. The full pipeline takes several hours to run.

---

## Known Limitations

1. **No web interface or API** - This is purely a training pipeline, not a deployable product
2. **Double-nested folder structure** - The code expects `content/dataset/earlyblt/earlyblt/` (class/class nesting)
3. **PLD label mapping is unverified** - The code assumes Fungi = Early Blight and Phytopthora = Late Blight, but this hasn't been confirmed by plant disease experts
4. **Near-duplicate detection is slow** - The all-pairs comparison scales poorly with large datasets
5. **Major domain gap** - Models work great on lab photos but poorly on real-world field photos
