# Potato Disease Classification — Training Data Q&A

---

## 1. Is the training data raw data?

**Yes.** The training data in `PlantVillage/` consists of **raw JPG image files** stored as-is from the PlantVillage dataset (Kaggle source). No preprocessed or transformed copies exist on disk.

- `PlantVillage/Potato___Early_blight/` — images as downloaded
- `PlantVillage/Potato___Late_blight/` — images as downloaded
- `PlantVillage/Potato___healthy/` — images as downloaded

The filenames are UUID-style (e.g., `001187a0-57ab-4329-baff-e7246a9edeb0___RS_Early.B 8178.JPG`), confirming they are the original PlantVillage format with no transformation applied.

---

## 2. Do we need preprocessing?

**Yes.** Preprocessing is required but is performed **on-the-fly** during training and inference — no preprocessed files are saved to disk.

| Preprocessing Step | CNN Baseline | MobileNetV2 | Transfer Learning (ResNet50V2) |
|--------------------|-------------|-------------|-------------------------------|
| Resize | 256×256 | 256×256 | 256×256 |
| Color conversion | RGB | RGB | RGB |
| Normalization | [0,255] → [0,1] | [0,255] → [-1,1] (via `preprocess_input`) | [0,255] → [-1,1] (via `preprocess_input`) |
| Data augmentation | RandomFlip + RandomRotation(0.2) | RandomFlip + RandomRotation(0.2) | RandomFlip + RandomRotation(0.2) |
| Train/Val/Test split | 80% / 10% / 10% | 80% / 10% / 10% | 80% / 10% / 10% |

**Key files:**
- Training preprocessing: `training/train_cnn_baseline.py:8-53`, `training/train_mobilenet.py:52-62`, `training/train_transfer_learning.py:42-47`
- Inference preprocessing: `backend/main.py:370-381`

---

## 3. Where is the clean data?

**The PlantVillage dataset is already a curated research dataset — it does not require traditional "cleaning."** The data at `PlantVillage/` is the clean data.

| Dataset | Location | Status |
|---------|----------|--------|
| PlantVillage (potato subset) | `PlantVillage/` | **Clean — actively used for training** |
| IrishPotato37G | `IrishPotato37G/` | Downloaded from Zenodo, partially integrated (used by `train_v2.py`) |
| PLD (Potato Leaf Disease) | `PLD/` | Downloaded but **not used** by current training scripts |

If additional cleaning were performed (e.g., removing corrupted images, deduplication), the output would go to a new directory such as `PlantVillage_Cleaned/` following the same 3-class structure.

---

## 4. How many samples of training data are required?

### Current dataset:

| Class | Image Count |
|-------|------------|
| Potato___Early_blight | ~632 |
| Potato___Late_blight | ~760 |
| Potato___healthy | ~760 |
| **Total** | **~2,152** |

### Split breakdown (80/10/10):

| Split | Percentage | Approximate Samples |
|-------|-----------|---------------------|
| Training | 80% | ~1,721 |
| Validation | 10% | ~215 |
| Test | 10% | ~216 |

### What's recommended:

- **Minimum viable**: ~2,000 images total (as currently used) — yields 92–99% accuracy depending on model.
- **For stability**: More data is better. The training report recommends augmenting PlantVillage with real-field images to close the lab-to-field gap.
- **The IrishPotato37G dataset** (`IrishPotato37G/`) contains significantly more images (2,800+ per class) and could be used to increase sample size.

---

## 5. What format is required?

### Image format:
- **Accepted**: JPEG, PNG, WebP (validated by magic bytes in `backend/main.py:36-37`)
- **Training**: Any standard format (JPEG, PNG, BMP, GIF, TIFF) — `image_dataset_from_directory()` handles conversion automatically

### Directory structure (for TensorFlow training):
```
PlantVillage/
├── Potato___Early_blight/
│   ├── image1.JPG
│   └── image2.JPG
├── Potato___Late_blight/
│   ├── image1.JPG
│   └── image2.JPG
└── Potato___healthy/
    ├── image1.JPG
    └── image2.JPG
```

### Input tensor format:
- **Shape**: `(batch_size, 256, 256, 3)` — height × width × 3 RGB channels
- **Data type**: Float32
- **Channel order**: RGB (not BGR)

---

## 6. Are the training data structured?

**Yes.** The data follows two established conventions:

### A. Directory-of-images structure (TensorFlow models 1, 2, 3)
The directory name IS the class label. `tf.keras.preprocessing.image_dataset_from_directory()` automatically infers classes from folder names.

```
PlantVillage/
  Potato___Early_blight/     → Class 0
  Potato___Late_blight/      → Class 1
  Potato___healthy/          → Class 2
```

### B. JSON split structure (PyTorch V2/V3)
Pre-computed train/val/test split stored in `results/split_cache.json`:
```json
{
  "X_train": ["path/to/img1.jpg", ...],
  "X_val": [...],
  "X_test": [...],
  "y_train": [0, 1, 2, ...],
  "y_val": [...],
  "y_test": [...]
}
```

Both structures are well-organized and follow standard ML conventions.

---

## Summary

| Question | Answer |
|----------|--------|
| Is the training data raw? | Yes — original PlantVillage JPGs, no transformations applied |
| Do we need preprocessing? | Yes — resize, normalization, augmentation (done on-the-fly) |
| Where is the clean data? | `PlantVillage/` is already clean/curated; no separate cleaned version needed |
| How many samples required? | ~2,152 total (80/10/10 split); more data recommended for stability |
| What format is required? | JPEG/PNG in 3-class directory structure; 256×256×3 RGB float32 tensors |
| Is the data structured? | Yes — directory-of-images (TF) and JSON split file (PyTorch) |

---

---

# IrishPotato37G Dataset — Full Analysis

---

## 1. What is the IrishPotato37G Dataset?

**Source:** Zenodo record `17553016` (DOI: `10.5281/zenodo.17553016`)
- **Full title:** "Irish Potato Leaf Image Dataset (Healthy, Early Blight, Late Blight)"
- **Authors:** Mduma, Neema; Mtali, Rahel — Nelson Mandela African Institution of Science and Technology (Tanzania)
- **License:** CC BY 4.0
- **Publication date:** 2025-11-16
- **URL:** https://zenodo.org/records/17553016
- **Purpose:** Research in diagnosing potato diseases, image classification and object detection for farmers

### Composition (21 zip files, ~37.5 GB total):

| Class | Zip Files | Images on Disk | Status |
|-------|-----------|---------------|--------|
| Healthy | HEALTHY_1 through HEALTHY_6 | **5,665** | All downloaded & extracted |
| Early Blight | EARLYBLT_1 through EARLYBLT_9 | **3,846** | 7 of 9 zips extracted; EARLYBLT_8 partial (42%), EARLYBLT_9 not started |
| Late Blight | LATEBLT_1 through LATEBLT_6 | **0** (directory missing) | None downloaded |
| **Total** | **21 zips** | **9,511 on disk** | **INCOMPLETE — ~15,000+ expected** |

### Image format:
- JPG/JPEG (mixed `.JPG` and `.jpg` extensions)
- High-resolution originals (~4 MB each, likely ~1500×1500px)
- Download script: `fetch_ipd37.py` (resumable HTTP downloads with progress tracking)

---

## 2. Is the IrishPotato37G Training Data Raw Data?

**Yes.** The images are raw, original photographs stored as-is after download and extraction. No preprocessing, resizing, or normalization has been applied to the files on disk.

- `IrishPotato37G/healthy/` — 5,665 raw JPGs
- `IrishPotato37G/earlyblt/` — 3,846 raw JPGs
- `IrishPotato37G/lateblt/` — **does not exist** (download incomplete)

---

## 3. Do We Need Preprocessing?

**Yes.** The `train_v3.py` script (designed for this dataset) applies the same preprocessing pipeline as the earlier versions:

| Step | Details |
|------|---------|
| Resize | 224×224 (for ConvNeXt-Tiny) |
| Color conversion | RGB via PIL |
| Normalization | ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| Augmentation | RandAugment, RandomGrayscale, RandomResizedCrop, MixUp, CutMix |
| Split | Stratified 70/15/15 (planned for `results/split_cache_v3.json`) |

All preprocessing is done on-the-fly during training. No preprocessed files are stored.

---

## 4. Where is the Clean Data?

| Location | Status |
|----------|--------|
| `IrishPotato37G/healthy/` | 5,665 images — **present** |
| `IrishPotato37G/earlyblt/` | 3,846 images — **present** |
| `IrishPotato37G/lateblt/` | **MISSING — directory does not exist** |
| `results/split_cache_v3.json` | **MISSING — never created** |

**The dataset is incomplete.** The download stopped during EARLYBLT_8 extraction. 12 of 21 zips were never processed. The `lateblt/` class has zero images, making training impossible.

**If complete**, the clean data would be:
```
IrishPotato37G/
├── healthy/     (~5,665 images)
├── earlyblt/    (~7,000+ images when fully downloaded)
└── lateblt/     (~5,000+ images when fully downloaded)
```

---

## 5. How Many Samples Are Required?

### What's on disk now:

| Class | Count | Status |
|-------|-------|--------|
| Healthy | 5,665 | Complete |
| Early Blight | 3,846 | Partial (2 zips remaining) |
| Late Blight | 0 | **Missing entirely** |
| **Total** | **9,511** | **Need ~15,000+** |

### What was planned (70/15/15 split):

| Split | Percentage | Approximate Samples |
|-------|-----------|---------------------|
| Training | 70% | ~10,500+ |
| Validation | 15% | ~2,250+ |
| Test | 15% | ~2,250+ |
| **Total** | **100%** | **~15,000+** |

### Why more data matters:

The earlier version (`train_all.py`) trained on **58,707 images** (a different, larger IPD dataset) and achieved 99.69% in-distribution accuracy — but collapsed to 36–51% on real field data (PLD dataset). The IrishPotato37G dataset was meant to be the "bigger, better" training set for v3, but it's incomplete.

---

## 6. What Format is Required?

### Directory structure (same as PlantVillage):
```
IrishPotato37G/
├── healthy/
│   ├── image1.jpg
│   └── image2.jpg
├── earlyblt/
│   ├── image1.JPG
│   └── image2.JPG
└── lateblt/
    ├── image1.jpg
    └── image2.jpg
```

### Training script expectations (`train_v3.py`):
- Classes: `["earlyblt", "healthy", "lateblt"]` (line 22)
- Input: 224×224×3 RGB float32 tensors
- Split cache: `results/split_cache_v3.json` with `X_train/X_val/X_test` (file paths) and `y_train/y_val/y_test` (integer labels)

---

## 7. Is the Data Structured?

**Yes — the same directory-of-images convention is used:**
```
IrishPotato37G/
  earlyblt/     → Class 0
  healthy/      → Class 1
  lateblt/      → Class 2
```

The `train_v3.py` script loads this via a custom `ImageFolder`-style dataset that reads file paths and labels from `split_cache_v3.json`.

---

## 8. What Training Has Been Done With This Dataset?

### `train_v3.py` — DESIGNED for IrishPotato37G, **NEVER RAN**

| Detail | Value |
|--------|-------|
| Script | `train_v3.py` (252 lines) |
| Model | ConvNeXt-Tiny (pretrained on ImageNet-22K) |
| Dataset | `IrishPotato37G/` |
| Split cache | `results/split_cache_v3.json` (**does not exist**) |
| Status | **BLOCKED — download incomplete** |
| Log | `train_v3_log.txt` — 155 lines of "waiting for download+extract..." |

The batch runner (`run_train_v3.bat`) waits for `results/EXTRACT_DONE.flag` which was never created because the download stopped at 42% of EARLYBLT_8.

### What was trained BEFORE on a different IPD dataset (58,707 images):

| Version | Script | Dataset | Models | Best Accuracy | Status |
|---------|--------|---------|--------|--------------|--------|
| V1 | `train_all.py` | 58,707 IPD images | EfficientNetV2-B3, ConvNeXt-Tiny, Swin-Tiny + ensemble | 99.65% (ensemble) | **DONE** |
| V2 | `train_v2.py` | Same 58,707 images | ConvNeXt-Tiny (regularized) | 99.69% | **DONE** |
| V3 | `train_v3.py` | IrishPotato37G | ConvNeXt-Tiny | — | **NEVER STARTED** |

### V1 Results (58,707 images):

| Model | Params | Test Acc | Test F1 |
|-------|--------|----------|---------|
| EfficientNetV2-B3 | 12.8M | 99.59% | 0.9958 |
| ConvNeXt-Tiny | 27.8M | 99.63% | 0.9962 |
| Swin-Tiny | 27.5M | 99.65% | 0.9964 |
| Ensemble (3) | — | 99.65% | 0.9964 |

### V2 Results (regularized ConvNeXt-Tiny):

| Model | Test Acc | Test F1 | IPD F1 | PLD Full F1 | PLD Clean F1 |
|-------|----------|---------|--------|-------------|--------------|
| ConvNeXt-Tiny v2 | **99.69%** | **0.9969** | 0.9955 | 0.320 | 0.319 |

### The Critical Problem — Domain Gap:

| Metric | Value |
|--------|-------|
| In-distribution (IPD) accuracy | **99.69%** |
| Cross-dataset (PLD) accuracy | **36–51%** |
| **Domain gap** | **~50 percentage points** |

The models are **highly overfit to the lab-controlled IPD images** and fail on real field data. The IrishPotato37G dataset was meant to address this, but it's incomplete.

---

## 9. Current Blockers & Next Steps

### Blockers:

| Issue | Impact |
|-------|--------|
| IrishPotato37G download incomplete | `lateblt/` missing, `train_v3.py` cannot run |
| `results/split_cache_v3.json` missing | No train/val/test split exists |
| `results/EXTRACT_DONE.flag` missing | Batch runner keeps waiting indefinitely |

### What needs to happen:

1. **Complete the download** — resume `fetch_ipd37.py` to finish EARLYBLT_8, download EARLYBLT_9, and all 6 LATEBLT zips
2. **Create the split** — `split_cache_v3.json` needs to be generated from the complete dataset
3. **Train v3** — `train_v3.py` will then be unblocked and can train on ~15,000+ images
4. **Evaluate domain gap** — test on PLD to see if the larger dataset reduces the ~50-point gap

### Summary:

| Question | Answer |
|----------|--------|
| What is IrishPotato37G? | Zenodo dataset: 15,000+ potato leaf images (healthy, early blight, late blight) from Tanzania |
| Is it raw data? | Yes — original JPGs, no preprocessing applied |
| Does it need preprocessing? | Yes — 224×224 resize, ImageNet normalization, augmentation (on-the-fly) |
| Where is the clean data? | `IrishPotato37G/` — but `lateblt/` is missing (download incomplete) |
| How many samples? | 9,511 on disk; ~15,000+ expected when complete |
| What format? | Directory-of-images structure, 224×224×3 RGB float32 tensors |
| Is it structured? | Yes — `earlyblt/`, `healthy/`, `lateblt/` subdirectories |
| Has training been done? | No — `train_v3.py` was written but never ran (blocked by incomplete download) |
| What was trained before? | V1 (3 models, 99.65%) and V2 (ConvNeXt, 99.69%) on a different 58,707-image IPD dataset |
| Main problem? | ~50-point domain gap — 99.69% in-distribution collapses to 36-51% on real field data |
