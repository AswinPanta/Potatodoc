# Potato Disease Detection — Implementation & Training Guide

**Cloned from:** https://github.com/academicbibek-star/potato-disease-detection.git  
**Cloned to:** `C:\Users\shadb\Downloads\potato-disease-detection`  
**Date:** September 2026

---

## 1. Repository Overview

### What's in the repo:

| File | Lines | Purpose |
|------|-------|---------|
| `potato_doc_production_v2_fixed.py` | 1,065 | Complete production pipeline — data audit, training, evaluation, error analysis |
| `setup.py` | 19 | Auto-installs missing Python packages |
| `requirements.txt` | 9 | pip dependencies list |
| `.gitignore` | 3 | Ignores venv and SSH keys |

### What the code does (17 cells):

```
Cell 1-2:   Setup, imports, configuration
Cell 3:     Locate datasets (IPD + PLD)
Cell 4:     Dataset audit (scan images, check corruption)
Cell 5:     SHA256 exact duplicate detection
Cell 6:     Perceptual hash (pHash) near-duplicate detection
Cell 7:     Leakage-safe group-aware train/val/test split
Cell 8:     Load PLD external dataset for cross-domain eval
Cell 8b:    Data quality gate (pass/fail checks)
Cell 9:     Augmentation pipelines (Albumentations or torchvision fallback)
Cell 10:    Dataset class + model factory (timm)
Cell 11:    Training loop (2-phase: head-only → full fine-tune)
Cell 12:    Train all 4 models
Cell 13-14: IPD test evaluation (accuracy, F1, classification report)
Cell 15:    PLD cross-domain evaluation
Cell 16:    PLD error analysis (per-class accuracy, confusion, visualization)
Cell 17:    Confusion matrix plots
```

---

## 2. Architecture & Models

### Models trained (4 total):

| # | Model | timm Name | Params | Input Size | Batch Size |
|---|-------|-----------|--------|-----------|------------|
| 1 | EfficientNetV2-B3 | `tf_efficientnetv2_b3` | 12.8M | 300×300 | 32 |
| 2 | ConvNeXt-Tiny v1 | `convnext_tiny.fb_in22k` | 27.8M | 224×224 | 64 |
| 3 | ConvNeXt-Tiny v2 | `convnext_tiny.fb_in22k` | 27.8M | 224×224 | 64 |
| 4 | Swin-Tiny | `swin_tiny_patch4_window7_224.ms_in22k` | 27.5M | 224×224 | 64 |

### Training Protocol (2-Phase):

```
Phase 1 — Head-Only Probe (10 epochs):
  - Freeze entire backbone
  - Train only classifier head
  - Optimizer: AdamW (lr=1e-3, weight_decay=0.01)
  - Scheduler: CosineAnnealing (T_max=10)
  - Loss: CrossEntropy (with class weights)

Phase 2 — Full Fine-Tune (up to 50 epochs):
  - Unfreeze all layers
  - Differential LR: backbone=1e-5, head=1e-4
  - Optimizer: AdamW (weight_decay=0.05)
  - Scheduler: CosineAnnealing (T_max=50)
  - Loss: SoftTargetCrossEntropy (for MixUp/CutMix)
  - Early stopping: patience=10 on val macro-F1
```

### Regularization Techniques:

| Technique | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Weight Decay | 0.01 | 0.05 |
| Label Smoothing | 0.1 | 0.1 |
| Drop Path | 0.1 | 0.1 |
| MixUp (α=0.2) | No | Yes (prob=1.0) |
| CutMix (α=1.0) | No | Yes (prob=1.0) |
| Gradient Clipping | 1.0 | 1.0 |
| AMP Mixed Precision | Yes | Yes |

---

## 3. Datasets Required

### Dataset 1: IPD — Irish Potato Leaf Image Dataset (TRAINING)

| Property | Value |
|----------|-------|
| Source | https://zenodo.org/records/17553016 |
| DOI | https://doi.org/10.5281/zenodo.17553016 |
| Size | ~37.5 GB (21 zip files) |
| Images | 58,709 photos |
| Classes | Early Blight, Healthy, Late Blight |
| License | CC-BY-4.0 |
| Paper | Laizer & Mduma (2025), Data in Brief 60, 111549 |

**Required folder structure:**
```
content/dataset/
  earlyblt/earlyblt/*.jpg    (17,772 images)
  healthy/healthy/*.jpg      (20,438 images)
  lateblt/lateblt/*.jpg      (20,499 images)
```

### Dataset 2: PLD — Potato Leaf Disease in Uncontrolled Environment (EVALUATION)

| Property | Value |
|----------|-------|
| Source 1 | https://data.mendeley.com/datasets/ptz377bwb8/1 |
| Source 2 | https://www.kaggle.com/datasets/warcoder/potato-leaf-disease-dataset |
| DOI | https://doi.org/10.17632/ptz377bwb8.1 |
| Size | ~180 MB |
| Images | 3,076 photos (7 categories, 3 mapped) |
| License | CC-BY-4.0 |
| Paper | Shabrina et al. (2024), Data in Brief 52, 109955 |

**Required folder structure:**
```
content/dataset/PLD/Potato Leaf Disease Dataset in Uncontrolled Environment/
  Fungi/*.jpg         → mapped to Early Blight (class 0)
  Healthy/*.jpg       → mapped to Healthy (class 1)
  Phytopthora/*.jpg   → mapped to Late Blight (class 2)
  Bacteria/*.jpg      → NOT used
  Virus/*.jpg         → NOT used
  Pest/*.jpg          → NOT used
  Nematode/*.jpg      → NOT used
```

---

## 4. Data Pipeline Workflow

### Step 1: Data Audit
```
Scan all images → Count per class → Check corruption (PIL verify)
```

### Step 2: Exact Duplicate Detection
```
Compute SHA256 hash per image → Group identical hashes → Flag cross-class duplicates
```

### Step 3: Near-Duplicate Detection
```
Compute perceptual hash (pHash) → Compare all pairs → Group by Hamming distance ≤ 5
Uses Union-Find to keep near-duplicates in same split (prevents leakage)
```

### Step 4: Leakage-Safe Split
```
Group-aware stratified split (70/15/15):
  - Near-duplicate groups kept together (no leakage)
  - Stratified by class label
  - Verified: no overlap between train/val/test groups
```

### Step 5: Augmentation
```
Training (Albumentations):
  - RandomResizedCrop (scale 0.5–1.0)
  - HorizontalFlip (p=0.5)
  - VerticalFlip (p=0.2)
  - ColorJitter / BrightnessContrast / HueSaturation
  - GaussianBlur / GaussNoise
  - Rotate (±15°)
  - Normalize (ImageNet mean/std)

Validation/Test:
  - Resize (1.14x) → CenterCrop
  - Normalize (ImageNet mean/std)
```

---

## 5. Hardware Requirements

### Tested on:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 3060 (8 GB VRAM) |
| CPU | Intel i7-12700K (12 cores / 20 threads) |
| RAM | 16 GB |
| CUDA | 12.4 |
| Python | 3.12.8 |
| PyTorch | 2.6.0+cu124 |

### Estimated training time:

| Model | Phase 1 | Phase 2 | Total |
|-------|---------|---------|-------|
| EfficientNetV2-B3 | ~10 min | ~30 min | ~40 min |
| ConvNeXt-Tiny | ~15 min | ~45 min | ~60 min |
| Swin-Tiny | ~15 min | ~45 min | ~60 min |
| **All 4 models** | — | — | **~3-4 hours** |

---

## 6. Installation & Setup

### Step 1: Clone the repo
```bash
git clone https://github.com/academicbibek-star/potato-disease-detection.git
cd potato-disease-detection
```

### Step 2: Create virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Or use the auto-installer:
```bash
python setup.py
```

### Step 4: Prepare datasets
1. Download IPD (21 zips, ~37.5 GB) from Zenodo
2. Download PLD (~180 MB) from Kaggle/Mendeley
3. Extract both into `content/dataset/` following the folder structure above

### Step 5: Run training
```bash
python potato_doc_production_v2_fixed.py
```

---

## 7. Output Files

All results saved to `content/results/`:

| File | Description |
|------|-------------|
| `efficientnetv2_b3_best.pth` | Best EfficientNetV2-B3 checkpoint |
| `convnext_tiny_v1_best.pth` | Best ConvNeXt-Tiny v1 checkpoint |
| `convnext_tiny_v2_best.pth` | Best ConvNeXt-Tiny v2 checkpoint |
| `swin_tiny_best.pth` | Best Swin-Tiny checkpoint |
| `split_cache.json` | Train/val/test split sizes |
| `ipd_near_duplicates.csv` | Near-duplicate pairs found |
| `mixed_label_duplicate_groups.csv` | Cross-class duplicate groups |
| `data_quality_gate.json` | Data quality pass/fail results |
| `ipd_results.json` | IPD test metrics per model |
| `pld_results.json` | PLD cross-domain metrics |
| `cross_domain.json` | IPD vs PLD comparison |
| `confusion_matrices.png` | Confusion matrix plots |
| `error_high_conf_errors.png` | High-confidence errors visualization |
| `error_low_conf_correct.png` | Low-confidence correct predictions |

---

## 8. Known Issues & Bugs

### Bug 1: Directory path double-nesting
```python
# Line 140: DATA_DIR points to content/dataset
DATA_DIR = BASE_DIR / "content" / "dataset"

# Line 219: Expects content/dataset/earlyblt/earlyblt/
ipd_ok = all((DATA_DIR / c / c).exists() for c in IPD_CLASSES)
```
The code expects `content/dataset/earlyblt/earlyblt/` (double nesting). Make sure your extraction matches this.

### Bug 2: scan_ipd() will crash if directories missing
```python
# Line 243: iterdir() on non-existent directory → FileNotFoundError
for f in sorted(cls_dir.iterdir()):
```
Fix: Ensure IPD is extracted before running, or add existence checks.

### Bug 3: PLD mapping unverified
```python
# Line 161: PLD_MAPPING_VERIFIED = False
# Line 498-499: Warning printed but training continues
```
The Fungi→EarlyBlight and Phytopthora→LateBlight mappings need expert verification.

### Bug 4: Coefficient of Variation formula (line referenced in older version)
The stability assessment uses `std/mean` which is correct, but the threshold (< 0.01 for "stable") may be too strict for small datasets.

---

## 9. Training Workflow Summary

```
┌─────────────────────────────────────────────────┐
│  1. DOWNLOAD DATA                               │
│     IPD (37.5 GB) + PLD (180 MB)               │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  2. DATA AUDIT                                  │
│     Scan → Count → Corruption check            │
│     → SHA256 dedup → pHash near-dedup          │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  3. SPLIT DATA                                  │
│     Group-aware stratified 70/15/15            │
│     (near-duplicates stay in same split)       │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  4. TRAIN 4 MODELS                              │
│     Phase 1: Head-only (10 epochs)             │
│     Phase 2: Full fine-tune (50 epochs max)    │
│     + MixUp/CutMix + label smoothing           │
│     + early stopping + AMP                      │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  5. EVALUATE                                    │
│     IPD test set → Accuracy, F1, Confusion     │
│     PLD cross-domain → Domain gap measurement  │
└──────────────────┬──────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────┐
│  6. OUTPUT                                      │
│     .pth checkpoints + metrics JSON + plots    │
│     Verdict: PRODUCTION READY or NOT           │
└─────────────────────────────────────────────────┘
```

---

## 10. What's Needed to Proceed

| # | Task | Status | Action Required |
|---|------|--------|-----------------|
| 1 | Clone repo | Done | — |
| 2 | Install deps | Done | `pip install -r requirements.txt` |
| 3 | Download IPD | **NOT DONE** | Download 21 zips from Zenodo (~37.5 GB) |
| 4 | Download PLD | **NOT DONE** | Download from Kaggle (~180 MB) |
| 5 | Extract to `content/dataset/` | **NOT DONE** | Follow folder structure in Section 3 |
| 6 | Fix directory paths | May need | Verify `DATA_DIR` matches your extraction |
| 7 | Run training | Blocked by #3-5 | `python potato_doc_production_v2_fixed.py` |
| 8 | Evaluate results | Blocked by #7 | Check `content/results/` |

---

## 11. Quick Start (If Datasets Already Downloaded)

```bash
cd C:\Users\shadb\Downloads\potato-disease-detection

# Install
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify data exists
dir content\dataset\earlyblt\earlyblt
dir content\dataset\healthy\healthy
dir content\dataset\lateblt\lateblt

# Run
python potato_doc_production_v2_fixed.py
```

---

## 12. Expected Results (from original author)

| Model | IPD Test Acc | IPD Test F1 | PLD Full F1 | Domain Gap |
|-------|-------------|-------------|-------------|------------|
| EfficientNetV2-B3 | 99.59% | 0.9958 | ~0.40 | ~0.60 |
| ConvNeXt-Tiny v1 | 99.63% | 0.9962 | ~0.36 | ~0.64 |
| ConvNeXt-Tiny v2 | 99.69% | 0.9969 | ~0.32 | ~0.68 |
| Swin-Tiny | 99.65% | 0.9964 | ~0.51 | ~0.49 |

**Verdict:** All models are **NOT PRODUCTION READY** due to ~50-point domain gap between IPD (lab) and PLD (field) data.
