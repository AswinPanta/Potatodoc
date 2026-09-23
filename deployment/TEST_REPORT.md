# Deployment Test Report — Potato Leaf Disease Classifier

**Date:** 2026-08-25
**Bundle:** `deployment\potato_bundle.pth` (385 MB) — all 4 trained models combined:

| Key | Architecture | img | Val macro-F1 |
|---|---|---|---|
| efficientnet_b3_v1 | tf_efficientnetv2_b3 | 300 | 0.9952 |
| convnext_tiny_v1 | convnext_tiny.fb_in22k | 224 | 0.9956 |
| swin_tiny_v1 | swin_tiny_patch4_window7_224.ms_in22k | 224 | 0.9953 |
| convnext_tiny_v2 | convnext_tiny.fb_in22k (regularized v2) | 224 | 0.9955 |

Prediction = soft-vote average of all 4 models' softmax probabilities.

---

## Test 1 — Real-world images (100 unseen field photos, PLD uncontrolled environment)

Sampled stratified from mapped classes: Fungi→Early Blight (58), Healthy→Healthy (15), Phytopthora→Late Blight (27).

| Model | Accuracy /100 |
|---|---|
| **Ensemble (all 4)** | **42%** |
| Swin-Tiny v1 | 42% |
| ConvNeXt-Tiny v1 | 40% |
| ConvNeXt-Tiny v2 | 39% |
| EfficientNet-B3 v1 | 37% |

**Per-class breakdown (ensemble):**

| True class | Correct | Accuracy |
|---|---|---|
| Late Blight | 27/27 | **100%** |
| Healthy | 6/15 | 40% |
| Early Blight | 9/58 | 16% |

Main error: **44 of 58 Early-Blight images predicted as Late Blight** (both diseases look similar in field photos); 9 Healthy called Late Blight.

## Test 2 — Same-distribution reference (100 held-out IPD test images)

| Model | Accuracy /100 |
|---|---|
| **Ensemble + every single model** | **100%** |

---

## Interpretation

- On lab-quality images (same source as training): **perfect 100%**.
- On genuine real-world field photos: **~42% overall**, but Late Blight detection stays perfect. The failure mode is specific: Early Blight ↔ Late Blight visual confusion under uncontrolled lighting/backgrounds, plus some Healthy misreads.
- This matches our earlier full-PLD evaluation (ensemble F1 ≈ 0.42). The v3 training on the 37 GB expanded Irish Potato dataset (currently downloading) is aimed exactly at closing this gap with more diverse imagery.

## How to use

```powershell
# one image or a folder of images:
python C:\Users\shadb\Downloads\dataset\deployment\predict.py "C:\path\to\image_or_folder" "C:\path\to\output.csv"
```

Requires: torch, timm, torchvision, PIL. Runs on GPU if available, else CPU.
Per-image CSV columns: image, true, pred_ensemble, each model's vote, confidence.
