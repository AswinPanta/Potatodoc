# Potato Disease Detection on Large Datasets — Approach Analysis & Recommendation

> **Date:** 2026-09-22
> **Scope:** Compare the proposed "Transfer-Learning CNN + robust pipeline + optimization" approach against what this project actually implements, using 2025–2026 literature + measured project metrics. Verdict + roadmap included.

---

## 1. The Proposed Approach (as stated)

> "Transfer Learning-based CNN combined with robust data pipelines and model optimization."

Four pillars:

### Pillar A — Data Ingestion & Caching
`tf.data` (TensorFlow) or `DataLoader` (PyTorch) with memory caching + prefetching so the GPU never idles on large batches from disk.

**Analysis:** Correct and standard. 2025–2026 consensus:
- TF: `map(..., num_parallel_calls=AUTOTUNE).cache().shuffle().batch().prefetch(AUTOTUNE)`; at 30–60k × 224px do **not** RAM-cache full-res — use `cache(filename)` on-disk or TFRecords + `interleave(deterministic=False)`.
- PyTorch: `num_workers=2–4/GPU, prefetch_factor=2, pin_memory=True, persistent_workers=True, to(device, non_blocking=True)`. Official gains: workers alone ~2.7×, +persistent ~3.7×, +batched `__getitems__` ~10×. Watch `/dev/shm` exhaustion.
- Move resize/normalize to GPU (DALI / torchvision-v2 / `tf.image`) once input-bound.

### Pillar B — Data Augmentation
Random rotations, scaling, flips, color adjustments dynamically → anti-overfit, robust to field lighting/angles.

**Analysis:** Correct but **minimum viable**. Proven winners 2024–2026 add:
- Baseline mandatory: rotation/flip/scale/crop + brightness/contrast/hue jitter (required for PlantVillage→field transfer).
- `CutMix > MixUp` for plants + `label_smoothing=0.1` + `AdamW + grad-clip 1.0`: Apple-MobileNet 94.6%→98.1% with CutMix; EfficientNet-B0+MixUp/CutMix 95.11%; AG-MobileNetV3 99.92%; Albumentations + dynamic MixUp α=0.1 → 99.95%/99.89 F1.
- **Potato caveat:** vanilla MixUp blurs tiny Early-Blight spots; CutMix cuts lesion continuity. Use Beta α≈0.1–0.2 pushing λ→0/1 to preserve lesions. RandAugment (n=2, m=9) is a good default at scale; avoid heavy erase on small spots.

### Pillar C — Model Architecture
Pre-trained `EfficientNetV2` or `MobileNetV3` for accuracy + speed; hybrids with ViT or multi-scale convs (CBSNet-style) for tiny spots / blurred edges.

**Analysis:** Directionally correct, choice depends on target:
- **Lab PlantVillage (~54k, 38-class), all >98%:** 2025 5-fold — EfficientNetV2 98.85%, ConvNeXt 98.92%, Swin 98.75%, ViT-B 98.92%, **Hybrid CNN-ViT 99.29% best**. 2025 55k — Swin-TL F1 1.00 > 11-layer CNN 0.99 > ViT 0.98.
- **Field (PlantDoc ~2.6k in-the-wild): 20–40% drop, ranking flips:** Swin-Base ~73–74% > ConvNeXt-Tiny ~71–73% > ViT-Base ~70% > EfficientNetV2-B3 ~56–61% single. Best pruned ensemble **Swin+ViT+EfficientNetV2 76.19% / F1 0.75 beats full 4-model 75.0%** (ConvNeXt corr 0.937 with Swin = redundant).
- **Potato 3-class:** 2026 K-fold — ResNet50 99.07±0.38% best overall; MobileNetV3-S 97.77±0.59% competitive. 2025 2k potato — MobileNetV3 96.8% beats ResNet50 89.8%, EfficientNet-B3/DenseNet121 ~91–92%. LDL-MobileNetV3S field-potato **94.89% / 1.5M params / 6.17 MB** beats EfficientNet-B0 93.23%/3.83M and ConvNeXt-Tiny 91.72%/27.8M/106MB.
- **Speed 2025–2026:** MobileNetV3-S 1.45M < EfficientNet-B0/V2-S 3.8–21M < ResNet50V2 25M < ConvNeXt-T 28M < ViT-B 86M / Swin-B 88M. ViT has highest GPU throughput via parallelization despite params; ConvNeXt highest latency. Mobile/drone → MobileNetV3 / EfficientNetV2-B0. Server accuracy → Swin / ConvNeXt / hybrid. Dual-branch CNN-ViT (PLOS 2025): **99.71% PlantVillage, 98.78% potato at 4.9M / 0.62 GFLOPs** — template winner.
- **Key hybrid datapoints:**
  - EfficientNetV2B3+ViT on real field potato (3,076 imgs, 7 classes): **85.06%** (+11.43% over prior), vs **98.15%** same model on PlantVillage — i.e. **~13% lab→field gap is normal** (PMC 2025).
  - EfficientNetB0-Swin hybrid 7-class potato: **91.73% / F1 91.71** vs EfficientNetB0 87.84%, Swin 87.36% alone (ISSC 2025).
  - MobileViT-v2: **99.69% potato at 4.39M params** — beats ResNet-101 (J. Comp. Sci. 2025).
  - CBSNet 2025 (potato-specific multi-scale + spatial triple attention): 92.04% / +7.8% precision over ResNeXt-50 — direct answer to fuzzy edges.
  - EfficientNet-LITE + KE-SVM: **87.82% field / 99.54% lab** vs 73.63%/98.15% EfficientNetV2B3 baseline (Frontiers 2025).
- **Image size:** 224 standard for 30–60k throughput. 300–384 adds +1–2% on <10px spots at ~1.8–2.5× cost. Practice: train 224, fine-tune 300/384 for 5–10 epochs.

### Pillar D — Deployment & Quantization
Post-training quantization (TFLite / ONNX Runtime) → smaller file, faster edge inference on mobile/drone.

**Analysis:** Correct, needs precision:
- **TFLite PTQ INT8:** ~4× smaller, ~1/3 latency/size (MobileNetV2 97.9% PlantVillage-2025). Typical drop 1–2% if calibrated on **field** images. Keep first/last layers FP16.
- **Dynamic Range Quantization (DRQ) beats naive PTQ:** rice-leaf study — baseline 94%/9MB → PTQ 83% vs **DRQ 92.23%/2.59MB** (−1.77%), 2.6–2.9× speedup on Pi 4.
- **QAT:** ~0.5–1.0% better than PTQ (sugarcane QAT-MobileNetV3 92% vs ResNet50 88.5%; tomato 98% vs 91%; AG-MobileNetV3 quantized 4.28MB/76.65ms/99.92%). Use if PTQ drop >1.5%.
- **Limits:** W8A8 PTQ ≈ lossless; W6A6 −2%; W4A4 collapse. Don't go <8-bit without SPQ/BRECQ.
- **TFLite vs ONNX:** TFLite+XNNPACK/GPU delegate best for Android offline + drone (this project's Expo mobile case). ONNX Runtime best for server/GPU/Windows + CUDA/TensorRT. Export **both** from same checkpoint; CoreML for iOS.

---

## 2. Where This Project Actually Is (audited)

### 2.1 Models — broader than proposed, and that's good
| Track | Architecture | Params | Measured |
|---|---|---|---|
| `training/train_cnn_baseline.py` → `saved_models/1` | 6-conv custom CNN | 232K | **92.58%** test (`TRAINING_REPORT.md:60-64`) |
| `training/train_transfer_learning.py` → `saved_models/2` | ResNet50V2 + Dropout+ Dense128 | ~25M | **97.40%** val Phase-1 (`:168-176`) |
| `training/train_mobilenet.py` → `saved_models/3` | MobileNetV2 + GAP/Dropout/Dense128 | 2.4M | **99.61%** test (`:221-224`) |
| `train_all.py` | `timm` EfficientNetV2-B3 (300) + ConvNeXt-Tiny (224) + Swin-Tiny (224) | 21M / 28M / 28M | **val-F1 0.9952 / 0.9956 / 0.9953; test-acc 0.9959 / 0.9963 / 0.9965** (`results/ipd_results.json`) |
| `train_v2/v3.py`, `potato-disease-detection/train_plantvillage.py`, `scripts/train_irish.py` | ConvNeXt-Tiny (+ EfficientNet-B0, MobileNetV2-100 in K-fold) | same | K-fold mean-F1 1.0; final 0.9954 acc (`plantvillage_results.json`); v2 0.9969 acc vs v1 0.9963 |
| `deployment/potato_bundle.pth` (385 MB) | 4-model bundle | — | IPD-100 **100%**; field-100 **42%** (`deployment/TEST_REPORT.md:17-44`) |

Project already covers Pillar C **and more**: custom CNN baseline + ResNet50V2 + MobileNetV2 (web API) **plus** EfficientNetV2/ConvNeXt/Swin (large-data track) — matching literature's top families. Missing vs literature: **no MobileNetV3, no explicit CNN×Transformer hybrid, no CBSNet-style multi-scale head** for tiny spots.

### 2.2 Data pipelines — matches Pillar A, PyTorch side is stronger
- **TF track** (`train_cnn_baseline.py:36-42`, `train_transfer_learning.py:39-45`, `train_mobilenet.py:42-48`): `image_dataset_from_directory(seed=42, 256, b32)` + 80/10/10 `take/skip` + `.cache().shuffle(1000).prefetch(AUTOTUNE)` (+`map(preprocess_input)`). Correct, no sampler/`num_workers` needed in TF.
- **PyTorch track** (`train_all.py:127-159`, `train_v2/v3.py`): custom `Dataset` (PIL + fallback), `Image.verify` integrity scan, stratified 70/15/15 + `split_cache*.json`, `DataLoader(num_workers=min(6,cpu), pin_memory, persistent_workers, prefetch_factor=3)`, class-weighted loss. **Exceeds** Pillar A.
- **Best practice in repo:** `potato_doc_production_v2_fixed_clean.py CELL 5–7` — SHA256 exact + pHash (Hamming≤5) Union-Find + **group-aware stratified split with no-leakage asserts**; mirrored in `potato-disease-detection/scripts/prepare_data.py`. Plus `WeightedRandomSampler` + `compute_class_weight(balanced)` + 5-fold CV in `train_plantvillage.py`. This is **above** the proposed approach (which says nothing about dedup/leakage/imbalance/K-fold).

### 2.3 Augmentation — matches + exceeds Pillar B on large-data track
- TF track: `RandomFlip(H+V) + RandomRotation(0.2)` only — **weaker than Pillar B** (no scale/color).
- PyTorch track: `RandomResizedCrop(0.7–1.0 / 0.5–1.0)` + H/V flip + `ColorJitter` + `Rotation(15)` + `RandAugment(2,9)` + `RandomGrayscale` + `Mixup(α0.2)+CutMix(α1.0)+LabelSmooth(0.1)` (stage-2; v3 uses MixUp prob 0.7 = cleaner batches); Albumentations variant adds `ShiftScaleRotate`, `CLAHE`, `MotionBlur`, `CoarseDropout`. **Exceeds** Pillar B and matches literature — with the potato caveat (α small to preserve lesions) already respected in v3.

### 2.4 Deployment & quantization — biggest gap vs Pillar D
- **Have:** FastAPI (`backend/main.py` — 3 models, thread-pool, rate-limit, warmup, Grad-CAM cache, `/predict` + `/gradcam`), registry server (`potato-disease-detection/server.py`), Expo mobile (`mobile/`), AMP (`autocast+GradScaler+clip 1.0`), checkpoints (`.pth`, SavedModel, `.h5`).
- **Missing:** **no `.tflite` / `.onnx` artifact in code** — only notebook names (`training/tf-lite-conversion-post-training.ipynb`, `tf-lite-conversion-quantized-aware-training.ipynb`, `tf-lite-converter.ipynb`, `f-lite-*.ipynb`). No PTQ/DRQ/QAT script, no representative-dataset calibration, no size/latency report. Pillar D is **aspirational, not shipped**.
- Consequence: MobileNetV2 (2.4M) is servable but not edge-optimized; ConvNeXt/Swin (28M, 111MB `.pth`) cannot go on drone/phone as-is.

### 2.5 Strengths the proposal lacks (keep them)
- **Unknown/OOD gate** (`backend/main.py:327-330,560-633`): green+texture pre-check AND temperature-scaled (T=1.5) confidence (`<0.85`) + norm-entropy (`>0.80`) + 2/3 agreement override. Correct direction for the lab→field gap (literature: best cross-domain ~68% MoE vs 41–48% baselines).
- **Grad-CAM** (cached sub-models, parallel, `/gradcam`) + per-class bars + treatment tips + history. Sibling `server.py:141-195` is input-gradient saliency, **not** true Grad-CAM — backend is the reference.
- **Ensemble = mean softmax** everywhere. Note: ensemble **hurts** here — TF ensemble 87.50% < MobileNetV2 99.61%; field ensemble 42% = Swin alone 42% (`TEST_REPORT.md`). Literature agrees: prune redundant members (ConvNeXt corr 0.937 with Swin).

### 2.6 The defining measurement
- Same-distribution: **99.6–100%** (IPD test, IPD-100 reference).
- Genuine field (PLD uncontrolled, 100 stratified): **~42% overall** — Late Blight 27/27 (100%), Healthy 6/15 (40%), Early Blight 9/58 (16%); 44/58 EB→LB confusion (`TEST_REPORT.md:29-37`). Full-PLD F1 ≈ 0.32–0.42.
- This **13–57% lab→field drop is the field norm** (EfficientNetV2B3 98.15%→73.63%; hybrid 98.15%→85.06%), not a bug. The v3 37 GB IrishPotato37G training targets exactly this.

---

## 3. Head-to-Head Comparison

| Dimension | Proposed approach | This project | Winner |
|---|---|---|---|
| Backbone strategy | EfficientNetV2 / MobileNetV3 + ViT/CBSNet hybrid | CNN-baseline + ResNet50V2 + MobileNetV2 **+ EfficientNetV2-B3 + ConvNeXt-T + Swin-T** | **Project (breadth)** — add MobileNetV3 + true hybrid next |
| Small-data (2k PlantVillage) | Transfer learning (generic) | MobileNetV2 **99.61%** > ResNet50V2 97.40% > CNN 92.58% | **Project (measured)** |
| Large-data (58k IPD) | Transfer learning (generic) | EffV2/ConvNeXt/Swin all **~0.996 F1** | **Tie** — matches SOTA |
| Field generalization | Not addressed | 42% field; EB↔LB confusion; OOD gate; v3 diverse-data retrain in progress | **Project (honest + gated)** — still the open problem |
| Data pipeline | tf.data / DataLoader + cache + prefetch | TF cache+prefetch **+** torch workers/pin/persistent/prefetch3 + verify + split-cache + weighted loss/sampler + 5-fold + SHA256/pHash dedup + group split | **Project (exceeds)** |
| Augmentation | rotate/scale/flip/color | TF track **weaker**; torch track RandAugment + MixUp/CutMix + smooth + Albumentations **exceeds** | **Split** — upgrade TF track |
| Tiny-spot / blurred edge | ViT / CBSNet multi-scale | 224px only; 300px only on EffV2-B3; no deformable/FPN/CBS module | **Proposal (gap to close)** |
| Deployment | TFLite / ONNX PTQ | FastAPI + mobile app shell, **no shipped quantized artifact** | **Proposal (gap to close)** |
| Quantization evidence | PTQ (generic) | Notebooks only; no PTQ vs DRQ vs QAT numbers | **Proposal** — implement DRQ→QAT with field calibration |
| Explainability / safety | Not mentioned | Grad-CAM + entropy/temperature unknown gate + agreement override | **Project** |
| Evaluation rigor | Not mentioned | Accuracy + macro-F1 + per-class + confusion + latency + K-fold + cross-domain | **Project** |

---

## 4. Verdict — Which Approach Is Best?

**Neither as-is. The best approach = project's foundation + proposal's missing pieces.**

1. **Keep the project's transfer-learning CNN core** — it already implements Pillar A–B (on the large-data track) and beats the proposal on pipeline rigor (dedup, grouping, imbalance, K-fold, OOD gate). The proposal's pipeline description is a **subset** of what's shipped.
2. **Adopt the proposal's architecture + deployment pointers** — they are the two genuine gaps: (a) edge-efficient + hybrid backbones (MobileNetV3, EfficientNetV2-B0/LITE, CNN×Swin hybrid, CBSNet-style multi-scale) for tiny spots; (b) real TFLite/ONNX quantization with field calibration.
3. **Do not chase lab accuracy** — 99.6%→100% IPD is saturated. All 2025–2026 evidence says the next +1% lab costs more than +10% field. Optimize for **field macro-F1 + calibration + latency**, with the OOD gate as a first-class metric.

**Recommended target system:**
- **Server:** fine-tuned **EfficientNetB0-Swin or EfficientNetV2B3-ViT hybrid** (literature 85–92% field vs 56–80% singles) + existing OOD gate + Grad-CAM.
- **Edge (phone/drone):** **MobileNetV3-S / EfficientNet-LITE** + DRQ (fallback QAT if drop >1.5%), dual TFLite + ONNX export, field representative set for calibration.
- **Data:** keep group-aware splits, pHash dedup, stratified K-fold, class-weighted loss + sampler; train 224, fine-tune 300/384 for 5–10 epochs for small lesions.
- **Evaluation:** report IPD-F1 **and** PLD-field-F1 **and** EB-vs-LB confusion **and** quantized size/latency/drop — never accuracy alone.

---

## 5. Action Roadmap (next steps)

- [ ] **Phase 1 — Close deployment gap (Pillar D):** export MobileNetV2 + best ConvNeXt to TFLite DRQ + ONNX; calibrate on field (PLD) images; publish size/latency/Δ-acc table. If Δ>1.5%, run QAT (`tf-lite-conversion-quantized-aware-training.ipynb` → script).
- [ ] **Phase 2 — Tiny-spot upgrade (Pillar C):** add MobileNetV3-S + EfficientNet-LITE baselines; prototype EfficientNetB0-Swin hybrid and/or CBSNet-style multi-scale head; 224→300/384 fine-tune ablation.
- [ ] **Phase 3 — Field-first training:** finish v3 IrishPotato37G run; add PLD field fine-tune split; track EB↔LB confusion as the primary metric; prune ensemble (drop redundant ConvNeXt if corr>0.9 with Swin).
- [ ] **Phase 4 — Harden TF track (Pillar B):** port RandAugment + color jitter + label smoothing to `training/train_*` TF scripts (currently flip+rot only).
- [ ] **Phase 5 — Unify explainability:** replace sibling `server.py` saliency with backend true Grad-CAM; keep temperature+entropy OOD gate everywhere.

---

## 7. Detailed Explanation of Each Approach Element

### 7.1 Data ingestion & caching — in detail
**Problem:** On 30–60k images the GPU starves while the CPU decodes JPEGs. Throughput = `min(disk decode, CPU preprocess, GPU compute)`.
**TF (`training/train_mobilenet.py:81-83`):**
`map(preprocess).cache().shuffle(1000).prefetch(AUTOTUNE)` means: decode once → keep in RAM → overlap next-batch prep with current GPU step. Correct order is map-cheap → cache → shuffle/augment → batch → prefetch-last. At 58k×256px do NOT RAM-cache augmented outputs; use `cache(filename)` or TFRecords + `interleave(deterministic=False)`.
**PyTorch (`train_all.py:164-175`):** `DataLoader(num_workers=min(6,cpu), pin_memory, persistent_workers, prefetch_factor=3)` means 6 CPU workers decode ahead, pinned memory enables async H→D copy (`non_blocking=True` to add), persistent workers avoid respawn. Measured literature gains: workers ~2.7×, +persistent ~3.7×, +batched fetch ~10×. Failure mode: `/dev/shm` exhaustion → drop to workers=2/prefetch=1.
**Project status:** PyTorch track already exceeds the proposal; TF track is correct but basic. No change needed except TFRecords if IPD grows past RAM.

### 7.2 Augmentation — in detail (formulas + why)
- **Geometric/photometric base** (`train_all.py:144-161`): `RandomResizedCrop(scale 0.7–1.0)` simulates drone distance; H/V flip + `Rotation(15)` simulates leaf angle; `ColorJitter(0.2)` simulates field lighting. This base is *mandatory* for PlantVillage→field transfer.
- **MixUp:** `x~=λxA+(1−λ)xB`, `y~=λyA+(1−λ)yB`, λ∼Beta(α,α). Forces smooth boundaries, anti-memorization. Potato risk: interpolation blurs <10px Early-Blight spots → use **α=0.1–0.2** (λ→0/1, preserves lesions). Project v3 already uses MixUp prob 0.7 + α 0.2 for this reason.
- **CutMix:** paste patch B into A, `y~=λyA+(1−λ)yB` with λ = patch-area ratio. Keeps pixels informative (vs Cutout black-box), forces attention to non-discriminative parts, improves localization + OOD. Risk: cuts lesion continuity → same small-α remedy; literature: CutMix > MixUp for plants (Apple-MobileNet 94.6%→98.1%).
- **Label smoothing (0.1):** `y_smooth=(1−ε)y_hot+ε/K`. Stops overconfident memorization; *highly recommended* with MixUp/CutMix (KerasCV, SDA-CAH 99.95%/99.89 F1 config: AdamW + smooth + clip 1.0 + ReduceLROnPlateau).
- **RandAugment(n=2,m=9):** apply 2 random ops at magnitude 9 per image. Good default at scale; avoid heavy erase/CoarseDropout on small-spot classes.
- **Albumentations tail** (`train_plantvillage.py:153-204`): `ShiftScaleRotate`, `CLAHE`, `MotionBlur/GaussNoise`, `CoarseDropout` simulate drone shake + sensor noise. Keep with p≤0.3 on potato.

### 7.3 Architectures — in detail (how each works, when to use)
- **EfficientNetV2 (proposal + `train_all.py` EffV2-B3/300):** NAS-searched mix of **MBConv** (1×1 expand → 3×3 depthwise → SE → 1×1 project) and **Fused-MBConv** (fused 3×3 replacing expand+depthwise) in early stages for accelerator utilization; smaller expansion ratios + 3×3 kernels + more layers; progressive learning (grow image + regularization jointly). Result: up to 6.8× smaller, 5–11× faster than ViT at same accuracy; 87.3% ImageNet top-1. Use when: best accuracy/param on server, 300px fine-tune for spots. Weakness: field single-model 56–61% (needs hybrid).
- **MobileNetV3 (proposal, missing in project):** inverted residuals + **Squeeze-and-Excitation** (global-pool → FC → scale channels) placed on the expanded representation + **h-swish/h-sigmoid** (quantization-friendly swish) + platform-aware NAS (Large/Small). MobileNetV3-S = 1.45M params; LDL-MobileNetV3S field-potato 94.89%/6.17MB beats Eff-B0/ConvNeXt-T at 106MB. Use when: phone/drone edge, offline Kotlin/Expo app.
- **ResNet50V2 (`saved_models/2`, 97.40% val):** pre-activation residuals, strong baseline, 25M params. 2026 K-fold potato best-overall 99.07±0.38%. Use when: stable server baseline.
- **MobileNetV2 (`saved_models/3`, 99.61% test, 2.4M):** linear bottleneck + inverted residual, no SE/h-swish. Current small-data winner. Direct TFLite path; 97.9% quantized at ~1/3 size in literature.
- **ConvNeXt-Tiny (`train_all/v2/v3`, ~0.996 F1, 28M/111MB):** modernized ResNet (7×7 depthwise, LayerNorm, GELU). Strong lab, highest latency, corr 0.937 with Swin → redundant in ensemble. Use when: server accuracy, then prune.
- **Swin-Tiny (0.9965 test, 28M):** hierarchical shifted-window self-attention = global context + local windows. Best field single (42% here, ~73% literature). Use when: field robustness matters most.
- **Hybrids (proposal, missing — highest priority):** EfficientNetV2B3+ViT 85.06% field (+11.43%) vs 98.15% lab; EfficientNetB0-Swin 91.73% 7-class; MobileViT-v2 99.69%/4.39M; dual-branch CNN-ViT 99.71%/98.78% at 4.9M/0.62GFLOPs; CBSNet multi-scale (1×1/3×3/5×5 + triple attention) +7.8% precision on fuzzy edges. Mechanism: CNN captures local lesion texture, transformer captures leaf-global context + background rejection. Use when: closing the 42%→70%+ field gap.
- **Image size rule:** 224 for throughput; 300/384 final 5–10-epoch fine-tune for <10px spots (+1–2% at 1.8–2.5× cost). Project already does 300 on EffV2-B3 — extend to winner.

### 7.4 Quantization & deployment — in detail
- **Dynamic Range Quantization (start here):** weights INT8, activations calibrated on-the-fly. 4× smaller, 2–3× CPU speedup. Rice-leaf: 9→2.5MB, 94%→92.23% (−1.77%), 2.6× on Pi 4. TFLite `Optimize.DEFAULT`.
- **Full-integer PTQ (for Coral/MCU):** weights + activations INT8 via representative dataset (100–1000 *field* images, not lab). ~4× smaller, 3×+ speedup, 1–2% drop if calibrated right; keep first/last layers FP16. Naive PTQ without calibration collapsed to 83% in the rice study — calibration is the difference.
- **FP16:** 2× smaller, GPU-delegate speedup, minimal drop; dequantizes on CPU so less latency gain. Good intermediate.
- **QAT (if PTQ drop >1.5%):** fake-quant nodes during training so weights adapt; +0.5–1.0% over PTQ (sugarcane 92% vs 88.5% ResNet; AG-MobileNetV3 4.28MB/76ms/99.92%). Cost: retrain + MOT toolkit.
- **Limits:** W8A8 ≈ lossless; W6A6 −2%; W4A4 collapse. Stay at 8-bit.
- **TFLite vs ONNX Runtime (2026 bench):** TFLite 2.16 INT8 22% faster on Cortex-A76 (Pi 5: 18.2 vs 23.1ms), 1.2 vs 2.1MB overhead, best for Android/ARM/Coral+XNNPACK/GPU delegate. ONNX 1.17 converts 87% faster (112s vs 14min for ViT), supports 3.8k vs 1.2k architectures, wins on x86/CUDA/TensorRT/OpenVINO and Qualcomm QNN (−12%). Rule: **TFLite for the Expo/Android/drone target; ONNX for the Windows/Linux server; ship both from the same checkpoint.**

## 8. Proper Suggestion — What Should Be Used (final)

**Decision:** Do NOT pick one approach. Run a **two-track system** — the project foundation is the keeper, the proposal fills the two real gaps.

| Track | Model | Pipeline | Quantization | Why |
|---|---|---|---|---|
| **Server (accuracy + explainability)** | **EfficientNetB0-Swin or EfficientNetV2B3-ViT hybrid**, initialized from `train_all.py` Swin/EffV2 weights; 224 train → 300/384 fine-tune; keep `backend/main.py` OOD gate (T=1.5, conf<0.85, entropy>0.80) + true Grad-CAM | Keep group-aware SHA256/pHash dedup + stratified K-fold + weighted loss/sampler; MixUp α0.1/CutMix + smooth 0.1 + AdamW/clip/scheduler | ONNX (CUDA/TensorRT) FP32 + INT8 static; report IPD-F1 + PLD-field-F1 + EB↔LB confusion | Literature 85–92% field vs 56–80% singles; directly attacks the 44/58 EB→LB error; OOD gate handles the residual gap |
| **Edge phone/drone (offline)** | **MobileNetV3-S (new) or EfficientNet-LITE**, distilled from server hybrid; MobileNetV2-99.61% as fallback | Same field-calibration set (PLD) as representative dataset; Albumentations-light (no heavy erase) | **TFLite DRQ first** (expect ~2.5MB, −1–2%); **QAT if drop>1.5%**; dual TFLite+ONNX export | LDL-MobileNetV3S 94.89%/1.5M/6.17MB; DRQ 92.23% vs PTQ 83%; TFLite 22% faster on ARM; matches Expo mobile target |
| **Retire/demote** | 4-model mean ensemble; ConvNeXt duplicate; TF flip+rot-only augmentation | Single 80/10/10 split reporting accuracy-only | Naive PTQ without field calibration; <8-bit | Ensemble 87.50%<99.61% and 42%=single; ConvNeXt–Swin corr 0.937; TF augment under-regularizes; uncalibrated PTQ collapses |

**Why this wins:**
1. Lab is saturated (99.6–100%) — extra lab points cost more than field points. The hybrid + field fine-tune is the only documented path from 42% toward 68–88% field.
2. Cost is bounded: one hybrid trainer + one edge student + two export scripts reuse existing `train_all/v3`, `prepare_data` dedup, FastAPI/Expo shells.
3. Safety is preserved: OOD gate + Grad-CAM + per-class bars stay; sibling saliency gets replaced by true Grad-CAM.
4. Numbers are verifiable: publish IPD-F1 / field-F1 / EB→LB / size / latency / Δ-quant for every release — never accuracy alone.

**Execution order:** (1) TFLite-DRQ + ONNX export of MobileNetV2 + best ConvNeXt with field calibration → size/latency/Δ table; QAT if needed. (2) MobileNetV3-S + EfficientNet-LITE baselines. (3) EffB0-Swin hybrid + 300px fine-tune; prune ensemble. (4) Port RandAugment/color/smoothing to TF trainers. (5) Finish v3 37GB run + PLD fine-tune; unify Grad-CAM.

## 9. Sources & Traceability

- Project metrics: `results/ipd_results.json:1-19`, `deployment/TEST_REPORT.md:17-51`, `training/TRAINING_REPORT.md:60-64,168-176,221-224,309-316`, `potato-disease-detection/results/plantvillage/plantvillage_results.json`, `backend/main.py:327-330,560-633,644-651,798-836`.
- Pipelines: `training/train_cnn_baseline.py:36-63`, `train_all.py:127-159,201-203`, `potato_doc_production_v2_fixed_clean.py:206-387 (CELL 5–7)`, `potato-disease-detection/scripts/prepare_data.py:120-253`, `train_plantvillage.py:153-246,336-339`.
- Literature 2025–2026: EfficientNetV2B3+ViT field 85.06% vs lab 98.15% (PMC/Springer 2025); EfficientNetB0-Swin 91.73% 7-class (ISSC 2025); MobileViT-v2 99.69%/4.39M (J. Comp. Sci. 2025); EfficientNet-LITE+KE-SVM 87.82% field/99.54% lab (Frontiers 2025); RTR-Lite-MobileNetV2 99.92%/82.00% PlantDoc (Sci. Direct 2025); QAT-MobileNetV3 sugarcane 92%/tomato 98% (Soft Comp. 2026); MobileNetV2-TFLite 97.9% ~1/3 size (ISCS 2025); DRQ 92.23%/2.59MB vs PTQ 83% (IJREEICE 2025); CBSNet potato multi-scale (2025); TCLeaf-Net +5.4 mAP (Dec 2025).
