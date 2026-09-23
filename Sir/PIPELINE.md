# Irish Main-Model Pipeline — Directory Structure & Data Flow

## Directory Layout

```
potato-disease-detection/          ← PIPELINE REPO (code, configs, checkpoints)
├── config/
│   ├── datasets.yaml              ← dataset_id → data root, classes, split
│   ├── models.yaml                ← model_id → checkpoint, preprocessing
│   └── server.yaml                ← host, port, default model
├── scripts/
│   ├── prepare_data.py            ← scan → quarantine → dedup → split
│   └── train_irish.py             ← Irish ConvNeXt-Tiny trainer
├── checkpoints/
│   ├── plantvillage/              ← 15 existing .pth files
│   └── irish/                     ← convnext_tiny_v3_best.pth (after training)
├── results/
│   ├── plantvillage/              ← JSON + confusion matrix
│   └── irish/                     ← irish_results.json (after training)
├── server.py                      ← registry-driven, reads config/*.yaml
└── TRAINING_GUIDE.md / PIPELINE.md

C:\Users\shadb\Downloads\dataset\  ← DATA REPO (raw images, source of truth)
├── IrishPotato37G/                ← raw images
│   ├── earlyblt/, healthy/, lateblt/
│   └── _quarantine/               ← corrupt files moved here
├── PlantVillage/                  ← existing (unchanged)
├── results/
│   ├── split_cache_irish.json     ← train/val/test paths (by prepare_data.py)
│   └── EXTRACT_DONE.flag          ← created when download completes
├── ipd37_zips/                    ← download staging (zips deleted after extract)
├── fetch_ipd37.py                 ← resumable downloader (running in background)
└── mobile/                        ← Expo app (auto-discovers models via /models)
```

## Data Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Zenodo     │────▶│ ipd37_zips/  │────▶│ IrishPotato37G/  │
│  (21 zips)   │     │ (staging)    │     │ earlyblt/healthy/│
└──────────────┘     └──────────────┘     │ lateblt/         │
                                          └───────┬──────────┘
                                                  │ prepare_data.py
                                                  ▼
                                          ┌──────────────────┐
                                          │ split_cache_     │
                                          │ irish.json       │
                                          └───────┬──────────┘
                                                  │ train_irish.py
                                                  ▼
                                          ┌──────────────────┐     ┌──────────────┐
                                          │ checkpoints/irish│────▶│ models.yaml  │
                                          │ convnext_tiny_   │     │ enabled:true │
                                          │ v3_best.pth      │     └──────┬───────┘
                                          └──────────────────┘            │
                                                                          ▼
                                          ┌──────────────────┐     ┌──────────────┐
                                          │   server.py      │◀────│ /models      │
                                          │ (multi-model)    │     │ /predict     │
                                          └──────────────────┘     │ /gradcam     │
                                                                   └──────┬───────┘
                                                                          │
                                                                          ▼
                                                                   ┌──────────────┐
                                                                   │ mobile/ Expo │
                                                                   │ auto-discov. │
                                                                   └──────────────┘
```

## Commands

```bash
# 1. Prepare data (after download completes)
python scripts/prepare_data.py --dataset irish

# 2. Train Irish model
python scripts/train_irish.py

# 3. Start server (serves all enabled models)
uvicorn server:app --host 0.0.0.0 --port 8000

# 4. Mobile app (auto-discovers models from server)
cd C:\Users\shadb\Downloads\dataset\mobile
npx expo start --tunnel
```

## Config Reference

| File | Purpose | Key Fields |
|------|---------|------------|
| `config/datasets.yaml` | Where raw data lives | `root`, `classes`, `split`, `split_cache` |
| `config/models.yaml` | Model registry | `checkpoint`, `timm_name`, `img_size`, `enabled` |
| `config/server.yaml` | Server settings | `host`, `port`, `default_model` |

## Current Status

| Item | Status |
|------|--------|
| Irish download | Resuming in background (PID 21356) |
| Directory structure | Done |
| Configs | Done (3 YAML files) |
| prepare_data.py | Done |
| train_irish.py | Done (waits for download + split cache) |
| server.py | Done (registry-driven, serves convnext_plantvillage) |
| Mobile app | Done (auto-discovery via /models) |
| Irish training | Blocked on download completion |
