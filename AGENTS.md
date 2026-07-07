# Potato Disease Classification — Session Log

## Models & Accuracy
| Model | Architecture | Accuracy | Parameters | Location |
|-------|-------------|----------|------------|----------|
| CNN Baseline | 6-conv custom CNN | 92.58% | 232K | `saved_models/1/` |
| Transfer Learning | ResNet50V2 (Phase 1 only) | 97.40% (val) | — | `saved_models/2/` (placeholder) |
| MobileNetV2 | MobileNetV2 + fine-tune | 99.61% | 2.4M | `saved_models/3/` |

## Key File Structure
```
project root/
├── api/main.py              # FastAPI — loads models from saved_models/{1,2,3}/
├── frontend/src/            # React app (refactored from single home.js)
│   ├── pages/HomePage.js
│   ├── components/ (Navbar, ModelSelector, ImageInput, WebcamCapture, PredictionResult, HistoryDialog)
│   ├── hooks/ (useApi, useHistory)
│   └── constants/ (diseaseInfo, theme)
├── mobile/                  # React Native (Expo) app
│   ├── App.js
│   └── src/screens/HomeScreen.js + components + hooks + constants
├── training/                # Training scripts + reports
│   ├── train_cnn_baseline.py
│   ├── train_mobilenet.py
│   ├── train_transfer_learning.py
│   ├── evaluate_models.py
│   ├── find_errors.py
│   ├── TRAINING_REPORT.md
│   └── TRAINING_REPORT.html
├── saved_models/            # Trained model files (git-tracked, ~33 MB)
├── PlantVillage/            # Dataset (not tracked)
├── Dockerfile               # For deployment (python:3.9-slim + TF 2.5.0)
└── README.md
```

## Bugs Fixed (6 total)
1. `train_mobilenet.py` — `x` used before assignment in data augmentation
2. `api/main.py` — `file.read()` in sync def → changed to async
3. `api/main-tf-serving.py` — same async issue
4. `train_mobilenet.py` — hard-coded relative dataset path
5. `frontend/src/home.js` — unused import removed
6. `api/main.py` — CORS from hardcoded localhost → `["*"]`

## Frontend Refactor
- `home.js` (696 lines) → 13 files in `constants/`, `hooks/`, `components/`, `pages/`
- Build passes with zero warnings
- Same functionality: upload, webcam, model selector, results, history

## Deployment (Hugging Face Spaces)
- **URL**: https://AswinPanta-potatodoc.hf.space
- **Dockerfile**: python:3.9-slim + pip install TF 2.5.0 + download models from GitHub during build
- **Models loaded**: CNN Baseline, Transfer Learning, MobileNetV2 (all 3)
- **Verified**: `/predict?model_id=mobilenetv2` returns predictions
- **HF Access Token**: stored separately — ask user if needed
- No credit card needed, always-on, 2 GB RAM free tier

## Git Remotes
- GitHub: `origin` → https://github.com/AswinPanta/Potatodoc.git (main branch pushed)
- Hugging Face: cloned at `/tmp/hf-potatodoc/` then deleted; files uploaded via `huggingface_hub` API

## Environment Notes
- macOS 12 (Monterey), Python 3.13 system, TF 2.5.0 in `api/venv/`
- SSL certs need `certifi` on macOS: `SSL_CERT_FILE=$(api/venv/bin/python -c "import certifi; print(certifi.where())")`
- `brew install git-lfs` needs hours of build deps — avoid
- Homebrew is Tier 3 on this OS version
- Frontend dev: `cd frontend && npm start`
- Mobile: `cd mobile && npm install && npx expo start`

## Known Limitations
1. Transfer Learning Phase 2 not trained (~20-30 min/epoch on CPU, needs GPU)
2. `find_errors.py` and `evaluate_models.py` use different test splits (can disagree on error counts)
3. Ensemble disabled — CNN baseline bias makes it worse than individual models
4. HF Space token is write-capable — rotate if exposed
