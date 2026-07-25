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
├── api/main.py              # FastAPI + GradCAM + unknown detection + entropy check
├── frontend/src/            # React app (14 files)
│   ├── pages/HomePage.js    # Manual save, GradCAM fetch, Next Image
│   ├── components/          # PredictionResult (heatmap, per-class bars, unknown card)
│   └── hooks/constants/     # fetchGradcam(), useHistory()
├── mobile/                  # React Native (Expo) — same features as web
├── training/                # Training scripts + reports
├── saved_models/            # Models (baked into Docker image)
├── Dockerfile               # tensorflow/tensorflow:2.15.0 base (Python 3.11)
└── README.md
```

## Features Implemented

### Unknown/Random Image Detection
- **API**: `UNKNOWN_THRESHOLD = 0.85` + Shannon entropy check (`norm_entropy > 0.80`)
- **Layers**: Color check (green ratio) → Texture check (intensity_std > 1.5) → Temperature scaling (T=1.5) → Confidence/entropy threshold
- Predict endpoint returns `is_unknown: true` for non-potato-leaf images
- Frontend/mobile show `UnknownImageCard` with warning + suggestions
- Unknown results cannot be saved to history

### Grad-CAM Heatmap (Explainable AI)
- `/gradcam` endpoint computes heatmaps for all 3 models
- Side-by-side display for ensemble mode, collapsible section
- Works for both single model and ensemble
- **Web + Mobile**: identical implementation

### Per-Class Probability Bar Chart
- API returns `probabilities` dict for all 3 classes
- Frontend/mobile display horizontal bar chart showing each class probability
- Predicted class highlighted with full opacity, others dimmed

### UI/UX Improvements
- **Manual save**: "Save to History" button (no auto-save)
- **Next Image**: replaces old "Clear" button
- **Save state**: button becomes "Saved!" and disables after save
- "Unknown" images cannot be saved
- Animations, hover effects, polished card styles

## Deployment
- **URL**: https://AswinPanta-potatodoc.hf.space
- **Dockerfile**: tensorflow/tensorflow:2.15.0 + `COPY saved_models/` (baked in, no runtime download)
- **HF Access Token**: stored in HF Space secrets (never commit to git)
- Models loaded from local disk → fast startup (~2s)

## Environment Notes
- macOS 12 (Monterey), Python 3.13 system, TF 2.15.0 in Docker image
- SSL certs: `SSL_CERT_FILE=$(api/venv/bin/python -c "import certifi; print(certifi.where())")`
- Frontend dev: `cd frontend && npm start`
- Mobile: `cd mobile && npm install && npx expo start`
