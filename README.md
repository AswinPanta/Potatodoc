# PotatoDoc: AI-Powered Potato Leaf Disease Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![React](https://img.shields.io/badge/react-17.0.2-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.100+-blue.svg)](https://fastapi.tiangolo.com/)

PotatoDoc is an AI-powered web application that helps farmers and gardeners identify potato leaf diseases (Early Blight, Late Blight, or Healthy) using deep learning.

## Features
- **3 Models + Ensemble**: Choose from CNN Baseline, Transfer Learning, MobileNetV2, or use Ensemble mode
- **Multiple Upload Options**: Drag & drop, click to upload, or webcam capture
- **Grad-CAM Heatmaps**: Visual explanations for model predictions
- **Unknown Image Detection**: Rejects non-potato-leaf images
- **Treatment Tips**: Disease information and treatment recommendations
- **Prediction History**: Past predictions stored locally

## Tech Stack
- **Backend**: FastAPI, TensorFlow 2.15.0, Uvicorn
- **Frontend**: React 17, Material-UI, Axios
- **Models**: TensorFlow/Keras (CNN custom, ResNet50V2, MobileNetV2)

## Models

| Model | Architecture | Test Accuracy | Params | Saved At |
|-------|-------------|--------------|--------|----------|
| CNN Baseline | 6-conv-layer custom CNN | **92.58%** | 232K | `saved_models/1/` |
| Transfer Learning | ResNet50V2 (Phase 1 only) | 97.40% (val) | — | `saved_models/2/` |
| MobileNetV2 | MobileNetV2 + fine-tuning | **99.61%** | 2.4M | `saved_models/3/` |

## Quick Start

### Prerequisites
- Python 3.11
- Node.js 16+
- npm

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000

## Project Structure
```
potato-disease-classification/
├── backend/                        # FastAPI backend
│   ├── main.py                     # API server (loads 3 models, Grad-CAM, unknown detection)
│   ├── test_models.py              # Comprehensive test script
│   └── requirements.txt            # Python dependencies
├── frontend/                       # React.js frontend
│   └── src/
│       ├── pages/                  # Page components (HomePage)
│       ├── components/             # Reusable UI components
│       ├── hooks/                  # Custom hooks (API, history)
│       └── constants/              # Disease info, theme
├── PlantVillage/                   # Dataset (3 class subdirs)
├── saved_models/
│   ├── 1/                          # CNN Baseline (92.58%)
│   ├── 2/                          # Transfer Learning / ResNet50V2 (97.40%)
│   └── 3/                          # MobileNetV2 (99.61%)
├── training/                       # Training scripts and reports
├── mobile/                         # React Native (Expo) mobile app
└── Dockerfile                      # Containerized API
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REACT_APP_API_URL` | `http://localhost:8000` | Backend URL for the web frontend |
| `EXPO_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL for the mobile app |

## API Endpoints

- `GET /ping` — Health check
- `GET /models` — List available models
- `POST /predict` — Predict disease from uploaded image
  - Query params: `model_id` (cnn-baseline, transfer-learning, mobilenetv2, ensemble)
  - Returns: class name, confidence scores, probabilities, treatment info
- `POST /gradcam` — Generate Grad-CAM heatmap
  - Query params: `model_id`
  - Returns: heatmap images for all models (or single model)

## License
MIT
