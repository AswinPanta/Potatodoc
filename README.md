# PotatoDoc: AI-Powered Potato Leaf Disease Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![React](https://img.shields.io/badge/react-17.0.2-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.68.0-blue.svg)](https://fastapi.tiangolo.com/)
[![Deploy to HF Spaces](https://img.shields.io/badge/deploy-%F0%9F%A4%97%20Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces)

PotatoDoc is an AI-powered web application that helps farmers and gardeners identify potato leaf diseases (Early Blight, Late Blight, or Healthy) using deep learning.

## Features
- **3 Models + Ensemble**: Choose from CNN Baseline, Transfer Learning, MobileNetV2, or use Ensemble mode
- **Multiple Upload Options**: Drag & drop, click to upload, or webcam capture
- **Treatment Tips**: Disease information and treatment recommendations
- **Prediction History**: Past predictions stored locally

## Tech Stack
- **Backend**: FastAPI, TensorFlow 2.5.0, Uvicorn
- **Frontend**: React 17, Material-UI, Axios
- **Models**: TensorFlow/Keras (CNN custom, ResNet50V2, MobileNetV2)

## Models

| Model | Architecture | Test Accuracy | Params | Saved At |
|-------|-------------|--------------|--------|----------|
| CNN Baseline | 6-conv-layer custom CNN | **92.58%** | 4.1M | `saved_models/1/` |
| Transfer Learning | ResNet50V2 (Phase 1 only) | 97.40% (val) | 23.9M | `saved_models/2/` *(placeholder)* |
| MobileNetV2 | MobileNetV2 + fine-tuning | **99.61%** | 3.5M | `saved_models/3/` |

> **Note**: `saved_models/2/` contains the original placeholder (duplicate CNN Baseline). Transfer Learning Phase 2 (fine-tuning) was skipped due to CPU time constraints (~12 min/epoch). See `training/TRAINING_REPORT.md` for full details.

## Dataset

[PlantVillage dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) — potato disease subset:
- **3 classes**: Early Blight, Late Blight, Healthy
- **2,152 images** (256×256)
- **Split**: 80% train / 10% val / 10% test (seed=42 for deterministic splits)

## Bug Fixes Applied

During training the following bugs were fixed:

1. **`training/train_mobilenet.py`** — `x` used before assignment in data augmentation pipeline (variable name mismatch)
2. **`api/main.py`** — `file.read()` called in synchronous `def` (blocking the async event loop); changed to `async def` with `await`
3. **`api/main-tf-serving.py`** — Same async issue
4. **`training/train_mobilenet.py`** — Hard-coded relative dataset path `"PlantVillage"` (failed depending on CWD); replaced with `Path(__file__).resolve().parent`
5. **`frontend/src/home.js`** — Unused import of `bg.png` removed

## Quick Start

### Prerequisites
- Python 3.9
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
cd api
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

## Deploy to Hugging Face Spaces (Free)

The API can be deployed for free on Hugging Face Spaces. The `Dockerfile` at the repo root packages everything.

### One-click deploy steps:
1. Go to https://huggingface.co/new-space
2. Name it `potatodoc`
3. **Space SDK**: select **Docker**
4. Under **Space Repository**: click **"Link GitHub Repository"** → select `AswinPanta/Potatodoc`
5. Click **"Create Space"**

HF will auto-build and deploy in ~3 minutes. Your API will be live at:
```
https://your-username-potatodoc.hf.space
```

> No credit card needed. The free tier includes 2 GB RAM, 16 GB disk, and always-on service.

## Project Structure
```
potato-disease-classification/
├── api/                          # FastAPI backend
│   ├── main.py                   # API server (loads 3 models)
│   ├── main-tf-serving.py        # TF Serving client version
│   └── venv/                     # Python venv (TF 2.5.0)
├── frontend/                     # React.js frontend
│   └── src/
│       ├── pages/                # Page components (HomePage)
│       ├── components/           # Reusable UI components
│       ├── hooks/                # Custom hooks (API, history)
│       └── constants/            # Disease info, theme
├── PlantVillage/                 # Dataset (3 class subdirs)
├── saved_models/
│   ├── 1/                        # CNN Baseline (92.58%)
│   ├── 2/                        # Placeholder (needs retraining)
│   └── 3/                        # MobileNetV2 (99.61%)
├── training/
│   ├── train_cnn_baseline.py     # CNN Baseline training script
│   ├── train_transfer_learning.py# ResNet50V2 training script
│   ├── train_mobilenet.py        # MobileNetV2 training script
│   └── TRAINING_REPORT.md        # Detailed training report
├── mobile/                       # React Native (Expo) mobile app
├── Dockerfile                    # Containerized API for deployment
└── .dockerignore                 # Docker build context exclusions
```

## Environment Notes

- **TensorFlow 2.5.0** — uses `tf.keras.layers.experimental.preprocessing.*` for augmentation
- **CPU-only** — no GPU available; training is 4–7× slower than GPU
- **SSL certs** — pre-trained weight downloads require `certifi` CA bundle on macOS

## API Endpoints

- `POST /predict` — Predict disease from uploaded image
  - Query params: `model_id` (1=cnn-baseline, 2=transfer-learning, 3=mobilenetv2, ensemble)
  - Returns: class name, confidence scores, treatment info

## License
MIT
