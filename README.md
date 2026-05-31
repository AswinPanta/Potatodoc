
# PotatoDoc: AI-Powered Potato Leaf Disease Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![React](https://img.shields.io/badge/react-17.0.2-blue.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/fastapi-0.68.0-blue.svg)](https://fastapi.tiangolo.com/)

PotatoDoc is an AI-powered web application that helps farmers and gardeners identify potato leaf diseases (Early Blight, Late Blight, or Healthy) using deep learning.

## Features
- 🧠 **3 Models + Ensemble**: Choose from CNN Baseline, Transfer Learning, MobileNetV2, or use our Ensemble system for better accuracy!
- 📸 **Multiple Upload Options**: Drag & drop, click to upload, or use webcam capture!
- 💊 **Treatment Tips**: Get detailed disease information and treatment recommendations
- 📜 **Prediction History**: Keep track of your past predictions (stored locally)
- ⚠️ **Low Confidence Alerts**: Get helpful tips when the model is uncertain
- 🎨 **Beautiful UI**: Clean, user-friendly interface with green/brown/white theme

## Tech Stack
- **Backend**: FastAPI, TensorFlow, Uvicorn
- **Frontend**: React.js, Material-UI, Axios
- **Models**: TensorFlow/Keras (CNN, Transfer Learning, MobileNetV2)

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

Now open http://localhost:3000 and start using PotatoDoc!

## Dataset
The models are trained on the PlantVillage dataset from Kaggle: https://www.kaggle.com/arjuntejaswi/plant-village

## Project Structure
```
potato-disease-classification/
├── api/                    # FastAPI backend
├── frontend/               # React.js frontend
├── saved_models/           # Trained TensorFlow models
├── training/               # Training notebooks and scripts
└── test_images_from_internet/
```

## License
MIT
