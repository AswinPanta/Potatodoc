
# Midterm Defense Report: PotatoDoc

## Title of the Project
PotatoDoc: AI-Powered Potato Leaf Disease Classification System

## Introduction / Background
Potato is one of the world's most important food crops, providing a staple diet for millions. However, potato plants are susceptible to various diseases, with Early Blight (Alternaria solani) and Late Blight (Phytophthora infestans) being the most common and destructive. These diseases can cause significant yield losses if not detected and treated early.

Traditional methods of disease detection rely on visual inspection by farmers or agricultural experts, which is time-consuming, subjective, and often inaccurate. With the advancement of deep learning and computer vision, automated disease detection systems have emerged as a viable solution to this problem. These systems can quickly and accurately identify diseases from plant images, enabling timely intervention and reducing crop losses.

## Problem Statement
1. Farmers often lack the expertise to accurately identify potato leaf diseases
2. Manual disease diagnosis is time-consuming and may lead to delayed treatment
3. Existing solutions may not be accessible or user-friendly for small-scale farmers
4. There is a need for an integrated system that combines disease detection with treatment recommendations

## Objectives of the Project
1. Develop an accurate deep learning model for classifying potato leaf diseases (Early Blight, Late Blight, Healthy)
2. Build a user-friendly web application for disease detection
3. Implement multiple model architectures and an ensemble system for improved accuracy
4. Provide detailed disease information and treatment recommendations
5. Add features like webcam capture and prediction history
6. Ensure the system is accessible and easy to use for farmers

## Methodology

### 1. Dataset
The dataset used is the PlantVillage dataset from Kaggle, specifically the subset containing potato leaf images:
- Classes: Early Blight, Late Blight, Healthy
- Total images: ~2000+ images per class

### 2. Model Development
Three different model architectures were implemented:
1. **CNN Baseline**: Custom convolutional neural network built from scratch
2. **Transfer Learning**: Using pre-trained models (e.g., VGG, ResNet)
3. **MobileNetV2**: Lightweight model optimized for mobile applications

### 3. Ensemble System
Combines predictions from all three models by averaging their confidence scores, resulting in more robust and accurate predictions.

### 4. Backend Development
- Framework: FastAPI (Python)
- Server: Uvicorn
- Model Serving: Direct model loading (with TensorFlow)

### 5. Frontend Development
- Framework: React.js
- UI Library: Material-UI
- Features: Drag & drop image upload, webcam capture, model selection, history, disease info, treatment recommendations

## Progress Completed So Far
1. ✅ Collected and preprocessed the dataset
2. ✅ Trained 3 different deep learning models (CNN Baseline, Transfer Learning, MobileNetV2)
3. ✅ Implemented ensemble prediction system
4. ✅ Developed FastAPI backend with model endpoints
5. ✅ Built React.js frontend with:
   - Image upload (drag & drop + click)
   - Webcam capture
   - Model selection (including ensemble)
   - Prediction results with confidence scores
   - Disease information and treatment tips
   - Prediction history (localStorage)
   - Image capture guidelines
   - Low-confidence warnings
6. ✅ Designed PotatoDoc branding with green/brown/white theme
7. ✅ Tested the system with sample images

## Results / Findings Achieved
1. All three models achieved high accuracy on the test dataset
2. The ensemble system provided more consistent predictions across different scenarios
3. The web application is user-friendly and intuitive
4. The treatment recommendations are practical and actionable for farmers
5. The low-confidence warning helps users identify when to retake photos

## Challenges Faced and Solutions Applied
1. **Model Compatibility Issues**:
   - Challenge: Different TensorFlow versions causing model loading errors
   - Solution: Standardized TensorFlow version and tested all models with compatible versions

2. **Frontend Theme and Design**:
   - Challenge: Original codebase theme was not user-friendly
   - Solution: Redesigned UI with green/brown/white theme, added clear visual hierarchy

3. **LocalStorage for History**:
   - Challenge: Persisting prediction history across sessions
   - Solution: Implemented localStorage to store and retrieve last 20 predictions

4. **Webcam Integration**:
   - Challenge: Accessing webcam and capturing images in React
   - Solution: Used HTML5 Canvas and MediaDevices API for webcam functionality

5. **Low Confidence Detection**:
   - Challenge: Providing feedback when model is uncertain
   - Solution: Added confidence thresholding (70%) with helpful tips

## Remaining Work / Future Plan
1. **Mobile Application**: Develop React Native app for on-device inference using TensorFlow Lite
2. **Model Optimization**: Further optimize models for speed and size
3. **Multi-Language Support**: Add support for multiple languages to reach more farmers
4. **Offline Mode**: Enable offline predictions using TensorFlow Lite
5. **User Authentication**: Add user accounts for personalized history
6. **Cloud Deployment**: Deploy the system on cloud platforms for wider accessibility
7. **More Diseases**: Expand the system to detect other potato diseases and pests

## Project Timeline (Updated)

| Phase | Description | Status | Timeline |
|-------|-------------|--------|----------|
| 1. Data Collection & Preprocessing | Download and prepare PlantVillage dataset | ✅ Completed | Week 1-2 |
| 2. Model Development | Train CNN Baseline, Transfer Learning, MobileNetV2 | ✅ Completed | Week 3-4 |
| 3. Ensemble System | Implement ensemble prediction | ✅ Completed | Week 5 |
| 4. Backend Development | Build FastAPI endpoints | ✅ Completed | Week 6 |
| 5. Frontend Development | Build React.js app with all features | ✅ Completed | Week 7-8 |
| 6. Testing & Validation | Test system with sample images | ✅ Completed | Week 9 |
| 7. Mobile App (Future) | Develop React Native app | ⏳ Pending | Week 10-12 |
| 8. Deployment (Future) | Deploy to cloud platforms | ⏳ Pending | Week 13-14 |
| 9. Documentation & Final Report | Complete final documentation | ⏳ Pending | Week 15 |

## Conclusion
PotatoDoc has successfully achieved its initial objectives by creating a comprehensive, user-friendly system for potato leaf disease detection. The combination of multiple deep learning models and an ensemble approach ensures robust and accurate predictions. The web application provides all the necessary tools for farmers to identify diseases early and take appropriate action, potentially reducing significant crop losses. Future work will focus on expanding the system's capabilities and making it even more accessible.

## References
1. PlantVillage Dataset. (n.d.). Retrieved from https://www.kaggle.com/arjuntejaswi/plant-village
2. Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv preprint arXiv:1704.04861.
3. FastAPI Documentation. (n.d.). Retrieved from https://fastapi.tiangolo.com/
4. React Documentation. (n.d.). Retrieved from https://reactjs.org/
5. TensorFlow Documentation. (n.d.). Retrieved from https://www.tensorflow.org/
6. Kamilaris, A., & Prenafeta-Boldú, F. X. (2018). Deep learning in agriculture: A survey. Computers and Electronics in Agriculture, 147, 70-90.
