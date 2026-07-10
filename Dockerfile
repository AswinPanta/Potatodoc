# Use TensorFlow 2.5 base image (Python 3.6)
FROM tensorflow/tensorflow:2.5.0

WORKDIR /app

# Install additional Python packages (tensorflow is already available)
# Pin uvicorn/Pillow to versions compatible with Python 3.6 in TF 2.5.0 image
RUN pip install --no-cache-dir \
    "numpy>=1.19.2,<1.20" \
    fastapi==0.79.1 \
    uvicorn==0.17.0 \
    python-multipart==0.0.5 \
    "Pillow>=8.0.0,<10.0.0" \
    "matplotlib>=3.3.0,<3.5" \
    && rm -rf /root/.cache/pip

# Copy application code and model files
COPY api/ ./api/
COPY saved_models/ ./saved_models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
