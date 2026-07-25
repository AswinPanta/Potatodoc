# Use TensorFlow 2.15 base image (Python 3.11, still uses Keras 2)
# (TF 2.16+ ships Keras 3 which may break SavedModels from TF 2.5)
FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

# Install additional Python packages
# TF 2.15 ships with numpy 1.24+ — no need for tight numpy pin
RUN pip install --no-cache-dir \
    "numpy>=1.24" \
    "fastapi>=0.100.0,<1" \
    "uvicorn>=0.24.0,<1" \
    "python-multipart>=0.0.6,<1" \
    "Pillow>=10.0.0,<11" \
    "matplotlib>=3.7.0,<4" \
    && rm -rf /root/.cache/pip

# Copy application code and model files
COPY api/ ./api/
COPY saved_models/ ./saved_models/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ping')" || exit 1

EXPOSE 8000

# Use 2 workers for better throughput.
# Note: each worker loads its own model copy (~33 MB total per worker).
# The in-memory rate limiter is per-process (not shared across workers).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
