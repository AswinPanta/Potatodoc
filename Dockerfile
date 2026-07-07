FROM tensorflow/tensorflow:2.5.0

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.68.0 \
    uvicorn==0.15.0 \
    python-multipart==0.0.5 \
    Pillow==9.5.0 \
    && rm -rf /root/.cache/pip

COPY api/ ./api/
COPY saved_models/ ./saved_models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
