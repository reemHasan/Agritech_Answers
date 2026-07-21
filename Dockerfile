# Crop Yield Prediction API — container image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first, separately from app code, so Docker can cache
# this layer and skip reinstalling on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY main.py .

# Trained model artifact, exported from MLflow before building this image:
#   mlflow artifacts download --run-id <RUN_ID> --artifact-path model --dst-path .
# or, from Python:
#   mlflow.sklearn.save_model(loaded_pipeline, "model")
# The resulting local "model/" directory (containing MLmodel, model.pkl,
# conda.yaml/requirements.txt, etc.) must sit next to this Dockerfile at
# build time.
COPY model/ ./model/

ENV MODEL_PATH=/app/model
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Basic container health check hitting the API's own health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]