# Crop Yield Prediction API — container image
# ── Stage 1: builder — install dependencies with uv ──────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
# Copy dependency files first (layer caching — only re-runs if deps change)
COPY pyproject.toml uv.lock* ./

# Install dependencies into /app/.venv
# --frozen: respect uv.lock exactly
# --no-dev:  skip dev/test tools in production image
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime — lean final image ──────────────────────────────────────
FROM python:3.12-slim AS runtime

# HuggingFace Spaces runs as a non-root user (uid=1000)
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY app/ ./app/
#RUN apt-get update && apt-get install -y git git-lfs && git lfs install && git lfs pull

# Copy model and data (large files — kept in final image for HF Spaces)
# paths relative to build context (project root)
RUN mkdir -p /app/ml/model
COPY ml/model/ridge_pipeline.joblib ./ml/model/ridge_pipeline.joblib

# debug commands
#RUN ls -lh ./ml/model
#RUN head -c 50 ./ml/model/lgbm_bestmodel_fbeta10_bundle.pkl || true
#RUN head -c 50 ./ml/model/lgbm_model_quantized.onnx || true

ENV MODEL_PATH=/app/ml/model/ridge_pipeline.joblib
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/app"
    
# HuggingFace Spaces exposes port 7860
EXPOSE 7860

# Basic container health check hitting the API's own health endpoint
#HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
#    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]