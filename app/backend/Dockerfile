# Crop Yield Prediction API — container image
# Build context: app/backend/  (this Dockerfile's own directory)
#   docker build -f app/backend/Dockerfile app/backend

# ── Stage 1: builder — install dependencies with uv ──────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

# ── Stage 2: runtime — lean final image ──────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# API source: main.py, pydantic_models.py, logger.py, helpers.py
COPY api/*.py ./api/

# Trained model artifact (plain joblib export -- see ml/src/train_final_model.py)
COPY model/ridge_pipeline.joblib ./model/ridge_pipeline.joblib

RUN chown -R appuser:appuser /app
USER appuser

ENV MODEL_PATH=/app/model/ridge_pipeline.joblib
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

#WORKDIR /app/api

# Render assigns the listen port dynamically via $PORT at runtime.
# Default to 8000 for local `docker run` where $PORT isn't set.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8000)}/')" || exit 1

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}