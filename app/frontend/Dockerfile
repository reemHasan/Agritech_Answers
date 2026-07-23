# Crop Yield Advisor — Streamlit frontend container image
# Build context: app/frontend/  (this Dockerfile's own directory)
#   docker build -f app/frontend/Dockerfile app/frontend

FROM python:3.12-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY ui_options .
COPY .streamlit/ ./.streamlit/

# ui_options.json (from ml/src/compute_field_bounds.py) is optional --
# app.py falls back to hardcoded defaults if it isn't present. Copy it in
# if you've generated it and want real data-derived slider bounds/dropdowns:
# COPY ui_options.json .

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

# Render assigns the listen port dynamically via $PORT. Default to 8501
# (Streamlit's own default) for local `docker run`.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8501)}/_stcore/health')" || exit 1

CMD streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true
