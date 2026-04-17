# News-to-Alpha production image (Railway / Render / Fly.io compatible).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Persistent HF cache so FinBERT / MiniLM downloads survive restarts.
    # Mount a volume at /data if you want it permanent across deploys.
    HF_HOME=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    TORCH_HOME=/data/torch-cache

WORKDIR /app

# System packages needed by torch / transformers wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

# Make sure cache + data dirs exist (volume will shadow if mounted).
RUN mkdir -p /data/hf-cache /data/torch-cache /app/data/processed /app/data/models

EXPOSE 8080

# Railway / Render inject $PORT; fall back to 8080 locally.
CMD ["sh", "-c", "gunicorn app.server:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8080}"]
