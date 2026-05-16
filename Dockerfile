FROM python:3.11-slim

# System deps. libgomp is required by sklearn/torch on slim images.
# curl is for the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch from the CPU-only index first. This saves ~2 GB and avoids
# pulling CUDA libraries we have no use for in a research prototype.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Project deps (the torch line in requirements.txt is a no-op since torch is
# already installed above).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source tree.
COPY . .

# Pre-create writable dirs so a fresh container with no bind-mount still
# has somewhere for `train_model.py` to write to.
RUN mkdir -p /app/checkpoints \
             /app/data/synthetic /app/data/processed \
             /app/results/figures /app/results/tables

EXPOSE 8000 8501

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src:/app

# Healthcheck so docker (or k8s) can tell when the API is actually ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default to the API. docker-compose overrides this for the dashboard service.
CMD ["uvicorn", "lhfm.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
