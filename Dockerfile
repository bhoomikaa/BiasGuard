# BiasGuard - Containerized AI pipeline
# Python 3.10+ · PyTorch · Transformers
FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .
RUN mkdir -p data/raw data/processed outputs models

ENV PYTHONUNBUFFERED=1
# Override CMD to run: ingest, preprocess, train, or inference
# Example: docker run biasguard python -m scripts.train --model bert-base-uncased
CMD ["python", "-m", "scripts.train", "--model", "bert-base-uncased", "--train", "data/processed/train.jsonl", "--val", "data/processed/val.jsonl", "-o", "outputs/biasguard"]
