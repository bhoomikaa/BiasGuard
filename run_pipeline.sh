#!/bin/bash
# BiasGuard - Full pipeline (ingest -> preprocess -> train)
set -e
cd "$(dirname "$0")"
echo "=== BiasGuard Pipeline ==="
echo "1. Ingest (HuggingFace civil_comments)..."
python3 -m scripts.ingest huggingface --dataset google/civil_comments --max-samples 3000 -o data/raw/ingested.jsonl
echo "2. Preprocess..."
python3 -m scripts.preprocess data/raw/ingested.jsonl -o data/processed
echo "3. Train (distilbert, 1 epoch for quick test)..."
python3 -m scripts.train --model distilbert-base-uncased --epochs 1 -o outputs/biasguard
echo "4. Inference test..."
python3 -m scripts.inference --model outputs/biasguard --text "This is a neutral comment."
echo "=== Done ==="
