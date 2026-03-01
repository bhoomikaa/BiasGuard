# BiasGuard

**Automated Bias Detection in Text using Generative & Transfer Learning**  
*Python · PyTorch · Hugging Face · AWS · Transformers (BERT | RoBERTa | LLaMA2)*

---

## 🚀 Project Overview  
BiasGuard detects and mitigates societal bias in large-scale text data through a cloud-native AI pipeline. The system:  
- Scrapes and preprocesses Reddit data for bias-related content  
- Fine-tunes BERT, RoBERTa, DistilBERT, and LLaMA2 for bias/toxicity detection  
- Achieves **86.8% accuracy** (BERT) on Civil Comments validation set  
- Employs layer-wise LR decay, LoRA (for LLaMA2), mixed-precision, early stopping

---

## 🧠 Key Features  
- **End-to-end pipeline**: Ingestion → Pre-processing → Model Training → Evaluation → Deployment  
- **Transfer-learning & LLMs**: Customised fine-tuning of state-of-the-art transformers  
- **Cloud infrastructure**: Built on AWS (Lambda, S3, SageMaker) for scalable processing  
- **Real-time results**: Rapid inference on new text data with low latency  

---

## 🛠️ Tech Stack  
- **Languages**: Python ≥ 3.9  
- **Deep-Learning**: PyTorch, Transformers (Hugging Face), Mixed Precision  
- **Data & Cloud**: AWS (S3, Lambda, SageMaker), Docker  
- **Best Practices**: Differential learning rates, early-stopping, clean modular architecture  

---

## 📁 Repo Structure  
```
BiasGuard/
├── notebooks/          # Exploratory analysis & bias-model prototyping
├── scripts/
│   ├── ingest.py       # Reddit/HuggingFace/local data ingestion
│   ├── preprocess.py   # Text cleaning & train/val split
│   ├── pseudo_label.py # Zero-shot pseudo-labeling (--min-confidence)
│   ├── train.py       # BERT/RoBERTa/LLaMA2 (LoRA) + layer-wise LR
│   ├── inference.py   # Real-time bias detection
│   ├── evaluate.py    # Benchmark accuracy/F1 on civil_comments
│   ├── api_server.py  # FastAPI real-time inference
│   ├── lambda_handler.py # AWS Lambda inference
│   ├── sagemaker_train.py # SageMaker training entrypoint
│   ├── model_utils.py # LLaMA2 + LoRA helpers
│   └── aws_utils.py   # S3 upload
├── results/            # Committed metrics (model_comparison.json)
├── tests/              # Pytest suite (pytest tests/ -v)
├── Dockerfile
├── run_pipeline.sh
├── README.md
└── requirements.txt
```

---

## 🔧 Quick Start  

```bash
git clone https://github.com/bhoomikaa/BiasGuard.git
cd BiasGuard
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt   # Required: PyTorch, transformers, etc.
```

**Note:** Use the project venv (or `pip install -r requirements.txt`) — training needs `torch` and `transformers`.

### 1. Ingest data
```bash
# Option A: HuggingFace dataset (Civil Comments - toxicity/bias)
python -m scripts.ingest huggingface --dataset google/civil_comments --max-samples 5000 -o data/raw/ingested.jsonl

# Option B: Reddit + pseudo-label (requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET)
python -m scripts.ingest reddit --subreddits politics news --limit 500 --pseudo-label -o data/raw/reddit.jsonl
# Then use data/raw/reddit_labeled.jsonl for preprocess

# Option C: Local CSV/JSONL
python -m scripts.ingest file path/to/your_data.csv -o data/raw/ingested.jsonl
```

### 2. Preprocess
```bash
python -m scripts.preprocess data/raw/ingested.jsonl -o data/processed
```

### 3. Train
```bash
# BERT (default)
python -m scripts.train --model bert-base-uncased -o outputs/biasguard

# RoBERTa, DistilBERT, or LLaMA2 (LoRA)
python -m scripts.train --model roberta-base -o outputs/roberta_biasguard
python -m scripts.train --model distilbert-base-uncased -o outputs/distilbert_biasguard
python -m scripts.train --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o outputs/llama_biasguard
```

### 4. Inference
```bash
python -m scripts.inference --model outputs/biasguard --text "Your sample text here"
```

### 5. Evaluation (benchmark accuracy)
```bash
python -m scripts.evaluate --model outputs/biasguard --max-samples 1000 -o results.json
```

### 6. API server (real-time)
```bash
python -m scripts.api_server --model outputs/biasguard --port 8000
# POST /predict with {"text": "..."}
```

### 7. Docker
```bash
docker build -t biasguard .
# Run training (mount data, or copy into image)
docker run -v $(pwd)/data:/app/data biasguard
```

---

## 🔬 Reproduce results

The model comparison table (DistilBERT 84.2%, BERT 86.8%, RoBERTa 83.8%, LLaMA2 85.6%) can be reproduced with:

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Full pipeline (~15–30 min) — downloads Civil Comments, trains all 4 models
python -m scripts.run_demo

# 3. Display the comparison table (uses results_demo.json or trained outputs)
python -m scripts.show_results
```

**Committed artifact:** `results/model_comparison.json` contains the comparison metrics so reviewers can verify without re-running.

**Fast smoke test (~2–5 min):** Mini run with 50 samples, 1 epoch, 2 models:
```bash
python -m scripts.smoke_test
```

---

## 📖 Implementation details

- **Layer-wise LR decay (LLRD):** Earlier transformer layers get lower learning rates; top layers and classifier head get higher LRs. This preserves lower-level linguistic features while adapting task-specific layers. Implemented in `train.py` via `_get_param_groups_layerwise()`.

- **LoRA fine-tuned (LLaMA2):** Low-Rank Adaptation — only small adapter matrices are trained instead of the full model. Reduces memory and training time while keeping the base LLaMA weights frozen. Uses `peft` library in `model_utils.py`.

---

## 🧪 Tests
```bash
pytest tests/ -v                    # all tests
pytest tests/ -v -m "not slow"      # skip slow integration test
```

## 📝 Notes
- **Reddit API**: Create a Reddit app at https://www.reddit.com/prefs/apps and set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`.
- **Reddit pseudo-labeling**: Use `--pseudo-label` with ingest reddit. Add `--min-confidence 0.7` to pseudo_label to filter low-confidence labels.
- **LLaMA2**: Uses LoRA (peft). Supports TinyLlama (no approval), Llama-2 (may need HF approval).
- **Layer-wise LR decay**: Enabled by default (`--no-llrd` to disable).
- **AWS Lambda**: Package `lambda_handler.py` + model; set `MODEL_PATH`. Use API Gateway to invoke.
- **SageMaker**: Use `sagemaker_train.py` as entrypoint; upload data to S3 channels `train`/`validation`.
- **AWS S3**: `python -m scripts.aws_utils upload ./outputs/biasguard s3://bucket/models/`
