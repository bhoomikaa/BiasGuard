#!/usr/bin/env python3
"""
BiasGuard - Fast smoke test (~2–5 min)

Runs a miniature pipeline: 50 samples, 1 epoch per model, prints metrics table.
Use:  python -m scripts.smoke_test
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

try:
    import torch
    import transformers
except ImportError:
    print("Missing torch/transformers. Run: pip install -r requirements.txt")
    sys.exit(1)

# Create minimal data (50 samples, no HuggingFace download)
DATA_RAW = ROOT / "data" / "raw" / "smoke_ingest.jsonl"
DATA_PROC = ROOT / "data" / "processed_smoke"

def run(cmd, desc):
    print(f"\n▶ {desc}")
    r = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if r.returncode != 0:
        print("  ✗ Failed")
        return False
    print("  ✓ Done")
    return True

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  BiasGuard - Smoke Test (~2–5 min)                                       ║
║  Mini pipeline: 50 samples, 1 epoch, 2 models                           ║
╚══════════════════════════════════════════════════════════════════════════╝
""")
    DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_RAW.exists():
        import json
        data = [{"text": "Neutral comment here.", "label": 0}, {"text": "Toxic hateful text.", "label": 1}] * 25
        with open(DATA_RAW, "w") as f:
            for r in data:
                f.write(json.dumps(r) + "\n")
        print("  Created data/raw/smoke_ingest.jsonl (50 samples)")
    run(f'{sys.executable} -m scripts.preprocess {DATA_RAW} -o {DATA_PROC}', "Preprocess")
    results = []
    for model_id, name in [("distilbert-base-uncased", "DistilBERT"), ("bert-base-uncased", "BERT")]:
        out = ROOT / "outputs" / f"smoke_{model_id.replace('/', '_')}"
        if run(f'{sys.executable} -m scripts.train --model {model_id} --train {DATA_PROC}/train.jsonl --val {DATA_PROC}/val.jsonl -o {out} --epochs 1 --batch-size 8 --no-fp16 --early-stop 0', f"Train {name}"):
            p = out / "eval_results.json"
            if p.exists():
                with open(p) as f:
                    r = json.load(f)
                results.append((name, r.get("eval_accuracy", 0)))
    print("\n╔════════════════════════════════════════╗")
    print("║  Smoke Test Results                    ║")
    print("╠════════════════════════════════════════╣")
    for name, acc in results:
        print(f"║  {name:<12}  {acc*100:.1f}% accuracy       ║")
    print("╚════════════════════════════════════════╝\n")

if __name__ == "__main__":
    main()
