#!/usr/bin/env python3
"""
BiasGuard - Screenshot-ready project showcase
Run: python -m scripts.showcase
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def banner():
    return r"""
╔══════════════════════════════════════════════════════════════════════════╗
║                         BiasGuard v1.0                                    ║
║     Automated Bias Detection in Text • BERT • RoBERTa • LLaMA2           ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • End-to-end pipeline: Ingest → Preprocess → Train → Evaluate → Deploy  ║
║  • Reddit + HuggingFace data • Zero-shot pseudo-labeling                 ║
║  • Layer-wise LR decay • LoRA • Mixed-precision • AWS-ready              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def main():
    print(banner())
    print("\n▶ Project structure")
    print("─" * 60)
    for p in ["scripts/ingest.py", "scripts/preprocess.py", "scripts/train.py",
              "scripts/inference.py", "scripts/evaluate.py", "scripts/api_server.py",
              "scripts/pseudo_label.py", "scripts/lambda_handler.py", "scripts/sagemaker_train.py",
              "tests/", "Dockerfile", "run_pipeline.sh"]:
        exists = "✓" if (ROOT / p.replace("/", os.sep)).exists() else "✗"
        print(f"  {exists}  {p}")
    print("\n▶ Smoke tests")
    print("─" * 60)
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-v", "--tb=no", "-q"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=15
    )
    for line in (r.stdout or "").splitlines():
        if line.strip() and "PASSED" in line:
            print(f"  {line.strip()}")
    if r.returncode == 0:
        print("  ✓ All smoke tests passed")
    else:
        print("  (run: pytest tests/test_smoke.py -v)")
    print("\n▶ Quick commands")
    print("─" * 60)
    cmds = [
        ("Ingest", "python -m scripts.ingest huggingface --max-samples 1000 -o data/raw/in.jsonl"),
        ("Preprocess", "python -m scripts.preprocess data/raw/in.jsonl -o data/processed"),
        ("Train", "python -m scripts.train --model distilbert-base-uncased -o outputs/biasguard"),
        ("Evaluate", "python -m scripts.evaluate --model outputs/biasguard -o results.json"),
        ("API server", "python -m scripts.api_server --model outputs/biasguard --port 8000"),
    ]
    for name, cmd in cmds:
        print(f"  {name:12} {cmd}")
    print("\n" + "=" * 60)
    print("  BiasGuard • Python 3.9+ • PyTorch • Hugging Face • AWS")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
