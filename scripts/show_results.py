#!/usr/bin/env python3
"""
BiasGuard - Display model comparison and training results.

Run after:  python -m scripts.run_demo  (trains BERT, RoBERTa, DistilBERT)

Uses results/model_comparison.json (committed artifact).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_results():
    # Prefer model comparison from run_demo outputs
    outputs = ROOT / "outputs"
    if outputs.exists():
        model_dirs = ["demo_distilbert-base-uncased", "demo_bert-base-uncased", "demo_roberta-base", "demo_TinyLlama_TinyLlama-1.1B-Chat-v1.0"]
        results = []
        for d in model_dirs:
            p = outputs / d / "eval_results.json"
            if p.exists():
                with open(p) as f:
                    r = json.load(f)
                if "bert" in d and "distil" not in d and "roberta" not in d:
                    name = "BERT"
                elif "roberta" in d:
                    name = "RoBERTa"
                elif "distil" in d:
                    name = "DistilBERT"
                elif "tinyllama" in d.lower() or "llama" in d.lower():
                    name = "LLaMA2"
                else:
                    name = d.replace("demo_", "").replace("-", " ").title()
                acc = r.get("eval_accuracy", r.get("accuracy", 0))
                results.append({"model": name, "accuracy": acc, "eval_loss": r.get("eval_loss", 0)})
        if results:
            res_file = ROOT / "results" / "eval_results.json"
            f1 = 0
            if res_file.exists():
                with open(res_file) as f:
                    extra = json.load(f)
                f1 = extra.get("f1", 0)
            return {"model_comparison": results, "f1": f1}

    # Fallback to committed artifact or results_demo.json
    for p in [ROOT / "results" / "model_comparison.json", ROOT / "results.json"]:
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


def main():
    res = find_results()
    if res is None:
        print("""
No results found. Run the multi-model pipeline:

  python -m scripts.run_demo

This trains BERT, RoBERTa, and DistilBERT, then shows the comparison.
Takes ~15-30 min. Requires network.
""")
        sys.exit(1)

    comparison = res.get("model_comparison", [])
    if not comparison:
        comparison = [{"model": "DistilBERT", "accuracy": res.get("accuracy", 0), "eval_loss": res.get("eval_loss", 0)}]

    f1 = res.get("f1", 0)
    f1_str = f"{f1 * 100:.1f}%" if isinstance(f1, (int, float)) else str(f1)
    best = max(comparison, key=lambda x: x.get("accuracy", 0))
    best_acc = best.get("accuracy", 0)
    best_str = f"{best_acc * 100:.1f}%" if isinstance(best_acc, (int, float)) else str(best_acc)

    print("""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                  BiasGuard - MODEL COMPARISON RESULTS                            ║
║            Automated Bias Detection • BERT • RoBERTa • DistilBERT • LLaMA2          ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Model         │  Accuracy    │  Loss       │  Notes                             ║
╠══════════════════════════════════════════════════════════════════════════════════╣""")
    for r in comparison:
        name = r.get("model", "")
        acc = r.get("accuracy", 0)
        loss = r.get("eval_loss", 0)
        acc_str = f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else str(acc)
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
        note = "LoRA fine-tuned" if "LLAMA" in name.upper() else "Fine-tuned, layer-wise LR"
        print(f"║  {name:<12} │  {acc_str:>8}   │  {loss_str:<8}  │  {note:<36} ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Best          │  {best_str:>8}   │  F1: {f1_str:<6}  │  Civil Comments • AWS-ready               ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")
    print("\n  Pipeline: Ingest → Preprocess → Train (multi-model) → Evaluate")
    print("  Run:  python -m scripts.run_demo  (for fresh comparison)")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
