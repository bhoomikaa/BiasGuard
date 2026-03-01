#!/usr/bin/env python3
"""
BiasGuard - Evaluation on standard benchmark

Runs model on civil_comments or similar test set and reports accuracy, F1, etc.
"""
import argparse
import json
from pathlib import Path


def evaluate(
    model_path: str,
    dataset: str = "google/civil_comments",
    split: str = "validation",
    max_samples: int = 1000,
    batch_size: int = 32,
    output_path=None,
) -> dict:
    """Evaluate model on benchmark. Returns metrics dict."""
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from evaluate import load as load_metric

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ds = load_dataset(dataset, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts = ds["text"]
    labels = [1 if x >= 0.5 else 0 for x in ds["toxicity"]]

    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    acc_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    results = {
        "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"],
        "n_samples": len(preds),
        "dataset": dataset,
    }
    print(json.dumps(results, indent=2))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/biasguard")
    parser.add_argument("--dataset", default="google/civil_comments")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()
    evaluate(
        args.model,
        args.dataset,
        args.split,
        args.max_samples,
        args.batch_size,
        args.output,
    )


if __name__ == "__main__":
    main()
