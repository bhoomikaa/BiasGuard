#!/usr/bin/env python3
"""
BiasGuard - Real-time Inference Script

Loads a fine-tuned model and runs bias detection on new text.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(model_path: str):
    """Load tokenizer and model from saved checkpoint (supports PEFT adapters)."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    try:
        from peft import PeftModel
        adapter_config = model_path / "adapter_config.json"
        if adapter_config.exists():
            import json
            cfg = json.loads(adapter_config.read_text())
            base_name = cfg.get("base_model_name_or_path", str(model_path))
            num_labels = 2
            if (model_path / "config.json").exists():
                cfg_json = json.loads((model_path / "config.json").read_text())
                num_labels = cfg_json.get("num_labels", 2)
            model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=num_labels)
            model = PeftModel.from_pretrained(model, model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except (ImportError, Exception):
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def predict(text: str, tokenizer, model, device: str, max_length: int = 128) -> dict:
    """Single-text inference."""
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = int(logits.argmax(dim=-1).cpu().item())
    return {"prediction": pred, "probabilities": probs.tolist(), "label": "biased" if pred == 1 else "neutral"}


def main():
    parser = argparse.ArgumentParser(description="BiasGuard inference")
    parser.add_argument("--model", default="outputs/biasguard", help="Path to fine-tuned model")
    parser.add_argument("--text", help="Single text to classify")
    parser.add_argument("--file", help="Text file with one line per sample")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Provide --text or --file. Example: --text 'Sample text here'")

    tokenizer, model, device = load_model(args.model)

    if args.text:
        out = predict(args.text, tokenizer, model, device)
        print(out)
    elif args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if line:
                    out = predict(line, tokenizer, model, device)
                    preview = line[:80] + "..." if len(line) > 80 else line
                    print(f"{out['label']}\t{out['probabilities'][1]:.3f}\t{preview}")


if __name__ == "__main__":
    main()
