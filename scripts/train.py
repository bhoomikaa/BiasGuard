#!/usr/bin/env python3
"""
BiasGuard - Model Training Script

Fine-tunes BERT, RoBERTa, DistilBERT, or LLaMA2 for bias detection with:
- Mixed-precision (fp16) training
- Full layer-wise learning rate decay (LLRD)
- LoRA for LLaMA2 (parameter-efficient)
- Early stopping
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
import evaluate


def load_data(train_path: str, val_path: str) -> tuple[Dataset, Dataset]:
    """Load preprocessed JSONL into HuggingFace Dataset."""
    import pandas as pd

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)


def tokenize_dataset(dataset, tokenizer, max_length: int = 128):
    def fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
        )
        out["labels"] = batch["label"]
        return out

    return dataset.map(fn, batched=True, remove_columns=dataset.column_names)


def _get_param_groups_layerwise(model, lr_base: float, decay_factor: float = 0.95, no_decay: Optional[List[str]] = None):
    """Build optimizer param groups with layer-wise learning rate decay (LLRD).
    Top layers get lr_base, earlier layers get lr_base * decay^depth.
    """
    no_decay = no_decay or ["bias", "LayerNorm.weight", "layer_norm.weight"]
    config = getattr(model, "config", None) or getattr(getattr(model, "base_model", model), "config", None)
    num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", 12) if config else 12
    layer_pattern = re.compile(r"encoder\.layer\.(\d+)|layers\.(\d+)|model\.layers\.(\d+)")

    def _layer_idx(name: str) -> int:
        m = layer_pattern.search(name)
        if m:
            for g in m.groups():
                if g is not None:
                    return int(g)
        if "classifier" in name or "score" in name or "lm_head" in name:
            return num_layers  # top
        return -1  # embeddings

    grouped = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        idx = _layer_idx(n)
        decay = 0.0 if any(nd in n for nd in no_decay) else 0.01
        depth = num_layers - idx
        lr = lr_base * (decay_factor ** max(0, depth))
        key = (lr, decay)
        grouped.setdefault(key, []).append(p)

    return [{"params": v, "lr": k[0], "weight_decay": k[1]} for k, v in grouped.items()]


def _get_param_groups_simple(model, lr: float):
    """Fallback: head 5x LR, encoder 1x."""
    no_decay = ["bias", "LayerNorm.weight"]
    opt_groups = []
    for prefix, lr_mult in [("encoder", 1.0), ("head", 5.0)]:
        for use_decay in [True, False]:
            params = [
                p for n, p in model.named_parameters()
                if ("classifier" in n or "score" in n) == (prefix == "head")
                and any(nd in n for nd in no_decay) == (not use_decay)
            ]
            if params:
                opt_groups.append({
                    "params": params,
                    "lr": lr * lr_mult,
                    "weight_decay": 0.01 if use_decay else 0.0
                })
    if not opt_groups:
        return [{"params": list(model.parameters()), "lr": lr, "weight_decay": 0.01}]
    return opt_groups


def train(
    model_name: str = "bert-base-uncased",
    train_path: str = "data/processed/train.jsonl",
    val_path: str = "data/processed/val.jsonl",
    output_dir: str = "outputs/biasguard",
    num_epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 128,
    fp16: bool = True,
    early_stopping_patience: int = 2,
    seed: int = 42,
    use_llrd: bool = True,
    use_lora: bool = True,
) -> dict:
    """Run training with mixed precision, LLRD, and optional LoRA."""
    train_ds, val_ds = load_data(train_path, val_path)
    num_labels = max(2, len(set(train_ds["label"])))

    is_llama = any(x in model_name.lower() for x in ["llama", "mistral", "phi", "tinyllama"])
    if is_llama:
        from scripts.model_utils import create_model_and_tokenizer
        model, tokenizer = create_model_and_tokenizer(model_name, num_labels, use_lora)
        max_length = min(max_length, 512)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    train_tok = tokenize_dataset(train_ds, tokenizer, max_length)
    val_tok = tokenize_dataset(val_ds, tokenizer, max_length)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    cfg = getattr(model, "config", None) or getattr(getattr(model, "base_model", model), "config", None)
    num_hidden = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", 12) if cfg else 12
    if use_llrd and cfg and num_hidden:
        opt_groups = _get_param_groups_layerwise(model, lr, decay_factor=0.95)
    else:
        opt_groups = _get_param_groups_simple(model, lr)

    optimizer = AdamW(opt_groups)
    num_devices = max(1, torch.cuda.device_count())
    num_update_steps = max(1, (len(train_tok) // (batch_size * num_devices)) * num_epochs)
    warmup_steps = int(0.1 * num_update_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_update_steps
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        fp16=fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=seed,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)] if early_stopping_patience else []

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    eval_res = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = Path(output_dir) / "eval_results.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_res, f, indent=2)
    print(f"Evaluation: {eval_res}")
    return eval_res


def main():
    parser = argparse.ArgumentParser(description="BiasGuard training")
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        help="bert-base-uncased | roberta-base | distilbert-base-uncased | TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--val", default="data/processed/val.jsonl")
    parser.add_argument("-o", "--output", default="outputs/biasguard")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-llrd", action="store_true", help="Disable layer-wise LR decay")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA for LLaMA")
    parser.add_argument("--early-stop", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        args.model,
        args.train,
        args.val,
        args.output,
        args.epochs,
        args.batch_size,
        args.lr,
        args.max_length,
        not args.no_fp16,
        args.early_stop,
        args.seed,
        use_llrd=not args.no_llrd,
        use_lora=not args.no_lora,
    )


if __name__ == "__main__":
    main()
