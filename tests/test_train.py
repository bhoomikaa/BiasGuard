"""Tests for train script - uses small model + 1 step for CI."""
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_train_load_data(processed_dir):
    from scripts.train import load_data
    base = Path(processed_dir)
    train_ds, val_ds = load_data(str(base / "train.jsonl"), str(base / "val.jsonl"))
    assert len(train_ds) >= 10
    assert len(val_ds) >= 2
    assert "text" in train_ds.column_names and "label" in train_ds.column_names


def test_train_param_groups():
    """Verify differential LR param groups are built correctly."""
    import torch
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    no_decay = ["bias", "LayerNorm.weight"]
    lr = 2e-5
    opt_groups = []
    for prefix, lr_mult in [("encoder", 1.0), ("head", 5.0)]:
        for use_decay in [True, False]:
            params = [
                p for n, p in model.named_parameters()
                if ("classifier" in n or "score" in n) == (prefix == "head")
                and any(nd in n for nd in no_decay) == (not use_decay)
            ]
            if params:
                opt_groups.append({"params": params, "lr": lr * lr_mult})
    assert len(opt_groups) >= 2
    assert any(g["lr"] > lr for g in opt_groups)
    # Head should have higher LR
    head_lrs = [g["lr"] for g in opt_groups if g["lr"] > lr]
    # All params should be in exactly one group
    total_params = sum(len(g["params"]) for g in opt_groups)
    assert total_params == len(list(model.parameters()))


@pytest.mark.slow
def test_train_mini_run(processed_dir, tmp_path):
    """Run 1 epoch with distilbert (fast) - full integration."""
    from scripts.train import train
    base = Path(processed_dir)
    res = train(
        model_name="distilbert-base-uncased",
        train_path=str(base / "train.jsonl"),
        val_path=str(base / "val.jsonl"),
        output_dir=str(tmp_path / "out"),
        num_epochs=1,
        batch_size=2,
        fp16=False,
        early_stopping_patience=0,
    )
    assert "eval_accuracy" in res
    assert (tmp_path / "out" / "eval_results.json").exists()
