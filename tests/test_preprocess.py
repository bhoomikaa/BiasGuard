"""Tests for preprocess script - no torch/transformers."""
import json
import tempfile
from pathlib import Path

import pytest

from scripts.preprocess import clean_text, preprocess


def test_clean_text():
    assert clean_text("  normal  text  ") == "normal text"
    assert clean_text("check https://example.com here") == "check [URL] here"
    assert clean_text("") == ""
    assert clean_text("x" * 3000).endswith("x" * 2000)  # truncated to 2000


def test_preprocess_jsonl(tmp_path, sample_jsonl):
    out = tmp_path / "out"
    n_train, n_val = preprocess(sample_jsonl, output_dir=str(out), val_ratio=0.2, seed=42)
    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()
    assert n_train + n_val == 20
    with open(out / "train.jsonl") as f:
        first = json.loads(f.readline())
    assert "text" in first and "label" in first


def test_preprocess_csv(tmp_path, sample_csv):
    out = tmp_path / "out"
    preprocess(sample_csv, output_dir=str(out))
    assert (out / "train.jsonl").exists()


def test_preprocess_missing_file():
    with pytest.raises(FileNotFoundError):
        preprocess("/nonexistent/path.jsonl")
