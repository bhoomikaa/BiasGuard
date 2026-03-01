"""Pytest fixtures - minimal synthetic data for fast tests without network."""
import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_jsonl(tmp_path):
    """Minimal labeled JSONL (20 samples) for pipeline tests."""
    data = [
        {"text": "This is a neutral and reasonable comment.", "label": 0},
        {"text": "That toxic and hateful rhetoric has no place here.", "label": 1},
        {"text": "I disagree but respect your perspective.", "label": 0},
        {"text": "You are an idiot and everyone hates you.", "label": 1},
    ] * 5  # 20 rows
    path = tmp_path / "sample.jsonl"
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")
    return str(path)


@pytest.fixture
def sample_csv(tmp_path):
    """Minimal CSV for ingest/preprocess tests."""
    path = tmp_path / "sample.csv"
    path.write_text("text,label\n"
                    "Clean neutral text here,0\n"
                    "Aggressive biased language,1\n"
                    "Another neutral example,0\n")
    return str(path)


@pytest.fixture
def processed_dir(tmp_path, sample_jsonl):
    """Preprocessed train/val split from sample data."""
    import pandas as pd
    df = pd.read_json(sample_jsonl, lines=True)
    train_df = df.iloc[:16]
    val_df = df.iloc[16:]
    (tmp_path / "train.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict()) for _, r in train_df.iterrows()) + "\n"
    )
    (tmp_path / "val.jsonl").write_text(
        "\n".join(json.dumps(r.to_dict()) for _, r in val_df.iterrows()) + "\n"
    )
    return str(tmp_path)
