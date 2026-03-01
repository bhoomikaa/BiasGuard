"""Tests for ingest script - file ingest only (no network)."""
import json
from pathlib import Path

import pytest

from scripts.ingest import ingest_from_file


def test_ingest_from_jsonl(tmp_path, sample_jsonl):
    out = tmp_path / "out.jsonl"
    df = ingest_from_file(sample_jsonl, str(out))
    assert out.exists()
    assert len(df) == 20
    with open(out) as f:
        row = json.loads(f.readline())
    assert "text" in row and "label" in row


def test_ingest_from_csv(tmp_path, sample_csv):
    out = tmp_path / "out.jsonl"
    df = ingest_from_file(sample_csv, str(out))
    assert len(df) == 3


def test_ingest_missing_file():
    with pytest.raises(FileNotFoundError):
        ingest_from_file("/nonexistent/file.jsonl")


def test_ingest_no_text_column(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"foo": 1}\n')
    with pytest.raises(ValueError, match="text"):
        ingest_from_file(str(bad))
