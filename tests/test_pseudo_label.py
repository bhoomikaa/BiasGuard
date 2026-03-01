"""Tests for pseudo_label script."""
import json
from pathlib import Path

import pytest


def test_pseudo_label_import():
    from scripts.pseudo_label import pseudo_label
    assert callable(pseudo_label)


def test_pseudo_label_missing_file():
    from scripts.pseudo_label import pseudo_label
    with pytest.raises(FileNotFoundError):
        pseudo_label("/nonexistent/file.jsonl", "/tmp/out.jsonl")


def test_pseudo_label_no_text_column(tmp_path):
    from scripts.pseudo_label import pseudo_label
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"foo": 1}\n')
    with pytest.raises(ValueError, match="text"):
        pseudo_label(str(bad), str(tmp_path / "out.jsonl"))
