#!/usr/bin/env python3
"""
BiasGuard - Preprocessing Script

Prepares text data for bias-detection training: cleaning, tokenization prep, train/val split.
"""

import argparse
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, remove URLs, reduce repetition."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.strip()
    t = re.sub(r"https?://\S+", "[URL]", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(.)\1{3,}", r"\1\1", t)  # reduce repeated chars
    return t[:2000].strip()


def preprocess(
    input_path: str,
    output_dir: str = "data/processed",
    text_col: str = "text",
    label_col: str = "label",
    val_ratio: float = 0.15,
    min_len: int = 10,
    seed: int = 42,
) -> Tuple[int, int]:
    """
    Load JSONL/CSV, clean text, filter short samples, split train/val.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_json(input_path, lines=True)

    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}'. Columns: {list(df.columns)}")

    df = df.rename(columns={text_col: "text"})
    if label_col in df.columns:
        df = df.rename(columns={label_col: "label"})
    else:
        df["label"] = 0

    df["text"] = df["text"].astype(str).apply(clean_text)
    df = df[df["text"].str.len() >= min_len].dropna(subset=["text", "label"])
    df = df[["text", "label"]].drop_duplicates(subset=["text"])

    if len(df) == 0:
        raise ValueError("No valid samples after preprocessing")

    # Stratify only when each class has at least 2 samples (sklearn requirement)
    stratify_arr = None
    if df["label"].nunique() > 1:
        counts = df["label"].value_counts()
        if counts.min() >= 2:
            stratify_arr = df["label"]
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=stratify_arr
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    val_path = out / "val.jsonl"

    train_df.to_json(train_path, orient="records", lines=True)
    val_df.to_json(val_path, orient="records", lines=True)

    print(f"Preprocessed: {len(train_df)} train, {len(val_df)} val -> {output_dir}")
    return len(train_df), len(val_df)


def main():
    parser = argparse.ArgumentParser(description="BiasGuard preprocessing")
    parser.add_argument("input", help="Path to ingested JSONL/CSV (from ingest.py)")
    parser.add_argument("-o", "--output-dir", default="data/processed")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    preprocess(
        args.input,
        args.output_dir,
        args.text_col,
        args.label_col,
        args.val_ratio,
        args.min_len,
        args.seed,
    )


if __name__ == "__main__":
    main()
