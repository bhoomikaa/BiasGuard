#!/usr/bin/env python3
"""
BiasGuard - Pseudo-labeling for unlabeled Reddit data

Uses a zero-shot classification model to assign bias/toxicity labels to unlabeled text.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def pseudo_label(
    input_path,
    output_path,
    model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    labels=("neutral", "biased/toxicity"),
    text_col="text",
    batch_size=16,
    threshold=0.5,
    min_confidence=None,
    drop_confidence_col=False,
) -> int:
    """
    Label unlabeled JSONL using zero-shot classification.
    Returns count of labeled samples.
    """
    from transformers import pipeline

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_json(input_path, lines=True)
    if text_col not in df.columns:
        raise ValueError(f"Expected column '{text_col}'. Columns: {list(df.columns)}")

    texts = df[text_col].astype(str).tolist()
    if not texts:
        raise ValueError("No text samples to label")

    import torch
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    labeled = []
    confidences = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = pipe(batch, candidate_labels=list(labels), multi_label=False)
        for res in out:
            pred = res["labels"][0]
            score = res["scores"][0]
            label = 1 if pred == labels[1] and score >= threshold else 0
            labeled.append(label)
            confidences.append(score)

    df["label"] = labeled
    df["confidence"] = [round(c, 4) for c in confidences]
    if min_confidence is not None:
        before = len(df)
        df = df[df["confidence"] >= min_confidence].copy()
        print(f"Kept {len(df)}/{before} samples with confidence >= {min_confidence}")
    if drop_confidence_col:
        df = df.drop(columns=["confidence"], errors="ignore")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True)
    print(f"Pseudo-labeled {len(df)} samples -> {output_path}")
    return len(df)


def main():
    parser = argparse.ArgumentParser(description="BiasGuard pseudo-labeling")
    parser.add_argument("input", help="Unlabeled JSONL (e.g. from ingest reddit)")
    parser.add_argument("-o", "--output", default="data/raw/pseudo_labeled.jsonl")
    parser.add_argument("--model", default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
    parser.add_argument("--labels", nargs=2, default=["neutral", "biased/toxicity"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=None, help="Drop samples below this score")
    parser.add_argument("--drop-confidence-col", action="store_true", help="Omit confidence column from output")
    args = parser.parse_args()

    pseudo_label(
        args.input,
        args.output,
        args.model,
        tuple(args.labels),
        batch_size=args.batch_size,
        threshold=args.threshold,
        min_confidence=args.min_confidence,
        drop_confidence_col=args.drop_confidence_col,
    )


if __name__ == "__main__":
    main()
