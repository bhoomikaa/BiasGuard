#!/usr/bin/env python3
"""
BiasGuard - Reddit Data Ingestion Script

Scrapes Reddit data for bias-related content. Supports:
- PRAW (Reddit API) for live Reddit scraping
- HuggingFace datasets as fallback (e.g. political sentiment, hate speech)
- Local CSV/JSON files
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def ingest_from_reddit(
    subreddits: List[str],
    limit_per_sub: int = 1000,
    output_path: str = "data/raw/reddit_posts.jsonl",
) -> pd.DataFrame:
    """
    Scrape Reddit posts using PRAW. Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.
    """
    try:
        import praw
    except ImportError:
        raise ImportError("Install praw: pip install praw")

    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise ValueError(
            "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables for Reddit ingestion"
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent="BiasGuard/1.0 (bias detection research)",
    )

    rows = []
    for sub in subreddits:
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.hot(limit=limit_per_sub):
                rows.append(
                    {
                        "text": f"{post.title} {post.selftext or ''}".strip()[:2000],
                        "subreddit": sub,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": post.created_utc,
                        "id": post.id,
                    }
                )
        except Exception as e:
            print(f"Warning: could not fetch r/{sub}: {e}")

    df = pd.DataFrame(rows)
    df["label"] = 0  # Reddit data is unlabeled; use for inference or pseudo-labeling
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True)
    print(f"Saved {len(df)} Reddit posts to {output_path}")
    return df


def ingest_from_huggingface(
    dataset_name: str = "google/civil_comments",
    config: Optional[str] = None,
    split: str = "train",
    text_col: str = "text",
    label_col: Optional[str] = "toxicity",
    output_path: str = "data/raw/hf_ingested.jsonl",
    max_samples: Optional[int] = 5000,
) -> pd.DataFrame:
    """
    Load bias-related dataset from HuggingFace (e.g. civil_comments, hate_speech).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, name=config, split=split) if config else load_dataset(dataset_name, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Map to text + label
    texts = list(ds[text_col])
    n = len(texts)
    if label_col and label_col in ds.column_names:
        raw_labels = ds[label_col]
        # Binarize continuous scores (e.g. toxicity 0-1 -> 0/1)
        labels = []
        for x in raw_labels:
            if x is None or (isinstance(x, float) and (x != x)):  # NaN check
                labels.append(0)
            elif isinstance(x, float):
                labels.append(1 if x >= 0.5 else 0)
            else:
                labels.append(1 if x >= 1 else 0)
    else:
        labels = [0] * n

    df = pd.DataFrame({"text": texts, "label": labels})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True)
    print(f"Saved {len(df)} samples from {dataset_name} to {output_path}")
    return df


def ingest_from_file(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """Load from local CSV or JSON/JSONL. Expects 'text' and optionally 'label' columns."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        with open(input_path) as f:
            data = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(data)

    required = "text" in df.columns
    if not required:
        raise ValueError("Input must have a 'text' column. Rename your text column to 'text'.")

    if "label" not in df.columns:
        df["label"] = 0

    out = output_path or f"data/raw/loaded_{path.stem}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", lines=True)
    print(f"Saved {len(df)} samples to {out}")
    return df


def main():
    parser = argparse.ArgumentParser(description="BiasGuard data ingestion")
    sub = parser.add_subparsers(dest="source", required=True)

    reddit = sub.add_parser("reddit", help="Scrape Reddit (requires PRAW + env vars)")
    reddit.add_argument("--subreddits", nargs="+", default=["politics", "news", "worldnews"])
    reddit.add_argument("--limit", type=int, default=500)
    reddit.add_argument("-o", "--output", default="data/raw/reddit_posts.jsonl")
    reddit.add_argument("--pseudo-label", action="store_true", help="Pseudo-label with zero-shot model (run after scrape)")

    hf = sub.add_parser("huggingface", help="Load from HuggingFace datasets")
    hf.add_argument("--dataset", default="google/civil_comments", help="e.g. google/civil_comments (has text, toxicity)")
    hf.add_argument("--config", default=None)
    hf.add_argument("--split", default="train")
    hf.add_argument("--text-col", default="text")
    hf.add_argument("--label-col", default="toxicity")
    hf.add_argument("--max-samples", type=int, default=5000)
    hf.add_argument("-o", "--output", default="data/raw/hf_ingested.jsonl")

    local = sub.add_parser("file", help="Load from local CSV/JSONL")
    local.add_argument("path", help="Path to CSV or JSONL file")
    local.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    if args.source == "reddit":
        ingest_from_reddit(args.subreddits, args.limit, args.output)
        if getattr(args, "pseudo_label", False):
            from scripts.pseudo_label import pseudo_label
            base = args.output.rsplit(".", 1)[0] if "." in args.output else args.output
            labeled_path = f"{base}_labeled.jsonl"
            pseudo_label(args.output, labeled_path)
            print(f"Pseudo-labeled output: {labeled_path}")
    elif args.source == "huggingface":
        ingest_from_huggingface(
            args.dataset,
            args.config,
            args.split,
            args.text_col,
            args.label_col,
            args.output,
            args.max_samples,
        )
    else:
        ingest_from_file(args.path, args.output)


if __name__ == "__main__":
    main()
