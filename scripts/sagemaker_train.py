#!/usr/bin/env python3
"""
BiasGuard - SageMaker training entrypoint

SageMaker expects this script in the container. Data is in /opt/ml/input/data/train/
and /opt/ml/input/data/validation/. Output to /opt/ml/model/.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args, _ = parser.parse_known_args()

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    train_path = os.path.join(train_dir, "train.jsonl")
    val_path = os.path.join(val_dir, "val.jsonl")
    if not os.path.exists(val_path):
        val_path = train_path

    _dir = os.path.dirname(os.path.abspath(__file__))
    if _dir not in sys.path:
        sys.path.insert(0, _dir)
    from train import train  # noqa: E402

    train(
        model_name=args.model,
        train_path=train_path,
        val_path=val_path,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
