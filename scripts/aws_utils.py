#!/usr/bin/env python3
"""
BiasGuard - AWS utilities for S3 and SageMaker integration

Upload data/models to S3, prepare for SageMaker training.
"""

import argparse
import os
from pathlib import Path


def upload_to_s3(local_path: str, s3_uri: str) -> str:
    """Upload file or directory to S3. Requires boto3 and AWS credentials."""
    try:
        import boto3
    except ImportError:
        raise ImportError("Install boto3: pip install boto3")

    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(local_path)

    bucket, *key_parts = s3_uri.replace("s3://", "").split("/")
    key_prefix = "/".join(key_parts) if key_parts else ""

    s3 = boto3.client("s3")
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file():
                rel = f.relative_to(path)
                s3_key = f"{key_prefix}/{rel}".lstrip("/")
                s3.upload_file(str(f), bucket, s3_key)
    else:
        s3_key = key_prefix or path.name
        s3.upload_file(str(path), bucket, s3_key)

    return f"s3://{bucket}/{key_prefix}"


def main():
    parser = argparse.ArgumentParser(description="BiasGuard AWS utils")
    parser.add_argument("action", choices=["upload"])
    parser.add_argument("local_path", help="Local file or directory")
    parser.add_argument("s3_uri", help="s3://bucket/key")
    args = parser.parse_args()

    if args.action == "upload":
        result = upload_to_s3(args.local_path, args.s3_uri)
        print(f"Uploaded to {result}")


if __name__ == "__main__":
    main()
