#!/usr/bin/env python3
"""Download GGUF models from S3 to local models/ (SageMaker backend)."""
import argparse
from pathlib import Path

import boto3
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def _load_config() -> dict:
    cfg_path = ROOT / "config" / "sagemaker.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text()) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "files", nargs="*",
        help="GGUF filenames (e.g. llama3.1-8b-educator-Q4_K_M.gguf). If empty, all.",
    )
    args = ap.parse_args()

    cfg = _load_config()
    bucket = cfg.get("s3_bucket")
    region = cfg.get("region", "us-east-1")
    if not bucket:
        raise ValueError("s3_bucket required in config/sagemaker.yaml")

    s3 = boto3.client("s3", region_name=region)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.files:
        keys = [f"gguf/{f}" for f in args.files]
    else:
        paginator = s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=bucket, Prefix="gguf/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".gguf"):
                    keys.append(obj["Key"])

    for key in keys:
        name = Path(key).name
        if "-trained" not in name:
            stem = Path(key).stem
            suffix = Path(key).suffix
            name = f"{stem}-trained{suffix}"
        local = MODELS_DIR / name
        s3.download_file(bucket, key, str(local))
        print(f"Downloaded {key} -> {local}")


if __name__ == "__main__":
    main()
