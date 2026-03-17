#!/usr/bin/env python3
"""Upload training data to S3 (SageMaker backend)."""
from pathlib import Path

import boto3
import yaml

ROOT = Path(__file__).resolve().parents[2]
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"
SRPO_TRAINING = ROOT / "data" / "srpo_training"

FILES = [
    (EDUCATOR_TRAINING / "train.jsonl", "data/educator_train.jsonl"),
    (EDUCATOR_TRAINING / "valid.jsonl", "data/educator_valid.jsonl"),
    (POET_TRAINING / "train.jsonl", "data/poet_train.jsonl"),
    (POET_TRAINING / "valid.jsonl", "data/poet_valid.jsonl"),
    # SRPO trajectories: prefer v2 if present, uploaded as trajectories.jsonl for SageMaker
    (
        SRPO_TRAINING / "trajectories_v2.jsonl"
        if (SRPO_TRAINING / "trajectories_v2.jsonl").exists()
        else SRPO_TRAINING / "trajectories.jsonl",
        "data/trajectories.jsonl",
    ),
]


def _load_config() -> dict:
    cfg_path = ROOT / "config" / "sagemaker.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text()) or {}


def main():
    cfg = _load_config()
    bucket = cfg.get("s3_bucket")
    region = cfg.get("region", "us-east-1")
    if not bucket:
        raise ValueError("s3_bucket in config/sagemaker.yaml")

    s3 = boto3.client("s3", region_name=region)
    for local, s3_key in FILES:
        if local.exists():
            s3.upload_file(str(local), bucket, s3_key)
            print(f"Uploaded {local} -> s3://{bucket}/{s3_key}")
        else:
            print(f"Skip {local} (not found)")
    print(f"Done. Bucket: {bucket}")


if __name__ == "__main__":
    main()
