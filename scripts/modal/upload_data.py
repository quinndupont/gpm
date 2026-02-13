#!/usr/bin/env python3
"""Upload training data to Modal volume."""
import modal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"

VOLUME_NAME = "poetry-data"


def main():
    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    files = [
        (EDUCATOR_TRAINING / "train.jsonl", "educator_train.jsonl"),
        (EDUCATOR_TRAINING / "valid.jsonl", "educator_valid.jsonl"),
        (POET_TRAINING / "train.jsonl", "poet_train.jsonl"),
        (POET_TRAINING / "valid.jsonl", "poet_valid.jsonl"),
    ]

    with vol.batch_upload(force=True) as batch:
        for local, remote in files:
            if local.exists():
                batch.put_file(str(local), remote)
                print(f"Uploaded {local} -> {remote}")
            else:
                print(f"Skip {local} (not found)")

    print(f"Done. Volume: {VOLUME_NAME}")


if __name__ == "__main__":
    main()
