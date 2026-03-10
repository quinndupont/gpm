#!/usr/bin/env python3
"""Modal: QLoRA fine-tune educator model. S3.2"""
from pathlib import Path

import modal

# Config path: works when run from project root (local or Modal workspace)
_CONFIG = Path("config/educator_training.yaml")
if not _CONFIG.exists():
    _p = Path(__file__).resolve()
    _CONFIG = _p.parents[1] / "config" / "educator_training.yaml" if len(_p.parents) > 1 else _CONFIG

_ROOT = Path(__file__).resolve().parents[2]
VOLUME_NAME = "poetry-data"
CHECKPOINT_VOLUME = "poetry-checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1",
        "git+https://github.com/huggingface/transformers.git",
        "peft>=0.10",
        "trl>=0.8",
        "bitsandbytes>=0.43",
        "datasets>=2.18",
        "accelerate>=0.28",
        "pyyaml>=6.0",
    )
    .add_local_file(str(_CONFIG), "/config/educator_training.yaml")
    .add_local_dir(str(_ROOT / "scripts" / "training"), "/opt/gpm/scripts/training")
    .env({"PYTHONPATH": "/opt/gpm"})
)

app = modal.App("poetry-educator-train")
data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 3600,
    volumes={"/vol/data": data_vol, "/vol/checkpoints": checkpoint_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_educator(num_epochs_override: int | None = None, base_model_override: str | None = None):
    from scripts.training.qlora_train import run_qlora_training

    data_vol.reload()
    path = run_qlora_training(
        config_path=Path("/config/educator_training.yaml"),
        data_dir=Path("/vol/data"),
        train_filename="educator_train.jsonl",
        valid_filename="educator_valid.jsonl",
        checkpoint_dir=Path("/vol/checkpoints/educator"),
        num_epochs_override=num_epochs_override,
        base_model_override=base_model_override,
    )
    checkpoint_vol.commit()
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-epochs-override", type=int, default=None)
    ap.add_argument("--base-model", type=str, default=None, dest="base_model_override", help="HuggingFace base model ID")
    a = ap.parse_args()
    with app.run():
        path = train_educator.remote(num_epochs_override=a.num_epochs_override, base_model_override=a.base_model_override)
        print(f"Done: {path}")
