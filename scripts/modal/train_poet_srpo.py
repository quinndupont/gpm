#!/usr/bin/env python3
"""Modal: SRPO (Stage 2) for poet model — self-refinement policy optimization."""
from pathlib import Path

import modal

_CONFIG = Path("config/srpo_training.yaml")
if not _CONFIG.exists():
    _p = Path(__file__).resolve()
    _CONFIG = (
        _p.parents[2] / "config" / "srpo_training.yaml"
        if len(_p.parents) > 2
        else _CONFIG
    )

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
        "pronouncing>=0.2",
    )
    .add_local_file(str(_CONFIG), "/config/srpo_training.yaml")
    .add_local_dir(str(_ROOT / "scripts" / "training"), "/opt/gpm/scripts/training")
    .add_local_dir(str(_ROOT / "scripts" / "eval"), "/opt/gpm/scripts/eval")
    .add_local_dir(str(_ROOT / "models" / "prompts"), "/opt/gpm/models/prompts")
    .env({"PYTHONPATH": "/opt/gpm"})
)

app = modal.App("poetry-poet-srpo")
data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=12 * 3600,
    volumes={"/vol/data": data_vol, "/vol/checkpoints": checkpoint_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_poet_srpo(
    num_epochs_override: int | None = None,
    base_model_override: str | None = None,
    sft_checkpoint: str | None = None,
    train_filename: str = "trajectories_v2.jsonl",
):
    """Run SRPO training on Modal.

    Args:
        num_epochs_override: Override epochs from config
        base_model_override: Override base model from config
        sft_checkpoint: Path to Stage 1 SFT checkpoint
        train_filename: Name of SRPO trajectories file in /vol/data/srpo_training/
    """
    from scripts.training.srpo_train import run_srpo_training

    data_vol.reload()
    checkpoint_vol.reload()

    sft_path = Path(sft_checkpoint or "/vol/checkpoints/poet/final")
    if not sft_path.exists():
        raise FileNotFoundError(
            f"SFT checkpoint not found: {sft_path}. Run Stage 1 (train_poet.py) first."
        )

    # SRPO trajectories should be in /vol/data/srpo_training/
    data_dir = Path("/vol/data/srpo_training")
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / train_filename
    if not train_file.exists():
        raise FileNotFoundError(
            f"SRPO training data not found: {train_file}. "
            "Run generate_srpo_data.py first."
        )

    path = run_srpo_training(
        config_path=Path("/config/srpo_training.yaml"),
        sft_checkpoint=sft_path,
        data_dir=data_dir,
        train_filename=train_filename,
        checkpoint_dir=Path("/vol/checkpoints/poet_srpo"),
        num_epochs_override=num_epochs_override,
        base_model_override=base_model_override,
    )
    checkpoint_vol.commit()
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-epochs-override", type=int, default=None)
    ap.add_argument(
        "--base-model", type=str, default=None, dest="base_model_override",
        help="HuggingFace base model ID",
    )
    ap.add_argument(
        "--sft-checkpoint", type=str, default=None,
        help="Path to SFT checkpoint (default: /vol/checkpoints/poet/final)",
    )
    ap.add_argument(
        "--train-file", type=str, default="trajectories_v2.jsonl",
        help="SRPO trajectories filename",
    )
    a = ap.parse_args()
    with app.run():
        path = train_poet_srpo.remote(
            num_epochs_override=a.num_epochs_override,
            base_model_override=a.base_model_override,
            sft_checkpoint=a.sft_checkpoint,
            train_filename=a.train_file,
        )
        print(f"Done: {path}")
