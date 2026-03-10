#!/usr/bin/env python3
"""SageMaker HuggingFace training entry point. Reads SM_* env vars and runs QLoRA."""
import os
import sys
from pathlib import Path

# Add /opt/ml/code to path for qlora_train import
sys.path.insert(0, "/opt/ml/code")
from qlora_train import run_qlora_training

TASK_CONFIG = {
    "educator": (
        "educator_training.yaml", "educator_train.jsonl", "educator_valid.jsonl",
        "educator",
    ),
    "poet": ("poet_training.yaml", "poet_train.jsonl", "poet_valid.jsonl", "poet"),
    "rhyme": ("rhyme_training.yaml", "rhyme_train.jsonl", "rhyme_valid.jsonl", "poet_rhyme"),
}


def main():
    train_dir = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    task = os.environ.get("SM_HP_TASK", "educator")
    num_epochs_override = os.environ.get("SM_HP_NUM_EPOCHS_OVERRIDE")
    num_epochs = int(num_epochs_override) if num_epochs_override else None
    base_model_override = os.environ.get("SM_HP_BASE_MODEL") or None

    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}. Must be educator|poet|rhyme")
    config_name, train_file, valid_file, subdir = TASK_CONFIG[task]
    config_path = Path("/opt/ml/code/config") / config_name
    checkpoint_dir = model_dir / subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_qlora_training(
        config_path=config_path,
        data_dir=train_dir,
        train_filename=train_file,
        valid_filename=valid_file,
        checkpoint_dir=checkpoint_dir,
        num_epochs_override=num_epochs,
        base_model_override=base_model_override,
    )
    print(f"Training complete. Model saved to {checkpoint_dir}/final")


if __name__ == "__main__":
    main()
