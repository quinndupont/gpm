#!/usr/bin/env python3
"""Launch SageMaker HuggingFace training job for educator, poet, or rhyme."""
import argparse
import shutil
import tempfile
from pathlib import Path

import boto3
import yaml
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput

ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict:
    cfg_path = ROOT / "config" / "sagemaker.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text()) or {}


def _get_hf_token(cfg: dict, region: str) -> str | None:
    secret_name = cfg.get("hf_secret_name")
    if secret_name:
        try:
            sm = boto3.client("secretsmanager", region_name=region)
            r = sm.get_secret_value(SecretId=secret_name)
            return r.get("SecretString")
        except Exception as e:
            print(f"Warning: Could not get HF token from Secrets Manager: {e}")
    return None


def _build_source_dir() -> Path:
    """Build source package with entry point, qlora_train, and configs."""
    d = Path(tempfile.mkdtemp(prefix="gpm-sagemaker-"))
    (d / "config").mkdir()
    shutil.copy(ROOT / "scripts" / "sagemaker" / "entrypoint" / "train.py", d / "train.py")
    shutil.copy(ROOT / "scripts" / "sagemaker" / "entrypoint" / "requirements.txt", d / "requirements.txt")
    shutil.copy(ROOT / "scripts" / "training" / "qlora_train.py", d / "qlora_train.py")
    for name in ["educator_training.yaml", "poet_training.yaml", "rhyme_training.yaml"]:
        shutil.copy(ROOT / "config" / name, d / "config" / name)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["educator", "poet", "rhyme"], required=True)
    ap.add_argument("--num-epochs-override", type=int, default=None)
    ap.add_argument("--base-model", type=str, default=None, help="HuggingFace base model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    args = ap.parse_args()

    cfg = _load_config()
    bucket = cfg.get("s3_bucket")
    role = cfg.get("iam_role")
    region = cfg.get("region", "us-east-1")
    instance_type = cfg.get("instance_type", "ml.g5.xlarge")
    if not bucket or not role:
        raise ValueError("s3_bucket and iam_role required in config/sagemaker.yaml")

    hf_token = _get_hf_token(cfg, region) or __import__("os").environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: No HuggingFace token. Gated models (e.g. Llama) may fail.")

    source_dir = _build_source_dir()
    try:
        env = {"HF_TOKEN": hf_token} if hf_token else {}
        if "ACCOUNT_ID" in role:
            raise ValueError("Replace ACCOUNT_ID in config/sagemaker.yaml iam_role with your AWS account ID")

        job_name = f"gpm-{args.task}-{__import__('time').strftime('%Y%m%d-%H%M%S', __import__('time').gmtime())}"
        output_path = f"s3://{bucket}/checkpoints/{args.task}/"

        huggingface_estimator = HuggingFace(
            entry_point="train.py",
            source_dir=str(source_dir),
            role=role,
            instance_type=instance_type,
            instance_count=1,
            transformers_version="4.36",
            pytorch_version="2.1",
            py_version="py310",
            output_path=output_path,
            hyperparameters={
                "task": args.task,
                **({"num_epochs_override": str(args.num_epochs_override)} if args.num_epochs_override is not None else {}),
                **({"base_model": args.base_model} if args.base_model else {}),
            },
            environment=env,
            sagemaker_session=__import__("sagemaker").Session(boto_session=boto3.Session(region_name=region)),
        )

        training_data = TrainingInput(s3_data=f"s3://{bucket}/data/", input_mode="File")

        huggingface_estimator.fit(
            inputs={"training": training_data},
            job_name=job_name,
        )
        print(f"Training complete. Checkpoints: {output_path}")
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
