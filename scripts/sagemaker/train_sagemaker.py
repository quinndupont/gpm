#!/usr/bin/env python3
"""Launch SageMaker HuggingFace training job for educator, poet, or REINFORCE."""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import boto3
import yaml
from sagemaker.huggingface import HuggingFace
from sagemaker.inputs import TrainingInput

ROOT = Path(__file__).resolve().parents[2]

# Fallback when model_registry.yaml is missing
DEFAULT_MODELS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct"),
    ("meta-llama/Llama-3.2-8B-Instruct", "Llama 3.2 8B Instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5 7B Instruct"),
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5 14B Instruct"),
    ("deepseek-ai/DeepSeek-V2-Lite-7B-Instruct", "DeepSeek V2 Lite 7B"),
]


def _load_model_choices() -> list[tuple[str, str]]:
    """Return [(hf_id, display_label), ...] from registry or default list."""
    registry = ROOT / "config" / "model_registry.yaml"
    if not registry.exists():
        return DEFAULT_MODELS
    data = yaml.safe_load(registry.read_text()) or {}
    models = data.get("models") or []
    out = []
    for m in models:
        if isinstance(m, dict) and m.get("hf_id"):
            label = m.get("short_name") or m["hf_id"].split("/")[-1]
            out.append((m["hf_id"], label))
    return out if out else DEFAULT_MODELS


def _prompt_model_choice(task: str) -> str:
    """Prompt user to select base model from list; return hf_id."""
    choices = _load_model_choices()
    print(f"\nBase model for task '{task}' (leave empty to use task default from config):\n")
    for i, (hf_id, label) in enumerate(choices, 1):
        print(f"  {i}. {label}  ({hf_id})")
    print(f"  {len(choices) + 1}. Use task default from config")
    while True:
        try:
            raw = input("\nChoice [1-%d]: " % (len(choices) + 1)).strip()
            if not raw:
                return None
            idx = int(raw)
            if 1 <= idx <= len(choices) + 1:
                return None if idx == len(choices) + 1 else choices[idx - 1][0]
        except ValueError:
            pass
        print("Invalid choice. Enter a number or leave empty for task default.")


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
    entry = ROOT / "scripts" / "sagemaker" / "entrypoint"
    shutil.copy(entry / "train.py", d / "train.py")
    shutil.copy(entry / "requirements.txt", d / "requirements.txt")
    shutil.copy(ROOT / "scripts" / "training" / "qlora_train.py", d / "qlora_train.py")
    for name in ["educator_training.yaml", "poet_training.yaml", "reinforce_training.yaml"]:
        cfg_file = ROOT / "config" / name
        if cfg_file.exists():
            shutil.copy(cfg_file, d / "config" / name)
    shutil.copy(ROOT / "scripts" / "training" / "reinforce_train.py", d / "reinforce_train.py")
    (d / "scripts").mkdir(exist_ok=True)
    (d / "scripts" / "eval").mkdir(parents=True, exist_ok=True)
    for eval_file in ["rhyme_analyzer.py", "form_registry.py", "meter_analyzer.py"]:
        src = ROOT / "scripts" / "eval" / eval_file
        if src.exists():
            shutil.copy(src, d / "scripts" / "eval" / eval_file)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["educator", "poet", "reinforce"], required=True)
    ap.add_argument("--num-epochs-override", type=int, default=None)
    ap.add_argument(
        "--base-model", type=str, default=None,
        help="HuggingFace base model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    ap.add_argument(
        "--no-prompt", action="store_true",
        help="Skip interactive model selection; use task default from config",
    )
    args = ap.parse_args()

    base_model = args.base_model
    if base_model is None and not args.no_prompt and sys.stdin.isatty():
        base_model = _prompt_model_choice(args.task)

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
            raise ValueError(
                "Replace ACCOUNT_ID in config/sagemaker.yaml iam_role with your AWS account ID",
            )

        import time as _t
        job_name = f"gpm-{args.task}-{_t.strftime('%Y%m%d-%H%M%S', _t.gmtime())}"
        output_path = f"s3://{bucket}/checkpoints/{args.task}/"

        huggingface_estimator = HuggingFace(
            entry_point="train.py",
            source_dir=str(source_dir),
            role=role,
            instance_type=instance_type,
            instance_count=1,
            transformers_version="4.46",
            pytorch_version="2.3",
            py_version="py311",
            output_path=output_path,
            hyperparameters={
                "task": args.task,
                **(
                    {"num_epochs_override": str(args.num_epochs_override)}
                    if args.num_epochs_override is not None else {}
                ),
                **({"base_model": base_model} if base_model else {}),
            },
            environment=env,
            sagemaker_session=__import__("sagemaker").Session(
                boto_session=boto3.Session(region_name=region),
            ),
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
