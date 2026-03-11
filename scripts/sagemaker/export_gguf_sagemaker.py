#!/usr/bin/env python3
"""Export LoRA checkpoint from S3 to GGUF and upload to S3. Run locally with GPU."""
import argparse
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import boto3
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict:
    cfg_path = ROOT / "config" / "sagemaker.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text()) or {}


def _find_latest_checkpoint(s3: object, bucket: str, task: str, region: str) -> str:
    """Find most recent model.tar.gz under checkpoints/{task}/."""
    prefix = f"checkpoints/{task}/"
    paginator = s3.get_paginator("list_objects_v2")
    latest = None
    latest_ts = None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("output/model.tar.gz"):
                ts = obj["LastModified"]
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                    latest = f"s3://{bucket}/{key}"
    if not latest:
        raise FileNotFoundError(f"No checkpoint found under s3://{bucket}/{prefix}")
    return latest


def _hf_to_short(hf_id: str) -> str:
    from scripts.training.model_registry import hf_to_short as _h
    return _h(hf_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        choices=["educator", "poet", "poet_reinforce", "educator-interim"],
        required=True,
    )
    ap.add_argument("--checkpoint-s3", help="s3://bucket/path/to/model.tar.gz (default: latest)")
    args = ap.parse_args()

    cfg = _load_config()
    bucket = cfg.get("s3_bucket")
    region = cfg.get("region", "us-east-1")
    if not bucket:
        raise ValueError("s3_bucket required in config/sagemaker.yaml")

    s3 = boto3.client("s3", region_name=region)
    checkpoint_s3 = args.checkpoint_s3
    if not checkpoint_s3:
        task_map = {"educator-interim": "educator"}
        s3_task = task_map.get(args.task, args.task)
        checkpoint_s3 = _find_latest_checkpoint(s3, bucket, s3_task, region)
    if checkpoint_s3.startswith("s3://"):
        parts = checkpoint_s3[5:].split("/", 1)
        ckpt_bucket, ckpt_key = parts[0], parts[1]
    else:
        raise ValueError("checkpoint-s3 must be s3:// URI")

    export_cfg = yaml.safe_load((ROOT / "config" / "export_pipeline.yaml").read_text()) or {}
    base_model = export_cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    quant = export_cfg.get("quantization", {}).get("primary_quant", "Q4_K_M")
    short = _hf_to_short(base_model)
    out_name = "educator-interim" if args.task == "educator-interim" else args.task
    gguf_name = f"{short}-{out_name}-{quant}.gguf"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model_tar = tmp / "model.tar.gz"
        s3.download_file(ckpt_bucket, ckpt_key, str(model_tar))
        with tarfile.open(model_tar) as tf:
            tf.extractall(tmp)
        ckpt_subdir = (
            "educator" if "educator" in args.task
            else ("poet_reinforce" if "reinforce" in args.task else "poet")
        )
        candidates = [tmp / ckpt_subdir / "final", tmp / "final", tmp / ckpt_subdir]
        ckpt_path = None
        for cand in candidates:
            if cand.exists() and (cand / "adapter_config.json").exists():
                ckpt_path = cand
                break
        if not ckpt_path:
            raise FileNotFoundError(
                f"Checkpoint not found in tar. Tried {candidates}. "
                f"Contents: {list(tmp.iterdir())}",
            )

        # Build llama.cpp
        llama_dir = tmp / "llama.cpp"
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/ggml-org/llama.cpp", str(llama_dir),
            ],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["cmake", "-B", "build"], cwd=llama_dir,
            check=True, capture_output=True,
        )
        subprocess.run(
            ["cmake", "--build", "build", "--config", "Release", "-j4"],
            cwd=llama_dir, check=True, capture_output=True,
        )

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Force CPU to avoid accelerate device map issues
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(ckpt_path))
        model = model.merge_and_unload()
        merge_dir = tmp / "merged"
        merge_dir.mkdir()
        model.save_pretrained(merge_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(merge_dir)
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        convert_script = llama_dir / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            subprocess.run(
                [
                    "curl", "-sL",
                    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py",
                    "-o", str(convert_script),
                ],
                check=True,
            )
        gguf_f16 = tmp / f"{out_name}-f16.gguf"
        subprocess.run(
            [
                sys.executable, str(convert_script), str(merge_dir),
                "--outfile", str(gguf_f16), "--outtype", "f16",
            ],
            cwd=str(llama_dir), check=True,
        )
        gguf_out = tmp / gguf_name
        quant_bin = str(llama_dir / "build" / "bin" / "llama-quantize")
        subprocess.run(
            [quant_bin, str(gguf_f16), str(gguf_out), quant],
            check=True,
        )
        s3.upload_file(str(gguf_out), bucket, f"gguf/{gguf_name}")
        print(f"Uploaded s3://{bucket}/gguf/{gguf_name}")


if __name__ == "__main__":
    main()
