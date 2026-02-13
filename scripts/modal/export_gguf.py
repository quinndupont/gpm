#!/usr/bin/env python3
"""Modal: Merge LoRA + convert to GGUF Q4_K_M. S3.4"""
import subprocess
import yaml
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "export_pipeline.yaml"

# Image with llama.cpp for conversion and quantization
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.36",
        "peft>=0.10",
        "accelerate>=0.28",
        "pyyaml>=6.0",
        "gguf>=0.8.0",
    )
    .add_local_file(str(ROOT / "config" / "export_pipeline.yaml"), "/config/export_pipeline.yaml")
    .run_commands(
        "cd /tmp && git clone --depth 1 https://github.com/ggerganov/llama.cpp",
        "cd /tmp/llama.cpp && make -j4",
    )
)

app = modal.App("poetry-export-gguf")
checkpoint_vol = modal.Volume.from_name("poetry-checkpoints", create_if_missing=True)
gguf_vol = modal.Volume.from_name("poetry-gguf", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=2 * 3600,
    volumes={"/vol/checkpoints": checkpoint_vol, "/vol/gguf": gguf_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def export_educator():
    """Merge educator LoRA and export to GGUF Q4_K_M."""
    return _export("educator", "/vol/checkpoints/educator/final")


@app.function(
    image=image,
    gpu="A10G",
    timeout=2 * 3600,
    volumes={"/vol/checkpoints": checkpoint_vol, "/vol/gguf": gguf_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def export_poet():
    """Merge poet LoRA and export to GGUF Q4_K_M."""
    return _export("poet", "/vol/checkpoints/poet/final")


def _export(model_name: str, checkpoint_path: str) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_vol.reload()
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first.")

    cfg = yaml.safe_load(Path("/config/export_pipeline.yaml").read_text()) if Path("/config/export_pipeline.yaml").exists() else {}
    base_model = "meta-llama/Llama-3.1-14B-Instruct"
    quant = cfg.get("quantization", {}).get("primary_quant", "Q4_K_M")
    out_dir = Path("/vol/gguf")

    # Load base and adapter, merge
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(ckpt))
    model = model.merge_and_unload()

    merge_dir = Path(f"/tmp/merged_{model_name}")
    merge_dir.mkdir(exist_ok=True)
    print("Saving merged model...")
    model.save_pretrained(merge_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    tokenizer.save_pretrained(merge_dir)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Convert to GGUF
    convert_script = "/tmp/llama.cpp/convert_hf_to_gguf.py"
    if not Path(convert_script).exists():
        subprocess.run(
            ["curl", "-sL", "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py", "-o", convert_script],
            check=True,
        )
    gguf_f32 = out_dir / f"llama3.1-14b-{model_name}-f32.gguf"
    subprocess.run(
        ["python3", convert_script, str(merge_dir), "--outfile", str(gguf_f32), "--outtype", "f16"],
        check=True,
        cwd="/tmp/llama.cpp",
    )

    # Quantize
    gguf_out = out_dir / f"llama3.1-14b-{model_name}-{quant}.gguf"
    subprocess.run(
        ["/tmp/llama.cpp/quantize", str(gguf_f32), str(gguf_out), quant],
        check=True,
    )
    gguf_f32.unlink(missing_ok=True)

    gguf_vol.commit()
    return str(gguf_out)


if __name__ == "__main__":
    import sys
    with app.run():
        if "poet" in sys.argv:
            p = export_poet.remote()
        else:
            p = export_educator.remote()
        print(f"Done: {p}")
