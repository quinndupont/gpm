#!/usr/bin/env python3
"""Modal: Merge LoRA + convert to GGUF Q4_K_M. S3.4"""
import subprocess
import yaml
from pathlib import Path

import modal

# Config path: works when run from project root (local or Modal workspace)
_EXPORT_CONFIG = Path("config/export_pipeline.yaml")
if not _EXPORT_CONFIG.exists():
    _p = Path(__file__).resolve()
    _EXPORT_CONFIG = _p.parents[1] / "config" / "export_pipeline.yaml" if len(_p.parents) > 1 else _EXPORT_CONFIG

# Image with llama.cpp for conversion and quantization (add_local_file last per Modal)
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
        "sentencepiece>=0.1.99",
    )
    .run_commands(
        "cd /tmp && git clone --depth 1 https://github.com/ggml-org/llama.cpp",
        "cd /tmp/llama.cpp && cmake -B build && cmake --build build --config Release -j4",
    )
    .add_local_file(str(_EXPORT_CONFIG), "/config/export_pipeline.yaml")
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
def export_educator_interim():
    """Export interim educator (trained on seed only) to educator-interim GGUF."""
    return _export("educator", "/vol/checkpoints/educator/final", out_name="educator-interim")


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


def _export(model_name: str, checkpoint_path: str, out_name: str | None = None) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint_vol.reload()
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first.")

    cfg = yaml.safe_load(Path("/config/export_pipeline.yaml").read_text()) if Path("/config/export_pipeline.yaml").exists() else {}
    base_model = cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
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
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(merge_dir)
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Convert to GGUF
    convert_script = "/tmp/llama.cpp/convert_hf_to_gguf.py"
    if not Path(convert_script).exists():
        subprocess.run(
            ["curl", "-sL", "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py", "-o", convert_script],
            check=True,
        )
    name = out_name or model_name
    gguf_f32 = out_dir / f"qwen2.5-7b-{name}-f32.gguf"
    subprocess.run(
        ["python3", convert_script, str(merge_dir), "--outfile", str(gguf_f32), "--outtype", "f16"],
        check=True,
        cwd="/tmp/llama.cpp",
    )

    # Quantize
    gguf_out = out_dir / f"qwen2.5-7b-{name}-{quant}.gguf"
    subprocess.run(
        ["/tmp/llama.cpp/build/bin/llama-quantize", str(gguf_f32), str(gguf_out), quant],
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
        elif "interim" in sys.argv:
            p = export_educator_interim.remote()
        else:
            p = export_educator.remote()
        print(f"Done: {p}")
