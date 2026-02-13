#!/usr/bin/env python3
"""Modal: QLoRA fine-tune poet model. S3.3"""
import json
import yaml
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "poet_training.yaml"
VOLUME_NAME = "poetry-data"
CHECKPOINT_VOLUME = "poetry-checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.36",
        "peft>=0.10",
        "trl>=0.8",
        "bitsandbytes>=0.43",
        "datasets>=2.18",
        "accelerate>=0.28",
        "pyyaml>=6.0",
    )
    .add_local_file(str(ROOT / "config" / "poet_training.yaml"), "/config/poet_training.yaml")
)

app = modal.App("poetry-poet-train")
data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=4 * 3600,
    volumes={"/vol/data": data_vol, "/vol/checkpoints": checkpoint_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_poet(num_epochs_override: int | None = None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    cfg = yaml.safe_load(Path("/config/poet_training.yaml").read_text())
    model_cfg = cfg.get("model_loading", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpointing", {})

    base_model = cfg.get("base_model", "meta-llama/Llama-3.1-14B-Instruct")
    num_epochs = num_epochs_override or train_cfg.get("num_epochs", 6)
    max_seq_len = train_cfg.get("max_seq_length", 512)
    save_path = Path(ckpt_cfg.get("save_path", "/vol/checkpoints/poet/"))

    train_file = Path("/vol/data/poet_train.jsonl")
    valid_file = Path("/vol/data/poet_valid.jsonl")
    data_vol.reload()
    if not train_file.exists():
        raise FileNotFoundError("Upload data first: python scripts/modal/upload_data.py")

    def load_jsonl(p: Path) -> list:
        out = []
        for line in p.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    train_data = load_jsonl(train_file)
    eval_data = load_jsonl(valid_file) if valid_file.exists() else None
    if not train_data:
        raise ValueError("No training data in poet_train.jsonl")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("quantization", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=model_cfg.get("double_quant", True),
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    save_path.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(save_path),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=train_cfg.get("learning_rate", 2e-4),
        num_train_epochs=num_epochs,
        max_steps=-1,
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=ckpt_cfg.get("save_total_limit", 4),
        eval_strategy="epoch" if eval_dataset else "no",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="messages",
        max_seq_length=max_seq_len,
        packing=False,
    )
    trainer.train()
    trainer.save_model(str(save_path / "final"))
    tokenizer.save_pretrained(str(save_path / "final"))
    checkpoint_vol.commit()
    return str(save_path / "final")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-epochs-override", type=int, default=None)
    a = ap.parse_args()
    with app.run():
        path = train_poet.remote(num_epochs_override=a.num_epochs_override)
        print(f"Done: {path}")
