#!/usr/bin/env python3
"""Shared QLoRA training logic for Modal and SageMaker backends."""
import json
from pathlib import Path

import yaml


def load_jsonl(p: Path) -> list:
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def run_qlora_training(
    config_path: Path,
    data_dir: Path,
    train_filename: str,
    valid_filename: str,
    checkpoint_dir: Path,
    num_epochs_override: int | None = None,
    base_model_override: str | None = None,
) -> str:
    """Run QLoRA fine-tuning. Backend-agnostic; caller handles volume reload/commit."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg.get("model_loading", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    ckpt_cfg = cfg.get("checkpointing", {})

    base_model = base_model_override or cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    num_epochs = num_epochs_override or train_cfg.get("num_epochs", 4)
    max_seq_len = train_cfg.get("max_seq_length", 1024)
    save_path = Path(checkpoint_dir)

    train_file = data_dir / train_filename
    valid_file = data_dir / valid_filename
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")

    train_data = load_jsonl(train_file)
    eval_data = load_jsonl(valid_file) if valid_file.exists() else None
    if not train_data:
        raise ValueError(f"No training data in {train_filename}")

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
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    save_path.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(save_path),
        per_device_train_batch_size=train_cfg.get("per_device_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        num_train_epochs=num_epochs,
        max_steps=-1,
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=ckpt_cfg.get("save_total_limit", 3),
        eval_strategy="epoch" if eval_dataset else "no",
        max_seq_length=max_seq_len,
        # dataset_text_field="messages",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(save_path / "final"))
    tokenizer.save_pretrained(str(save_path / "final"))
    return str(save_path / "final")
