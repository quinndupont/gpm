#!/usr/bin/env python3
"""REINFORCE (reward-weighted regression) for poet model — Stage 2.

Loads the SFT checkpoint from Stage 1, generates completions per prompt,
scores them with the deterministic rhyme analyzer, and trains with
advantage-weighted log-prob loss + KL penalty against the frozen SFT ref.

Algorithm:
    For each batch of prompts:
        1. Generate N completions from the current policy
        2. Score each with compute_reward() (partial credit rhyme scoring)
        3. Normalize: Â = (score - mean) / std
        4. L = -mean(Â * log P_policy(completion | prompt)) + beta * KL(policy || ref)
"""
import json
import math
from pathlib import Path

import yaml


def load_jsonl(p: Path) -> list:
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _extract_prompts(data: list[dict]) -> list[dict]:
    """Extract system + user messages from chat-format training data."""
    prompts = []
    for entry in data:
        msgs = entry.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if user.strip():
            prompts.append({"system": system, "user": user})
    return prompts


def _detect_form_from_prompt(user_msg: str) -> str | None:
    """Try to detect the expected poetic form from the prompt text."""
    lower = user_msg.lower()
    for form in ["sonnet", "villanelle", "limerick", "ballad", "couplets", "tercets"]:
        if form in lower:
            return form
    return None


def run_reinforce_training(
    config_path: Path,
    sft_checkpoint: Path,
    data_dir: Path,
    train_filename: str,
    checkpoint_dir: Path,
    num_epochs_override: int | None = None,
    base_model_override: str | None = None,
) -> str:
    """Run REINFORCE (reward-weighted regression) starting from SFT checkpoint."""
    import torch
    import torch.nn.functional as F
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )

    from scripts.eval.rhyme_analyzer import compute_reward

    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg.get("model_loading", {})
    train_cfg = cfg.get("training", {})
    rl_cfg = cfg.get("reinforce", {})

    base_model = base_model_override or cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    num_epochs = num_epochs_override or train_cfg.get("num_epochs", 2)
    max_seq_len = train_cfg.get("max_seq_length", 1024)
    save_path = Path(checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    num_completions = rl_cfg.get("num_completions", 4)
    beta_kl = float(rl_cfg.get("beta_kl", 0.1))
    max_gen_tokens = rl_cfg.get("max_gen_tokens", 512)
    gen_temperature = float(rl_cfg.get("temperature", 0.8))
    gen_top_p = float(rl_cfg.get("top_p", 0.95))

    lr = float(train_cfg.get("learning_rate", 1e-5))
    grad_accum = train_cfg.get("gradient_accumulation_steps", 8)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))

    train_file = data_dir / train_filename
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    train_data = load_jsonl(train_file)
    if not train_data:
        raise ValueError(f"No training data in {train_filename}")
    prompts = _extract_prompts(train_data)
    print(f"REINFORCE: {len(prompts)} prompts, {num_completions} completions each, "
          f"{num_epochs} epochs, beta_kl={beta_kl}")

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Load policy (trainable, from SFT checkpoint) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("quantization", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=model_cfg.get("double_quant", True),
    )
    policy = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    policy = prepare_model_for_kbit_training(policy)
    policy = PeftModel.from_pretrained(policy, str(sft_checkpoint), is_trainable=True)
    policy.train()

    # --- Load reference model (frozen SFT copy for KL) ---
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, str(sft_checkpoint), is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # --- Optimizer + scheduler ---
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = (len(prompts) * num_epochs) // grad_accum
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    device = next(policy.parameters()).device

    def _build_chat_ids(system: str, user: str) -> torch.Tensor:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=max_seq_len - max_gen_tokens).input_ids.to(device)

    def _generate_completions(prompt_ids: torch.Tensor, n: int) -> list[str]:
        with torch.no_grad():
            out = policy.generate(
                input_ids=prompt_ids.expand(n, -1),
                max_new_tokens=max_gen_tokens,
                temperature=gen_temperature,
                top_p=gen_top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_len = prompt_ids.shape[1]
        texts = []
        for seq in out:
            completion_ids = seq[prompt_len:]
            texts.append(tokenizer.decode(completion_ids, skip_special_tokens=True))
        return texts

    def _compute_log_probs(model: torch.nn.Module, prompt_ids: torch.Tensor,
                           completion_text: str) -> torch.Tensor:
        """Compute per-token log probs of completion under model, summed."""
        comp_ids = tokenizer(
            completion_text, return_tensors="pt", truncation=True,
            max_length=max_gen_tokens, add_special_tokens=False,
        ).input_ids.to(device)
        full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        with torch.set_grad_enabled(model.training):
            logits = model(full_ids).logits
        prompt_len = prompt_ids.shape[1]
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = comp_ids
        min_len = min(shift_logits.shape[1], shift_labels.shape[1])
        log_probs = F.log_softmax(shift_logits[:, :min_len, :], dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels[:, :min_len].unsqueeze(-1)).squeeze(-1)
        return token_log_probs.sum(dim=1)

    # --- Training loop ---
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        import random
        random.shuffle(prompts)
        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for i, prompt in enumerate(prompts):
            prompt_ids = _build_chat_ids(prompt["system"], prompt["user"])
            form = _detect_form_from_prompt(prompt["user"])

            completions = _generate_completions(prompt_ids, num_completions)
            rewards = [compute_reward(c, expected_form=form) for c in completions]

            mean_r = sum(rewards) / len(rewards)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards) - 1, 1))
            if std_r < 1e-8:
                std_r = 1.0
            advantages = [(r - mean_r) / std_r for r in rewards]

            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_kl = 0.0

            for comp, adv in zip(completions, advantages):
                if not comp.strip():
                    continue
                log_p = _compute_log_probs(policy, prompt_ids, comp)
                with torch.no_grad():
                    log_ref = _compute_log_probs(ref_model, prompt_ids, comp)
                kl = (log_p - log_ref).mean()
                loss_i = -(adv * log_p.mean()) + beta_kl * kl
                batch_loss = batch_loss + loss_i / num_completions
                batch_kl += kl.item()

            scaled_loss = batch_loss / grad_accum
            scaled_loss.backward()

            epoch_loss += batch_loss.item()
            epoch_reward += mean_r
            epoch_kl += batch_kl / num_completions
            n_batches += 1

            if (i + 1) % grad_accum == 0 or (i + 1) == len(prompts):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (i + 1) % 10 == 0:
                avg_loss = epoch_loss / n_batches
                avg_r = epoch_reward / n_batches
                avg_kl = epoch_kl / n_batches
                print(f"  [{epoch+1}/{num_epochs}] step {i+1}/{len(prompts)} "
                      f"loss={avg_loss:.4f} reward={avg_r:.3f} kl={avg_kl:.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_r = epoch_reward / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f} "
              f"avg_reward={avg_r:.3f} avg_kl={avg_kl:.4f}")

        epoch_dir = save_path / f"checkpoint-epoch-{epoch+1}"
        policy.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))

    final_dir = save_path / "final"
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"REINFORCE training complete. Final checkpoint: {final_dir}")
    return str(final_dir)
