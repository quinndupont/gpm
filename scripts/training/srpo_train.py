#!/usr/bin/env python3
"""SRPO (Self-Refinement Policy Optimization) for poet model — Stage 2.

Loads the SFT checkpoint from Stage 1, trains on (prompt, draft_0, critique, draft_1)
trajectories to learn both generation AND self-revision.

Algorithm:
    For each trajectory:
        1. Compute generation loss: L_gen = -log P(draft_0 | prompt)
        2. Compute revision loss: L_rev = -w(r) * log P(draft_1 | prompt, draft_0, critique)
        3. Compute KL penalty against frozen SFT reference
        4. L = alpha * L_gen + (1-alpha) * L_rev + beta * KL(policy || ref)

Key insight: The poet learns two skills in one training run:
    - π (generation): Write poems from briefs
    - π† (self-revision): Improve drafts based on critique
"""
import json
import math
from pathlib import Path

import yaml


def load_jsonl(p: Path) -> list:
    """Load JSONL file into list of dicts."""
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def run_srpo_training(
    config_path: Path,
    sft_checkpoint: Path,
    data_dir: Path,
    train_filename: str,
    checkpoint_dir: Path,
    num_epochs_override: int | None = None,
    base_model_override: str | None = None,
) -> str:
    """Run SRPO training starting from SFT checkpoint.

    Args:
        config_path: Path to srpo_training.yaml
        sft_checkpoint: Path to Stage 1 SFT checkpoint
        data_dir: Directory containing training data
        train_filename: Name of SRPO trajectories JSONL file
        checkpoint_dir: Where to save checkpoints
        num_epochs_override: Override epochs from config
        base_model_override: Override base model from config

    Returns:
        Path to final checkpoint
    """
    import copy
    import torch
    import torch.nn.functional as F
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_cosine_schedule_with_warmup,
    )

    from models.prompts.loader import get_persona, render_prompt
    from scripts.training.timer import TrainingTimer

    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg.get("model_loading", {})
    train_cfg = cfg.get("training", {})
    srpo_cfg = cfg.get("srpo", {})

    base_model = base_model_override or cfg.get("base_model", "meta-llama/Llama-3.1-8B-Instruct")
    num_epochs = num_epochs_override or train_cfg.get("num_epochs", 3)
    max_seq_len = train_cfg.get("max_seq_length", 1536)
    save_path = Path(checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # SRPO hyperparameters
    alpha = float(srpo_cfg.get("alpha", 0.4))
    beta_kl = float(srpo_cfg.get("beta_kl", 0.08))
    reward_norm = float(srpo_cfg.get("reward_normalization", 0.2))
    max_reward_weight = float(srpo_cfg.get("max_reward_weight", 2.0))

    # Training hyperparameters
    lr = float(train_cfg.get("learning_rate", 5e-6))
    grad_accum = train_cfg.get("gradient_accumulation_steps", 8)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))

    # Load training data
    train_file = data_dir / train_filename
    if not train_file.exists():
        raise FileNotFoundError(f"Training data not found: {train_file}")
    trajectories = load_jsonl(train_file)
    if not trajectories:
        raise ValueError(f"No training data in {train_filename}")
    # Exclude trajectories with empty critique (e.g. v2 variance before refactor)
    before = len(trajectories)
    trajectories = [t for t in trajectories if t.get("critique", "").strip()]
    if len(trajectories) < before:
        print(f"Filtered {before - len(trajectories)} trajectories with empty critique")
    if not trajectories:
        raise ValueError(f"No valid trajectories (all had empty critique)")
    print(f"SRPO: {len(trajectories)} trajectories, {num_epochs} epochs, "
          f"alpha={alpha}, beta_kl={beta_kl}")

    # Initialize training timer
    timer = TrainingTimer()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # --- Load base model and policy (trainable, from SFT checkpoint) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("quantization", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=model_cfg.get("double_quant", True),
    )
    print("Loading base model and SFT checkpoint...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)
    policy = PeftModel.from_pretrained(base, str(sft_checkpoint), is_trainable=True)
    policy.train()

    # --- Create reference model via deepcopy (faster than reloading from HF) ---
    print("Creating reference model via deepcopy (saves 1-2 min vs. reloading)...")
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # --- Optimizer + scheduler ---
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    total_steps = (len(trajectories) * num_epochs) // grad_accum
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    device = next(policy.parameters()).device
    poet_persona = get_persona("poet")

    def _tokenize_chat(system: str, user: str, max_len: int) -> torch.Tensor:
        """Tokenize a chat prompt (system + user) for the model."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_len
        ).input_ids.to(device)

    def _compute_log_probs(
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        completion_text: str,
        max_comp_len: int = 512,
    ) -> torch.Tensor:
        """Compute sum of per-token log probs for completion under model."""
        comp_ids = tokenizer(
            completion_text, return_tensors="pt", truncation=True,
            max_length=max_comp_len, add_special_tokens=False,
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

    def _compute_log_probs_batch(
        model: torch.nn.Module,
        prompt_ids_list: list[torch.Tensor],
        completion_texts: list[str],
        max_comp_len: int = 512,
    ) -> torch.Tensor:
        """Compute log probs for multiple (prompt, completion) pairs in batched forward pass.

        Note: For SRPO, we may have different prompts (generation vs revision),
        so this function handles varying prompt lengths.

        Args:
            model: Model to compute log probs with
            prompt_ids_list: List of prompt tensors (each of shape (1, prompt_len_i))
            completion_texts: List of completion strings
            max_comp_len: Maximum completion length for tokenization

        Returns:
            Tensor of shape (N,) with summed log probs
        """
        if not completion_texts or not prompt_ids_list:
            return torch.tensor([], device=device)

        batch_size = len(completion_texts)
        assert len(prompt_ids_list) == batch_size, "Mismatch between prompts and completions"

        # Tokenize all completions
        comp_ids_list = [
            tokenizer(text, return_tensors="pt", truncation=True,
                      max_length=max_comp_len, add_special_tokens=False).input_ids.to(device)
            for text in completion_texts
        ]

        # For SRPO, prompt lengths may vary, so we process each separately
        # This is still faster than the old approach due to reduced overhead
        log_probs_results = []
        for prompt_ids, comp_ids in zip(prompt_ids_list, comp_ids_list):
            full_ids = torch.cat([prompt_ids, comp_ids], dim=1)
            with torch.set_grad_enabled(model.training):
                logits = model(full_ids).logits

            prompt_len = prompt_ids.shape[1]
            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = comp_ids
            min_len = min(shift_logits.shape[1], shift_labels.shape[1])

            log_probs = F.log_softmax(shift_logits[:, :min_len, :], dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels[:, :min_len].unsqueeze(-1)).squeeze(-1)
            log_probs_results.append(token_log_probs.sum(dim=1))

        return torch.stack(log_probs_results).squeeze()

    def _build_generation_prompt(trajectory: dict) -> tuple[str, str]:
        """Build input for generation objective: system + brief."""
        system = trajectory.get("system", poet_persona)
        user = trajectory["prompt"]
        return system, user

    def _build_revision_prompt(trajectory: dict) -> tuple[str, str]:
        """Build input for revision objective: system + (brief, draft, critique)."""
        system = poet_persona
        # Use SRPO revision prompt template
        user = render_prompt(
            "tuning", "poet_revision_srpo",
            brief=trajectory["prompt"],
            draft=trajectory["draft_0"],
            critique=trajectory["critique"],
            rhyme_ctx="",
        )
        return system, user

    def _compute_reward_weight(trajectory: dict) -> float:
        """Compute reward weight w(r) based on improvement."""
        r0 = trajectory.get("reward_0", 0.0)
        r1 = trajectory.get("reward_1", 0.0)
        improvement = r1 - r0
        w_r = max(0.0, improvement) / reward_norm
        w_r = min(w_r, max_reward_weight)
        return w_r

    # --- Training loop ---
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        timer.start_epoch()
        import random
        random.shuffle(trajectories)

        epoch_loss_gen = 0.0
        epoch_loss_rev = 0.0
        epoch_kl = 0.0
        epoch_loss_total = 0.0
        n_batches = 0

        for i, traj in enumerate(trajectories):
            timer.start_step()

            # Skip if missing required fields
            if not all(k in traj for k in ["prompt", "draft_0", "critique", "draft_1"]):
                continue

            # --- Generation objective ---
            with timer.phase("generation"):
                gen_system, gen_user = _build_generation_prompt(traj)
                gen_prompt_ids = _tokenize_chat(gen_system, gen_user, max_seq_len // 2)
                draft_0 = traj["draft_0"]

                if not draft_0.strip():
                    continue

                log_p_gen = _compute_log_probs(policy, gen_prompt_ids, draft_0)
                L_gen = -log_p_gen.mean()

            # --- Revision objective ---
            with timer.phase("revision"):
                rev_system, rev_user = _build_revision_prompt(traj)
                rev_prompt_ids = _tokenize_chat(rev_system, rev_user, max_seq_len)
                draft_1 = traj["draft_1"]

                if not draft_1.strip():
                    continue

                log_p_rev = _compute_log_probs(policy, rev_prompt_ids, draft_1)

                # Reward weight
                w_r = _compute_reward_weight(traj)
                L_rev = -w_r * log_p_rev.mean()

                # KL penalty (on revision, most important to constrain)
                with torch.no_grad():
                    log_ref = _compute_log_probs(ref_model, rev_prompt_ids, draft_1)
                kl = (log_p_rev - log_ref).mean()

            # --- Gradients ---
            with timer.phase("gradients"):
                # Combined loss
                loss = alpha * L_gen + (1 - alpha) * L_rev + beta_kl * kl
                scaled_loss = loss / grad_accum
                scaled_loss.backward()

            epoch_loss_gen += L_gen.item()
            epoch_loss_rev += L_rev.item()
            epoch_kl += kl.item()
            epoch_loss_total += loss.item()
            n_batches += 1

            if (i + 1) % grad_accum == 0 or (i + 1) == len(trajectories):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if (i + 1) % 10 == 0:
                avg_gen = epoch_loss_gen / n_batches
                avg_rev = epoch_loss_rev / n_batches
                avg_kl = epoch_kl / n_batches
                avg_total = epoch_loss_total / n_batches
                metrics = {
                    "loss": avg_total,
                    "gen": avg_gen,
                    "rev": avg_rev,
                    "kl": avg_kl,
                }
                print(f"  {timer.format_summary(i+1, len(trajectories), metrics, epoch+1, num_epochs)}")

        avg_gen = epoch_loss_gen / max(n_batches, 1)
        avg_rev = epoch_loss_rev / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)
        avg_total = epoch_loss_total / max(n_batches, 1)
        metrics = {
            "loss": avg_total,
            "gen": avg_gen,
            "rev": avg_rev,
            "kl": avg_kl,
        }
        print(f"{timer.format_epoch_summary(epoch+1, num_epochs, metrics)}")

        # Save epoch checkpoint
        epoch_dir = save_path / f"checkpoint-epoch-{epoch+1}"
        policy.save_pretrained(str(epoch_dir))
        tokenizer.save_pretrained(str(epoch_dir))

    # Save final checkpoint
    final_dir = save_path / "final"
    policy.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"SRPO training complete. Final checkpoint: {final_dir}")
    return str(final_dir)


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    parser = argparse.ArgumentParser(description="SRPO training for poet model")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "srpo_training.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        required=True,
        help="Path to Stage 1 SFT checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "srpo_training",
        help="Directory containing SRPO trajectories",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="trajectories_v2.jsonl",
        help="Training data filename",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "poet_srpo",
        help="Where to save checkpoints",
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--base-model", type=str, help="Override base model")
    args = parser.parse_args()

    run_srpo_training(
        config_path=args.config,
        sft_checkpoint=args.sft_checkpoint,
        data_dir=args.data_dir,
        train_filename=args.train_file,
        checkpoint_dir=args.checkpoint_dir,
        num_epochs_override=args.epochs,
        base_model_override=args.base_model,
    )
