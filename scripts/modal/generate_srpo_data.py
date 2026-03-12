#!/usr/bin/env python3
"""Modal: Generate SRPO training data using GPU for inference."""
from pathlib import Path

import modal

_CONFIG = Path("config/srpo_data_generation.yaml")
if not _CONFIG.exists():
    _p = Path(__file__).resolve()
    _CONFIG = (
        _p.parents[2] / "config" / "srpo_data_generation.yaml"
        if len(_p.parents) > 2
        else _CONFIG
    )

_ROOT = Path(__file__).resolve().parents[2]
VOLUME_NAME = "poetry-data"
CHECKPOINT_VOLUME = "poetry-checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1",
        "git+https://github.com/huggingface/transformers.git",
        "peft>=0.10",
        "bitsandbytes>=0.43",
        "accelerate>=0.28",
        "pyyaml>=6.0",
        "pronouncing>=0.2",
        "anthropic[bedrock]>=0.18",
        "python-dotenv>=1.0",
        "ollama>=0.1",
    )
    .add_local_file(str(_CONFIG), "/config/srpo_data_generation.yaml")
    .add_local_dir(str(_ROOT / "scripts" / "data"), "/opt/gpm/scripts/data")
    .add_local_dir(str(_ROOT / "scripts" / "data_generation"), "/opt/gpm/scripts/data_generation")
    .add_local_dir(str(_ROOT / "scripts" / "eval"), "/opt/gpm/scripts/eval")
    .add_local_dir(str(_ROOT / "models" / "prompts"), "/opt/gpm/models/prompts")
    .env({"PYTHONPATH": "/opt/gpm"})
)

app = modal.App("poetry-srpo-data")
data_vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # Smaller GPU sufficient for inference
    timeout=6 * 3600,
    volumes={"/vol/data": data_vol, "/vol/checkpoints": checkpoint_vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        # Use "aws-secret" for Bedrock, "anthropic-secret" for direct API
        # Set USE_BEDROCK=1 in your secret to enable Bedrock
        modal.Secret.from_name("anthropic-secret"),
    ],
)
def generate_srpo_data(
    max_trajectories: int = 5000,
    source_file: str = "poet_train.jsonl",
    output_file: str = "trajectories.jsonl",
):
    """Generate SRPO training trajectories on Modal.

    Args:
        max_trajectories: Maximum number of trajectories to generate
        source_file: Source poet training data (in /vol/data/)
        output_file: Output trajectories file (in /vol/data/srpo_training/)
    """
    import json
    import os
    import random
    import sys
    import time

    import yaml

    sys.path.insert(0, "/opt/gpm")

    from models.prompts.loader import get_persona, render_prompt
    from scripts.data_generation.claude_utils import call_claude
    from scripts.eval.form_registry import detect_form, is_rhyming_form
    from scripts.eval.rhyme_analyzer import (
        analyze as analyze_rhyme,
        compute_reward,
        format_analysis_for_prompt,
    )

    data_vol.reload()
    checkpoint_vol.reload()

    # Load config
    config_path = Path("/config/srpo_data_generation.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    draft_cfg = cfg.get("draft_generation", {})
    critique_cfg = cfg.get("critique_generation", {})
    revision_cfg = cfg.get("revision_generation", {})
    filter_cfg = cfg.get("filtering", {})
    rate_cfg = cfg.get("rate_limits", {})

    min_improvement = filter_cfg.get("min_improvement", 0.05)
    min_draft_1_score = filter_cfg.get("min_draft_1_score", 0.5)
    candidates_per_draft = revision_cfg.get("candidates_per_draft", 2)
    requests_per_minute = rate_cfg.get("requests_per_minute", 50)

    # Load source prompts
    source_path = Path("/vol/data") / source_file
    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_path}")

    prompts = []
    for line in source_path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        msgs = entry.get("messages", [])
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if user.strip():
            form = detect_form(user)
            prompts.append({"system": system, "user": user, "form": form})

    random.shuffle(prompts)
    prompts = prompts[:max_trajectories * 2]  # Get more than needed for filtering
    print(f"Loaded {len(prompts)} prompts")

    # Prepare output
    output_dir = Path("/vol/data/srpo_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file

    poet_persona = get_persona("poet")
    educator_persona = get_persona("educator_neutral")

    trajectories_written = 0
    trajectories_filtered = 0
    min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
    last_request_time = 0.0

    with open(output_path, "w") as f:
        for i, prompt in enumerate(prompts):
            if trajectories_written >= max_trajectories:
                break

            print(f"[{i+1}/{len(prompts)}] Generating trajectory...", flush=True)

            try:
                # 1. Generate draft_0 using Claude (simulating poet)
                draft_0 = call_claude(
                    prompt["user"],
                    prompt["system"],
                    model="claude-sonnet-4-20250514",
                    max_tokens=draft_cfg.get("max_tokens", 512),
                )

                elapsed = time.time() - last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_request_time = time.time()

                # 2. Generate critique
                form_ctx = ""
                if prompt.get("form") and is_rhyming_form(prompt["form"]):
                    rhyme = analyze_rhyme(draft_0, expected_form=prompt["form"])
                    form_ctx = f"\n\nRhyme analysis (automated):\n{format_analysis_for_prompt(rhyme)}\n"

                critique_prompt = render_prompt(
                    "inference", "critique",
                    brief=prompt["user"], draft=draft_0,
                    history_ctx="", form_ctx=form_ctx,
                )
                critique = call_claude(
                    critique_prompt,
                    educator_persona,
                    model="claude-sonnet-4-20250514",
                    max_tokens=600,
                )

                elapsed = time.time() - last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_request_time = time.time()

                # 3. Generate revision candidates
                best_revision = None
                best_reward_1 = -1.0

                for _ in range(candidates_per_draft):
                    revision_prompt = render_prompt(
                        "tuning", "poet_revision_srpo",
                        brief=prompt["user"], draft=draft_0,
                        critique=critique, rhyme_ctx="",
                    )
                    revision = call_claude(
                        revision_prompt,
                        poet_persona,
                        model=revision_cfg.get("model", "claude-sonnet-4-20250514"),
                        max_tokens=revision_cfg.get("max_tokens", 600),
                    )

                    elapsed = time.time() - last_request_time
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                    last_request_time = time.time()

                    reward = compute_reward(revision, expected_form=prompt.get("form"))
                    if reward > best_reward_1:
                        best_reward_1 = reward
                        best_revision = revision

                draft_1 = best_revision

                # 4. Score draft_0
                reward_0 = compute_reward(draft_0, expected_form=prompt.get("form"))
                reward_1 = best_reward_1

                # 5. Filter
                improvement = reward_1 - reward_0
                if improvement < min_improvement:
                    print(f"  Filtered: improvement {improvement:.3f} < {min_improvement}")
                    trajectories_filtered += 1
                    continue
                if reward_1 < min_draft_1_score:
                    print(f"  Filtered: reward_1 {reward_1:.3f} < {min_draft_1_score}")
                    trajectories_filtered += 1
                    continue

                # 6. Write trajectory
                trajectory = {
                    "prompt": prompt["user"],
                    "system": prompt["system"],
                    "draft_0": draft_0,
                    "critique": critique,
                    "draft_1": draft_1,
                    "reward_0": round(reward_0, 4),
                    "reward_1": round(reward_1, 4),
                    "expected_form": prompt.get("form"),
                }
                f.write(json.dumps(trajectory) + "\n")
                f.flush()

                trajectories_written += 1
                print(f"  Saved: r0={reward_0:.3f} -> r1={reward_1:.3f} (+{improvement:.3f})")

            except Exception as e:
                print(f"  Error: {e}", flush=True)
                continue

    data_vol.commit()
    print(f"\nDone: {trajectories_written} trajectories, {trajectories_filtered} filtered")
    print(f"Output: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-trajectories", type=int, default=5000)
    ap.add_argument("--source-file", type=str, default="poet_train.jsonl")
    ap.add_argument("--output-file", type=str, default="trajectories.jsonl")
    a = ap.parse_args()
    with app.run():
        path = generate_srpo_data.remote(
            max_trajectories=a.max_trajectories,
            source_file=a.source_file,
            output_file=a.output_file,
        )
        print(f"Done: {path}")
