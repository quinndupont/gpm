#!/usr/bin/env python3
"""SRPO Training Data Generator.

Generates trajectories for Self-Refinement Policy Optimization:
    (prompt, draft_0, critique, draft_1)

Pipeline:
1. Load prompts from existing poet training data
2. Generate draft_0 using current SFT poet checkpoint
3. Generate critique using Educator model (with rhyme analysis)
4. Generate draft_1 (gold revision) using Claude API
5. Score both drafts with rhyme_analyzer.compute_reward()
6. Filter: keep only trajectories where reward_1 > reward_0 + threshold
7. Output JSONL in SRPO format
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.prompts.loader import get_persona, render_prompt
from scripts.data_generation.claude_utils import call_claude, load_env
from scripts.eval.form_registry import detect_form, get_scheme, is_rhyming_form
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme
from scripts.eval.rhyme_analyzer import compute_reward, format_analysis_for_prompt


def load_config(config_path: Path) -> dict:
    """Load SRPO data generation config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_poet_prompts(source_path: Path, limit: int | None = None) -> list[dict]:
    """Extract prompts from poet training data.

    Returns list of dicts with 'system', 'user' (the brief), and optionally 'form'.
    """
    prompts = []
    if not source_path.exists():
        raise FileNotFoundError(f"Source data not found: {source_path}")

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

    if limit:
        prompts = prompts[:limit]
    return prompts


def generate_draft_with_model(
    prompt: dict,
    model_path: str,
    temperature: float = 0.8,
    max_tokens: int = 512,
) -> str:
    """Generate a draft poem using a local model via Ollama or llama.cpp."""
    # Try Ollama first
    try:
        from ollama import chat

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        response = chat(
            model=model_path,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        if hasattr(response, "message"):
            return getattr(response.message, "content", "") or ""
        return response.get("message", {}).get("content", "") or ""
    except ImportError:
        pass
    except Exception as e:
        print(f"  Ollama error: {e}", file=sys.stderr)

    # Fallback to llama.cpp if available
    try:
        from llama_cpp import Llama

        llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]
    except ImportError:
        raise RuntimeError("Neither ollama nor llama-cpp-python available")


def generate_critique(
    draft: str,
    brief: str,
    form: str | None,
    use_educator: bool = True,
    educator_model: str = "qwen2.5:7b-instruct",
    include_rhyme_analysis: bool = True,
) -> str:
    """Generate a critique of the draft poem."""
    # Build critique prompt with form analysis if applicable
    form_ctx = ""
    if form and is_rhyming_form(form) and include_rhyme_analysis:
        rhyme = analyze_rhyme(draft, expected_form=form)
        form_ctx = f"\n\nRhyme analysis (automated):\n{format_analysis_for_prompt(rhyme)}\n"

    critique_prompt = render_prompt(
        "inference", "critique",
        brief=brief, draft=draft, history_ctx="", form_ctx=form_ctx,
    )

    system = get_persona("educator_neutral")

    if use_educator:
        # Use local educator model
        try:
            from ollama import chat

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": critique_prompt},
            ]
            response = chat(
                model=educator_model,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 600},
            )
            if hasattr(response, "message"):
                return getattr(response.message, "content", "") or ""
            return response.get("message", {}).get("content", "") or ""
        except Exception as e:
            print(f"  Educator error, falling back to Claude: {e}", file=sys.stderr)

    # Fallback to Claude
    return call_claude(critique_prompt, system, model="claude-sonnet-4-20250514", max_tokens=600)


def generate_revision(
    brief: str,
    draft: str,
    critique: str,
    form: str | None,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.7,
    max_tokens: int = 600,
) -> str:
    """Generate a gold-standard revision using Claude."""
    # Build rhyme context if applicable
    rhyme_ctx = ""
    if form and is_rhyming_form(form):
        rhyme = analyze_rhyme(draft, expected_form=form)
        if rhyme.get("deviations"):
            dev_lines = []
            for d in rhyme["deviations"]:
                dev_lines.append(
                    f'- Line {d["line"]}: "{d["word"]}" does not rhyme with '
                    f'"{d["expected_rhyme_with"]}" (needs {d["expected_label"]} rhyme)'
                )
            rhyme_ctx = "\n\nRhyme errors to fix:\n" + "\n".join(dev_lines) + "\n"

    # Use SRPO revision prompt
    revision_prompt = render_prompt(
        "tuning", "poet_revision_srpo",
        brief=brief, draft=draft, critique=critique, rhyme_ctx=rhyme_ctx,
    )

    system = get_persona("poet")
    return call_claude(revision_prompt, system, model=model, max_tokens=max_tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate SRPO training data")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "srpo_data_generation.yaml",
        help="Config file path",
    )
    parser.add_argument("--limit", type=int, help="Limit number of trajectories")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print config and exit"
    )
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config)

    source_data = ROOT / cfg["source_data"]
    output_dir = ROOT / cfg["output_dir"]
    output_file = output_dir / cfg["output_filename"]

    draft_cfg = cfg.get("draft_generation", {})
    critique_cfg = cfg.get("critique_generation", {})
    revision_cfg = cfg.get("revision_generation", {})
    filter_cfg = cfg.get("filtering", {})
    rate_cfg = cfg.get("rate_limits", {})

    max_trajectories = args.limit or filter_cfg.get("max_trajectories", 5000)
    min_improvement = filter_cfg.get("min_improvement", 0.05)
    min_draft_1_score = filter_cfg.get("min_draft_1_score", 0.5)
    candidates_per_draft = revision_cfg.get("candidates_per_draft", 2)
    requests_per_minute = rate_cfg.get("requests_per_minute", 50)

    if args.dry_run:
        print(f"Source: {source_data}")
        print(f"Output: {output_file}")
        print(f"Max trajectories: {max_trajectories}")
        print(f"Min improvement: {min_improvement}")
        print(f"Candidates per draft: {candidates_per_draft}")
        return

    # Load prompts
    print(f"Loading prompts from {source_data}...")
    prompts = load_poet_prompts(source_data, limit=max_trajectories * 2)
    random.shuffle(prompts)
    print(f"  Loaded {len(prompts)} prompts")

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories_written = 0
    trajectories_filtered = 0

    # Rate limiting
    min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
    last_request_time = 0.0

    with open(output_file, "w") as f:
        for i, prompt in enumerate(prompts):
            if trajectories_written >= max_trajectories:
                break

            print(f"[{i+1}/{len(prompts)}] Generating trajectory...", flush=True)

            try:
                # 1. Generate draft_0 using SFT poet
                # For now, use Claude to simulate poet (in production, use local checkpoint)
                draft_0 = call_claude(
                    prompt["user"],
                    prompt["system"],
                    model="claude-sonnet-4-20250514",
                    max_tokens=draft_cfg.get("max_tokens", 512),
                )

                # Rate limit
                elapsed = time.time() - last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_request_time = time.time()

                # 2. Generate critique
                critique = generate_critique(
                    draft_0,
                    prompt["user"],
                    prompt.get("form"),
                    use_educator=critique_cfg.get("use_educator", True),
                    include_rhyme_analysis=critique_cfg.get("include_rhyme_analysis", True),
                )

                # Rate limit
                elapsed = time.time() - last_request_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)
                last_request_time = time.time()

                # 3. Generate revision candidates
                best_revision = None
                best_reward_1 = -1.0

                for _ in range(candidates_per_draft):
                    revision = generate_revision(
                        prompt["user"],
                        draft_0,
                        critique,
                        prompt.get("form"),
                        model=revision_cfg.get("model", "claude-sonnet-4-20250514"),
                        temperature=revision_cfg.get("temperature", 0.7),
                        max_tokens=revision_cfg.get("max_tokens", 600),
                    )

                    # Rate limit
                    elapsed = time.time() - last_request_time
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                    last_request_time = time.time()

                    # Score revision
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
                print(f"  Error: {e}", file=sys.stderr)
                continue

    print(f"\nDone: {trajectories_written} trajectories written, {trajectories_filtered} filtered")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
