#!/usr/bin/env python3
"""SRPO Training Data Generator v2 - Hybrid Strategy.

Educator for critique only; rhyme-only rewards; shared frontier rotation.

Strategies:
- variance: Poet N samples → pick worst → educator critique → frontier revise
- model_gap: Weak model draft_0 → educator critique → frontier revise
- hybrid: Combine variance (30%) and model_gap (70%)

Usage:
    python generate_srpo_data_v2.py --config config/srpo_data_generation_v2.yaml
    python generate_srpo_data_v2.py --strategy variance --limit 10
    python generate_srpo_data_v2.py --strategy model_gap --limit 10
    python generate_srpo_data_v2.py --strategy hybrid --limit 20
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Literal

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.prompts.loader import get_persona, render_prompt
from scripts.data_generation.claude_utils import call_claude, load_env
from scripts.eval.form_registry import detect_form, get_scheme, is_rhyming_form
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme
from scripts.eval.rhyme_analyzer import compute_reward, format_analysis_for_prompt

EducatorScorer = None  # Lazy import


def _get_educator_critic(cfg: dict):
    """Load educator model for critique when use_educator_critique is enabled."""
    global EducatorScorer
    if not cfg.get("use_educator_critique", True):
        return None
    if EducatorScorer is None:
        from scripts.eval.educator_scorer import EducatorScorer as _ES
        EducatorScorer = _ES
    edu_cfg = cfg.get("educator_critique", cfg.get("educator_scoring", {}))
    model_path = edu_cfg.get("model_path", "llama3.1-8b-educator-Q4_K_M.gguf")
    path = Path(model_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Educator model not found: {path}")
    return EducatorScorer(path)


def load_config(config_path: Path) -> dict:
    """Load SRPO data generation config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_poet_prompts(source_path: Path, limit: int | None = None) -> list[dict]:
    """Extract prompts from poet training data."""
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


def call_model(user: str, system: str, cfg: dict) -> str:
    """Unified model caller supporting ollama and bedrock backends."""
    backend = cfg.get("backend", "bedrock")
    model = cfg.get("model", "llama3.1:8b")
    temperature = cfg.get("temperature", 0.8)
    max_tokens = cfg.get("max_tokens", 512)

    if backend == "ollama":
        from ollama import chat

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response = chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        if hasattr(response, "message"):
            return getattr(response.message, "content", "") or ""
        return response.get("message", {}).get("content", "") or ""
    else:
        # Bedrock via call_claude
        return call_claude(
            user,
            system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def generate_critique_bedrock(
    draft: str,
    brief: str,
    form: str | None,
    cfg: dict,
) -> str:
    """Generate critique using Bedrock (when educator GGUF not used)."""
    form_ctx = ""
    if form and is_rhyming_form(form) and cfg.get("include_rhyme_analysis", True):
        rhyme = analyze_rhyme(draft, expected_form=form)
        form_ctx = f"\n\nRhyme analysis (automated):\n{format_analysis_for_prompt(rhyme)}\n"

    critique_prompt = render_prompt(
        "inference", "critique",
        brief=brief, draft=draft, history_ctx="", form_ctx=form_ctx,
    )
    system = get_persona("educator_neutral")
    return call_model(critique_prompt, system, cfg).strip()


def generate_critique_for_draft(
    draft: str,
    brief: str,
    form: str | None,
    educator_critic,
    critique_cfg: dict,
    use_educator: bool,
) -> str:
    """Generate critique using educator GGUF or Bedrock fallback."""
    if use_educator and educator_critic:
        return educator_critic.generate_critique(
            poem=draft,
            brief=brief,
            expected_form=form,
            include_rhyme_analysis=critique_cfg.get("include_rhyme_analysis", True),
        )
    return generate_critique_bedrock(draft, brief, form, critique_cfg)


def generate_revision(
    brief: str,
    draft: str,
    critique: str,
    form: str | None,
    cfg: dict,
) -> str:
    """Generate revision using configured model."""
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

    revision_prompt = render_prompt(
        "tuning", "poet_revision_srpo",
        brief=brief, draft=draft, critique=critique, rhyme_ctx=rhyme_ctx,
    )
    system = get_persona("poet")

    return call_model(revision_prompt, system, cfg)


def generate_variance_trajectory(
    prompt: dict,
    variance_cfg: dict,
    frontier_models: list,
    frontier_idx: int,
    educator_critic,
    critique_cfg: dict,
    use_educator_critique: bool,
    rate_limiter: "RateLimiter",
) -> tuple[dict | None, int]:
    """Variance: poet N samples → pick worst → educator critique → frontier revise.

    Returns (trajectory, next_frontier_idx).
    """
    num_candidates = variance_cfg.get("num_candidates", 5)
    min_score_gap = variance_cfg.get("min_score_gap", 0.10)
    model_cfg = variance_cfg.get("model", {})

    drafts = []
    for i in range(num_candidates):
        rate_limiter.wait()
        try:
            draft = call_model(prompt["user"], prompt["system"], model_cfg)
            score = compute_reward(draft, expected_form=prompt.get("form"))
            drafts.append({"draft": draft, "score": score})
            print(f"    Candidate {i+1}/{num_candidates}: score={score:.3f}")
        except Exception as e:
            print(f"    Candidate {i+1} failed: {e}")
            continue

    if len(drafts) < 2:
        print("    Not enough valid candidates")
        return None, frontier_idx

    drafts.sort(key=lambda x: x["score"])
    rejected = drafts[0]
    score_gap = drafts[-1]["score"] - rejected["score"]
    if score_gap < min_score_gap:
        print(f"    Score gap {score_gap:.3f} < {min_score_gap}")
        return None, frontier_idx

    draft_0 = rejected["draft"]
    reward_0 = rejected["score"]

    # Educator critique
    rate_limiter.wait()
    try:
        critique = generate_critique_for_draft(
            draft_0, prompt["user"], prompt.get("form"),
            educator_critic, critique_cfg, use_educator_critique,
        )
    except Exception as e:
        print(f"    Failed to generate critique: {e}")
        return None, frontier_idx

    # Frontier revise
    frontier_cfg = frontier_models[frontier_idx % len(frontier_models)]
    frontier_idx = (frontier_idx + 1) % len(frontier_models)
    rate_limiter.wait()
    try:
        draft_1 = generate_revision(
            prompt["user"], draft_0, critique, prompt.get("form"), frontier_cfg,
        )
    except Exception as e:
        print(f"    Failed to generate revision: {e}")
        return None, frontier_idx

    reward_1 = compute_reward(draft_1, expected_form=prompt.get("form"))
    print(f"    Variance: r0={reward_0:.3f} -> r1={reward_1:.3f}")

    return {
        "prompt": prompt["user"],
        "system": prompt["system"],
        "draft_0": draft_0,
        "critique": critique,
        "draft_1": draft_1,
        "reward_0": round(reward_0, 4),
        "reward_1": round(reward_1, 4),
        "expected_form": prompt.get("form"),
        "strategy": "variance",
    }, frontier_idx


def generate_model_gap_trajectory(
    prompt: dict,
    model_gap_cfg: dict,
    frontier_models: list,
    frontier_idx: int,
    educator_critic,
    critique_cfg: dict,
    use_educator_critique: bool,
    rate_limiter: "RateLimiter",
) -> tuple[dict | None, int]:
    """Model-gap: weak model draft_0 → educator critique → frontier revise.

    Returns (trajectory, next_frontier_idx).
    """
    rejected_cfg = model_gap_cfg.get("rejected_model", {})
    if not frontier_models:
        raise ValueError("No frontier_models configured")

    frontier_cfg = frontier_models[frontier_idx % len(frontier_models)]
    chosen_model_name = frontier_cfg.get("model", "unknown")

    # 1. Generate draft_0 with weak model
    rate_limiter.wait()
    try:
        draft_0 = call_model(prompt["user"], prompt["system"], rejected_cfg)
    except Exception as e:
        print(f"    Failed to generate rejected draft: {e}")
        return None, frontier_idx

    reward_0 = compute_reward(draft_0, expected_form=prompt.get("form"))
    print(f"    Rejected (8B): score={reward_0:.3f}")

    # 2. Generate critique (educator or Bedrock)
    rate_limiter.wait()
    try:
        critique = generate_critique_for_draft(
            draft_0, prompt["user"], prompt.get("form"),
            educator_critic, critique_cfg, use_educator_critique,
        )
    except Exception as e:
        print(f"    Failed to generate critique: {e}")
        return None, frontier_idx

    # 3. Frontier revise
    frontier_idx = (frontier_idx + 1) % len(frontier_models)
    rate_limiter.wait()
    try:
        draft_1 = generate_revision(
            prompt["user"], draft_0, critique, prompt.get("form"), frontier_cfg,
        )
    except Exception as e:
        print(f"    Failed to generate revision with {chosen_model_name}: {e}")
        return None, frontier_idx

    reward_1 = compute_reward(draft_1, expected_form=prompt.get("form"))
    print(f"    Chosen ({chosen_model_name}): score={reward_1:.3f}")

    return {
        "prompt": prompt["user"],
        "system": prompt["system"],
        "draft_0": draft_0,
        "critique": critique,
        "draft_1": draft_1,
        "reward_0": round(reward_0, 4),
        "reward_1": round(reward_1, 4),
        "expected_form": prompt.get("form"),
        "strategy": "model_gap",
        "chosen_model": chosen_model_name,
    }, frontier_idx


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 100):
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request_time = 0.0

    def wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


def main():
    parser = argparse.ArgumentParser(description="Generate SRPO training data v2")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "srpo_data_generation_v2.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--strategy",
        choices=["variance", "model_gap", "hybrid"],
        help="Override trajectory strategy from config",
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

    strategy = args.strategy or cfg.get("trajectory_strategy", "hybrid")
    variance_cfg = cfg.get("variance", {})
    model_gap_cfg = cfg.get("model_gap", {})
    hybrid_cfg = cfg.get("hybrid", {})
    critique_cfg = cfg.get("critique", {})
    filter_cfg = cfg.get("filtering", {})
    rate_cfg = cfg.get("rate_limits", {})
    frontier_models = cfg.get("frontier_models", model_gap_cfg.get("chosen_models", []))

    max_trajectories = args.limit or filter_cfg.get("max_trajectories", 5000)
    min_improvement = filter_cfg.get("min_improvement", 0.05)
    min_chosen_score = filter_cfg.get("min_chosen_score", 0.55)
    requests_per_minute = rate_cfg.get("requests_per_minute", 100)
    use_educator_critique = cfg.get("use_educator_critique", True)

    if args.dry_run:
        print(f"Strategy: {strategy}")
        print(f"Source: {source_data}")
        print(f"Output: {output_file}")
        print(f"Max trajectories: {max_trajectories}")
        print(f"Min improvement: {min_improvement}")
        print(f"Min chosen score: {min_chosen_score}")
        print(f"Educator critique: {use_educator_critique}")
        print(f"Frontier models: {[m.get('model') for m in frontier_models]}")
        return

    # Load prompts
    print(f"Loading prompts from {source_data}...")
    prompts = load_poet_prompts(source_data, limit=max_trajectories * 3)
    random.shuffle(prompts)
    print(f"  Loaded {len(prompts)} prompts")

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing trajectories
    existing_count = 0
    if output_file.exists():
        with open(output_file) as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"Found {existing_count} existing trajectories in {output_file.name}")
        print(f"Appending up to {max_trajectories} new trajectories...")
    else:
        print(f"Creating new file: {output_file.name}")

    rate_limiter = RateLimiter(requests_per_minute)

    educator_critic = None
    if use_educator_critique:
        print("Loading educator model for critique...")
        educator_critic = _get_educator_critic(cfg)
        print("  Educator loaded successfully")
    else:
        print("Using Bedrock for critique (use_educator_critique: false)")

    trajectories_written = 0
    trajectories_filtered = 0
    variance_count = 0
    model_gap_count = 0
    frontier_idx = 0

    variance_ratio = hybrid_cfg.get("variance_ratio", 0.3)

    with open(output_file, "a") as f:
        for i, prompt in enumerate(prompts):
            if trajectories_written >= max_trajectories:
                break

            print(f"\n[{i+1}/{len(prompts)}] Generating trajectory...", flush=True)

            try:
                if strategy == "variance":
                    use_variance = True
                elif strategy == "model_gap":
                    use_variance = False
                else:
                    use_variance = random.random() < variance_ratio

                if use_variance:
                    print("  Strategy: variance")
                    trajectory, frontier_idx = generate_variance_trajectory(
                        prompt, variance_cfg, frontier_models, frontier_idx,
                        educator_critic, critique_cfg, use_educator_critique,
                        rate_limiter,
                    )
                else:
                    print("  Strategy: model_gap")
                    trajectory, frontier_idx = generate_model_gap_trajectory(
                        prompt, model_gap_cfg, frontier_models, frontier_idx,
                        educator_critic, critique_cfg, use_educator_critique,
                        rate_limiter,
                    )

                if trajectory is None:
                    print("  Skipped: strategy returned None")
                    continue

                improvement = trajectory["reward_1"] - trajectory["reward_0"]
                if improvement < min_improvement:
                    print(f"  Filtered: improvement {improvement:.3f} < {min_improvement}")
                    trajectories_filtered += 1
                    continue
                if trajectory["reward_1"] < min_chosen_score:
                    print(f"  Filtered: chosen score {trajectory['reward_1']:.3f} < {min_chosen_score}")
                    trajectories_filtered += 1
                    continue

                f.write(json.dumps(trajectory) + "\n")
                f.flush()

                trajectories_written += 1
                if trajectory.get("strategy") == "variance":
                    variance_count += 1
                else:
                    model_gap_count += 1

                print(f"  Saved: r0={trajectory['reward_0']:.3f} -> r1={trajectory['reward_1']:.3f} (+{improvement:.3f})")

            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                continue

    print(f"\n{'='*50}")
    print(f"Done: {trajectories_written} new trajectories written, {trajectories_filtered} filtered")
    print(f"  Variance: {variance_count}")
    print(f"  Model-gap: {model_gap_count}")
    total_count = existing_count + trajectories_written
    print(f"Total trajectories in file: {total_count} ({existing_count} existing + {trajectories_written} new)")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
