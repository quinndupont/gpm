#!/usr/bin/env python3
"""SRPO Training Data Generator v2 - Hybrid Strategy.

Supports three trajectory generation strategies:
- variance: Generate N completions from same model, pick best/worst
- model_gap: Weak model for rejected, strong model for chosen
- hybrid: Combine variance (40%) and model_gap (60%)

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

EducatorScorer = None  # Lazy import to avoid loading llama_cpp when not needed


def _get_educator_scorer(cfg: dict):
    """Load educator scorer if use_educator_scoring is enabled."""
    global EducatorScorer
    if not cfg.get("use_educator_scoring", False):
        return None
    if EducatorScorer is None:
        from scripts.eval.educator_scorer import EducatorScorer as _ES
        EducatorScorer = _ES
    edu_cfg = cfg.get("educator_scoring", {})
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


def generate_critique(
    draft: str,
    brief: str,
    form: str | None,
    cfg: dict,
) -> str:
    """Generate critique using configured model."""
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
    rate_limiter: "RateLimiter",
    educator_scorer=None,
    edu_weights: dict | None = None,
) -> dict | None:
    """Generate trajectory using variance strategy.

    Generate N candidates from same model, score each, pick best/worst.
    """
    num_candidates = variance_cfg.get("num_candidates", 5)
    min_score_gap = variance_cfg.get("min_score_gap", 0.15)
    model_cfg = variance_cfg.get("model", {})

    drafts = []
    for i in range(num_candidates):
        rate_limiter.wait()
        try:
            draft = call_model(prompt["user"], prompt["system"], model_cfg)
            if educator_scorer:
                scores = educator_scorer.score_poem(
                    poem=draft,
                    brief=prompt["user"],
                    expected_form=prompt.get("form"),
                )
                score = educator_scorer.compute_aggregate_reward(scores, edu_weights)
            else:
                score = compute_reward(draft, expected_form=prompt.get("form"))
            drafts.append({"draft": draft, "score": score, "scores": scores if educator_scorer else None})
            print(f"    Candidate {i+1}/{num_candidates}: score={score:.3f}")
        except Exception as e:
            print(f"    Candidate {i+1} failed: {e}")
            continue

    if len(drafts) < 2:
        print("    Not enough valid candidates")
        return None

    # Sort by score
    drafts.sort(key=lambda x: x["score"])
    rejected = drafts[0]
    chosen = drafts[-1]

    score_gap = chosen["score"] - rejected["score"]
    if score_gap < min_score_gap:
        print(f"    Score gap {score_gap:.3f} < {min_score_gap}")
        return None

    out = {
        "prompt": prompt["user"],
        "system": prompt["system"],
        "draft_0": rejected["draft"],
        "critique": "",
        "draft_1": chosen["draft"],
        "reward_0": round(rejected["score"], 4),
        "reward_1": round(chosen["score"], 4),
        "expected_form": prompt.get("form"),
        "strategy": "variance",
    }
    if educator_scorer and rejected.get("scores") and chosen.get("scores"):
        out["scores_0"] = {k: round(v, 3) for k, v in rejected["scores"].items()}
        out["scores_1"] = {k: round(v, 3) for k, v in chosen["scores"].items()}
    return out


def generate_model_gap_trajectory(
    prompt: dict,
    model_gap_cfg: dict,
    critique_cfg: dict,
    chosen_model_idx: int,
    rate_limiter: "RateLimiter",
    educator_scorer=None,
    edu_weights: dict | None = None,
) -> dict | None:
    """Generate trajectory using model-gap strategy.

    Use weak model (8B) for rejected, strong model (frontier) for chosen.
    Frontier model revises draft_0 based on critique to create draft_1.
    This teaches both generation and self-revision skills.
    """
    rejected_cfg = model_gap_cfg.get("rejected_model", {})
    chosen_models = model_gap_cfg.get("chosen_models", [])

    if not chosen_models:
        raise ValueError("No chosen_models configured for model_gap strategy")

    # Round-robin selection of chosen model
    chosen_cfg = chosen_models[chosen_model_idx % len(chosen_models)]
    chosen_model_name = chosen_cfg.get("model", "unknown")

    # 1. Generate rejected draft with weak model (8B)
    rate_limiter.wait()
    try:
        draft_0 = call_model(prompt["user"], prompt["system"], rejected_cfg)
    except Exception as e:
        print(f"    Failed to generate rejected draft: {e}")
        return None

    if educator_scorer:
        scores_0 = educator_scorer.score_poem(
            poem=draft_0,
            brief=prompt["user"],
            expected_form=prompt.get("form"),
        )
        reward_0 = educator_scorer.compute_aggregate_reward(scores_0, edu_weights)
    else:
        reward_0 = compute_reward(draft_0, expected_form=prompt.get("form"))
        scores_0 = None
    print(f"    Rejected (8B): score={reward_0:.3f}")

    # 2. Generate critique
    rate_limiter.wait()
    try:
        critique = generate_critique(
            draft_0,
            prompt["user"],
            prompt.get("form"),
            critique_cfg,
        )
    except Exception as e:
        print(f"    Failed to generate critique: {e}")
        return None

    # 3. Generate chosen draft with strong model (frontier)
    rate_limiter.wait()
    try:
        draft_1 = generate_revision(
            prompt["user"],
            draft_0,
            critique,
            prompt.get("form"),
            chosen_cfg,
        )
    except Exception as e:
        print(f"    Failed to generate chosen revision with {chosen_model_name}: {e}")
        return None

    if educator_scorer:
        scores_1 = educator_scorer.score_poem(
            poem=draft_1,
            brief=prompt["user"],
            expected_form=prompt.get("form"),
        )
        reward_1 = educator_scorer.compute_aggregate_reward(scores_1, edu_weights)
    else:
        reward_1 = compute_reward(draft_1, expected_form=prompt.get("form"))
        scores_1 = None
    print(f"    Chosen ({chosen_model_name}): score={reward_1:.3f}")

    out = {
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
    }
    if educator_scorer and scores_0 and scores_1:
        out["scores_0"] = {k: round(v, 3) for k, v in scores_0.items()}
        out["scores_1"] = {k: round(v, 3) for k, v in scores_1.items()}
    return out


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

    max_trajectories = args.limit or filter_cfg.get("max_trajectories", 5000)
    min_improvement = filter_cfg.get("min_improvement", 0.10)
    min_chosen_score = filter_cfg.get("min_chosen_score", 0.55)
    requests_per_minute = rate_cfg.get("requests_per_minute", 100)

    if args.dry_run:
        print(f"Strategy: {strategy}")
        print(f"Source: {source_data}")
        print(f"Output: {output_file}")
        print(f"Max trajectories: {max_trajectories}")
        print(f"Min improvement: {min_improvement}")
        print(f"Min chosen score: {min_chosen_score}")
        if strategy in ("model_gap", "hybrid"):
            chosen_models = model_gap_cfg.get("chosen_models", [])
            print(f"Chosen models: {[m.get('model') for m in chosen_models]}")
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

    use_educator_scoring = cfg.get("use_educator_scoring", False)
    educator_scorer = None
    edu_weights = None
    if use_educator_scoring:
        print("Loading educator model for multi-dimensional scoring...")
        educator_scorer = _get_educator_scorer(cfg)
        edu_weights = cfg.get("educator_scoring", {}).get("weights")
        print("  Educator loaded successfully")

    trajectories_written = 0
    trajectories_filtered = 0
    variance_count = 0
    model_gap_count = 0
    chosen_model_idx = 0

    # Hybrid ratios
    variance_ratio = hybrid_cfg.get("variance_ratio", 0.4)

    with open(output_file, "a") as f:
        for i, prompt in enumerate(prompts):
            if trajectories_written >= max_trajectories:
                break

            print(f"\n[{i+1}/{len(prompts)}] Generating trajectory...", flush=True)

            try:
                # Decide strategy for this trajectory
                if strategy == "variance":
                    use_variance = True
                elif strategy == "model_gap":
                    use_variance = False
                else:  # hybrid
                    use_variance = random.random() < variance_ratio

                if use_variance:
                    print("  Strategy: variance")
                    trajectory = generate_variance_trajectory(
                        prompt, variance_cfg, rate_limiter,
                        educator_scorer=educator_scorer,
                        edu_weights=edu_weights,
                    )
                else:
                    print("  Strategy: model_gap")
                    trajectory = generate_model_gap_trajectory(
                        prompt, model_gap_cfg, critique_cfg, chosen_model_idx, rate_limiter,
                        educator_scorer=educator_scorer,
                        edu_weights=edu_weights,
                    )
                    chosen_model_idx += 1

                if trajectory is None:
                    print("  Skipped: strategy returned None")
                    continue

                # Apply filters
                improvement = trajectory["reward_1"] - trajectory["reward_0"]
                if improvement < min_improvement:
                    print(f"  Filtered: improvement {improvement:.3f} < {min_improvement}")
                    trajectories_filtered += 1
                    continue
                if trajectory["reward_1"] < min_chosen_score:
                    print(f"  Filtered: chosen score {trajectory['reward_1']:.3f} < {min_chosen_score}")
                    trajectories_filtered += 1
                    continue

                # Write trajectory
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
