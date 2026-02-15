#!/usr/bin/env python3
"""
RevFlux test harness: run many revision cycles across prompt categories, revision lengths,
and model configs (trained GGUF vs vanilla Ollama). Evaluates process dynamics, not outcome.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODELS_CONFIG = ROOT / "config" / "rev_flux_models.yaml"

from scripts.benchmarks.rev_flux.prompts import CATEGORIES
from scripts.benchmarks.rev_flux.line_change import revision_round_changes, aggregate_line_changes


def _load_models_config() -> list[dict]:
    import yaml
    if not MODELS_CONFIG.exists():
        return [{"id": "trained", "label": "Trained (GGUF)", "educator": "gguf", "poet": "gguf"}]
    with open(MODELS_CONFIG) as f:
        return yaml.safe_load(f).get("models", [])


def _slug(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")[:32]


def run_single(
    pipeline,
    user_request: str,
    max_revisions: int,
    category: str,
    prompt_idx: int,
    model_id: str = "trained",
    verbose: bool = False,
) -> dict:
    """Run one pipeline invocation and attach RevFlux metrics."""
    result = pipeline.generate(
        user_request,
        max_revisions=max_revisions,
        verbose=verbose,
        interactive=False,
    )
    rounds = revision_round_changes(result["revision_history"])
    change_pcts = aggregate_line_changes(rounds)
    hist = result["revision_history"]
    meta = result.get("metadata", {})
    return {
        "user_request": user_request,
        "category": category,
        "prompt_idx": prompt_idx,
        "max_revisions": max_revisions,
        "model_id": model_id,
        "revisions_actual": len(hist),
        "change_pcts": change_pcts,
        "revision_history": hist,
        "final_poem": result["final_poem"],
        "metadata": {**meta, "approved": meta.get("approved", False), "approved_at_round": meta.get("approved_at_round")},
    }


def main():
    parser = argparse.ArgumentParser(
        description="RevFlux harness: run revision cycles across prompts and revision lengths"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(CATEGORIES),
        choices=list(CATEGORIES),
        help="Prompt categories to run",
    )
    parser.add_argument(
        "--max-revisions",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Revision cycle lengths to test (e.g. 1 2 3 4)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max prompts per category (default: all)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids to run (from config/rev_flux_models.yaml). Default: all.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model configs and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "rev_flux",
        help="Directory for output JSON files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose pipeline output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Run harness visualizations after completion",
    )
    args = parser.parse_args()

    models_config = _load_models_config()
    model_ids = args.models or [m["id"] for m in models_config]
    models_to_run = [m for m in models_config if m["id"] in model_ids]
    if not models_to_run:
        print("No matching models. Use --list-models to see available.")
        return

    if args.list_models:
        for m in models_config:
            mark = " *" if m["id"] in model_ids else ""
            print(f"  {m['id']}: {m.get('label', m['id'])}{mark}")
        return

    if args.dry_run:
        total = 0
        for cat in args.categories:
            prompts = CATEGORIES[cat]
            n = min(len(prompts), args.limit or len(prompts))
            total += n * len(args.max_revisions) * len(models_to_run)
        print(f"Would run {total} pipeline invocations")
        print(f"Categories: {args.categories}")
        print(f"Max revisions: {args.max_revisions}")
        print(f"Models: {[m['id'] for m in models_to_run]}")
        return

    from scripts.inference.pipeline import PoetryPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for model_cfg in models_to_run:
        mid = model_cfg["id"]
        edu = model_cfg.get("educator", "gguf")
        poet = model_cfg.get("poet", "gguf")
        edu_override = None if edu == "gguf" else edu
        poet_override = None if poet == "gguf" else poet
        pipeline = PoetryPipeline(
            educator_model_override=edu_override,
            poet_model_override=poet_override,
        )
        for category in args.categories:
            prompts = CATEGORIES[category]
            limit = args.limit or len(prompts)
            for idx, request in enumerate(prompts[:limit]):
                for max_rev in args.max_revisions:
                    print(f"[{mid}] [{category}] prompt {idx + 1}/{limit}, max_revisions={max_rev}...", flush=True)
                    run = run_single(
                        pipeline,
                        request,
                        max_revisions=max_rev,
                        category=category,
                        prompt_idx=idx,
                        model_id=mid,
                        verbose=args.verbose,
                    )
                    runs.append(run)
                    slug = _slug(mid)
                    out_file = args.output_dir / f"{category}_{idx}_rev{max_rev}_{slug}.json"
                    with open(out_file, "w") as f:
                        json.dump(
                            {
                                "user_request": run["user_request"],
                                "category": run["category"],
                                "prompt_idx": run["prompt_idx"],
                                "max_revisions": run["max_revisions"],
                                "model_id": run["model_id"],
                                "revisions_actual": run["revisions_actual"],
                                "change_pcts": run["change_pcts"],
                                "revision_history": run["revision_history"],
                                "final_poem": run["final_poem"],
                                "metadata": run["metadata"],
                            },
                            f,
                            indent=2,
                        )
                    print(f"  -> {out_file}", flush=True)

    summary = {
        "total_runs": len(runs),
        "categories": args.categories,
        "max_revisions_tested": args.max_revisions,
        "models_tested": model_ids,
        "aggregate_change_pcts": [p for r in runs for p in r["change_pcts"]],
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    if args.visualize and runs:
        import subprocess
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "benchmarks" / "rev_flux" / "visualize_harness.py"), str(args.output_dir), "-o", str(args.output_dir / "plots")],
            cwd=str(ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
