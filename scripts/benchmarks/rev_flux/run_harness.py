#!/usr/bin/env python3
"""
RevFlux test harness: run many revision cycles across prompt categories and revision lengths.
Outputs JSON per run for visualization. Evaluates process dynamics, not outcome quality.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.rev_flux.prompts import CATEGORIES
from scripts.benchmarks.rev_flux.line_change import revision_round_changes, aggregate_line_changes


def run_single(
    pipeline,
    user_request: str,
    max_revisions: int,
    category: str,
    prompt_idx: int,
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
    return {
        "user_request": user_request,
        "category": category,
        "prompt_idx": prompt_idx,
        "max_revisions": max_revisions,
        "revisions_actual": len(result["revision_history"]),
        "change_pcts": change_pcts,
        "revision_history": result["revision_history"],
        "final_poem": result["final_poem"],
        "metadata": result.get("metadata", {}),
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
    args = parser.parse_args()

    from scripts.inference.pipeline import PoetryPipeline

    pipeline = PoetryPipeline()

    if args.dry_run:
        total = 0
        for cat in args.categories:
            prompts = CATEGORIES[cat]
            n = min(len(prompts), args.limit or len(prompts))
            total += n * len(args.max_revisions)
        print(f"Would run {total} pipeline invocations")
        print(f"Categories: {args.categories}")
        print(f"Max revisions: {args.max_revisions}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for category in args.categories:
        prompts = CATEGORIES[category]
        limit = args.limit or len(prompts)
        for idx, request in enumerate(prompts[:limit]):
            for max_rev in args.max_revisions:
                print(f"[{category}] prompt {idx + 1}/{limit}, max_revisions={max_rev}...", flush=True)
                run = run_single(
                    pipeline,
                    request,
                    max_revisions=max_rev,
                    category=category,
                    prompt_idx=idx,
                    verbose=args.verbose,
                )
                runs.append(run)
                out_file = args.output_dir / f"{category}_{idx}_rev{max_rev}.json"
                with open(out_file, "w") as f:
                    json.dump(
                        {
                            "user_request": run["user_request"],
                            "category": run["category"],
                            "prompt_idx": run["prompt_idx"],
                            "max_revisions": run["max_revisions"],
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
        "aggregate_change_pcts": [p for r in runs for p in r["change_pcts"]],
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
