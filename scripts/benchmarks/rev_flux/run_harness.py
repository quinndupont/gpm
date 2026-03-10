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
from scripts.benchmarks.rev_flux.line_change import (
    revision_round_changes,
    aggregate_line_changes,
    revised_lines_per_round,
    lines_changed_per_round,
)


def _load_models_config() -> tuple[list[dict], list[str] | None]:
    import yaml
    default = [{"id": "trained", "label": "Trained (GGUF)", "educator": "gguf", "poet": "gguf", "revisions": [0, 1, 3, 5]}]
    if not MODELS_CONFIG.exists():
        return default, None
    data = yaml.safe_load(open(MODELS_CONFIG)) or {}
    models = data.get("models", default)
    test_models = data.get("test_models")
    return models, test_models


def _slug(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")[:32]


def run_single(
    pipeline,
    user_request: str,
    max_revisions: int,
    category: str,
    prompt_idx: int,
    model_id: str = "trained",
    min_revisions: int = 1,
    verbose: bool = False,
) -> dict:
    """Run one pipeline invocation and attach RevFlux metrics."""
    result = pipeline.generate(
        user_request,
        max_revisions=max_revisions,
        min_revisions=min_revisions if max_revisions > 0 else 0,
        verbose=verbose,
        interactive=False,
    )
    hist = result["revision_history"]
    rounds = revision_round_changes(hist)
    change_pcts = aggregate_line_changes(rounds)
    revised_per_round = revised_lines_per_round(rounds)
    lines_changed = lines_changed_per_round(rounds)
    meta = result.get("metadata", {})
    return {
        "user_request": user_request,
        "category": category,
        "prompt_idx": prompt_idx,
        "max_revisions": max_revisions,
        "model_id": model_id,
        "revisions_actual": len([h for h in hist if h.get("critique")]),
        "change_pcts": change_pcts,
        "per_round_changes": rounds,
        "revised_lines_per_round": [[[i, p] for i, p in r] for r in revised_per_round],
        "lines_changed_per_round": lines_changed,
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
        default=[0, 1, 3, 5],
        help="Revision cycle lengths (0=poet only). Overrides per-model config when set.",
    )
    parser.add_argument(
        "--num-revisions",
        type=int,
        default=None,
        help="Number of revision levels for all models (1-4, from [0,1,3,5]). Default in --test: 3.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Short test: trained only, limit 1 prompt per category",
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
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all vanilla (rev0) + all trained (full sweep)",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Discover GGUF educator+poet pairs from models/ and run them",
    )
    parser.add_argument(
        "--min-revisions",
        type=int,
        default=1,
        help="Minimum revision rounds before honoring educator approval (default: 1). Use 0 to allow early approval.",
    )
    args = parser.parse_args()

    CANONICAL_REVS = [0, 1, 3, 5]
    if args.num_revisions is not None:
        revs_to_run_override = CANONICAL_REVS[: args.num_revisions]
    elif args.test:
        revs_to_run_override = CANONICAL_REVS[:3]
    else:
        revs_to_run_override = None

    models_config, test_models = _load_models_config()

    if args.auto_discover:
        from scripts.training.model_discovery import discover_local_gguf
        discovered = discover_local_gguf()
        by_base = {}
        for m in discovered:
            base = m.base_model_hf_id
            from scripts.training.model_registry import hf_to_short
            short = hf_to_short(base)
            if short not in by_base:
                by_base[short] = {}
            by_base[short][m.task] = m.path
        models_config = []
        for short, tasks in by_base.items():
            edu = tasks.get("educator") or tasks.get("educator-interim")
            poet = tasks.get("poet_rhyme") or tasks.get("poet")
            if edu and poet:
                models_config.append({
                    "id": f"trained-{short}",
                    "label": f"Fine-tuned {short}",
                    "educator": f"gguf:{edu}",
                    "poet": f"gguf:{poet}",
                    "revisions": [0, 1, 3, 5],
                })
        if not models_config:
            print("No educator+poet pairs found in models/*.gguf")
            return
        test_models = None  # discovery overrides config; use all discovered
        print("Auto-discovered:", [m["id"] for m in models_config])

    if args.models:
        model_ids = args.models
    elif args.test and test_models:
        model_ids = test_models
    else:
        model_ids = [m["id"] for m in models_config]
    models_to_run = [m for m in models_config if m["id"] in model_ids]
    if not models_to_run:
        print("No matching models. Use --list-models to see available.")
        return

    if args.compare:
        model_ids = [m["id"] for m in models_config]
        models_to_run = models_config
        args.max_revisions = None
        revs_to_run_override = None  # use per-model config for full sweep

    if args.list_models:
        for m in models_config:
            mark = " *" if m["id"] in model_ids else ""
            print(f"  {m['id']}: {m.get('label', m['id'])}{mark}")
        return

    if args.dry_run:
        total = 0
        limit = 1 if args.test else (args.limit or 999)
        n_prompts = sum(min(len(CATEGORIES[cat]), limit) for cat in args.categories)
        for m in models_to_run:
            if revs_to_run_override is not None:
                revs = revs_to_run_override
            elif args.max_revisions is not None:
                revs = args.max_revisions
            else:
                revs = m.get("revisions", [0])
            total += n_prompts * len(revs)
        print(f"Would run {total} pipeline invocations")
        print(f"Categories: {args.categories}")
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
        if revs_to_run_override is not None:
            revs_to_run = revs_to_run_override
        elif args.max_revisions is not None:
            revs_to_run = args.max_revisions
        else:
            revs_to_run = model_cfg.get("revisions", [0])
        for category in args.categories:
            prompts = CATEGORIES[category]
            n_prompts = 1 if args.test else (args.limit or len(prompts))
            for idx, request in enumerate(prompts[:n_prompts]):
                for max_rev in revs_to_run:
                    print(f"[{mid}] [{category}] prompt {idx + 1}/{n_prompts}, max_revisions={max_rev}...", flush=True)
                    run = run_single(
                        pipeline,
                        request,
                        max_revisions=max_rev,
                        category=category,
                        prompt_idx=idx,
                        model_id=mid,
                        min_revisions=args.min_revisions,
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
                                "per_round_changes": run["per_round_changes"],
                                "revised_lines_per_round": run["revised_lines_per_round"],
                                "lines_changed_per_round": run["lines_changed_per_round"],
                                "revision_history": run["revision_history"],
                                "final_poem": run["final_poem"],
                                "metadata": run["metadata"],
                            },
                            f,
                            indent=2,
                        )
                    print(f"  -> {out_file}", flush=True)

    import statistics
    by_model = {}
    by_model_category: dict[str, dict[str, dict]] = {}
    for r in runs:
        mid = r["model_id"]
        cat = r["category"]
        if mid not in by_model:
            by_model[mid] = {"change_pcts": [], "perf_sec": []}
        by_model[mid]["change_pcts"].extend(r.get("change_pcts", []))
        perf = r.get("metadata", {}).get("perf_total_sec")
        if perf is not None:
            by_model[mid]["perf_sec"].append(perf)
        if mid not in by_model_category:
            by_model_category[mid] = {}
        if cat not in by_model_category[mid]:
            by_model_category[mid][cat] = {"change_pcts": [], "perf_sec": []}
        by_model_category[mid][cat]["change_pcts"].extend(r.get("change_pcts", []))
        if perf is not None:
            by_model_category[mid][cat]["perf_sec"].append(perf)
    model_stats = {}
    for mid, data in by_model.items():
        cps = data["change_pcts"]
        secs = data["perf_sec"]
        model_stats[mid] = {
            "change_pcts_mean": round(statistics.mean(cps), 2) if cps else None,
            "change_pcts_median": round(statistics.median(cps), 2) if cps else None,
            "change_pcts_std": round(statistics.stdev(cps), 2) if len(cps) > 1 else None,
            "perf_total_sec_mean": round(statistics.mean(secs), 2) if secs else None,
            "perf_total_sec_median": round(statistics.median(secs), 2) if secs else None,
        }
    model_stats_by_category: dict[str, dict[str, dict]] = {}
    for mid, cat_data in by_model_category.items():
        model_stats_by_category[mid] = {}
        for cat, data in cat_data.items():
            cps = data["change_pcts"]
            secs = data["perf_sec"]
            model_stats_by_category[mid][cat] = {
                "change_pcts_mean": round(statistics.mean(cps), 2) if cps else None,
                "change_pcts_median": round(statistics.median(cps), 2) if cps else None,
                "change_pcts_std": round(statistics.stdev(cps), 2) if len(cps) > 1 else None,
                "perf_total_sec_mean": round(statistics.mean(secs), 2) if secs else None,
                "perf_total_sec_median": round(statistics.median(secs), 2) if secs else None,
            }
    summary = {
        "total_runs": len(runs),
        "categories": args.categories,
        "models_tested": model_ids,
        "aggregate_change_pcts": [p for r in runs for p in r.get("change_pcts", [])],
        "model_stats": model_stats,
        "model_stats_by_category": model_stats_by_category,
    }
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    if model_stats:
        print("\nComparison (overall):")
        for mid in sorted(model_stats):
            s = model_stats[mid]
            cp = s.get("change_pcts_mean", "N/A")
            pf = s.get("perf_total_sec_mean", "N/A")
            print(f"  {mid}: {cp} | {pf}s")
    if model_stats_by_category:
        print("\nBy category:")
        for cat in args.categories:
            print(f"  {cat}:")
            for mid in sorted(model_stats_by_category):
                if cat not in model_stats_by_category[mid]:
                    continue
                s = model_stats_by_category[mid][cat]
                cp = s.get("change_pcts_mean", "N/A")
                pf = s.get("perf_total_sec_mean", "N/A")
                print(f"    {mid}: {cp} | {pf}s")

    if args.visualize and runs:
        import subprocess
        viz_py = ["uv", "run", "--with", "matplotlib", "python"]
        harness_script = str(ROOT / "scripts" / "benchmarks" / "rev_flux" / "visualize_harness.py")
        dashboard_script = str(ROOT / "scripts" / "benchmarks" / "rev_flux" / "visualize_dashboard.py")
        out_plots = str(args.output_dir / "plots")
        subprocess.run(
            viz_py + [harness_script, str(args.output_dir), "-o", out_plots, "--comparison-only"],
            cwd=str(ROOT),
            check=True,
        )
        subprocess.run(
            viz_py + [dashboard_script, str(args.output_dir), "-o", out_plots],
            cwd=str(ROOT),
            check=True,
        )


if __name__ == "__main__":
    main()
