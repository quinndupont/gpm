#!/usr/bin/env python3
"""
Rhyme benchmark: run prompts that request rhyming forms, analyze outputs with rhyme_analyzer.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from scripts.benchmarks.rhyme_bench.prompts import RHYME_PROMPTS
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme

ROOT = Path(__file__).resolve().parents[3]
MODELS_CONFIG = ROOT / "config" / "rev_flux_models.yaml"


def _load_models_config() -> list[dict]:
    import yaml
    if not MODELS_CONFIG.exists():
        return [{"id": "trained", "label": "Trained (GGUF)", "educator": "gguf", "poet": "gguf"}]
    with open(MODELS_CONFIG) as f:
        return yaml.safe_load(f).get("models", [])


def _slug(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")[:32]


def _is_educator_finetuned(model_config: dict) -> bool:
    """Check if a model has a fine-tuned educator.

    Args:
        model_config: Model config dict with 'educator' field

    Returns:
        True if educator is fine-tuned (uses gguf or a path), False otherwise
    """
    educator = model_config.get("educator", "gguf")

    # "gguf" or paths starting with "gguf:" or "./" indicate fine-tuned local models
    # Ollama models (ollama:) and Bedrock models (bedrock:) are not fine-tuned
    if educator == "gguf":
        return True
    if educator.startswith("gguf:"):
        return True
    if educator.startswith("./"):
        return True

    # Ollama and Bedrock models are vanilla/frontier
    return False


def run_single(
    pipeline,
    user_request: str,
    form: str,
    variant: str | None,
    max_revisions: int,
    prompt_idx: int,
    model_id: str,
    verbose: bool = False,
) -> dict:
    result = pipeline.generate(
        user_request,
        max_revisions=max_revisions,
        verbose=verbose,
        interactive=False,
    )
    poem = result.get("final_poem", "")
    analysis = analyze_rhyme(poem, expected_form=form, expected_variant=variant)
    return {
        "user_request": user_request,
        "form": form,
        "variant": variant,
        "prompt_idx": prompt_idx,
        "model_id": model_id,
        "max_revisions": max_revisions,
        "final_poem": poem,
        "rhyme_analysis": {
            "strict_rhyme_density": analysis.get("strict_rhyme_density", 0),
            "rhyme_density": analysis.get("rhyme_density", 0),
            "matches_form": analysis.get("matches_form"),
            "deviations_count": len(analysis.get("deviations", [])),
            "strict_rhyme_pairs": len(analysis.get("strict_rhyme_pairs", [])),
            "rhyme_pairs": len(analysis.get("rhyme_pairs", [])),
            "line_count": analysis.get("line_count", 0),
            "detected_scheme": analysis.get("detected_scheme", ""),
            "expected_scheme": analysis.get("expected_scheme"),
        },
        "metadata": result.get("metadata", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Rhyme benchmark: test rhyming form adherence")
    parser.add_argument(
        "--prompts",
        nargs="+",
        type=int,
        default=None,
        help="Indices of prompts to run (default: all)",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=0,
        help="Revision cycles (0=poet only)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Short test: 2 prompts, trained model only",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model ids from config/rev_flux_models.yaml",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available models and exit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "rhyme_bench",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose pipeline output",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Run diagnostic analysis with failure categorization",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=None,
        help="Specific forms to test (default: all)",
    )
    parser.add_argument(
        "--exclude-educator-finetuned",
        action="store_true",
        help="Exclude models with fine-tuned educators (only vanilla or poet-only fine-tuned)",
    )
    args = parser.parse_args()

    models_config = _load_models_config()

    # Filter out educator fine-tuned models if requested
    if args.exclude_educator_finetuned:
        models_config = [m for m in models_config if not _is_educator_finetuned(m)]
        if not models_config:
            print("No models available after filtering educator fine-tuned models.")
            return

    model_ids = args.models or [m["id"] for m in models_config]
    models_to_run = [m for m in models_config if m["id"] in model_ids]
    if not models_to_run:
        print("No matching models. Use --list-models to see available.")
        return

    if args.list_models:
        for m in models_config:
            mark = " *" if m["id"] in model_ids else ""
            educator_tag = " [educator-finetuned]" if _is_educator_finetuned(m) else ""
            print(f"  {m['id']}: {m.get('label', m['id'])}{mark}{educator_tag}")
        if args.exclude_educator_finetuned:
            filtered_count = sum(1 for m in _load_models_config() if _is_educator_finetuned(m))
            print(f"\n  ({filtered_count} educator fine-tuned models excluded)")
        return

    prompt_indices = args.prompts
    if args.test:
        prompt_indices = prompt_indices or [0, 1]
        models_to_run = [m for m in models_to_run if m["id"] == "trained"]
    if prompt_indices is None:
        prompt_indices = list(range(len(RHYME_PROMPTS)))

    # Filter prompts by form if requested
    if args.forms:
        forms_to_test = set(args.forms)
        filtered_indices = [
            idx for idx in prompt_indices
            if idx < len(RHYME_PROMPTS) and RHYME_PROMPTS[idx][0] in forms_to_test
        ]
        prompt_indices = filtered_indices
        if not prompt_indices:
            print(f"No prompts found for forms: {args.forms}")
            return

    from scripts.inference.pipeline import PoetryPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        for idx in prompt_indices:
            if idx >= len(RHYME_PROMPTS):
                continue
            form, variant, request = RHYME_PROMPTS[idx]
            print(f"[{mid}] form={form} prompt {idx}...", flush=True)
            run = run_single(
                pipeline,
                request,
                form=form,
                variant=variant,
                max_revisions=args.max_revisions,
                prompt_idx=idx,
                model_id=mid,
                verbose=args.verbose,
            )
            runs.append(run)
            slug = _slug(mid)
            out_file = args.output_dir / f"rhyme_{form}_{idx}_{slug}_{run_timestamp}.json"
            with open(out_file, "w") as f:
                json.dump(run, f, indent=2)

            # Enhanced logging with tokens/s and abbreviated performance metrics
            ra = run["rhyme_analysis"]
            meta = run.get("metadata", {})
            tok_s = meta.get("perf_tokens_per_sec", 0)
            time_s = meta.get("perf_total_sec", 0)
            match_symbol = "T" if ra['matches_form'] is True else "F" if ra['matches_form'] is False else "?"
            detected_scheme = ra.get("detected_scheme", "")[:20]  # Truncate long schemes
            expected_scheme = ra.get("expected_scheme", "")

            print(
                f"  ({tok_s:.1f} tok/s, {time_s:.1f}s) "
                f"sd={ra['strict_rhyme_density']:.2f} match={match_symbol} "
                f"devs={ra['deviations_count']} lines={ra['line_count']} "
                f"scheme={detected_scheme} expect={expected_scheme}",
                flush=True,
            )
            print(f"  -> {out_file}", flush=True)

    # Diagnostic analysis if requested
    if args.diagnostic:
        from scripts.benchmarks.rhyme_bench.diagnostic import (
            DiagnosticAnalyzer,
            DiagnosticReport,
        )

        print("\nRunning diagnostic analysis...", flush=True)
        analyzer = DiagnosticAnalyzer()
        report = DiagnosticReport(runs, analyzer)

        # Save JSON report
        diag_path = args.output_dir / "diagnostic_report.json"
        with open(diag_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        # Save markdown summary
        md_path = args.output_dir / "diagnostic_summary.md"
        with open(md_path, "w") as f:
            f.write(report.to_markdown())

        print(f"\nDiagnostic report saved:")
        print(f"  JSON: {diag_path}")
        print(f"  Markdown: {md_path}")

    # Summary
    matches = sum(1 for r in runs if r["rhyme_analysis"].get("matches_form") is True)
    strict_densities = [r["rhyme_analysis"]["strict_rhyme_density"] for r in runs]
    summary = {
        "run_timestamp": run_timestamp,
        "total_runs": len(runs),
        "matches_form_count": matches,
        "matches_form_rate": round(matches / len(runs), 2) if runs else 0,
        "mean_strict_rhyme_density": (
            round(sum(strict_densities) / len(strict_densities), 2) if strict_densities else 0
        ),
        "models_tested": model_ids,
        "prompt_indices": prompt_indices,
        "max_revisions": args.max_revisions,
    }
    # Save timestamped summary
    summary_timestamped_path = args.output_dir / f"summary_{run_timestamp}.json"
    with open(summary_timestamped_path, "w") as f:
        json.dump(summary, f, indent=2)
    # Save latest summary for backward compatibility
    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nSummary: matches_form {matches}/{len(runs)}, "
        f"mean strict_density={summary['mean_strict_rhyme_density']}",
    )
    print(f"  Latest: {summary_path}")
    print(f"  Timestamped: {summary_timestamped_path}")


if __name__ == "__main__":
    main()
