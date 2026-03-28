#!/usr/bin/env python3
"""
Rhyme benchmark: run prompts that request rhyming forms, analyze outputs with rhyme_analyzer.

Default: interactive configuration. Use --non-interactive for scripts/CI (argparse + optional --bench-config JSON).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts.benchmarks.rhyme_bench.prompts import RHYME_PROMPTS
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme

ROOT = Path(__file__).resolve().parents[3]
MODELS_CONFIG = ROOT / "config" / "rev_flux_models.yaml"
STUDIES_DIR = Path(__file__).resolve().parent / "studies"

STUDY_REGISTRY: dict[str, dict[str, Any]] = {
    "baseline_default": {
        "label": "Baseline — default poet prompt (forward scheme contract)",
        "poet_generation_mode": "default",
        "cmu_second_pass": False,
    },
    "ablate_backward": {
        "label": "Ablate — backward construction instructions",
        "poet_generation_mode": "backward",
        "cmu_second_pass": False,
    },
    "ablate_cmu_two_pass": {
        "label": "Ablate — CMU analysis then second poet pass",
        "poet_generation_mode": "default",
        "cmu_second_pass": True,
    },
}


def _load_models_config() -> list[dict]:
    import yaml
    if not MODELS_CONFIG.exists():
        return [{"id": "trained", "label": "Trained (GGUF)", "educator": "gguf", "poet": "gguf"}]
    with open(MODELS_CONFIG) as f:
        return yaml.safe_load(f).get("models", [])


def _slug(s: str) -> str:
    return s.replace(":", "-").replace("/", "_")[:32]


def _is_educator_finetuned(model_config: dict) -> bool:
    """True if this config uses a local/GGUF educator (rhyme bench never runs these)."""
    educator = model_config.get("educator", "gguf")
    if educator == "gguf":
        return True
    if educator.startswith("gguf:"):
        return True
    if educator.startswith("./"):
        return True
    return False


def _models_config_for_rhyme_bench() -> list[dict]:
    """Load rev_flux models and drop entries that use a GGUF/local educator."""
    raw = _load_models_config()
    return [m for m in raw if not _is_educator_finetuned(m)]


@dataclass
class BenchConfig:
    study_id: str = "baseline_default"
    output_dir: Path = field(
        default_factory=lambda: ROOT / "data" / "rhyme_bench" / "studies" / "baseline_default",
    )
    models: list[str] | None = None  # None = all selected from config
    prompt_indices: list[int] | None = None  # None = all
    test: bool = False
    max_revisions: int = 0
    diagnostic: bool = False
    forms: list[str] | None = None
    verbose: bool = False
    list_models_only: bool = False

    def to_summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        reg = STUDY_REGISTRY.get(self.study_id, {})
        d["poet_generation_mode"] = reg.get("poet_generation_mode", "default")
        d["cmu_second_pass"] = reg.get("cmu_second_pass", False)
        return d


def _default_output_dir_for_study(study_id: str) -> Path:
    return ROOT / "data" / "rhyme_bench" / "studies" / study_id


def _prompt_line(prompt: str, default: str = "") -> str:
    if default:
        raw = input(f"{prompt} [{default}]: ").strip()
        return raw or default
    return input(f"{prompt}: ").strip()


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suf = "Y/n" if default else "y/N"
    raw = input(f"{prompt} ({suf}): ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.replace(",", " ").split():
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def prompt_bench_config() -> BenchConfig:
    print("\n=== Rhyme benchmark — interactive setup ===\n")
    print("Study / condition:")
    keys = list(STUDY_REGISTRY.keys())
    for i, sid in enumerate(keys, start=1):
        print(f"  {i}. {sid} — {STUDY_REGISTRY[sid]['label']}")
    choice = _prompt_line("Enter number or study_id", "1")
    if choice.isdigit():
        idx = int(choice) - 1
        study_id = keys[idx] if 0 <= idx < len(keys) else keys[0]
    else:
        study_id = choice if choice in STUDY_REGISTRY else "baseline_default"

    default_out = str(_default_output_dir_for_study(study_id))
    out_raw = _prompt_line("Output directory", default_out)
    output_dir = Path(out_raw).expanduser()

    models_config = _models_config_for_rhyme_bench()
    if not models_config:
        print("No models left after skipping GGUF/local educators. Check config/rev_flux_models.yaml.")
        sys.exit(1)

    if _prompt_yes_no("List models (preview only)?", default=False):
        for m in models_config:
            print(f"  {m['id']}: {m.get('label', m['id'])}")
        print()

    print("Models: comma-separated ids, or 'all'")
    mid_raw = _prompt_line("Model id(s)", "all").lower()
    print("Prompts: 'all' | 'test' (first two prompts, first eligible model) | comma-separated indices")
    pr_raw = _prompt_line("Prompt selection", "all").lower()
    prompt_indices: list[int] | None
    test = pr_raw == "test"
    if mid_raw in ("all", "*", ""):
        models = None if test else [m["id"] for m in models_config]
    else:
        models = [x.strip() for x in mid_raw.split(",") if x.strip()]

    if test:
        prompt_indices = [0, 1]
    elif pr_raw in ("all", "*", ""):
        prompt_indices = None
    else:
        prompt_indices = _parse_int_list(pr_raw)

    max_rev = int(_prompt_line("Max revisions (0 = poet only)", "0"))
    diagnostic = _prompt_yes_no("Run diagnostic report after?", default=False)

    forms: list[str] | None = None
    if _prompt_yes_no("Filter by form(s)?", default=False):
        forms_raw = _prompt_line("Forms (space-separated, e.g. sonnet villanelle)", "")
        forms = [f.strip() for f in forms_raw.split() if f.strip()] or None

    verbose = _prompt_yes_no("Verbose pipeline output?", default=False)

    cfg = BenchConfig(
        study_id=study_id,
        output_dir=output_dir,
        models=models,
        prompt_indices=prompt_indices,
        test=test,
        max_revisions=max_rev,
        diagnostic=diagnostic,
        forms=forms,
        verbose=verbose,
    )

    print("\n--- Confirm ---")
    print(json.dumps(cfg.to_summary_dict(), indent=2))
    if not _prompt_yes_no("Run with these settings?", default=True):
        print("Aborted.")
        sys.exit(0)
    return cfg


def load_bench_config_json(path: Path) -> BenchConfig:
    with open(path) as f:
        raw = json.load(f)
    out = raw.get(
        "output_dir",
        str(ROOT / "data" / "rhyme_bench" / "studies" / "baseline_default"),
    )
    return BenchConfig(
        study_id=raw.get("study_id", "baseline_default"),
        output_dir=Path(out).expanduser(),
        models=raw.get("models"),
        prompt_indices=raw.get("prompt_indices"),
        test=raw.get("test", False),
        max_revisions=raw.get("max_revisions", 0),
        diagnostic=raw.get("diagnostic", False),
        forms=raw.get("forms"),
        verbose=raw.get("verbose", False),
        list_models_only=raw.get("list_models_only", False),
    )


def run_single(
    pipeline,
    user_request: str,
    form: str,
    variant: str | None,
    max_revisions: int,
    prompt_idx: int,
    model_id: str,
    verbose: bool = False,
    poet_generation_mode: str = "default",
    cmu_second_pass: bool = False,
    study_id: str = "baseline_default",
) -> dict:
    result = pipeline.generate(
        user_request,
        max_revisions=max_revisions,
        verbose=verbose,
        interactive=False,
        poet_generation_mode=poet_generation_mode,
        expected_form=form,
        expected_variant=variant,
        cmu_second_pass=cmu_second_pass,
    )
    poem = result.get("final_poem", "")
    analysis = analyze_rhyme(poem, expected_form=form, expected_variant=variant)
    row: dict[str, Any] = {
        "user_request": user_request,
        "form": form,
        "variant": variant,
        "prompt_idx": prompt_idx,
        "model_id": model_id,
        "study_id": study_id,
        "max_revisions": max_revisions,
        "poet_generation_mode": poet_generation_mode,
        "cmu_second_pass": cmu_second_pass,
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
    if result.get("pass1_poem") is not None:
        row["pass1_poem"] = result["pass1_poem"]
    if result.get("rhyme_analysis_pass1") is not None:
        row["rhyme_analysis_pass1"] = result["rhyme_analysis_pass1"]
    return row


def _resolve_prompt_indices(cfg: BenchConfig) -> list[int]:
    prompt_indices = cfg.prompt_indices
    if cfg.test:
        prompt_indices = prompt_indices or [0, 1]
    if prompt_indices is None:
        prompt_indices = list(range(len(RHYME_PROMPTS)))
    if cfg.forms:
        forms_to_test = set(cfg.forms)
        prompt_indices = [
            idx for idx in prompt_indices
            if idx < len(RHYME_PROMPTS) and RHYME_PROMPTS[idx][0] in forms_to_test
        ]
    return prompt_indices


def execute_bench(cfg: BenchConfig) -> None:
    models_config = _models_config_for_rhyme_bench()
    if not models_config:
        print("No models available (all entries use GGUF/local educators; rhyme bench skips those).")
        return

    model_ids = cfg.models or [m["id"] for m in models_config]
    models_to_run = [m for m in models_config if m["id"] in model_ids]
    if not models_to_run:
        print("No matching models. Use --list-models to see available.")
        return

    if cfg.list_models_only:
        print("  (GGUF/local educator entries from rev_flux_models.yaml are never run on rhyme bench.)")
        for m in models_config:
            mark = " *" if m["id"] in model_ids else ""
            print(f"  {m['id']}: {m.get('label', m['id'])}{mark}")
        return

    study = STUDY_REGISTRY.get(cfg.study_id, STUDY_REGISTRY["baseline_default"])
    poet_generation_mode = study["poet_generation_mode"]
    cmu_second_pass = study["cmu_second_pass"]

    prompt_indices = _resolve_prompt_indices(cfg)
    if not prompt_indices:
        print("No prompts to run (check --forms / indices).")
        return

    if cfg.test and cfg.models is None:
        models_to_run = models_to_run[:1]
        if not models_to_run:
            print("No models available for test run.")
            return

    from scripts.inference.pipeline import PoetryPipeline

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
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
            print(f"[{mid}] study={cfg.study_id} form={form} prompt {idx}...", flush=True)
            run = run_single(
                pipeline,
                request,
                form=form,
                variant=variant,
                max_revisions=cfg.max_revisions,
                prompt_idx=idx,
                model_id=mid,
                verbose=cfg.verbose,
                poet_generation_mode=poet_generation_mode,
                cmu_second_pass=cmu_second_pass,
                study_id=cfg.study_id,
            )
            runs.append(run)
            slug = _slug(mid)
            out_file = cfg.output_dir / f"rhyme_{form}_{idx}_{slug}_{run_timestamp}.json"
            with open(out_file, "w") as f:
                json.dump(run, f, indent=2)

            ra = run["rhyme_analysis"]
            meta = run.get("metadata", {})
            tok_s = meta.get("perf_tokens_per_sec", 0)
            time_s = meta.get("perf_total_sec", 0)
            match_symbol = "T" if ra['matches_form'] is True else "F" if ra['matches_form'] is False else "?"
            detected_scheme = ra.get("detected_scheme", "")[:20]
            expected_scheme = ra.get("expected_scheme", "")

            print(
                f"  ({tok_s:.1f} tok/s, {time_s:.1f}s) "
                f"sd={ra['strict_rhyme_density']:.2f} match={match_symbol} "
                f"devs={ra['deviations_count']} lines={ra['line_count']} "
                f"scheme={detected_scheme} expect={expected_scheme}",
                flush=True,
            )
            print(f"  -> {out_file}", flush=True)

    if cfg.diagnostic:
        from scripts.benchmarks.rhyme_bench.diagnostic import (
            DiagnosticAnalyzer,
            DiagnosticReport,
        )

        print("\nRunning diagnostic analysis...", flush=True)
        analyzer = DiagnosticAnalyzer()
        report = DiagnosticReport(runs, analyzer)

        diag_path = cfg.output_dir / "diagnostic_report.json"
        with open(diag_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        md_path = cfg.output_dir / "diagnostic_summary.md"
        with open(md_path, "w") as f:
            f.write(report.to_markdown())

        print(f"\nDiagnostic report saved:")
        print(f"  JSON: {diag_path}")
        print(f"  Markdown: {md_path}")

    matches = sum(1 for r in runs if r["rhyme_analysis"].get("matches_form") is True)
    strict_densities = [r["rhyme_analysis"]["strict_rhyme_density"] for r in runs]
    summary = {
        "run_timestamp": run_timestamp,
        "study_id": cfg.study_id,
        "bench_config": cfg.to_summary_dict(),
        "total_runs": len(runs),
        "matches_form_count": matches,
        "matches_form_rate": round(matches / len(runs), 2) if runs else 0,
        "mean_strict_rhyme_density": (
            round(sum(strict_densities) / len(strict_densities), 2) if strict_densities else 0
        ),
        "models_tested": model_ids,
        "prompt_indices": prompt_indices,
        "max_revisions": cfg.max_revisions,
    }
    summary_timestamped_path = cfg.output_dir / f"summary_{run_timestamp}.json"
    with open(summary_timestamped_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary_path = cfg.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"\nSummary: matches_form {matches}/{len(runs)}, "
        f"mean strict_density={summary['mean_strict_rhyme_density']}",
    )
    print(f"  Latest: {summary_path}")
    print(f"  Timestamped: {summary_timestamped_path}")


def main() -> None:
    if "--non-interactive" not in sys.argv:
        if len(sys.argv) > 1:
            if sys.argv[1] in ("-h", "--help"):
                parser = argparse.ArgumentParser(
                    description="Rhyme benchmark: interactive by default; use --non-interactive for scripts.",
                )
                parser.add_argument(
                    "--non-interactive",
                    action="store_true",
                    help="Use argparse flags (required for CI); run with no args for the interactive wizard",
                )
                parser.print_help()
                print(
                    "\nInteractive mode: run with no arguments to configure study, models, output dir, etc.\n",
                )
                sys.exit(0)
            print(
                "Rhyme bench: run with no arguments for interactive setup, or pass "
                "--non-interactive for command-line / script use.",
                file=sys.stderr,
            )
            sys.exit(2)
        cfg = prompt_bench_config()
        execute_bench(cfg)
        return

    parser = argparse.ArgumentParser(description="Rhyme benchmark: test rhyming form adherence")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use argparse / --bench-config only (for CI and scripts)",
    )
    parser.add_argument(
        "--bench-config",
        type=Path,
        default=None,
        help="JSON file with bench options (merged with CLI flags)",
    )
    parser.add_argument("--prompts", nargs="+", type=int, default=None, help="Prompt indices")
    parser.add_argument("--max-revisions", type=int, default=0, help="Revision cycles (0=poet only)")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Short test: 2 prompts, first eligible model (GGUF educator configs are skipped)",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Model ids from rev_flux_models.yaml")
    parser.add_argument("--list-models", action="store_true", help="Print models and exit")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose pipeline output")
    parser.add_argument("--diagnostic", action="store_true", help="Run diagnostic analysis")
    parser.add_argument("--forms", nargs="+", default=None, help="Limit to these forms")
    parser.add_argument(
        "--study",
        type=str,
        default="baseline_default",
        choices=list(STUDY_REGISTRY.keys()),
        help="Study / ablation id",
    )
    args = parser.parse_args()

    cfg = load_bench_config_json(args.bench_config) if args.bench_config else BenchConfig()
    cfg.study_id = args.study
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.prompts is not None:
        cfg.prompt_indices = args.prompts
    cfg.max_revisions = args.max_revisions
    if args.test:
        cfg.test = True
    if args.models is not None:
        cfg.models = args.models
    if args.verbose:
        cfg.verbose = True
    if args.diagnostic:
        cfg.diagnostic = True
    if args.forms is not None:
        cfg.forms = list(args.forms)
    if args.list_models:
        cfg.list_models_only = True
    execute_bench(cfg)


if __name__ == "__main__":
    main()
