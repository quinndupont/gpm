#!/usr/bin/env python3
"""Re-run rhyme_analyzer on existing rhyme_*.json runs (in-place).

Use after changing ``scripts/eval/rhyme_analyzer.py`` so stored ``rhyme_analysis``
matches the current logic without regenerating poems from models.

Examples::

    uv run python scripts/benchmarks/rhyme_bench/rescore_rhyme_runs.py
    uv run python scripts/benchmarks/rhyme_bench/rescore_rhyme_runs.py --study ablate_cmu_two_pass
    uv run python scripts/benchmarks/rhyme_bench/rescore_rhyme_runs.py --dry-run
    uv run python scripts/benchmarks/rhyme_bench/rescore_rhyme_runs.py --regenerate-summaries
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme


def _slim_rhyme_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """Same shape as ``run_bench.run_single`` stores under ``rhyme_analysis``."""
    return {
        "strict_rhyme_density": analysis.get("strict_rhyme_density", 0),
        "rhyme_density": analysis.get("rhyme_density", 0),
        "matches_form": analysis.get("matches_form"),
        "deviations_count": len(analysis.get("deviations", [])),
        "strict_rhyme_pairs": len(analysis.get("strict_rhyme_pairs", [])),
        "rhyme_pairs": len(analysis.get("rhyme_pairs", [])),
        "line_count": analysis.get("line_count", 0),
        "detected_scheme": analysis.get("detected_scheme", ""),
        "expected_scheme": analysis.get("expected_scheme"),
    }


def rescore_study_dir(study_dir: Path, dry_run: bool) -> tuple[int, int, int]:
    """Returns (files_touched, files_changed, files_skipped_empty)."""
    touched = changed = skipped = 0
    for json_file in sorted(study_dir.glob("rhyme_*.json")):
        try:
            data = json.loads(json_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip (read error) {json_file}: {e}", file=sys.stderr)
            skipped += 1
            continue
        poem = data.get("final_poem", "")
        if not poem:
            skipped += 1
            continue
        touched += 1
        form = data.get("form")
        variant = data.get("variant")
        analysis = analyze_rhyme(poem, expected_form=form, expected_variant=variant)
        new_ra = _slim_rhyme_analysis(analysis)
        old_ra = data.get("rhyme_analysis", {})
        if old_ra != new_ra:
            changed += 1
        data["rhyme_analysis"] = new_ra

        if data.get("pass1_poem"):
            p1 = analyze_rhyme(
                data["pass1_poem"],
                expected_form=form,
                expected_variant=variant,
            )
            data["rhyme_analysis_pass1"] = p1

        if not dry_run:
            with open(json_file, "w") as f:
                json.dump(data, f, indent=2)
    return touched, changed, skipped


def main() -> None:
    p = argparse.ArgumentParser(description="Rescore rhyme_*.json with current rhyme_analyzer")
    p.add_argument(
        "--studies-root",
        type=Path,
        default=ROOT / "data" / "rhyme_bench" / "studies",
        help="Root directory containing per-study folders",
    )
    p.add_argument(
        "--study",
        type=str,
        default=None,
        help="Only this study id (subfolder name); default: all subfolders",
    )
    p.add_argument("--dry-run", action="store_true", help="Analyze but do not write files")
    p.add_argument(
        "--regenerate-summaries",
        action="store_true",
        help="After rescoring, run reporting.regenerate_all_study_summaries",
    )
    args = p.parse_args()
    root: Path = args.studies_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    study_dirs = [root / args.study] if args.study else sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")
    )
    total_touched = total_changed = total_skip = 0
    for sd in study_dirs:
        if not sd.is_dir():
            print(f"skip (not a dir): {sd}", file=sys.stderr)
            continue
        t, c, s = rescore_study_dir(sd, dry_run=args.dry_run)
        total_touched += t
        total_changed += c
        total_skip += s
        print(f"{sd.name}: touched={t} analysis_changed={c} skipped_empty={s}")

    print(
        f"total: touched={total_touched} analysis_changed={total_changed} "
        f"skipped_empty={total_skip}" + (" (dry-run)" if args.dry_run else "")
    )

    if args.regenerate_summaries and not args.dry_run:
        from scripts.benchmarks.rhyme_bench.reporting import regenerate_all_study_summaries

        out = regenerate_all_study_summaries(root)
        print("summaries:", json.dumps({k: v for k, v in out.items() if k != "studies"}))


if __name__ == "__main__":
    main()
