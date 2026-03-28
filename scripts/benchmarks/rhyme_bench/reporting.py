#!/usr/bin/env python3
"""Regenerate rhyme bench summaries from on-disk rhyme_*.json runs (per study + combined)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def _load_runs_in_dir(study_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(study_dir.glob("rhyme_*.json")):
        try:
            with open(f) as fp:
                runs.append(json.load(fp))
        except (json.JSONDecodeError, OSError):
            pass
    return runs


def _compute_summary(
    runs: list[dict],
    study_id: str,
    output_dir: Path,
    regenerate_ts: str,
) -> dict[str, Any]:
    matches = sum(1 for r in runs if r.get("rhyme_analysis", {}).get("matches_form") is True)
    strict_densities = [
        r["rhyme_analysis"]["strict_rhyme_density"]
        for r in runs
        if "rhyme_analysis" in r and r["rhyme_analysis"].get("strict_rhyme_density") is not None
    ]
    model_ids: list[str] = sorted({r.get("model_id", "") for r in runs if r.get("model_id")})
    prompt_indices = sorted({r.get("prompt_idx", -1) for r in runs if r.get("prompt_idx") is not None})
    if prompt_indices and prompt_indices[0] == -1:
        prompt_indices = []

    poet_mode = None
    cmu = None
    for r in runs:
        meta = r.get("metadata") or {}
        if poet_mode is None and meta.get("poet_generation_mode"):
            poet_mode = meta["poet_generation_mode"]
        if cmu is None and meta.get("cmu_second_pass") is not None:
            cmu = meta["cmu_second_pass"]
        if r.get("poet_generation_mode") and poet_mode is None:
            poet_mode = r["poet_generation_mode"]
        if r.get("cmu_second_pass") is not None and cmu is None:
            cmu = r["cmu_second_pass"]

    bench_config = {
        "study_id": study_id,
        "output_dir": str(output_dir),
        "regenerated_from_runs": True,
        "regenerate_timestamp": regenerate_ts,
        "poet_generation_mode": poet_mode,
        "cmu_second_pass": cmu,
    }

    return {
        "run_timestamp": regenerate_ts,
        "study_id": study_id,
        "bench_config": bench_config,
        "total_runs": len(runs),
        "matches_form_count": matches,
        "matches_form_rate": round(matches / len(runs), 4) if runs else 0,
        "mean_strict_rhyme_density": (
            round(sum(strict_densities) / len(strict_densities), 4) if strict_densities else 0
        ),
        "models_tested": model_ids,
        "prompt_indices": prompt_indices,
        "max_revisions": max(
            (r.get("max_revisions", 0) for r in runs),
            default=0,
        ),
    }


def infer_study_id(run: dict, study_dir_name: str) -> str:
    return run.get("study_id") or study_dir_name


def load_tagged_runs_from_studies_root(studies_root: Path) -> list[dict]:
    """Load all rhyme runs under studies/<id>/; set study_id on each run for reporting."""
    out: list[dict] = []
    if not studies_root.is_dir():
        return out
    for sub in sorted(studies_root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith("."):
            continue
        for r in _load_runs_in_dir(sub):
            rr = dict(r)
            rr["_study_dir"] = sub.name
            rr["study_id"] = infer_study_id(r, sub.name)
            out.append(rr)
    return out


def remove_legacy_summaries(study_dir: Path) -> None:
    for pattern in ("summary.json", "summary_*.json"):
        for f in study_dir.glob(pattern):
            try:
                f.unlink()
            except OSError:
                pass


def regenerate_all_study_summaries(studies_root: Path | None = None) -> dict[str, Any]:
    """
    For each studies/<study_id>/ with rhyme_*.json: delete old summary*.json, write fresh summary.json.
    Also writes studies_root/SUMMARY_BY_STUDY.json (all studies).
    """
    root = studies_root or (ROOT / "data" / "rhyme_bench" / "studies")
    regenerate_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    by_study: dict[str, dict[str, Any]] = {}

    if not root.is_dir():
        return {"error": f"not a directory: {root}", "studies": {}}

    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        runs = _load_runs_in_dir(sub)
        if not runs:
            continue
        remove_legacy_summaries(sub)
        study_id = sub.name
        summary = _compute_summary(runs, study_id, sub, regenerate_ts)
        by_study[study_id] = summary
        out_path = sub / "summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        ts_path = sub / f"summary_{regenerate_ts}.json"
        with open(ts_path, "w") as f:
            json.dump(summary, f, indent=2)

    combined = {
        "regenerate_timestamp": regenerate_ts,
        "studies_root": str(root),
        "studies": by_study,
    }
    combined_path = root / "SUMMARY_BY_STUDY.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)

    return combined


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Regenerate rhyme bench summaries from rhyme_*.json files")
    p.add_argument(
        "--studies-root",
        type=Path,
        default=ROOT / "data" / "rhyme_bench" / "studies",
    )
    args = p.parse_args()
    out = regenerate_all_study_summaries(args.studies_root)
    print(json.dumps({k: v for k, v in out.items() if k != "studies"}, indent=2))
    for sid, s in out.get("studies", {}).items():
        print(f"  {sid}: runs={s['total_runs']} mean_sd={s['mean_strict_rhyme_density']} form={s['matches_form_rate']}")


if __name__ == "__main__":
    main()
