#!/usr/bin/env python3
"""Rebuild summary.json from all run files in data dir. Use after partial harness runs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, nargs="?", default=ROOT / "data" / "rev_flux")
    args = parser.parse_args()
    runs = []
    for f in sorted(args.data_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        if "_rev" not in f.name:
            continue
        try:
            with open(f) as fp:
                runs.append(json.load(fp))
        except (json.JSONDecodeError, IOError):
            pass
    if not runs:
        print("No run files found")
        return
    categories = sorted({r.get("category") for r in runs if r.get("category")})
    max_revisions = sorted({r.get("max_revisions") for r in runs if r.get("max_revisions")})
    model_ids = sorted({r.get("model_id") for r in runs if r.get("model_id")})
    summary = {
        "total_runs": len(runs),
        "categories": categories,
        "max_revisions_tested": list(max_revisions),
        "models_tested": model_ids,
        "aggregate_change_pcts": [p for r in runs for p in r.get("change_pcts", [])],
    }
    out = args.data_dir / "summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Merged {len(runs)} runs -> {out}")


if __name__ == "__main__":
    main()
