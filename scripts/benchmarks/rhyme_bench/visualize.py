#!/usr/bin/env python3
"""Rhyme benchmark visualizations: per-model strict rhyme density, matches_form rate."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load_runs(data_dir: Path) -> list[dict]:
    runs = []
    for f in sorted(data_dir.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            with open(f) as fp:
                runs.append(json.load(fp))
        except (json.JSONDecodeError, IOError):
            pass
    return runs


def plot_model_comparison(
    runs: list[dict],
    out_path: Path,
    title: str = "Rhyme: Strict Rhyme Density by Model",
) -> None:
    """Box plot: strict_rhyme_density by model_id."""
    plt = _ensure_matplotlib()
    by_model: dict[str, list[float]] = {}
    for r in runs:
        ra = r.get("rhyme_analysis", {})
        density = ra.get("strict_rhyme_density")
        if density is None:
            continue
        mid = r.get("model_id", "unknown")
        by_model.setdefault(mid, []).append(density)
    if not by_model:
        return
    models = list(by_model)
    data = [by_model[m] for m in models]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set_ylabel("Strict rhyme density (0–1)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_matches_form_rate(
    runs: list[dict],
    out_path: Path,
    title: str = "Rhyme: Form Adherence Rate by Model",
) -> None:
    """Bar chart: fraction of runs where matches_form=True, per model."""
    plt = _ensure_matplotlib()
    by_model: dict[str, tuple[int, int]] = {}  # (matches, total)
    for r in runs:
        ra = r.get("rhyme_analysis", {})
        matches = ra.get("matches_form") is True
        mid = r.get("model_id", "unknown")
        m, t = by_model.get(mid, (0, 0))
        by_model[mid] = (m + (1 if matches else 0), t + 1)
    if not by_model:
        return
    models = list(by_model)
    rates = [by_model[m][0] / by_model[m][1] if by_model[m][1] else 0 for m in models]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
    colors = ["#27ae60" if r >= 0.5 else "#e67e22" if r >= 0.25 else "#e74c3c" for r in rates]
    ax.bar(range(len(models)), rates, color=colors, alpha=0.8, width=0.7)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Matches form rate (0–1)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Rhyme benchmark visualizations")
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=ROOT / "data" / "rhyme_bench",
        help="Directory with rhyme bench output JSONs",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory (default: data_dir/plots)")
    parser.add_argument("--title", type=str, default=None, help="Plot title prefix")
    args = parser.parse_args()

    runs = _load_runs(args.data_dir)
    if not runs:
        print("No run data found.", file=sys.stderr)
        return 1

    out_dir = args.output or (args.data_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (args.title + ": ") if args.title else ""

    plot_model_comparison(runs, out_dir / "model_comparison.png", title=prefix + "Strict Rhyme Density by Model")
    plot_matches_form_rate(runs, out_dir / "matches_form_rate.png", title=prefix + "Form Adherence Rate by Model")
    print(f"Saved plots to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
