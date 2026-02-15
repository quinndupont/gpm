#!/usr/bin/env python3
"""
RevFlux harness visualizations: stanza map, line stability, per-category,
revision-length comparison, approval timing. Requires harness output (runs + summary).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

REV_FLUX = Path(__file__).resolve().parent


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    return plt, mcolors


def _load_runs(data_dir: Path) -> list[dict]:
    """Load all run JSONs (exclude summary)."""
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


def plot_stanza_map(
    revision_history: list[dict],
    out_path: Path,
    title: str = "RevFlux: Stanza Structure Map",
) -> None:
    """2D blocks: each stanza colored by mean change %."""
    from scripts.benchmarks.rev_flux.line_change import stanza_change_map

    plt, mcolors = _ensure_matplotlib()
    stanzas, means = stanza_change_map(revision_history)
    if not stanzas:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.RdYlGn_r  # red = high change, green = low
    norm = mcolors.Normalize(vmin=0, vmax=100)
    y = 0
    for i, (s, m) in enumerate(zip(stanzas, means)):
        h = max(len(s) * 0.5, 0.8)
        color = cmap(norm(m))
        rect = plt.Rectangle((0, y), 1, h, facecolor=color, edgecolor="black", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(0.5, y + h / 2, f"S{i+1} ({m:.0f}%)", ha="center", va="center", fontsize=10)
        y += h + 0.15
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, y + 0.1)
    ax.axis("off")
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean change (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_line_stability(
    revision_history: list[dict],
    out_path: Path,
    title: str = "RevFlux: Line Stability Index",
) -> None:
    """Bar chart: each line's stability (rounds unchanged)."""
    from scripts.benchmarks.rev_flux.line_change import line_stability_indices

    plt = _ensure_matplotlib()[0]
    stability = line_stability_indices(revision_history)
    if not stability:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No revision data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.savefig(out_path, dpi=150)
        plt.close()
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(stability)))
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in stability]
    ax.bar(x, stability, color=colors, alpha=0.8, width=0.9)
    ax.set_xlabel("Line index")
    ax.set_ylabel("Rounds unchanged")
    ax.set_title(title)
    ax.set_ylim(0, max(stability) + 1 if stability else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_category_comparison(
    runs: list[dict],
    out_path: Path,
    title: str = "RevFlux: Per-Category Change Distribution",
) -> None:
    """Box plot: change_pcts by category. Only runs with revision data."""
    plt = _ensure_matplotlib()[0]
    by_cat: dict[str, list[float]] = {}
    for r in runs:
        pcts = r.get("change_pcts", [])
        if not pcts:
            continue
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).extend(pcts)
    if not by_cat:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = list(by_cat)
    data = [by_cat[c] for c in cats]
    bp = ax.boxplot(data, tick_labels=cats, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set_ylabel("Line change (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_revision_length_comparison(
    runs: list[dict],
    out_path: Path,
    title: str = "RevFlux: Change by Max Revision Length",
) -> None:
    """Box plot: change_pcts by max_revisions. Only runs with revision data."""
    plt = _ensure_matplotlib()[0]
    by_rev: dict[int, list[float]] = {}
    for r in runs:
        pcts = r.get("change_pcts", [])
        if not pcts:
            continue
        rev = r.get("max_revisions", 0)
        by_rev.setdefault(rev, []).extend(pcts)
    if not by_rev:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    revs = sorted(by_rev)
    data = [by_rev[r] for r in revs]
    labels = [str(r) for r in revs]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("coral")
        patch.set_alpha(0.7)
    ax.set_xlabel("Max revisions")
    ax.set_ylabel("Line change (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(
    runs: list[dict],
    out_path: Path,
    title: str = "RevFlux: Per-Model Change Distribution",
) -> None:
    """Box plot: change_pcts by model_id. Only includes models with revision data (trained)."""
    plt = _ensure_matplotlib()[0]
    by_model: dict[str, list[float]] = {}
    for r in runs:
        pcts = r.get("change_pcts", [])
        if not pcts:
            continue
        mid = r.get("model_id") or r.get("metadata", {}).get("model_poet", "unknown")
        by_model.setdefault(mid, []).extend(pcts)
    if not by_model:
        return
    models = list(by_model)
    data = [by_model[m] for m in models]
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    bp = ax.boxplot(data, tick_labels=models, patch_artist=True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    for patch in bp["boxes"]:
        patch.set_facecolor("mediumseagreen")
        patch.set_alpha(0.7)
    ax.set_ylabel("Line change (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_approval_timing(
    runs: list[dict],
    out_path: Path,
    title: str = "RevFlux: Approval Timing",
) -> None:
    """Bar chart: when did educator approve (round 1, 2, 3, ...) or not at all. Only runs with revision."""
    plt = _ensure_matplotlib()[0]
    by_round: dict[int | str, int] = {}
    for r in runs:
        if not r.get("revision_history"):
            continue
        meta = r.get("metadata", {})
        if meta.get("approved"):
            rd = meta.get("approved_at_round", 1)
            by_round[rd] = by_round.get(rd, 0) + 1
        else:
            by_round["max_reached"] = by_round.get("max_reached", 0) + 1
    if not by_round:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    approved = [(k, v) for k, v in by_round.items() if k != "max_reached"]
    approved.sort(key=lambda x: x[0])
    labels = [str(x[0]) for x in approved]
    vals = [x[1] for x in approved]
    x_pos = list(range(len(labels)))
    ax.bar(x_pos, vals, color="steelblue", alpha=0.8, width=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    if "max_reached" in by_round:
        ax.bar(len(labels), by_round["max_reached"], color="gray", alpha=0.7, width=0.8, label="Max revisions reached")
        ax.set_xticks(list(x_pos) + [len(labels)])
        ax.set_xticklabels(labels + ["—"])
    ax.set_xlabel("Approval round (1-indexed)")
    ax.set_ylabel("Count of runs")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="RevFlux harness visualizations")
    parser.add_argument(
        "data_dir",
        type=Path,
        nargs="?",
        default=ROOT / "data" / "rev_flux",
        help="Directory with harness output JSONs",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data_dir/plots)",
    )
    parser.add_argument(
        "--single",
        type=Path,
        help="Single run JSON for stanza map + stability only",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or args.data_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        with open(args.single) as f:
            data = json.load(f)
        hist = data.get("revision_history", [])
        if hist:
            plot_stanza_map(hist, out_dir / "stanza_map.png", title=f"RevFlux: {args.single.stem}")
            plot_line_stability(hist, out_dir / "line_stability.png", title=f"RevFlux: {args.single.stem}")
        print(f"Saved to {out_dir}")
        return

    runs = _load_runs(args.data_dir)
    if not runs:
        print("No runs found")
        return

    plot_category_comparison(runs, out_dir / "category_comparison.png")
    plot_revision_length_comparison(runs, out_dir / "revision_length_comparison.png")
    plot_model_comparison(runs, out_dir / "model_comparison.png")
    plot_approval_timing(runs, out_dir / "approval_timing.png")

    # Per-run stanza + stability for first few
    for r in runs[:5]:
        hist = r.get("revision_history", [])
        if hist:
            cat = r.get("category", "unknown")
            idx = r.get("prompt_idx", 0)
            rev = r.get("max_revisions", 0)
            plot_stanza_map(hist, out_dir / f"stanza_{cat}_{idx}_rev{rev}.png")
            plot_line_stability(hist, out_dir / f"stability_{cat}_{idx}_rev{rev}.png")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
