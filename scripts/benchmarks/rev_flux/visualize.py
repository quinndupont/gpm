"""RevFlux visualization: histogram and bar chart of per-line change (revised lines only)."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("pip install matplotlib")


def _revised_only(change_pcts: list[float], threshold: float = 0.5) -> list[tuple[int, float]]:
    """(line_idx, change_pct) for lines that changed."""
    return [(i, p) for i, p in enumerate(change_pcts) if p > threshold]


def plot_line_change_histogram(
    revised: list[tuple[int, float]],
    lines_changed_per_round: list[int],
    out_path: Path,
    title: str = "RevFlux: Revised Lines Only",
    bins: int = 20,
) -> None:
    """Histogram: x = % change buckets, y = count. Only revised lines. Summary of lines/round."""
    plt = _ensure_matplotlib()
    if not revised:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No revised lines", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.savefig(out_path, dpi=150)
        plt.close()
        return
    pcts = [p for _, p in revised]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pcts, bins=bins, range=(0, 100), color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Line change (%)")
    ax.set_ylabel("Count of revised lines")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    summary = " | ".join(f"R{r+1}: {n} lines" for r, n in enumerate(lines_changed_per_round) if n > 0)
    if summary:
        fig.text(0.5, -0.05, summary, ha="center", fontsize=9, transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_line_change_bars(
    revised: list[tuple[int, float]],
    lines_changed_per_round: list[int],
    out_path: Path,
    title: str = "RevFlux: Revised Lines (line index, change %)",
) -> None:
    """Bar chart: x = line index, y = change %. Only revised lines."""
    plt = _ensure_matplotlib()
    if not revised:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "No revised lines", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        fig.savefig(out_path, dpi=150)
        plt.close()
        return
    indices = [i for i, _ in revised]
    pcts = [p for _, p in revised]
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71" if p < 10 else "#f39c12" if p < 50 else "#e74c3c" for p in pcts]
    ax.bar(indices, pcts, color=colors, alpha=0.8, width=0.9)
    ax.set_xlabel("Line index")
    ax.set_ylabel("Change (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    summary = " | ".join(f"R{r+1}: {n} lines" for r, n in enumerate(lines_changed_per_round) if n > 0)
    if summary:
        fig.text(0.5, -0.05, summary, ha="center", fontsize=9, transform=fig.transFigure)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="RevFlux: visualize revised lines only")
    parser.add_argument("input", type=Path, help="JSON with revision data")
    parser.add_argument("-o", "--output", type=Path, help="Output image path")
    parser.add_argument("--title", type=str, default="RevFlux: Revised Lines")
    parser.add_argument("--bars", action="store_true", help="Bar chart (line index vs change pct)")
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins")
    parser.add_argument("--threshold", type=float, default=0.5, help="Min change pct to count as revised")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Get revised lines and per-round counts
    revised = []
    lines_changed_per_round = []
    if "revised_lines_per_round" in data and "lines_changed_per_round" in data:
        lines_changed_per_round = data["lines_changed_per_round"]
        for r in data["revised_lines_per_round"]:
            revised.extend([(int(i), float(p)) for i, p in r])
    elif "per_round_changes" in data:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from line_change import revised_lines_per_round, lines_changed_per_round as lcpr
        rounds = data["per_round_changes"]
        revised_per = revised_lines_per_round(rounds, args.threshold)
        lines_changed_per_round = lcpr(rounds, args.threshold)
        for r in revised_per:
            revised.extend([(i, p) for i, p in r])
    elif "change_pcts" in data:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from line_change import revised_lines_per_round, lines_changed_per_round as lcpr
        # Flatten - treat as single round
        pcts = data["change_pcts"]
        revised = _revised_only(pcts, args.threshold)
        lines_changed_per_round = [len(revised)] if revised else [0]
    elif "aggregate_change_pcts" in data:
        pcts = data["aggregate_change_pcts"]
        revised = _revised_only(pcts, args.threshold)
        lines_changed_per_round = [len(revised)]
    elif "revision_history" in data:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from line_change import revision_round_changes, revised_lines_per_round, lines_changed_per_round as lcpr
        rounds = revision_round_changes(data["revision_history"])
        revised_per = revised_lines_per_round(rounds, args.threshold)
        lines_changed_per_round = lcpr(rounds, args.threshold)
        for r in revised_per:
            revised.extend([(i, p) for i, p in r])
    else:
        raise ValueError("Input must have revision data")

    out = args.output or args.input.with_suffix(".png")
    if args.bars:
        plot_line_change_bars(revised, lines_changed_per_round, out, title=args.title)
    else:
        plot_line_change_histogram(revised, lines_changed_per_round, out, title=args.title, bins=args.bins)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
