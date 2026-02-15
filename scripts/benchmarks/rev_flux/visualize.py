"""RevFlux visualization: histogram of per-line change percentages."""
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


def plot_line_change_histogram(
    change_pcts: list[float],
    out_path: Path,
    title: str = "RevFlux: Per-Line Change During Revision",
    bins: int = 20,
) -> None:
    """Histogram: x = % change buckets, y = count of lines."""
    plt = _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(change_pcts, bins=bins, range=(0, 100), color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Line change (%)")
    ax.set_ylabel("Count of lines")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_line_change_bars(
    change_pcts: list[float],
    out_path: Path,
    title: str = "RevFlux: Per-Line Change (ordered)",
    max_bars: int = 200,
) -> None:
    """Bar chart: each bar = one line's change percentage (vertical bars)."""
    plt = _ensure_matplotlib()
    data = change_pcts[:max_bars]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(data)))
    colors = ["#2ecc71" if p < 10 else "#f39c12" if p < 50 else "#e74c3c" for p in data]
    ax.bar(x, data, color=colors, alpha=0.8, width=0.9)
    ax.set_xlabel("Line index (across all revision rounds)")
    ax.set_ylabel("Change (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="RevFlux: visualize line change percentages")
    parser.add_argument("input", type=Path, help="JSON with revision_history or change_pcts list")
    parser.add_argument("-o", "--output", type=Path, help="Output image path")
    parser.add_argument("--title", type=str, default="RevFlux: Per-Line Change During Revision")
    parser.add_argument("--bars", action="store_true", help="Bar chart (per-line) instead of histogram")
    parser.add_argument("--bins", type=int, default=20, help="Histogram bins")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if "change_pcts" in data:
        change_pcts = data["change_pcts"]
    elif "aggregate_change_pcts" in data:
        change_pcts = data["aggregate_change_pcts"]
    elif "revision_history" in data:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from line_change import revision_round_changes, aggregate_line_changes
        rounds = revision_round_changes(data["revision_history"])
        change_pcts = aggregate_line_changes(rounds)
    else:
        raise ValueError("Input must have 'change_pcts' or 'revision_history'")

    out = args.output or args.input.with_suffix(".png")
    if args.bars:
        plot_line_change_bars(change_pcts, out, title=args.title)
    else:
        plot_line_change_histogram(change_pcts, out, title=args.title, bins=args.bins)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
