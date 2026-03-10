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
MODELS_CONFIG = ROOT / "config" / "rev_flux_models.yaml"


def _load_model_labels() -> dict[str, str]:
    """Load model_id -> label from config. Falls back to model_id if not found."""
    labels = {}
    if MODELS_CONFIG.exists():
        import yaml
        data = yaml.safe_load(open(MODELS_CONFIG)) or {}
        for m in data.get("models", []):
            labels[m["id"]] = m.get("label", m["id"])
    return labels


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
    """2D blocks: each stanza colored by mean change %. Stanza = lines separated by blank lines."""
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
    fig.text(0.5, -0.02, "Stanza = lines separated by blank lines. Color = mean % change (prev→final draft).", ha="center", fontsize=9, transform=fig.transFigure)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Mean change (%)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
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
    model_labels: dict[str, str] | None = None,
) -> None:
    """Grouped box plot: change_pcts by (category, model). Only runs with revision data."""
    plt = _ensure_matplotlib()[0]
    by_cat_model: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        pcts = r.get("change_pcts", [])
        if not pcts:
            continue
        cat = r.get("category", "unknown")
        mid = r.get("model_id") or r.get("metadata", {}).get("model_poet", "unknown")
        by_cat_model.setdefault((cat, mid), []).extend(pcts)
    if not by_cat_model:
        return
    labels = model_labels or {}
    cat_order = ["famous_poetry", "short_generic", "cliche"]
    models = sorted({m for _, m in by_cat_model})
    cats = [c for c in cat_order if any((c, m) in by_cat_model for m in models)]
    cats.extend([c for c in sorted({c for c, _ in by_cat_model}) if c not in cats])
    n_cats, n_models = len(cats), len(models)
    data, positions, colors = [], [], []
    colors_list = plt.cm.Set3.colors
    for i, cat in enumerate(cats):
        for j, mid in enumerate(models):
            key = (cat, mid)
            if key not in by_cat_model:
                continue
            data.append(by_cat_model[key])
            pos = i * (n_models + 1) + j
            positions.append(pos)
            colors.append(colors_list[j % len(colors_list)])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(10, n_cats * (n_models + 1) * 0.8), 5))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.8)
    ax.set_xticks([i * (n_models + 1) + (n_models - 1) / 2 for i in range(n_cats)])
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("Line change (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=colors_list[j % len(colors_list)], alpha=0.8, label=labels.get(m, m)) for j, m in enumerate(models)]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_revision_length_comparison(
    runs: list[dict],
    out_path: Path,
    title: str = "RevFlux: Change by Model and Revision Length",
    model_labels: dict[str, str] | None = None,
) -> None:
    """Grouped box plot: x = model, grouped by revision level. Captures multiple models and revision process."""
    plt = _ensure_matplotlib()[0]
    by_key: dict[tuple[str, int], list[float]] = {}
    for r in runs:
        pcts = r.get("change_pcts", [])
        if not pcts:
            continue
        rev = r.get("max_revisions", 0)
        if rev == 0:
            continue
        mid = r.get("model_id") or r.get("metadata", {}).get("model_poet", "unknown")
        by_key.setdefault((mid, rev), []).extend(pcts)
    if not by_key:
        return
    labels = model_labels or {}
    models = sorted({m for m, _ in by_key})
    revs = sorted({r for _, r in by_key})
    n_models, n_revs = len(models), len(revs)
    data, positions, colors = [], [], []
    colors_list = plt.cm.Set3.colors
    for i, mid in enumerate(models):
        for j, rev in enumerate(revs):
            if (mid, rev) in by_key:
                data.append(by_key[(mid, rev)])
                pos = i * (n_revs + 1) + j
                positions.append(pos)
                colors.append(colors_list[j % len(colors_list)])
    fig, ax = plt.subplots(figsize=(max(10, n_models * (n_revs + 1) * 0.8), 5))
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[idx])
        patch.set_alpha(0.8)
    ax.set_xticks([i * (n_revs + 1) + (n_revs - 1) / 2 for i in range(n_models)])
    ax.set_xticklabels([labels.get(m, m) for m in models], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Line change (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=colors_list[j % len(colors_list)], alpha=0.8, label=f"rev{r}") for j, r in enumerate(revs)]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_model_comparison(
    runs: list[dict],
    out_path: Path,
    title: str = "RevFlux: Per-Model Change Distribution",
    model_labels: dict[str, str] | None = None,
) -> None:
    """Box plot: change_pcts by model_id. Only includes models with revision data."""
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
    labels = model_labels or {}
    models = list(by_model)
    data = [by_model[m] for m in models]
    tick_labels = [labels.get(m, m) for m in models]
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 5))
    bp = ax.boxplot(data, tick_labels=tick_labels, patch_artist=True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    colors = plt.cm.Set3.colors
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.8)
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
    model_labels: dict[str, str] | None = None,
) -> None:
    """Grouped bar chart: when did educator approve, per model. Only runs with revision."""
    plt = _ensure_matplotlib()[0]
    by_round_model: dict[tuple[int | str, str], int] = {}
    for r in runs:
        if not r.get("revision_history"):
            continue
        mid = r.get("model_id") or r.get("metadata", {}).get("model_poet", "unknown")
        meta = r.get("metadata", {})
        if meta.get("approved"):
            rd = meta.get("approved_at_round", 1)
            key = (rd, mid)
        else:
            key = ("max_reached", mid)
        by_round_model[key] = by_round_model.get(key, 0) + 1
    if not by_round_model:
        return
    labels_map = model_labels or {}
    round_vals = {k for k, _ in by_round_model}
    rounds = sorted([k for k in round_vals if k != "max_reached"])
    if "max_reached" in round_vals:
        rounds.append("max_reached")
    models = sorted({m for _, m in by_round_model})
    n_rounds, n_models = len(rounds), len(models)
    width = 0.8 / max(1, n_models)
    x = list(range(n_rounds))
    colors = plt.cm.Set3.colors
    fig, ax = plt.subplots(figsize=(max(8, n_rounds * 1.5), 5))
    for j, mid in enumerate(models):
        vals = [by_round_model.get((r, mid), 0) for r in rounds]
        offset = (j - n_models / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width, label=labels_map.get(mid, mid), color=colors[j % len(colors)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) if r != "max_reached" else "—" for r in rounds])
    ax.set_xlabel("Approval round (1-indexed)")
    ax.set_ylabel("Count of runs")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
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
    parser.add_argument(
        "--comparison-only",
        action="store_true",
        help="Only produce model comparison plots (skip per-run stanza/stability)",
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

    model_labels = _load_model_labels()
    plot_category_comparison(runs, out_dir / "category_comparison.png", model_labels=model_labels)
    plot_revision_length_comparison(runs, out_dir / "revision_length_comparison.png", model_labels=model_labels)
    plot_model_comparison(runs, out_dir / "model_comparison.png", model_labels=model_labels)
    plot_approval_timing(runs, out_dir / "approval_timing.png", model_labels=model_labels)

    if args.comparison_only:
        print(f"Plots saved to {out_dir}")
        return

    # Per-run stanza + stability: one per (category, rev) for rev in [1, 3]
    seen: set[tuple[str, int]] = set()
    for r in runs:
        hist = r.get("revision_history", [])
        if not hist:
            continue
        cat = r.get("category", "unknown")
        idx = r.get("prompt_idx", 0)
        rev = r.get("max_revisions", 0)
        if rev not in (1, 3):
            continue
        key = (cat, rev)
        if key in seen:
            continue
        seen.add(key)
        plot_stanza_map(hist, out_dir / f"stanza_{cat}_{idx}_rev{rev}.png")
        plot_line_stability(hist, out_dir / f"stability_{cat}_{idx}_rev{rev}.png")

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
