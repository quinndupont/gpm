#!/usr/bin/env python3
"""
RevFlux quantitative revision dashboard: positional analysis, latent-space metrics,
and model comparison. Produces a 2x3 multi-panel figure.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

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
    import matplotlib.patches as mpatches
    return plt, mpatches


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


def _get_model_id(run: dict) -> str:
    return run.get("model_id") or run.get("metadata", {}).get("model_poet", "unknown")


def _normalize(values: list[float], low: float | None = None, high: float | None = None) -> list[float]:
    """Normalize to [0, 1]. If low/high given, use those; else min/max of values."""
    if not values:
        return []
    lo = low if low is not None else min(values)
    hi = high if high is not None else max(values)
    span = hi - lo or 1.0
    return [(v - lo) / span for v in values]


def main():
    parser = argparse.ArgumentParser(description="RevFlux quantitative revision dashboard")
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
    args = parser.parse_args()

    out_dir = args.output_dir or args.data_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dashboard.png"

    runs = _load_runs(args.data_dir)
    runs_with_revision = [r for r in runs if r.get("change_pcts")]

    if not runs_with_revision:
        print("No runs with revision data; dashboard requires at least one rev1+ run")
        return

    from scripts.benchmarks.rev_flux.line_change import (
        positional_change_profile,
        revision_coverage,
        head_preservation,
        tail_attention,
        structural_growth,
        change_entropy,
    )

    plt, mpatches = _ensure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Collect per-model and per-run metrics
    by_model: dict[str, list[dict]] = {}
    for r in runs_with_revision:
        mid = _get_model_id(r)
        by_model.setdefault(mid, []).append(r)

    models = sorted(by_model.keys())
    model_labels = _load_model_labels()
    colors = plt.cm.tab10.colors[: len(models)]
    model_color = {m: colors[i] for i, m in enumerate(models)}

    # 1. Positional Change Profile (top-left)
    ax = axes[0, 0]
    n_bins = 10
    x_bins = [(i + 0.5) / n_bins for i in range(n_bins)]
    for mid in models:
        profiles = []
        for r in by_model[mid]:
            p = positional_change_profile(r["change_pcts"], n_bins)
            profiles.append(p)
        if profiles:
            mean_profile = [sum(p[i] for p in profiles) / len(profiles) for i in range(n_bins)]
            ax.plot(x_bins, mean_profile, label=model_labels.get(mid, mid), color=model_color[mid], linewidth=2)
    ax.set_xlabel("Normalized position (0=start, 1=end)")
    ax.set_ylabel("Mean change (%)")
    ax.set_title("Positional Change Profile")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 1)

    # 2. Revision Locality Heatmap (top-center)
    ax = axes[0, 1]
    rows = []
    row_labels = []
    for mid in models:
        for cat in sorted({r.get("category", "?") for r in by_model[mid]}):
            combo_runs = [r for r in by_model[mid] if r.get("category") == cat]
            if not combo_runs:
                continue
            profiles = [positional_change_profile(r["change_pcts"], n_bins) for r in combo_runs]
            mean_row = [sum(p[i] for p in profiles) / len(profiles) for i in range(n_bins)]
            rows.append(mean_row)
            row_labels.append(f"{model_labels.get(mid, mid)}\n{cat}")
    if rows:
        im = ax.imshow(rows, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6 if len(row_labels) > 8 else 7)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f"{i/n_bins:.1f}" for i in range(n_bins)])
        ax.set_xlabel("Normalized position")
        ax.set_title("Revision Locality Heatmap")
        plt.colorbar(im, ax=ax, label="Change (%)")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # 3. Coverage vs Tail Attention scatter (top-right)
    ax = axes[0, 2]
    for mid in models:
        covs = []
        tails = []
        for r in by_model[mid]:
            covs.append(revision_coverage(r["change_pcts"]))
            tails.append(tail_attention(r["change_pcts"]))
        if covs:
            ax.scatter(covs, tails, c=[model_color[mid]], label=model_labels.get(mid, mid), alpha=0.8, s=40)
    ax.set_xlabel("Revision coverage")
    ax.set_ylabel("Tail attention (mean change % in last 33%)")
    ax.set_title("Coverage vs Tail Attention")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)

    # 4. Model Comparison Radar (bottom-left)
    axes[1, 0].remove()
    ax_radar = fig.add_subplot(2, 3, 4, projection="polar")
    n_radar = 6
    angles = [2 * math.pi * i / n_radar for i in range(n_radar)]
    angles += angles[:1]

    radar_labels = ["coverage", "tail_attn", "head_pres", "growth", "entropy", "1/perf"]
    agg = {}
    for mid in models:
        covs, tails, heads, grows, ents, perfs = [], [], [], [], [], []
        for r in by_model[mid]:
            covs.append(revision_coverage(r["change_pcts"]))
            tails.append(tail_attention(r["change_pcts"]) / 100.0)
            heads.append(head_preservation(r["change_pcts"]))
            hist = r.get("revision_history", [])
            grows.append(min(2.0, max(0, structural_growth(hist))) if hist else 1.0)
            ents.append(change_entropy(r["change_pcts"]))
            p = r.get("metadata", {}).get("perf_total_sec")
            perfs.append(1.0 / p if p and p > 0 else 0.0)
        agg[mid] = {
            "coverage": sum(covs) / len(covs) if covs else 0,
            "tail_attn": sum(tails) / len(tails) if tails else 0,
            "head_pres": sum(heads) / len(heads) if heads else 1,
            "growth": sum(grows) / len(grows) if grows else 1,
            "entropy": sum(ents) / len(ents) if ents else 0,
            "perf": sum(perfs) / len(perfs) if perfs else 0,
        }
    # Normalize each axis to 0-1 across models
    for key in radar_labels:
        k = key.replace("1/perf", "perf")
        vals = [agg[m].get(k, 0) for m in models]
        if vals and max(vals) > min(vals):
            for m in models:
                agg[m][f"{k}_norm"] = (agg[m].get(k, 0) - min(vals)) / (max(vals) - min(vals))
        else:
            for m in models:
                agg[m][f"{k}_norm"] = 0.5

    for mid in models:
        vals = [
            agg[mid].get("coverage_norm", 0.5),
            agg[mid].get("tail_attn_norm", 0.5),
            agg[mid].get("head_pres_norm", 0.5),
            agg[mid].get("growth_norm", 0.5),
            agg[mid].get("entropy_norm", 0.5),
            agg[mid].get("perf_norm", 0.5),
        ]
        vals += vals[:1]
        ax_radar.plot(angles, vals, "o-", linewidth=2, label=model_labels.get(mid, mid), color=model_color[mid])
        ax_radar.fill(angles, vals, alpha=0.15, color=model_color[mid])
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=8)
    ax_radar.set_title("Model Comparison Radar")
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=7 if len(models) <= 4 else 6)

    # 5. Structural Growth by Category (bottom-center)
    ax = axes[1, 1]
    cats = sorted({r.get("category", "?") for r in runs_with_revision})
    x = range(len(cats))
    width = 0.8 / max(1, len(models))
    for i, mid in enumerate(models):
        vals = []
        for cat in cats:
            combo = [r for r in by_model[mid] if r.get("category") == cat]
            grows = [structural_growth(r.get("revision_history", [])) for r in combo if r.get("revision_history")]
            vals.append(sum(grows) / len(grows) if grows else 1.0)
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width, label=model_labels.get(mid, mid), color=model_color[mid], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("Structural growth (final/initial lines)")
    ax.set_title("Structural Growth by Category")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7 if len(models) <= 4 else 6)

    # 6. Composite Score Table (bottom-right)
    ax = axes[1, 2]
    ax.axis("off")
    scores = []
    for mid in models:
        covs = [revision_coverage(r["change_pcts"]) for r in by_model[mid]]
        tails = [tail_attention(r["change_pcts"]) / 100.0 for r in by_model[mid]]
        heads = [head_preservation(r["change_pcts"]) for r in by_model[mid]]
        ents = [change_entropy(r["change_pcts"]) for r in by_model[mid]]
        cov = sum(covs) / len(covs) if covs else 0
        tail = sum(tails) / len(tails) if tails else 0
        head = sum(heads) / len(heads) if heads else 1
        ent = sum(ents) / len(ents) if ents else 0
        # 0.3*cov + 0.3*tail + 0.2*ent + 0.2*(1-head)
        score = 0.3 * cov + 0.3 * tail + 0.2 * ent + 0.2 * (1 - head)
        scores.append((mid, score, cov, tail, head, ent))
    scores.sort(key=lambda x: -x[1])
    table_data = [[model_labels.get(s[0], s[0]), f"{s[1]:.3f}", f"{s[2]:.2f}", f"{s[3]:.2f}", f"{s[4]:.2f}", f"{s[5]:.2f}"] for s in scores]
    col_labels = ["Model", "Score", "Cov", "Tail", "Head", "Ent"]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8 if len(models) > 4 else 9)
    table.scale(1.2, 1.6 if len(models) > 4 else 1.8)
    ax.set_title("Composite Revision Score (ranked)\n0.3*cov + 0.3*tail + 0.2*ent + 0.2*(1-head)")

    fig.suptitle("RevFlux Quantitative Revision Dashboard", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Dashboard saved: {out_path}")


if __name__ == "__main__":
    main()
