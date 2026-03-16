#!/usr/bin/env python3
"""Rhyme benchmark visualizations: per-model strict rhyme density, matches_form rate."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _ensure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load_runs(data_dir: Path) -> list[dict]:
    runs = []
    # Only load rhyme_*.json files (individual run files)
    for f in sorted(data_dir.glob("rhyme_*.json")):
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
        if mid == "unknown":
            continue  # Skip unknown models
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
        if mid == "unknown":
            continue  # Skip unknown models
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


def plot_failure_breakdown(
    diagnostic_report: dict,
    out_path: Path,
    title: str = "Failure Category Distribution",
) -> None:
    """Stacked bar chart: failure counts by category with severity coloring."""
    plt = _ensure_matplotlib()
    import matplotlib.patches as mpatches

    breakdown = diagnostic_report.get("failure_breakdown", {})
    severity_dist = diagnostic_report.get("severity_distribution", {})

    if not breakdown:
        return

    # Sort by count
    categories = sorted(breakdown.keys(), key=lambda k: breakdown[k], reverse=True)
    counts = [breakdown[cat] for cat in categories]
    severities = [severity_dist.get(cat, 0.5) for cat in categories]

    # Color by severity (red = high, yellow = medium, green = low)
    def severity_color(sev):
        if sev >= 0.7:
            return "#e74c3c"  # Red
        elif sev >= 0.4:
            return "#f39c12"  # Orange
        else:
            return "#27ae60"  # Green

    colors = [severity_color(sev) for sev in severities]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(categories)), counts, color=colors, alpha=0.8, width=0.7)

    # Add category labels
    category_labels = [cat.replace("_", " ").title() for cat in categories]
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(category_labels, rotation=30, ha="right")

    ax.set_ylabel("Failure Count")
    ax.set_title(title)
    ax.set_ylim(0, max(counts) * 1.1 if counts else 1)

    # Add legend for severity colors
    red_patch = mpatches.Patch(color="#e74c3c", label="High severity (≥0.7)")
    orange_patch = mpatches.Patch(color="#f39c12", label="Medium severity (0.4-0.7)")
    green_patch = mpatches.Patch(color="#27ae60", label="Low severity (<0.4)")
    ax.legend(handles=[red_patch, orange_patch, green_patch], loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_severity_heatmap(
    diagnostic_report: dict,
    out_path: Path,
    title: str = "Failure Severity by Form and Category",
) -> None:
    """Heatmap: rows=forms, cols=categories, color=mean severity."""
    plt = _ensure_matplotlib()
    import numpy as np

    failures = diagnostic_report.get("failures", [])
    if not failures:
        return

    # Group by form and category
    form_category_severities = {}
    for failure in failures:
        form = failure.get("form", "unknown")
        category = failure.get("category", "unknown")
        severity = failure.get("severity", 0.0)

        key = (form, category)
        if key not in form_category_severities:
            form_category_severities[key] = []
        form_category_severities[key].append(severity)

    # Calculate means
    form_category_means = {
        key: np.mean(severities)
        for key, severities in form_category_severities.items()
    }

    # Get unique forms and categories
    forms = sorted(set(form for form, _ in form_category_means.keys()))
    categories = sorted(set(cat for _, cat in form_category_means.keys()))

    if not forms or not categories:
        return

    # Build matrix
    matrix = np.zeros((len(forms), len(categories)))
    for i, form in enumerate(forms):
        for j, category in enumerate(categories):
            matrix[i, j] = form_category_means.get((form, category), 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.2), max(6, len(forms) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(forms)))
    category_labels = [cat.replace("_", " ").title() for cat in categories]
    ax.set_xticklabels(category_labels, rotation=30, ha="right")
    ax.set_yticklabels([f.title() for f in forms])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Severity", rotation=270, labelpad=20)

    # Add values to cells
    for i in range(len(forms)):
        for j in range(len(categories)):
            value = matrix[i, j]
            if value > 0:
                text_color = "white" if value > 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_near_miss_analysis(
    runs: list[dict],
    out_path: Path,
    title: str = "Perfect vs Slant Rhyme Distribution",
) -> None:
    """Scatter plot: perfect rhyme pairs (x) vs slant pairs (y) per run."""
    plt = _ensure_matplotlib()

    perfect_counts = []
    slant_counts = []
    forms = []

    for run in runs:
        ra = run.get("rhyme_analysis", {})
        # Handle both list and integer formats
        strict_pairs_data = ra.get("strict_rhyme_pairs", [])
        all_pairs_data = ra.get("rhyme_pairs", [])
        strict_pairs = len(strict_pairs_data) if isinstance(strict_pairs_data, list) else strict_pairs_data
        all_pairs = len(all_pairs_data) if isinstance(all_pairs_data, list) else all_pairs_data
        slant = all_pairs - strict_pairs

        if all_pairs > 0:  # Only include runs with some rhymes
            perfect_counts.append(strict_pairs)
            slant_counts.append(slant)
            forms.append(run.get("form", "unknown"))

    if not perfect_counts:
        return

    # Color by form
    unique_forms = sorted(set(forms))
    form_colors = {
        form: plt.cm.tab10(i % 10)
        for i, form in enumerate(unique_forms)
    }
    colors = [form_colors[f] for f in forms]

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(perfect_counts, slant_counts, c=colors, alpha=0.6, s=100)

    # Add diagonal line (equal perfect and slant)
    max_val = max(max(perfect_counts), max(slant_counts)) if perfect_counts and slant_counts else 1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="Equal perfect/slant")

    ax.set_xlabel("Perfect Rhyme Pairs")
    ax.set_ylabel("Slant Rhyme Pairs")
    ax.set_title(title)

    # Add legend for forms
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=form_colors[f], label=f.title()) for f in unique_forms]
    ax.legend(handles=patches, loc="upper right", title="Form")

    # Add annotation
    ax.text(
        0.05, 0.95,
        "Points above diagonal: More slant than perfect",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        alpha=0.7,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_model_performance_comparison(
    diagnostic_report: dict,
    out_path: Path,
    title: str = "Model Performance Comparison (Success Rate)",
) -> None:
    """Grouped bar chart: success rates by model, grouped by type (trained/vanilla/frontier)."""
    plt = _ensure_matplotlib()
    import matplotlib.patches as mpatches

    by_model = diagnostic_report.get("by_model", {})
    if not by_model:
        return

    # Filter out unknown models
    by_model = {k: v for k, v in by_model.items() if k != "unknown"}
    if not by_model:
        return

    # Group models by type
    trained = {}
    vanilla = {}
    frontier = {}

    for model_id, stats in by_model.items():
        model_type = stats.get("model_type", "frontier")
        if model_type == "trained":
            trained[model_id] = stats
        elif model_type == "vanilla":
            vanilla[model_id] = stats
        else:
            frontier[model_id] = stats

    # Sort by success rate within each group
    def sort_dict(d):
        return dict(sorted(d.items(), key=lambda x: x[1]["success_rate"], reverse=True))

    trained = sort_dict(trained)
    vanilla = sort_dict(vanilla)
    frontier = sort_dict(frontier)

    # Combine all models in order: trained, vanilla, frontier
    all_models = list(trained.keys()) + list(vanilla.keys()) + list(frontier.keys())
    success_rates = [by_model[m]["success_rate"] for m in all_models]

    # Color by type
    colors = (
        ["#3498db"] * len(trained)  # Blue for trained
        + ["#95a5a6"] * len(vanilla)  # Gray for vanilla
        + ["#e74c3c"] * len(frontier)  # Red for frontier
    )

    fig, ax = plt.subplots(figsize=(max(10, len(all_models) * 0.8), 6))
    bars = ax.bar(range(len(all_models)), success_rates, color=colors, alpha=0.8, width=0.7)

    # Add model labels
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_ylabel("Success Rate (0-1)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Add legend
    trained_patch = mpatches.Patch(color="#3498db", label="Fine-Tuned")
    vanilla_patch = mpatches.Patch(color="#95a5a6", label="Vanilla Baseline")
    frontier_patch = mpatches.Patch(color="#e74c3c", label="Frontier")
    ax.legend(handles=[trained_patch, vanilla_patch, frontier_patch], loc="upper right")

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{rate:.0%}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_trained_vs_vanilla_comparison(
    diagnostic_report: dict,
    out_path: Path,
    title: str = "Trained vs Vanilla Model Comparison",
) -> None:
    """Side-by-side comparison of trained models vs their vanilla counterparts."""
    plt = _ensure_matplotlib()
    import numpy as np

    by_model = diagnostic_report.get("by_model", {})
    if not by_model:
        return

    # Filter out unknown models
    by_model = {k: v for k, v in by_model.items() if k != "unknown"}
    if not by_model:
        return

    # Find pairs of trained/vanilla models
    pairs = []
    for model_id, stats in by_model.items():
        if stats.get("model_type") == "trained":
            # Try to find corresponding vanilla model
            # trained-llama3.1-8b -> llama3.1-8b-vanilla
            base_name = model_id.replace("trained-", "")
            vanilla_id = f"{base_name}-vanilla"

            if vanilla_id in by_model:
                pairs.append({
                    "base": base_name,
                    "trained_id": model_id,
                    "vanilla_id": vanilla_id,
                    "trained_success": stats["success_rate"],
                    "vanilla_success": by_model[vanilla_id]["success_rate"],
                    "trained_severity": stats["mean_severity"],
                    "vanilla_severity": by_model[vanilla_id]["mean_severity"],
                })

    if not pairs:
        # No paired models found, skip this plot
        return

    # Create grouped bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(pairs))
    width = 0.35

    # Success rate comparison
    trained_success = [p["trained_success"] for p in pairs]
    vanilla_success = [p["vanilla_success"] for p in pairs]
    base_names = [p["base"] for p in pairs]

    bars1 = ax1.bar(x - width/2, trained_success, width, label='Fine-Tuned', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, vanilla_success, width, label='Vanilla', color='#95a5a6', alpha=0.8)

    ax1.set_ylabel('Success Rate (0-1)')
    ax1.set_title('Success Rate: Trained vs Vanilla')
    ax1.set_xticks(x)
    ax1.set_xticklabels(base_names, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{height:.0%}', ha='center', va='bottom', fontsize=8)

    # Mean severity comparison (lower is better)
    trained_severity = [p["trained_severity"] for p in pairs]
    vanilla_severity = [p["vanilla_severity"] for p in pairs]

    bars3 = ax2.bar(x - width/2, trained_severity, width, label='Fine-Tuned', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, vanilla_severity, width, label='Vanilla', color='#95a5a6', alpha=0.8)

    ax2.set_ylabel('Mean Severity (0-1, lower is better)')
    ax2.set_title('Failure Severity: Trained vs Vanilla')
    ax2.set_xticks(x)
    ax2.set_xticklabels(base_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_severity_comparison(
    diagnostic_report: dict,
    out_path: Path,
    title: str = "Model Performance: Severity Comparison",
) -> None:
    """Bar chart showing mean severity by model (lower is better)."""
    plt = _ensure_matplotlib()

    by_model = diagnostic_report.get("by_model", {})
    if not by_model:
        return

    # Filter out unknown models
    by_model = {k: v for k, v in by_model.items() if k != "unknown"}
    if not by_model:
        return

    # Sort models by severity (ascending - lower is better)
    models_sorted = sorted(by_model.items(), key=lambda x: x[1]["mean_severity"])
    model_ids = [m[0] for m in models_sorted]
    severities = [m[1]["mean_severity"] for m in models_sorted]
    model_types = [m[1]["model_type"] for m in models_sorted]

    # Color by type
    color_map = {"trained": "#3498db", "vanilla": "#95a5a6", "frontier": "#e74c3c"}
    colors = [color_map.get(t, "#95a5a6") for t in model_types]

    fig, ax = plt.subplots(figsize=(max(10, len(model_ids) * 0.8), 6))
    bars = ax.bar(range(len(model_ids)), severities, color=colors, alpha=0.8, width=0.7)

    ax.set_xticks(range(len(model_ids)))
    ax.set_xticklabels(model_ids, rotation=45, ha="right")
    ax.set_ylabel("Mean Severity (0-1, lower is better)")
    ax.set_title(title)
    ax.set_ylim(0, max(severities) * 1.15 if severities else 1)

    # Add severity threshold lines
    ax.axhline(y=0.4, color='#27ae60', linestyle='--', alpha=0.3, linewidth=1, label='Low severity (<0.4)')
    ax.axhline(y=0.7, color='#e67e22', linestyle='--', alpha=0.3, linewidth=1, label='High severity (≥0.7)')

    # Add value labels on bars
    for bar, sev in zip(bars, severities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{sev:.2f}', ha='center', va='bottom', fontsize=8)

    import matplotlib.patches as mpatches
    trained_patch = mpatches.Patch(color="#3498db", label="Fine-Tuned")
    vanilla_patch = mpatches.Patch(color="#95a5a6", label="Vanilla Baseline")
    frontier_patch = mpatches.Patch(color="#e74c3c", label="Frontier")
    ax.legend(handles=[trained_patch, vanilla_patch, frontier_patch], loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_model_performance_dimensions(
    diagnostic_report: dict,
    out_path: Path,
    title: str = "Model Performance Across Dimensions",
) -> None:
    """Stacked bar chart showing failure category distribution per model."""
    plt = _ensure_matplotlib()
    import numpy as np

    by_model = diagnostic_report.get("by_model", {})
    failures = diagnostic_report.get("failures", [])

    if not by_model or not failures:
        return

    # Filter out unknown models from failures
    failures = [f for f in failures if f.get("model_id", "unknown") != "unknown"]
    if not failures:
        return

    # Count failure categories per model
    from collections import defaultdict
    model_category_counts = defaultdict(lambda: defaultdict(int))

    for failure in failures:
        model_id = failure.get("model_id", "unknown")
        category = failure.get("category", "unknown")
        model_category_counts[model_id][category] += 1

    # Get all unique categories
    all_categories = sorted(set(
        cat for cats in model_category_counts.values() for cat in cats.keys()
    ))

    # Sort models by total failures
    models_sorted = sorted(
        model_category_counts.keys(),
        key=lambda m: sum(model_category_counts[m].values()),
        reverse=True
    )

    if not models_sorted:
        return

    # Build data matrix
    data = np.zeros((len(all_categories), len(models_sorted)))
    for i, category in enumerate(all_categories):
        for j, model in enumerate(models_sorted):
            data[i, j] = model_category_counts[model].get(category, 0)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(max(10, len(models_sorted) * 0.8), 7))

    # Color palette for categories
    colors_palette = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    category_colors = {cat: colors_palette[i % len(colors_palette)] for i, cat in enumerate(all_categories)}

    bottom = np.zeros(len(models_sorted))
    bars = []
    for i, category in enumerate(all_categories):
        bar = ax.bar(range(len(models_sorted)), data[i], bottom=bottom,
                     label=category.replace('_', ' ').title(),
                     color=category_colors[category], alpha=0.8, width=0.7)
        bars.append(bar)
        bottom += data[i]

    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels(models_sorted, rotation=45, ha="right")
    ax.set_ylabel("Failure Count")
    ax.set_title(title)
    ax.legend(loc="upper right", title="Failure Category", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def _scheme_similarity(detected: str, expected: str) -> float:
    """Calculate similarity between detected and expected rhyme schemes (0-1)."""
    if not detected or not expected:
        return 0.0

    # Normalize schemes (remove spaces)
    det = detected.replace(" ", "")
    exp = expected.replace(" ", "")

    # If lengths don't match, penalize
    if len(det) != len(exp):
        return 0.0

    # Calculate character-wise match
    matches = sum(1 for d, e in zip(det, exp) if d == e)
    return matches / len(exp) if exp else 0.0


def plot_scheme_similarity_heatmap(
    runs: list[dict],
    out_path: Path,
    title: str = "Rhyme Scheme Accuracy by Model and Form",
) -> None:
    """Heatmap: rows=models, cols=forms, color=average scheme similarity."""
    plt = _ensure_matplotlib()
    import numpy as np
    from collections import defaultdict

    # Group by model and form
    model_form_similarities = defaultdict(list)

    for run in runs:
        model_id = run.get("model_id", "unknown")
        form = run.get("form", "unknown")
        ra = run.get("rhyme_analysis", {})

        detected = ra.get("detected_scheme", "")
        expected = ra.get("expected_scheme", "")

        if detected and expected:
            similarity = _scheme_similarity(detected, expected)
            model_form_similarities[(model_id, form)].append(similarity)

    if not model_form_similarities:
        return

    # Calculate means
    model_form_means = {
        key: np.mean(sims)
        for key, sims in model_form_similarities.items()
    }

    # Get unique models and forms
    models = sorted(set(model for model, _ in model_form_means.keys()))
    forms = sorted(set(form for _, form in model_form_means.keys()))

    if not models or not forms:
        return

    # Build matrix
    matrix = np.zeros((len(models), len(forms)))
    for i, model in enumerate(models):
        for j, form in enumerate(forms):
            matrix[i, j] = model_form_means.get((model, form), 0.0)

    fig, ax = plt.subplots(figsize=(max(10, len(forms) * 1.2), max(6, len(models) * 0.8)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(forms)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([f.title() for f in forms], rotation=30, ha="right")
    ax.set_yticklabels(models)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Scheme Accuracy (0-1)", rotation=270, labelpad=20)

    # Add values to cells
    for i in range(len(models)):
        for j in range(len(forms)):
            value = matrix[i, j]
            if value > 0:
                text_color = "white" if value < 0.5 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_xlabel("Poetic Form")
    ax.set_ylabel("Model")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_form_difficulty_ranking(
    runs: list[dict],
    out_path: Path,
    title: str = "Form Difficulty Ranking (by Average Scheme Accuracy)",
) -> None:
    """Bar chart showing average scheme accuracy by form (harder forms have lower accuracy)."""
    plt = _ensure_matplotlib()
    from collections import defaultdict

    # Group by form
    form_similarities = defaultdict(list)

    for run in runs:
        form = run.get("form", "unknown")
        ra = run.get("rhyme_analysis", {})

        detected = ra.get("detected_scheme", "")
        expected = ra.get("expected_scheme", "")

        if detected and expected:
            similarity = _scheme_similarity(detected, expected)
            form_similarities[form].append(similarity)

    if not form_similarities:
        return

    # Calculate means and sort by difficulty (ascending accuracy = descending difficulty)
    import numpy as np
    form_means = {form: np.mean(sims) for form, sims in form_similarities.items()}
    forms_sorted = sorted(form_means.items(), key=lambda x: x[1])

    forms = [f[0] for f in forms_sorted]
    accuracies = [f[1] for f in forms_sorted]

    # Color by difficulty
    def difficulty_color(acc):
        if acc >= 0.7:
            return "#27ae60"  # Green (easy)
        elif acc >= 0.4:
            return "#f39c12"  # Orange (medium)
        else:
            return "#e74c3c"  # Red (hard)

    colors = [difficulty_color(acc) for acc in accuracies]

    fig, ax = plt.subplots(figsize=(max(10, len(forms) * 0.8), 6))
    bars = ax.bar(range(len(forms)), accuracies, color=colors, alpha=0.8, width=0.7)

    ax.set_xticks(range(len(forms)))
    ax.set_xticklabels([f.title() for f in forms], rotation=45, ha="right")
    ax.set_ylabel("Average Scheme Accuracy (0-1)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    # Add horizontal line at 0.5
    ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{acc:.0%}', ha='center', va='bottom', fontsize=9)

    # Add legend
    import matplotlib.patches as mpatches
    green_patch = mpatches.Patch(color="#27ae60", label="Easy (≥70% accuracy)")
    orange_patch = mpatches.Patch(color="#f39c12", label="Medium (40-70% accuracy)")
    red_patch = mpatches.Patch(color="#e74c3c", label="Hard (<40% accuracy)")
    ax.legend(handles=[green_patch, orange_patch, red_patch], loc="upper left")

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
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output directory (default: data_dir/plots)",
    )
    parser.add_argument("--title", type=str, default=None, help="Plot title prefix")
    args = parser.parse_args()

    runs = _load_runs(args.data_dir)
    if not runs:
        print("No run data found.", file=sys.stderr)
        return 1

    out_dir = args.output or (args.data_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = (args.title + ": ") if args.title else ""

    plot_model_comparison(
        runs, out_dir / "model_comparison.png",
        title=prefix + "Strict Rhyme Density by Model",
    )
    plot_matches_form_rate(
        runs, out_dir / "matches_form_rate.png",
        title=prefix + "Form Adherence Rate by Model",
    )

    # Scheme similarity analysis plots (always generate if we have runs)
    plot_scheme_similarity_heatmap(
        runs, out_dir / "scheme_similarity_heatmap.png",
        title=prefix + "Rhyme Scheme Accuracy by Model and Form",
    )
    plot_form_difficulty_ranking(
        runs, out_dir / "form_difficulty_ranking.png",
        title=prefix + "Form Difficulty Ranking (by Average Scheme Accuracy)",
    )

    # Check for diagnostic report and generate diagnostic plots
    diag_path = args.data_dir / "diagnostic_report.json"
    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)

        plot_failure_breakdown(
            diag, out_dir / "failure_breakdown.png",
            title=prefix + "Failure Category Distribution",
        )
        plot_severity_heatmap(
            diag, out_dir / "severity_heatmap.png",
            title=prefix + "Failure Severity by Form and Category",
        )
        plot_near_miss_analysis(
            runs, out_dir / "near_miss_analysis.png",
            title=prefix + "Perfect vs Slant Rhyme Distribution",
        )

        # Model performance comparison plots
        plot_model_performance_comparison(
            diag, out_dir / "model_performance_comparison.png",
            title=prefix + "Model Performance Comparison (Success Rate)",
        )
        plot_trained_vs_vanilla_comparison(
            diag, out_dir / "trained_vs_vanilla.png",
            title=prefix + "Trained vs Vanilla Model Comparison",
        )
        plot_model_severity_comparison(
            diag, out_dir / "model_severity_comparison.png",
            title=prefix + "Model Performance: Severity Comparison",
        )
        plot_model_performance_dimensions(
            diag, out_dir / "model_performance_dimensions.png",
            title=prefix + "Model Performance Across Dimensions",
        )

        print(f"Diagnostic plots included (diagnostic_report.json found)")
        print(f"  - Model performance comparison plots generated")

    print(f"\nScheme similarity analysis plots generated:")
    print(f"  - Scheme accuracy heatmap (model x form)")
    print(f"  - Form difficulty ranking")
    print(f"\nSaved all plots to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
