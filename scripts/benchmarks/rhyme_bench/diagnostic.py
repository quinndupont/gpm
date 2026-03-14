#!/usr/bin/env python3
"""Diagnostic analysis module for rhyme benchmark failures.

Categorizes rhyme failures, computes severity scores, and generates actionable insights
for model improvement.
"""
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from scripts.eval.form_registry import get_line_count, get_scheme, parse_scheme


class FailureCategory(Enum):
    """Types of rhyme failures with diagnostic value."""

    SCHEME_VIOLATION = "scheme_violation"
    NEAR_MISS = "near_miss"
    DENSITY_ISSUE = "density_issue"
    FORM_CONFUSION = "form_confusion"
    LINE_COUNT_ERROR = "line_count_error"
    NO_RHYME_DETECTED = "no_rhyme_detected"


@dataclass
class FailureInstance:
    """Single failure instance with categorization and context."""

    category: str  # FailureCategory.value
    form: str
    variant: str | None
    prompt_idx: int
    model_id: str
    severity: float  # 0.0 (minor) to 1.0 (critical)
    details: dict
    user_request: str
    final_poem: str | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


# Default thresholds for failure categorization
DEFAULT_THRESHOLDS = {
    "strict_density_min": 0.7,  # Below this = density_issue
    "near_miss_slant_ratio": 0.3,  # If >30% slant vs perfect = near_miss
    "scheme_deviation_tolerance": 0.2,  # Allow 20% deviation before violation
    "line_count_tolerance": 2,  # +/- lines for variable forms
}


class DiagnosticAnalyzer:
    """Analyzes benchmark runs and categorizes rhyme failures."""

    def __init__(self, threshold_config: dict | None = None):
        """Initialize with configurable thresholds.

        Args:
            threshold_config: Custom thresholds (merges with defaults)
        """
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if threshold_config:
            self.thresholds.update(threshold_config)

    def analyze_run(self, run_data: dict) -> FailureInstance | None:
        """Analyze a single benchmark run for failures.

        Args:
            run_data: Run dict from run_bench.py containing rhyme_analysis

        Returns:
            FailureInstance if failed, None if passed
        """
        rhyme_analysis = run_data.get("rhyme_analysis", {})
        form = run_data.get("form")
        variant = run_data.get("variant")

        # Check if this run failed
        if not self._is_failure(rhyme_analysis, form, variant):
            return None

        # Categorize the failure
        category = self.categorize_failure(run_data, rhyme_analysis)

        # Compute severity score
        severity = self.compute_severity(run_data, rhyme_analysis, category)

        # Extract details specific to the failure category
        details = self._extract_details(run_data, rhyme_analysis, category)

        return FailureInstance(
            category=category.value,
            form=form,
            variant=variant,
            prompt_idx=run_data.get("prompt_idx", -1),
            model_id=run_data.get("model_id", "unknown"),
            severity=severity,
            details=details,
            user_request=run_data.get("user_request", ""),
            final_poem=run_data.get("final_poem"),
        )

    def _is_failure(self, analysis: dict, form: str, variant: str | None) -> bool:
        """Determine if a run represents a failure."""
        strict_density = analysis.get("strict_rhyme_density", 0.0)
        matches_form = analysis.get("matches_form")
        rhyme_density = analysis.get("rhyme_density", 0.0)

        # Multiple failure conditions
        if strict_density < self.thresholds["strict_density_min"]:
            return True
        if matches_form is False:
            return True
        if rhyme_density < 0.3:  # Catastrophic: almost no rhymes
            return True

        # Check line count
        expected_lines = get_line_count(form, variant)
        actual_lines = analysis.get("line_count", 0)
        if expected_lines and abs(actual_lines - expected_lines) > self.thresholds["line_count_tolerance"]:
            return True

        return False

    def categorize_failure(self, run_data: dict, analysis: dict) -> FailureCategory:
        """Determine primary failure category.

        Args:
            run_data: Full run data
            analysis: rhyme_analysis dict

        Returns:
            Primary FailureCategory
        """
        form = run_data.get("form")
        variant = run_data.get("variant")
        rhyme_density = analysis.get("rhyme_density", 0.0)
        strict_density = analysis.get("strict_rhyme_density", 0.0)
        matches_form = analysis.get("matches_form")
        expected_lines = get_line_count(form, variant)
        actual_lines = analysis.get("line_count", 0)

        # Priority order: most critical failures first

        # 1. No rhyme detected at all
        if rhyme_density < 0.1:
            return FailureCategory.NO_RHYME_DETECTED

        # 2. Line count error (check early, very obvious failure)
        if expected_lines and abs(actual_lines - expected_lines) > self.thresholds["line_count_tolerance"]:
            return FailureCategory.LINE_COUNT_ERROR

        # 3. Form confusion (wrong form or doesn't match expected)
        if matches_form is False:
            # Check if it's a scheme violation vs form confusion
            expected_scheme = analysis.get("expected_scheme")
            detected_scheme = analysis.get("detected_scheme")
            if expected_scheme and detected_scheme:
                deviation_ratio = self._scheme_deviation_ratio(expected_scheme, detected_scheme)
                if deviation_ratio > 0.5:  # More than 50% different
                    return FailureCategory.FORM_CONFUSION
                # Otherwise continue to check if it's a scheme violation

        # 4. Scheme violation (matches_form is False and deviation is significant)
        if matches_form is False:
            expected_scheme = analysis.get("expected_scheme")
            detected_scheme = analysis.get("detected_scheme")
            if expected_scheme and detected_scheme:
                deviation_ratio = self._scheme_deviation_ratio(expected_scheme, detected_scheme)
                if deviation_ratio > self.thresholds["scheme_deviation_tolerance"]:
                    return FailureCategory.SCHEME_VIOLATION

        # 5. Near-miss (lots of slant rhymes instead of perfect)
        # Handle both list and integer formats
        strict_pairs = analysis.get("strict_rhyme_pairs", 0)
        all_pairs = analysis.get("rhyme_pairs", 0)
        strict_count = len(strict_pairs) if isinstance(strict_pairs, list) else strict_pairs
        all_count = len(all_pairs) if isinstance(all_pairs, list) else all_pairs

        if all_count > 0:
            slant_ratio = 1.0 - (strict_count / all_count)
            if slant_ratio > self.thresholds["near_miss_slant_ratio"]:
                return FailureCategory.NEAR_MISS

        # 6. Density issue (default if nothing else)
        if strict_density < self.thresholds["strict_density_min"]:
            return FailureCategory.DENSITY_ISSUE

        # Fallback (shouldn't reach here if _is_failure returned True)
        return FailureCategory.DENSITY_ISSUE

    def _scheme_deviation_ratio(self, expected: str, detected: str) -> float:
        """Compute ratio of scheme positions that differ.

        Args:
            expected: Expected scheme string (e.g., "ABAB CDCD")
            detected: Detected scheme string

        Returns:
            Ratio 0.0-1.0 of positions that don't match
        """
        expected_parsed = parse_scheme(expected)
        detected_parsed = parse_scheme(detected)

        if not expected_parsed or not detected_parsed:
            return 1.0  # Treat empty as total mismatch

        # Compare position by position
        max_len = max(len(expected_parsed), len(detected_parsed))
        mismatches = 0

        for i in range(max_len):
            exp_letter = expected_parsed[i] if i < len(expected_parsed) else None
            det_letter = detected_parsed[i] if i < len(detected_parsed) else None
            if exp_letter != det_letter:
                mismatches += 1

        return mismatches / max_len if max_len > 0 else 0.0

    def compute_severity(
        self, run_data: dict, analysis: dict, category: FailureCategory
    ) -> float:
        """Compute severity score 0.0 (minor) to 1.0 (critical).

        Weighted combination of:
        - Deviation count (30%)
        - Scheme match distance (40%)
        - Density shortfall (30%)

        Args:
            run_data: Full run data
            analysis: rhyme_analysis dict
            category: Failure category

        Returns:
            Severity score 0.0-1.0
        """
        deviations_count = analysis.get("deviations_count", 0)
        strict_density = analysis.get("strict_rhyme_density", 0.0)
        expected_scheme = analysis.get("expected_scheme")
        detected_scheme = analysis.get("detected_scheme")

        # Component scores (0.0 = no issue, 1.0 = severe issue)

        # 1. Deviation score (normalized by line count)
        line_count = analysis.get("line_count", 1)
        deviation_score = min(deviations_count / max(line_count, 1), 1.0)

        # 2. Scheme distance score
        if expected_scheme and detected_scheme:
            scheme_distance = self._scheme_deviation_ratio(expected_scheme, detected_scheme)
        else:
            scheme_distance = 1.0 if category == FailureCategory.FORM_CONFUSION else 0.0

        # 3. Density shortfall score
        target_density = self.thresholds["strict_density_min"]
        density_shortfall = max(0.0, target_density - strict_density) / target_density

        # Weighted combination
        weights = {
            "deviation": 0.3,
            "scheme": 0.4,
            "density": 0.3,
        }

        severity = (
            weights["deviation"] * deviation_score
            + weights["scheme"] * scheme_distance
            + weights["density"] * density_shortfall
        )

        # Boost severity for catastrophic categories
        if category == FailureCategory.NO_RHYME_DETECTED:
            severity = max(severity, 0.9)
        elif category == FailureCategory.FORM_CONFUSION:
            severity = max(severity, 0.8)

        return min(severity, 1.0)

    def _extract_details(
        self, run_data: dict, analysis: dict, category: FailureCategory
    ) -> dict:
        """Extract category-specific failure details.

        Args:
            run_data: Full run data
            analysis: rhyme_analysis dict
            category: Failure category

        Returns:
            Dict of relevant details for this failure
        """
        details = {
            "expected_scheme": analysis.get("expected_scheme"),
            "detected_scheme": analysis.get("detected_scheme"),
            "strict_rhyme_density": analysis.get("strict_rhyme_density", 0.0),
            "rhyme_density": analysis.get("rhyme_density", 0.0),
            "deviations_count": analysis.get("deviations_count", 0),
            "line_count": analysis.get("line_count", 0),
        }

        # Add category-specific details
        if category == FailureCategory.SCHEME_VIOLATION:
            details["deviations"] = analysis.get("deviations", [])

        elif category == FailureCategory.NEAR_MISS:
            # Handle both list and integer formats for backwards compatibility
            strict_pairs = analysis.get("strict_rhyme_pairs", [])
            all_pairs = analysis.get("rhyme_pairs", [])
            strict_count = len(strict_pairs) if isinstance(strict_pairs, list) else strict_pairs
            all_count = len(all_pairs) if isinstance(all_pairs, list) else all_pairs

            details["strict_pairs_count"] = strict_count
            details["all_pairs_count"] = all_count
            details["slant_count"] = all_count - strict_count

        elif category == FailureCategory.LINE_COUNT_ERROR:
            form = run_data.get("form")
            variant = run_data.get("variant")
            expected_lines = get_line_count(form, variant)
            details["expected_line_count"] = expected_lines
            details["actual_line_count"] = analysis.get("line_count", 0)

        return details


class DiagnosticReport:
    """Aggregated diagnostic report with insights."""

    def __init__(self, runs: list[dict], analyzer: DiagnosticAnalyzer | None = None):
        """Generate report from benchmark runs.

        Args:
            runs: List of run dicts from run_bench.py
            analyzer: DiagnosticAnalyzer instance (creates default if None)
        """
        self.runs = runs
        self.analyzer = analyzer or DiagnosticAnalyzer()
        self.failures: list[FailureInstance] = []

        # Analyze all runs
        for run in runs:
            failure = self.analyzer.analyze_run(run)
            if failure:
                self.failures.append(failure)

    def failure_breakdown(self) -> dict[str, int]:
        """Count failures by category.

        Returns:
            Dict mapping category name to count
        """
        breakdown = defaultdict(int)
        for failure in self.failures:
            breakdown[failure.category] += 1
        return dict(breakdown)

    def by_form(self) -> dict[str, dict]:
        """Group failures by poetic form with statistics.

        Returns:
            Dict mapping form name to stats dict
        """
        form_data = defaultdict(lambda: {
            "total": 0,
            "failures": 0,
            "failure_instances": [],
            "severities": [],
        })

        # Count totals by form
        for run in self.runs:
            form = run.get("form", "unknown")
            form_data[form]["total"] += 1

        # Add failures
        for failure in self.failures:
            form = failure.form
            form_data[form]["failures"] += 1
            form_data[form]["failure_instances"].append(failure)
            form_data[form]["severities"].append(failure.severity)

        # Compute stats
        result = {}
        for form, data in form_data.items():
            total = data["total"]
            failures = data["failures"]
            severities = data["severities"]

            # Get primary issues (most common categories)
            category_counts = defaultdict(int)
            for f in data["failure_instances"]:
                category_counts[f.category] += 1
            primary_issues = sorted(category_counts.keys(), key=lambda k: category_counts[k], reverse=True)

            result[form] = {
                "total": total,
                "failures": failures,
                "failure_rate": round(failures / total, 2) if total > 0 else 0.0,
                "primary_issues": primary_issues[:3],  # Top 3
                "mean_severity": round(sum(severities) / len(severities), 2) if severities else 0.0,
            }

        return result

    def _categorize_model(self, model_id: str) -> str:
        """Categorize model as trained, vanilla, or frontier.

        Args:
            model_id: Model identifier

        Returns:
            "trained" | "vanilla" | "frontier"
        """
        if model_id.startswith("trained-"):
            return "trained"
        elif model_id.endswith("-vanilla"):
            return "vanilla"
        else:
            return "frontier"

    def by_model(self) -> dict[str, dict]:
        """Group failures by model with statistics and categorization.

        Returns:
            Dict mapping model_id to stats dict with keys:
            - total: total runs for this model
            - failures: failure count
            - success_rate: float (0.0-1.0)
            - failure_rate: float (0.0-1.0)
            - mean_severity: average severity across failures
            - primary_issues: top 3 failure categories
            - model_type: "trained" | "vanilla" | "frontier"
        """
        model_data = defaultdict(lambda: {
            "total": 0,
            "failures": 0,
            "failure_instances": [],
            "severities": [],
        })

        # Count totals by model
        for run in self.runs:
            model_id = run.get("model_id", "unknown")
            model_data[model_id]["total"] += 1

        # Add failures
        for failure in self.failures:
            model_id = failure.model_id
            model_data[model_id]["failures"] += 1
            model_data[model_id]["failure_instances"].append(failure)
            model_data[model_id]["severities"].append(failure.severity)

        # Compute stats
        result = {}
        for model_id, data in model_data.items():
            total = data["total"]
            failures = data["failures"]
            severities = data["severities"]

            # Get primary issues (most common categories)
            category_counts = defaultdict(int)
            for f in data["failure_instances"]:
                category_counts[f.category] += 1
            primary_issues = sorted(
                category_counts.keys(),
                key=lambda k: category_counts[k],
                reverse=True
            )

            result[model_id] = {
                "total": total,
                "failures": failures,
                "success_rate": round((total - failures) / total, 2) if total > 0 else 0.0,
                "failure_rate": round(failures / total, 2) if total > 0 else 0.0,
                "primary_issues": primary_issues[:3],  # Top 3
                "mean_severity": round(sum(severities) / len(severities), 2) if severities else 0.0,
                "model_type": self._categorize_model(model_id),
            }

        return result

    def severity_distribution(self) -> dict[str, float]:
        """Mean severity by category.

        Returns:
            Dict mapping category name to mean severity
        """
        category_severities = defaultdict(list)
        for failure in self.failures:
            category_severities[failure.category].append(failure.severity)

        return {
            category: round(sum(severities) / len(severities), 2)
            for category, severities in category_severities.items()
        }

    def actionable_insights(self) -> list[dict]:
        """Generate prioritized recommendations.

        Returns:
            List of insight dicts with priority, description, recommendation
        """
        insights = []

        # Group by category and form
        category_form_failures = defaultdict(lambda: defaultdict(list))
        for failure in self.failures:
            category_form_failures[failure.category][failure.form].append(failure)

        # Generate insights per category
        for category, form_failures in category_form_failures.items():
            for form, failures in form_failures.items():
                if not failures:
                    continue

                mean_severity = sum(f.severity for f in failures) / len(failures)
                count = len(failures)

                # Generate category-specific recommendations
                recommendation = self._get_recommendation(category, form, failures)

                # Compute priority (higher = more important)
                # Priority = severity * count * novelty_factor
                priority_score = mean_severity * count

                insights.append({
                    "priority_score": priority_score,
                    "category": category,
                    "form": form,
                    "description": f"{count} {form} failures ({category})",
                    "mean_severity": round(mean_severity, 2),
                    "recommendation": recommendation,
                    "affected_runs": count,
                })

        # Sort by priority score (descending) and assign priority ranks
        insights.sort(key=lambda x: x["priority_score"], reverse=True)
        for i, insight in enumerate(insights):
            insight["priority"] = i + 1
            del insight["priority_score"]  # Remove internal score

        return insights

    def _get_recommendation(
        self, category: str, form: str, failures: list[FailureInstance]
    ) -> str:
        """Generate category-specific recommendation.

        Args:
            category: Failure category
            form: Poetic form
            failures: List of failures for this category+form

        Returns:
            Actionable recommendation string
        """
        if category == FailureCategory.SCHEME_VIOLATION.value:
            return f"Review {form} rhyme scheme patterns in training data; add more examples with correct scheme adherence"

        elif category == FailureCategory.NEAR_MISS.value:
            return f"Improve {form} rhyme quality: filter training data for perfect rhymes, reduce slant rhyme examples"

        elif category == FailureCategory.DENSITY_ISSUE.value:
            return f"Increase rhyme density signal for {form}: emphasize rhyme requirements in prompts, use SRPO with rhyme-focused trajectories"

        elif category == FailureCategory.FORM_CONFUSION.value:
            return f"Add more {form} examples to training data; strengthen form recognition in brief generation"

        elif category == FailureCategory.LINE_COUNT_ERROR.value:
            return f"Fix {form} line count understanding: add structured examples showing correct line counts"

        elif category == FailureCategory.NO_RHYME_DETECTED.value:
            return f"Critical: Model not rhyming for {form} at all; verify form detection and add abundant rhyme examples"

        return "Review training data and model configuration"

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        total_runs = len(self.runs)
        total_failures = len(self.failures)

        # Extract model and form info
        models_tested = list(set(run.get("model_id", "unknown") for run in self.runs))
        forms_tested = list(set(run.get("form", "unknown") for run in self.runs))

        return {
            "summary": {
                "total_runs": total_runs,
                "total_failures": total_failures,
                "failure_rate": round(total_failures / total_runs, 2) if total_runs > 0 else 0.0,
                "mean_severity": (
                    round(sum(f.severity for f in self.failures) / len(self.failures), 2)
                    if self.failures
                    else 0.0
                ),
                "models_tested": models_tested,
                "forms_tested": forms_tested,
            },
            "failure_breakdown": self.failure_breakdown(),
            "severity_distribution": self.severity_distribution(),
            "by_form": self.by_form(),
            "by_model": self.by_model(),
            "failures": [f.to_dict() for f in self.failures],
            "insights": self.actionable_insights(),
            "thresholds": self.analyzer.thresholds,
        }

    def to_markdown(self) -> str:
        """Generate human-readable markdown report."""
        data = self.to_dict()
        summary = data["summary"]
        breakdown = data["failure_breakdown"]
        severity_dist = data["severity_distribution"]
        by_form = data["by_form"]
        by_model = data["by_model"]
        insights = data["insights"]

        lines = [
            "# Rhyme Benchmark Diagnostic Report",
            "",
            f"**Total Runs**: {summary['total_runs']}",
            f"**Failures**: {summary['total_failures']} ({summary['failure_rate']:.0%})",
            f"**Mean Severity**: {summary['mean_severity']:.2f}",
            f"**Models**: {', '.join(summary['models_tested'])}",
            f"**Forms**: {', '.join(summary['forms_tested'])}",
            "",
            "## Failure Breakdown",
            "",
            "| Category | Count | % of Failures | Mean Severity |",
            "|----------|-------|---------------|---------------|",
        ]

        # Sort by count
        total_failures = summary["total_failures"]
        sorted_categories = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories:
            pct = (count / total_failures * 100) if total_failures > 0 else 0
            severity = severity_dist.get(category, 0.0)
            lines.append(f"| {category.replace('_', ' ').title()} | {count} | {pct:.1f}% | {severity:.2f} |")

        lines.extend([
            "",
            "## Performance by Model",
            "",
        ])

        # Group models by type for better organization
        trained_models = {}
        vanilla_models = {}
        frontier_models = {}

        for model_id, stats in by_model.items():
            if stats["model_type"] == "trained":
                trained_models[model_id] = stats
            elif stats["model_type"] == "vanilla":
                vanilla_models[model_id] = stats
            else:
                frontier_models[model_id] = stats

        # Sort each group by success rate (descending)
        def sort_by_success(items):
            return sorted(items, key=lambda x: x[1]["success_rate"], reverse=True)

        # Display trained models first
        if trained_models:
            lines.append("### Fine-Tuned Models")
            lines.append("")
            for model_id, stats in sort_by_success(trained_models.items()):
                success_count = stats["total"] - stats["failures"]
                primary_issues_str = ", ".join(stats["primary_issues"]) if stats["primary_issues"] else "None"
                lines.extend([
                    f"#### {model_id}",
                    f"- **Success rate**: {stats['success_rate'] * 100:.1f}% ({success_count}/{stats['total']})",
                    f"- **Failures**: {stats['failures']}",
                    f"- **Mean severity**: {stats['mean_severity']:.2f}",
                    f"- **Primary issues**: {primary_issues_str}",
                    "",
                ])

        # Then vanilla baselines
        if vanilla_models:
            lines.append("### Vanilla Baselines")
            lines.append("")
            for model_id, stats in sort_by_success(vanilla_models.items()):
                success_count = stats["total"] - stats["failures"]
                primary_issues_str = ", ".join(stats["primary_issues"]) if stats["primary_issues"] else "None"
                lines.extend([
                    f"#### {model_id}",
                    f"- **Success rate**: {stats['success_rate'] * 100:.1f}% ({success_count}/{stats['total']})",
                    f"- **Failures**: {stats['failures']}",
                    f"- **Mean severity**: {stats['mean_severity']:.2f}",
                    f"- **Primary issues**: {primary_issues_str}",
                    "",
                ])

        # Finally frontier models
        if frontier_models:
            lines.append("### Frontier Models")
            lines.append("")
            for model_id, stats in sort_by_success(frontier_models.items()):
                success_count = stats["total"] - stats["failures"]
                primary_issues_str = ", ".join(stats["primary_issues"]) if stats["primary_issues"] else "None"
                lines.extend([
                    f"#### {model_id}",
                    f"- **Success rate**: {stats['success_rate'] * 100:.1f}% ({success_count}/{stats['total']})",
                    f"- **Failures**: {stats['failures']}",
                    f"- **Mean severity**: {stats['mean_severity']:.2f}",
                    f"- **Primary issues**: {primary_issues_str}",
                    "",
                ])

        lines.extend([
            "",
            "## Performance by Form",
            "",
        ])

        # Sort by failure rate
        sorted_forms = sorted(by_form.items(), key=lambda x: x[1]["failure_rate"], reverse=True)
        for form, stats in sorted_forms:
            lines.extend([
                f"### {form.title()}",
                f"- **Success rate**: {(1 - stats['failure_rate']) * 100:.1f}% ({stats['total'] - stats['failures']}/{stats['total']})",
                f"- **Primary issues**: {', '.join(stats['primary_issues'])}",
                f"- **Mean severity**: {stats['mean_severity']:.2f}",
                "",
            ])

        lines.extend([
            "## Actionable Insights",
            "",
        ])

        # Emoji indicators for priority
        priority_emoji = {
            1: "🔴",
            2: "🟠",
            3: "🟡",
        }

        for insight in insights[:10]:  # Top 10
            priority = insight["priority"]
            emoji = priority_emoji.get(priority, "🔵")
            lines.extend([
                f"### {emoji} Priority {priority}: {insight['form'].title()} - {insight['category'].replace('_', ' ').title()}",
                f"**Issue**: {insight['description']}",
                f"**Severity**: {insight['mean_severity']:.2f} (affects {insight['affected_runs']} runs)",
                f"**Recommendation**: {insight['recommendation']}",
                "",
            ])

        return "\n".join(lines)


def load_runs(data_dir: Path) -> list[dict]:
    """Load all benchmark run JSONs from directory.

    Args:
        data_dir: Path to data/rhyme_bench directory

    Returns:
        List of run dicts
    """
    runs = []
    for json_file in data_dir.glob("rhyme_*.json"):
        with open(json_file) as f:
            runs.append(json.load(f))
    return runs


def run_diagnostic(
    data_dir: Path,
    output_path: Path | None = None,
    threshold_config: dict | None = None,
) -> DiagnosticReport:
    """Main entry point: analyze all runs and generate report.

    Args:
        data_dir: Path to data/rhyme_bench directory
        output_path: Optional path to save diagnostic_report.json
        threshold_config: Optional custom thresholds

    Returns:
        DiagnosticReport instance
    """
    runs = load_runs(data_dir)
    analyzer = DiagnosticAnalyzer(threshold_config)
    report = DiagnosticReport(runs, analyzer)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    return report
