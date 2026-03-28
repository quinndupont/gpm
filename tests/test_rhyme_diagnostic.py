"""Diagnostic analysis tests for rhyme benchmark."""
import json
from pathlib import Path

import pytest

from scripts.benchmarks.rhyme_bench.diagnostic import (
    DEFAULT_THRESHOLDS,
    DiagnosticAnalyzer,
    DiagnosticReport,
    FailureCategory,
    FailureInstance,
)

ROOT = Path(__file__).resolve().parent.parent
RHYME_BENCH_DIR = ROOT / "data" / "rhyme_bench" / "studies" / "baseline_default"


@pytest.mark.eval
class TestFailureCategorization:
    """Test failure category detection logic."""

    def test_scheme_violation_detected(self):
        """Verify scheme violations are correctly identified."""
        run_data = {
            "form": "sonnet",
            "variant": "shakespearean",
            "prompt_idx": 0,
            "model_id": "test",
            "user_request": "Write a sonnet",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "ABAB CDCD EFEF GG",
                "detected_scheme": "ABAB CDCD EFGH GI",  # Wrong third quatrain (4/14 = 29% deviation)
                "matches_form": False,
                "strict_rhyme_density": 0.75,
                "rhyme_density": 0.75,
                "deviations_count": 4,
                "line_count": 14,
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is not None
        assert failure.category == FailureCategory.SCHEME_VIOLATION.value

    def test_near_miss_classification(self):
        """Verify slant-heavy patterns classified as near-miss."""
        run_data = {
            "form": "quatrain",
            "variant": None,
            "prompt_idx": 1,
            "model_id": "test",
            "user_request": "Write a quatrain",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "ABAB",
                "detected_scheme": "ABAB",
                "matches_form": True,
                "strict_rhyme_density": 0.5,  # Low strict density
                "rhyme_density": 0.9,  # High overall density (lots of slant)
                "deviations_count": 0,
                "strict_rhyme_pairs": 1,  # Only 1 perfect rhyme pair
                "rhyme_pairs": 4,  # But 4 total pairs (3 slant)
                "line_count": 4,
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is not None
        # 1/4 = 25% strict, 75% slant -> exceeds 30% threshold
        assert failure.category == FailureCategory.NEAR_MISS.value

    def test_density_issue_threshold(self):
        """Verify density issues detected below threshold."""
        run_data = {
            "form": "couplets",
            "variant": None,
            "prompt_idx": 2,
            "model_id": "test",
            "user_request": "Write couplets",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "AA BB CC DD",
                "detected_scheme": "AA BB CC DD",
                "matches_form": True,
                "strict_rhyme_density": 0.5,  # Below 0.7 threshold
                "rhyme_density": 0.5,
                "deviations_count": 2,
                "line_count": 8,
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is not None
        assert failure.category == FailureCategory.DENSITY_ISSUE.value

    def test_form_confusion_detection(self):
        """Verify wrong form detection."""
        run_data = {
            "form": "sonnet",
            "variant": "shakespearean",
            "prompt_idx": 3,
            "model_id": "test",
            "user_request": "Write a sonnet",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "ABAB CDCD EFEF GG",
                "detected_scheme": "ABCB DEFE GHGH",  # Completely different (ballad-like)
                "matches_form": False,
                "strict_rhyme_density": 0.6,
                "rhyme_density": 0.6,
                "deviations_count": 8,
                "line_count": 14,
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is not None
        # >50% deviation = form confusion
        assert failure.category == FailureCategory.FORM_CONFUSION.value

    def test_line_count_error(self):
        """Verify line count mismatches."""
        run_data = {
            "form": "sonnet",
            "variant": "shakespearean",
            "prompt_idx": 4,
            "model_id": "test",
            "user_request": "Write a sonnet",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "ABAB CDCD EFEF GG",
                "detected_scheme": "ABAB CDCD EFEF",
                "matches_form": False,
                "strict_rhyme_density": 0.75,
                "rhyme_density": 0.75,
                "deviations_count": 0,
                "line_count": 11,  # Should be 14 (difference of 3 > tolerance of 2)
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is not None
        assert failure.category == FailureCategory.LINE_COUNT_ERROR.value

    def test_no_rhyme_detected(self):
        """Verify catastrophic no-rhyme case."""
        run_data = {
            "form": "limerick",
            "variant": None,
            "prompt_idx": 5,
            "model_id": "test",
            "user_request": "Write a limerick",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "AABBA",
                "detected_scheme": "",
                "matches_form": False,
                "strict_rhyme_density": 0.0,
                "rhyme_density": 0.05,  # Almost no rhymes
                "deviations_count": 5,
                "line_count": 5,
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is not None
        assert failure.category == FailureCategory.NO_RHYME_DETECTED.value


@pytest.mark.eval
class TestDiagnosticAnalyzer:
    """Test DiagnosticAnalyzer core functionality."""

    def test_analyze_perfect_run(self):
        """No failures for perfect adherence."""
        run_data = {
            "form": "couplets",
            "variant": None,
            "prompt_idx": 0,
            "model_id": "test",
            "user_request": "Write couplets",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "AA BB",
                "detected_scheme": "AA BB",
                "matches_form": True,
                "strict_rhyme_density": 1.0,
                "rhyme_density": 1.0,
                "deviations_count": 0,
                "line_count": 4,
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is None  # Perfect run, no failure

    def test_analyze_multiple_failures(self):
        """Multiple failure types in one run (picks primary)."""
        run_data = {
            "form": "sonnet",
            "variant": "shakespearean",
            "prompt_idx": 0,
            "model_id": "test",
            "user_request": "Write a sonnet",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "ABAB CDCD EFEF GG",
                "detected_scheme": "ABAB CDCD EFEF",  # Wrong
                "matches_form": False,
                "strict_rhyme_density": 0.5,  # Low
                "rhyme_density": 0.5,
                "deviations_count": 2,
                "line_count": 11,  # Wrong count (difference of 3 > tolerance of 2)
            },
        }

        analyzer = DiagnosticAnalyzer()
        failure = analyzer.analyze_run(run_data)

        assert failure is not None
        # Should prioritize line count error (checked earlier in priority)
        assert failure.category == FailureCategory.LINE_COUNT_ERROR.value

    def test_severity_scoring(self):
        """Severity scores match expected ranges."""
        # Minor failure: just barely below density threshold
        minor_run = {
            "form": "couplets",
            "variant": None,
            "prompt_idx": 0,
            "model_id": "test",
            "user_request": "Write couplets",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "AA BB",
                "detected_scheme": "AA BB",
                "matches_form": True,
                "strict_rhyme_density": 0.68,  # Just below 0.7
                "rhyme_density": 0.68,
                "deviations_count": 0,
                "line_count": 4,
            },
        }

        # Severe failure: complete form confusion
        severe_run = {
            "form": "sonnet",
            "variant": "shakespearean",
            "prompt_idx": 1,
            "model_id": "test",
            "user_request": "Write a sonnet",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "ABAB CDCD EFEF GG",
                "detected_scheme": "",
                "matches_form": False,
                "strict_rhyme_density": 0.0,
                "rhyme_density": 0.0,
                "deviations_count": 14,
                "line_count": 14,
            },
        }

        analyzer = DiagnosticAnalyzer()

        minor_failure = analyzer.analyze_run(minor_run)
        severe_failure = analyzer.analyze_run(severe_run)

        assert minor_failure is not None
        assert severe_failure is not None

        # Minor should have low severity, severe should have high
        assert minor_failure.severity < 0.3
        assert severe_failure.severity > 0.8

    def test_threshold_configuration(self):
        """Custom thresholds work correctly."""
        # Custom threshold: only fail if density < 0.5 (instead of 0.7)
        custom_thresholds = {"strict_density_min": 0.5}

        run_data = {
            "form": "couplets",
            "variant": None,
            "prompt_idx": 0,
            "model_id": "test",
            "user_request": "Write couplets",
            "final_poem": "Test poem",
            "rhyme_analysis": {
                "expected_scheme": "AA BB",
                "detected_scheme": "AA BB",
                "matches_form": True,
                "strict_rhyme_density": 0.6,  # Between 0.5 and 0.7
                "rhyme_density": 0.6,
                "deviations_count": 0,
                "line_count": 4,
            },
        }

        # Default analyzer should flag this as failure
        default_analyzer = DiagnosticAnalyzer()
        default_failure = default_analyzer.analyze_run(run_data)
        assert default_failure is not None

        # Custom analyzer should pass
        custom_analyzer = DiagnosticAnalyzer(custom_thresholds)
        custom_failure = custom_analyzer.analyze_run(run_data)
        assert custom_failure is None


@pytest.mark.eval
class TestDiagnosticReport:
    """Test report generation and aggregation."""

    @pytest.fixture
    def sample_runs(self):
        """Sample runs with mixed success/failure."""
        return [
            # Success
            {
                "form": "couplets",
                "variant": None,
                "prompt_idx": 0,
                "model_id": "model1",
                "user_request": "Write couplets",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "AA BB",
                    "detected_scheme": "AA BB",
                    "matches_form": True,
                    "strict_rhyme_density": 1.0,
                    "rhyme_density": 1.0,
                    "deviations_count": 0,
                    "line_count": 4,
                },
            },
            # Scheme violation
            {
                "form": "sonnet",
                "variant": "shakespearean",
                "prompt_idx": 1,
                "model_id": "model1",
                "user_request": "Write a sonnet",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "ABAB CDCD EFEF GG",
                    "detected_scheme": "ABAB CDCD EFGH GI",  # 29% deviation
                    "matches_form": False,
                    "strict_rhyme_density": 0.75,
                    "rhyme_density": 0.75,
                    "deviations_count": 4,
                    "line_count": 14,
                },
            },
            # Near-miss
            {
                "form": "quatrain",
                "variant": None,
                "prompt_idx": 2,
                "model_id": "model1",
                "user_request": "Write a quatrain",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "ABAB",
                    "detected_scheme": "ABAB",
                    "matches_form": True,
                    "strict_rhyme_density": 0.5,
                    "rhyme_density": 0.9,
                    "deviations_count": 0,
                    "strict_rhyme_pairs": 1,
                    "rhyme_pairs": 4,
                    "line_count": 4,
                },
            },
            # Another scheme violation (sonnet)
            {
                "form": "sonnet",
                "variant": "shakespearean",
                "prompt_idx": 3,
                "model_id": "model1",
                "user_request": "Write a sonnet",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "ABAB CDCD EFEF GG",
                    "detected_scheme": "ABAB CDCD GHGH IJ",  # 36% deviation (5/14)
                    "matches_form": False,
                    "strict_rhyme_density": 0.75,
                    "rhyme_density": 0.75,
                    "deviations_count": 5,
                    "line_count": 14,
                },
            },
        ]

    def test_failure_breakdown_counts(self, sample_runs):
        """Accurate category counts."""
        report = DiagnosticReport(sample_runs)
        breakdown = report.failure_breakdown()

        assert breakdown[FailureCategory.SCHEME_VIOLATION.value] == 2
        assert breakdown[FailureCategory.NEAR_MISS.value] == 1
        assert len(report.failures) == 3  # 3 failures out of 4 runs

    def test_by_form_grouping(self, sample_runs):
        """Failures grouped by form correctly."""
        report = DiagnosticReport(sample_runs)
        by_form = report.by_form()

        assert "sonnet" in by_form
        assert "couplets" in by_form
        assert "quatrain" in by_form

        # Sonnet: 2 runs, 2 failures
        assert by_form["sonnet"]["total"] == 2
        assert by_form["sonnet"]["failures"] == 2
        assert by_form["sonnet"]["failure_rate"] == 1.0

        # Couplets: 1 run, 0 failures
        assert by_form["couplets"]["total"] == 1
        assert by_form["couplets"]["failures"] == 0
        assert by_form["couplets"]["failure_rate"] == 0.0

    def test_actionable_insights_generation(self, sample_runs):
        """Insights prioritize high-severity issues."""
        report = DiagnosticReport(sample_runs)
        insights = report.actionable_insights()

        assert len(insights) > 0

        # Top insight should have priority 1
        assert insights[0]["priority"] == 1

        # Should have recommendations
        for insight in insights:
            assert "recommendation" in insight
            assert len(insight["recommendation"]) > 0

    def test_serialization_roundtrip(self, sample_runs):
        """to_dict produces valid JSON."""
        report = DiagnosticReport(sample_runs)
        data = report.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(data)
        roundtrip = json.loads(json_str)

        assert roundtrip["summary"]["total_runs"] == 4
        assert roundtrip["summary"]["total_failures"] == 3
        assert "failure_breakdown" in roundtrip
        assert "insights" in roundtrip

    def test_markdown_format_valid(self, sample_runs):
        """Markdown output is valid and readable."""
        report = DiagnosticReport(sample_runs)
        markdown = report.to_markdown()

        assert "# Rhyme Benchmark Diagnostic Report" in markdown
        assert "Failure Breakdown" in markdown
        assert "Performance by Form" in markdown
        assert "Actionable Insights" in markdown
        assert len(markdown) > 100  # Should be substantial


@pytest.mark.data
class TestDiagnosticData:
    """Integration tests with actual diagnostic data."""

    @pytest.fixture
    def diagnostic_path(self):
        return RHYME_BENCH_DIR / "diagnostic_report.json"

    def test_diagnostic_report_structure_if_exists(self, diagnostic_path):
        """Validate structure of diagnostic_report.json."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        assert "summary" in diag
        assert "failure_breakdown" in diag
        assert "severity_distribution" in diag
        assert "by_form" in diag
        assert "failures" in diag
        assert "insights" in diag
        assert "thresholds" in diag

        # Validate summary structure
        summary = diag["summary"]
        assert "total_runs" in summary
        assert "total_failures" in summary
        assert "failure_rate" in summary
        assert "mean_severity" in summary

    def test_severity_bounds(self, diagnostic_path):
        """Assert severity scores within bounds."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        for failure in diag.get("failures", []):
            severity = failure.get("severity", 0)
            assert 0.0 <= severity <= 1.0, f"Severity {severity} out of bounds"

        # Check summary mean severity
        mean_severity = diag["summary"].get("mean_severity", 0)
        assert 0.0 <= mean_severity <= 1.0

    def test_failure_categories_valid(self, diagnostic_path):
        """All categories are recognized."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        valid_categories = {cat.value for cat in FailureCategory}

        for failure in diag.get("failures", []):
            category = failure.get("category")
            assert category in valid_categories, f"Unknown category: {category}"

        for category in diag.get("failure_breakdown", {}).keys():
            assert category in valid_categories, f"Unknown category in breakdown: {category}"


@pytest.mark.eval
class TestByModelGrouping:
    """Test per-model performance aggregation."""

    @pytest.fixture
    def multi_model_runs(self):
        """Sample runs from multiple models with different patterns."""
        return [
            # Trained model - success
            {
                "form": "couplets",
                "variant": None,
                "prompt_idx": 0,
                "model_id": "trained-llama3.1-8b",
                "user_request": "Write couplets",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "AA BB",
                    "detected_scheme": "AA BB",
                    "matches_form": True,
                    "strict_rhyme_density": 1.0,
                    "rhyme_density": 1.0,
                    "deviations_count": 0,
                    "line_count": 4,
                },
            },
            # Trained model - success
            {
                "form": "sonnet",
                "variant": "shakespearean",
                "prompt_idx": 1,
                "model_id": "trained-llama3.1-8b",
                "user_request": "Write a sonnet",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "ABAB CDCD EFEF GG",
                    "detected_scheme": "ABAB CDCD EFEF GG",
                    "matches_form": True,
                    "strict_rhyme_density": 0.93,
                    "rhyme_density": 0.93,
                    "deviations_count": 0,
                    "line_count": 14,
                },
            },
            # Vanilla model - failure (scheme violation)
            {
                "form": "sonnet",
                "variant": "shakespearean",
                "prompt_idx": 1,
                "model_id": "llama3.1-8b-vanilla",
                "user_request": "Write a sonnet",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "ABAB CDCD EFEF GG",
                    "detected_scheme": "ABAB CDCD EFGH GI",
                    "matches_form": False,
                    "strict_rhyme_density": 0.75,
                    "rhyme_density": 0.75,
                    "deviations_count": 4,
                    "line_count": 14,
                },
            },
            # Vanilla model - failure (density issue)
            {
                "form": "couplets",
                "variant": None,
                "prompt_idx": 0,
                "model_id": "llama3.1-8b-vanilla",
                "user_request": "Write couplets",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "AA BB",
                    "detected_scheme": "AA BB",
                    "matches_form": True,
                    "strict_rhyme_density": 0.5,  # Below threshold
                    "rhyme_density": 0.5,
                    "deviations_count": 2,
                    "line_count": 4,
                },
            },
            # Frontier model - success
            {
                "form": "quatrain",
                "variant": None,
                "prompt_idx": 2,
                "model_id": "claude-sonnet-4",
                "user_request": "Write a quatrain",
                "final_poem": "Test",
                "rhyme_analysis": {
                    "expected_scheme": "ABAB",
                    "detected_scheme": "ABAB",
                    "matches_form": True,
                    "strict_rhyme_density": 1.0,
                    "rhyme_density": 1.0,
                    "deviations_count": 0,
                    "line_count": 4,
                },
            },
        ]

    def test_by_model_counts(self, multi_model_runs):
        """Verify accurate run and failure counts per model."""
        report = DiagnosticReport(multi_model_runs)
        by_model = report.by_model()

        # Check all models present
        assert "trained-llama3.1-8b" in by_model
        assert "llama3.1-8b-vanilla" in by_model
        assert "claude-sonnet-4" in by_model

        # Trained model: 2 runs, 0 failures
        assert by_model["trained-llama3.1-8b"]["total"] == 2
        assert by_model["trained-llama3.1-8b"]["failures"] == 0
        assert by_model["trained-llama3.1-8b"]["success_rate"] == 1.0

        # Vanilla model: 2 runs, 2 failures
        assert by_model["llama3.1-8b-vanilla"]["total"] == 2
        assert by_model["llama3.1-8b-vanilla"]["failures"] == 2
        assert by_model["llama3.1-8b-vanilla"]["success_rate"] == 0.0

        # Frontier model: 1 run, 0 failures
        assert by_model["claude-sonnet-4"]["total"] == 1
        assert by_model["claude-sonnet-4"]["failures"] == 0
        assert by_model["claude-sonnet-4"]["success_rate"] == 1.0

    def test_model_type_categorization(self, multi_model_runs):
        """Verify model type categorization."""
        report = DiagnosticReport(multi_model_runs)
        by_model = report.by_model()

        assert by_model["trained-llama3.1-8b"]["model_type"] == "trained"
        assert by_model["llama3.1-8b-vanilla"]["model_type"] == "vanilla"
        assert by_model["claude-sonnet-4"]["model_type"] == "frontier"

    def test_primary_issues_per_model(self, multi_model_runs):
        """Verify primary issues correctly identified per model."""
        report = DiagnosticReport(multi_model_runs)
        by_model = report.by_model()

        # Vanilla has 2 failures: scheme_violation and density_issue
        vanilla_issues = by_model["llama3.1-8b-vanilla"]["primary_issues"]
        assert "scheme_violation" in vanilla_issues
        assert "density_issue" in vanilla_issues
        assert len(vanilla_issues) <= 3  # Top 3 only

        # Trained has no failures
        trained_issues = by_model["trained-llama3.1-8b"]["primary_issues"]
        assert len(trained_issues) == 0

    def test_mean_severity_calculation(self, multi_model_runs):
        """Verify mean severity computed correctly."""
        report = DiagnosticReport(multi_model_runs)
        by_model = report.by_model()

        # Models with no failures should have 0.0 mean severity
        assert by_model["trained-llama3.1-8b"]["mean_severity"] == 0.0
        assert by_model["claude-sonnet-4"]["mean_severity"] == 0.0

        # Vanilla model has failures, should have positive severity
        assert by_model["llama3.1-8b-vanilla"]["mean_severity"] > 0.0

    def test_markdown_includes_model_section(self, multi_model_runs):
        """Verify markdown output includes Performance by Model section."""
        report = DiagnosticReport(multi_model_runs)
        markdown = report.to_markdown()

        assert "## Performance by Model" in markdown
        assert "### Fine-Tuned Models" in markdown
        assert "### Vanilla Baselines" in markdown
        assert "### Frontier Models" in markdown

        # Check specific model entries
        assert "#### trained-llama3.1-8b" in markdown
        assert "#### llama3.1-8b-vanilla" in markdown
        assert "#### claude-sonnet-4" in markdown

        # Verify stats displayed
        assert "**Success rate**:" in markdown
        assert "**Mean severity**:" in markdown
        assert "**Primary issues**:" in markdown

    def test_model_categorization_edge_cases(self):
        """Test edge cases in model categorization."""
        report = DiagnosticReport([])

        # Test categorization helper
        assert report._categorize_model("trained-model-name") == "trained"
        assert report._categorize_model("model-vanilla") == "vanilla"
        assert report._categorize_model("claude-opus-4") == "frontier"
        assert report._categorize_model("qwen3-32b") == "frontier"
        assert report._categorize_model("unknown") == "frontier"  # Default

        # Edge case: model with both prefix and suffix (shouldn't happen, but test)
        assert report._categorize_model("trained-model-vanilla") == "trained"  # Prefix wins


@pytest.mark.data
class TestByModelIntegration:
    """Integration tests with actual diagnostic data."""

    @pytest.fixture
    def diagnostic_path(self):
        return RHYME_BENCH_DIR / "diagnostic_report.json"

    def test_by_model_in_diagnostic_report(self, diagnostic_path):
        """Verify by_model section exists in diagnostic_report.json."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        assert "by_model" in diag
        by_model = diag["by_model"]

        # Should have at least one model
        assert len(by_model) > 0

        # Check structure of each model entry
        for model_id, stats in by_model.items():
            assert "total" in stats
            assert "failures" in stats
            assert "success_rate" in stats
            assert "failure_rate" in stats
            assert "mean_severity" in stats
            assert "primary_issues" in stats
            assert "model_type" in stats

            # Validate data types and ranges
            assert isinstance(stats["total"], int)
            assert isinstance(stats["failures"], int)
            assert 0.0 <= stats["success_rate"] <= 1.0
            assert 0.0 <= stats["failure_rate"] <= 1.0
            assert 0.0 <= stats["mean_severity"] <= 1.0
            assert isinstance(stats["primary_issues"], list)
            assert stats["model_type"] in ["trained", "vanilla", "frontier"]

            # Consistency checks
            assert abs(stats["success_rate"] + stats["failure_rate"] - 1.0) < 0.01

    def test_markdown_summary_includes_model_section(self):
        """Verify diagnostic_summary.md includes model performance."""
        summary_path = RHYME_BENCH_DIR / "diagnostic_summary.md"
        if not summary_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_summary.md not present")

        with open(summary_path) as f:
            content = f.read()

        assert "## Performance by Model" in content
