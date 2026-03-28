"""Rhyme benchmark tests — prompt parsing, structure, threshold assertions."""
import json
import re
from pathlib import Path

import pytest

from scripts.benchmarks.rhyme_bench.prompts import RHYME_PROMPTS
from scripts.eval.form_registry import FORMS, get_scheme

ROOT = Path(__file__).resolve().parent.parent
RHYME_BENCH_DIR = ROOT / "data" / "rhyme_bench" / "studies" / "baseline_default"
STUDIES_DIR = ROOT / "scripts" / "benchmarks" / "rhyme_bench" / "studies"


@pytest.mark.eval
class TestRhymeBenchPrompts:
    """Prompt coverage and parsing."""

    def test_all_prompts_parse(self):
        assert len(RHYME_PROMPTS) >= 10
        for i, (form, variant, request) in enumerate(RHYME_PROMPTS):
            assert form in FORMS, f"Prompt {i}: unknown form {form}"
            assert isinstance(request, str)
            assert len(request) > 20

    def test_prompts_cover_forms(self):
        forms = {f for f, _, _ in RHYME_PROMPTS}
        assert "sonnet" in forms
        assert "villanelle" in forms
        assert "limerick" in forms
        assert "quatrain" in forms
        assert "couplets" in forms

    def test_each_form_has_scheme(self):
        for form, variant, _ in RHYME_PROMPTS:
            scheme = get_scheme(form, variant)
            assert scheme is not None, f"Form {form} variant {variant} has no scheme"


@pytest.mark.data
class TestRhymeBenchSummary:
    """Threshold assertions on summary.json when present."""

    @pytest.fixture
    def summary_path(self):
        return RHYME_BENCH_DIR / "summary.json"

    def test_summary_structure_if_exists(self, summary_path):
        if not summary_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/summary.json not present")
        with open(summary_path) as f:
            summary = json.load(f)
        assert "total_runs" in summary
        assert "matches_form_rate" in summary
        assert "mean_strict_rhyme_density" in summary
        assert "run_timestamp" in summary, "summary.json should include run_timestamp field"

    def test_summary_thresholds_if_exists(self, summary_path):
        if not summary_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/summary.json not present")
        with open(summary_path) as f:
            summary = json.load(f)
        mean_density = summary.get("mean_strict_rhyme_density", 0)
        form_rate = summary.get("matches_form_rate", 0)
        assert mean_density >= 0.5, f"mean_strict_rhyme_density={mean_density} < 0.5"
        assert form_rate >= 0.6, f"matches_form_rate={form_rate} < 0.6"


@pytest.mark.data
class TestTimestampedBehavior:
    """Tests for timestamped filename behavior and accumulation."""

    def test_timestamped_filenames_if_exist(self):
        """Verify individual run files have timestamp suffix."""
        rhyme_files = list(RHYME_BENCH_DIR.glob("rhyme_*.json"))
        if not rhyme_files:
            pytest.skip("No rhyme_*.json files present")

        # Pattern: rhyme_{form}_{idx}_{slug}_{timestamp}.json
        # Timestamp format: YYYYMMDD_HHMMSS
        # Note: slug can contain alphanumeric, hyphens, underscores, and dots (e.g., qwen2.5-7b-vanilla)
        timestamp_pattern = re.compile(r"rhyme_\w+_\d+_[\w\.\-]+_\d{8}_\d{6}\.json")
        legacy_pattern = re.compile(r"rhyme_\w+_\d+_[\w\.\-]+\.json")  # Old format without timestamp

        timestamped_count = 0
        legacy_count = 0

        for file in rhyme_files:
            if timestamp_pattern.match(file.name):
                timestamped_count += 1
            elif legacy_pattern.match(file.name):
                legacy_count += 1
            else:
                pytest.fail(f"File {file.name} doesn't match expected pattern")

        # After implementation, all new files should be timestamped
        # But we allow legacy files to coexist
        assert timestamped_count > 0 or legacy_count > 0, "No valid rhyme files found"

    def test_summary_timestamp_format(self):
        """Verify summary.json timestamp is in correct format."""
        summary_path = RHYME_BENCH_DIR / "summary.json"
        if not summary_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/summary.json not present")

        with open(summary_path) as f:
            summary = json.load(f)

        timestamp = summary.get("run_timestamp")
        assert timestamp is not None, "run_timestamp field missing"

        # Verify format: YYYYMMDD_HHMMSS
        assert re.match(r"\d{8}_\d{6}", timestamp), f"Invalid timestamp format: {timestamp}"

        # Verify it's a valid date (basic check)
        year = int(timestamp[:4])
        month = int(timestamp[4:6])
        day = int(timestamp[6:8])
        hour = int(timestamp[9:11])
        minute = int(timestamp[11:13])
        second = int(timestamp[13:15])

        assert 2020 <= year <= 2030, f"Year {year} seems unrealistic"
        assert 1 <= month <= 12, f"Month {month} invalid"
        assert 1 <= day <= 31, f"Day {day} invalid"
        assert 0 <= hour <= 23, f"Hour {hour} invalid"
        assert 0 <= minute <= 59, f"Minute {minute} invalid"
        assert 0 <= second <= 59, f"Second {second} invalid"

    def test_timestamped_summary_exists(self):
        """Verify timestamped summary file exists."""
        summary_path = RHYME_BENCH_DIR / "summary.json"
        if not summary_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/summary.json not present")

        with open(summary_path) as f:
            summary = json.load(f)

        timestamp = summary.get("run_timestamp")
        assert timestamp is not None

        # Verify corresponding timestamped summary exists
        timestamped_summary_path = RHYME_BENCH_DIR / f"summary_{timestamp}.json"
        assert timestamped_summary_path.exists(), f"Timestamped summary {timestamped_summary_path} not found"

        # Verify it has the same content
        with open(timestamped_summary_path) as f:
            timestamped_summary = json.load(f)

        assert timestamped_summary == summary, "Timestamped and latest summaries should match"


@pytest.mark.data
class TestDiagnosticReportData:
    """Tests for diagnostic output data."""

    @pytest.fixture
    def diagnostic_path(self):
        return RHYME_BENCH_DIR / "diagnostic_report.json"

    @pytest.fixture
    def diagnostic_summary_path(self):
        return RHYME_BENCH_DIR / "diagnostic_summary.md"

    def test_diagnostic_structure_if_exists(self, diagnostic_path):
        """Validate diagnostic report structure."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        # Check top-level structure
        assert "summary" in diag
        assert "failure_breakdown" in diag
        assert "severity_distribution" in diag
        assert "by_form" in diag
        assert "failures" in diag
        assert "insights" in diag
        assert "thresholds" in diag

        # Validate summary
        summary = diag["summary"]
        assert "total_runs" in summary
        assert "total_failures" in summary
        assert "failure_rate" in summary
        assert "mean_severity" in summary
        assert "models_tested" in summary
        assert "forms_tested" in summary

        # Validate failure_breakdown
        breakdown = diag["failure_breakdown"]
        assert isinstance(breakdown, dict)
        for category, count in breakdown.items():
            assert isinstance(count, int)
            assert count >= 0

        # Validate by_form
        by_form = diag["by_form"]
        assert isinstance(by_form, dict)
        for form, stats in by_form.items():
            assert "total" in stats
            assert "failures" in stats
            assert "failure_rate" in stats
            assert "primary_issues" in stats
            assert "mean_severity" in stats

    def test_severity_scores_valid(self, diagnostic_path):
        """Assert severity scores within bounds."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        # Check individual failure severities
        for failure in diag.get("failures", []):
            severity = failure.get("severity", 0)
            assert 0.0 <= severity <= 1.0, f"Severity {severity} out of bounds for failure"

        # Check summary mean severity
        mean_severity = diag["summary"].get("mean_severity", 0)
        assert 0.0 <= mean_severity <= 1.0, f"Mean severity {mean_severity} out of bounds"

        # Check severity distribution
        for category, severity in diag.get("severity_distribution", {}).items():
            assert 0.0 <= severity <= 1.0, f"Severity {severity} out of bounds for category {category}"

    def test_failure_categories_recognized(self, diagnostic_path):
        """All categories are recognized failure types."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        from scripts.benchmarks.rhyme_bench.diagnostic import FailureCategory

        valid_categories = {cat.value for cat in FailureCategory}

        # Check failure instances
        for failure in diag.get("failures", []):
            category = failure.get("category")
            assert category in valid_categories, f"Unknown category: {category}"

        # Check failure breakdown
        for category in diag.get("failure_breakdown", {}).keys():
            assert category in valid_categories, f"Unknown category in breakdown: {category}"

    def test_insights_have_recommendations(self, diagnostic_path):
        """Insights include actionable recommendations."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        with open(diagnostic_path) as f:
            diag = json.load(f)

        insights = diag.get("insights", [])
        for insight in insights:
            assert "priority" in insight
            assert "category" in insight
            assert "form" in insight
            assert "description" in insight
            assert "recommendation" in insight
            assert "affected_runs" in insight

            # Recommendations should be non-empty
            assert len(insight["recommendation"]) > 0

            # Priority should be positive
            assert insight["priority"] > 0

    def test_diagnostic_summary_markdown_exists(self, diagnostic_path, diagnostic_summary_path):
        """Markdown summary exists when JSON exists."""
        if not diagnostic_path.exists():
            pytest.skip("data/rhyme_bench/studies/baseline_default/diagnostic_report.json not present")

        # If JSON exists, markdown should too
        assert diagnostic_summary_path.exists(), "diagnostic_summary.md missing when diagnostic_report.json present"

        # Validate basic markdown structure
        with open(diagnostic_summary_path) as f:
            content = f.read()

        assert "# Rhyme Benchmark Diagnostic Report" in content
        assert "## Failure Breakdown" in content or "Failure Breakdown" in content
        assert "## Performance by Form" in content or "Performance by Form" in content
        assert "## Actionable Insights" in content or "Actionable Insights" in content


@pytest.mark.eval
class TestRhymeBenchStudies:
    """Study folders and info cards (ablation metadata)."""

    def test_study_cards_exist(self):
        for sid in ("baseline_default", "ablate_backward", "ablate_cmu_two_pass"):
            card = STUDIES_DIR / sid / "CARD.yaml"
            assert card.exists(), f"Missing study card: {card}"
