"""Rhyme benchmark tests — prompt parsing, structure, threshold assertions."""
import json
import pytest
from pathlib import Path

from scripts.benchmarks.rhyme_bench.prompts import RHYME_PROMPTS
from scripts.eval.form_registry import get_scheme, FORMS

ROOT = Path(__file__).resolve().parent.parent
RHYME_BENCH_DIR = ROOT / "data" / "rhyme_bench"


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
            pytest.skip("data/rhyme_bench/summary.json not present")
        with open(summary_path) as f:
            summary = json.load(f)
        assert "total_runs" in summary
        assert "matches_form_rate" in summary
        assert "mean_strict_rhyme_density" in summary

    def test_summary_thresholds_if_exists(self, summary_path):
        if not summary_path.exists():
            pytest.skip("data/rhyme_bench/summary.json not present")
        with open(summary_path) as f:
            summary = json.load(f)
        mean_density = summary.get("mean_strict_rhyme_density", 0)
        form_rate = summary.get("matches_form_rate", 0)
        assert mean_density >= 0.5, f"mean_strict_rhyme_density={mean_density} < 0.5"
        assert form_rate >= 0.6, f"matches_form_rate={form_rate} < 0.6"
