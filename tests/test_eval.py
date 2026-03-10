"""Rhyme, meter, and form evaluation tests."""
from pathlib import Path

import pytest

from scripts.eval.form_registry import (
    FORMS,
    detect_form,
    form_description,
    get_line_count,
    get_meter,
    get_scheme,
    is_metered_form,
    is_rhyming_form,
    parse_scheme,
)
from scripts.eval.meter_analyzer import analyze as analyze_meter
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme
from scripts.eval.rhyme_analyzer import format_analysis_for_prompt

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.eval
class TestRhymeAnalyzer:
    """Rhyme analyzer unit tests with known poems."""

    def test_analyze_limerick(self):
        poem = """There once was a cat from Maine
Who refused to go out in the rain.
He slept on the mat
Next to the fat rat
And never got up again."""
        result = analyze_rhyme(poem, expected_form="limerick")
        assert result["line_count"] == 5
        assert result["expected_scheme"] == "AABBA"
        assert "detected_scheme" in result
        assert result["strict_rhyme_density"] >= 0
        assert result["rhyme_density"] >= 0

    def test_analyze_quatrain_abab(self):
        poem = """The rain on the tin
makes a sound like a din.
When the storm lets up
we fill the old cup."""
        result = analyze_rhyme(poem, expected_form="quatrain")
        assert result["line_count"] == 4
        assert result["expected_scheme"] == "ABAB"
        assert "end_words" in result
        assert len(result["end_words"]) == 4

    def test_analyze_empty_poem(self):
        result = analyze_rhyme("")
        assert result["line_count"] == 0
        assert result["detected_scheme"] == ""
        assert result["rhyme_density"] == 0.0
        assert result.get("strict_rhyme_density", 0.0) == 0.0
        assert result["matches_form"] is None

    def test_analyze_single_line(self):
        result = analyze_rhyme("A single line with no rhyme.")
        assert result["line_count"] == 1
        assert result["rhyme_density"] == 0.0

    def test_analyze_prose_paragraph(self):
        text = "This is a paragraph of prose. It has no line breaks that would indicate poetry."
        result = analyze_rhyme(text)
        assert result["line_count"] == 1
        assert "end_words" in result

    def test_format_analysis_for_prompt(self):
        poem = "cat\nmat"
        result = analyze_rhyme(poem)
        formatted = format_analysis_for_prompt(result)
        assert "Detected scheme" in formatted
        assert "Rhyme density" in formatted


@pytest.mark.eval
class TestFormRegistry:
    """Form registry validation."""

    def test_all_forms_have_valid_scheme_or_none(self):
        for form_name, spec in FORMS.items():
            scheme = spec.get("rhyme_scheme")
            if scheme is not None:
                parsed = parse_scheme(scheme)
                assert len(parsed) > 0, f"{form_name}: scheme {scheme} parses to empty"

    def test_sonnet_has_variants(self):
        assert "variants" in FORMS["sonnet"]
        assert "shakespearean" in FORMS["sonnet"]["variants"]
        assert get_scheme("sonnet", "shakespearean") == "ABAB CDCD EFEF GG"

    def test_get_scheme_limerick(self):
        assert get_scheme("limerick") == "AABBA"

    def test_get_scheme_villanelle(self):
        assert get_scheme("villanelle") == "ABA ABA ABA ABA ABA ABAA"

    def test_get_line_count(self):
        assert get_line_count("limerick") == 5
        assert get_line_count("sonnet", "shakespearean") == 14
        assert get_line_count("sonnet") == 14

    def test_get_meter(self):
        assert get_meter("sonnet") == "iambic pentameter"
        assert get_meter("limerick") == "anapestic"

    def test_is_rhyming_form(self):
        assert is_rhyming_form("sonnet") is True
        assert is_rhyming_form("limerick") is True
        assert is_rhyming_form("free_verse") is False

    def test_is_metered_form(self):
        assert is_metered_form("sonnet") is True
        assert is_metered_form("quatrain") is False

    def test_detect_form_from_text(self):
        assert detect_form("Write a Shakespearean sonnet") == "sonnet"
        assert detect_form("Write a villanelle about time") == "villanelle"
        assert detect_form("Write a limerick") == "limerick"

    def test_form_description(self):
        desc = form_description("limerick")
        assert "limerick" in desc.lower()
        assert "5" in desc or "AABBA" in desc


@pytest.mark.eval
class TestMeterAnalyzer:
    """Meter analyzer tests."""

    def test_analyze_empty_poem(self):
        result = analyze_meter("")
        assert result["line_stresses"] == []
        assert result["consistency"] == 0.0

    def test_analyze_simple_poem(self):
        poem = """The cat sat on the mat
And watched the mouse run past."""
        result = analyze_meter(poem)
        assert "line_stresses" in result
        assert "dominant_foot" in result or "dominant_foot_name" in result

    def test_analyze_does_not_crash_on_prose(self):
        text = "This is a paragraph of prose with no particular meter."
        result = analyze_meter(text)
        assert isinstance(result, dict)
        assert "consistency" in result


@pytest.mark.eval
class TestEdgeCases:
    """Edge cases — should not crash."""

    def test_rhyme_empty_string(self):
        result = analyze_rhyme("")
        assert result["line_count"] == 0

    def test_rhyme_whitespace_only(self):
        result = analyze_rhyme("   \n\n   ")
        assert result["line_count"] == 0

    def test_rhyme_unicode(self):
        poem = "café\nthé"
        result = analyze_rhyme(poem)
        assert "end_words" in result

    def test_meter_empty_string(self):
        result = analyze_meter("")
        assert result["consistency"] == 0.0
