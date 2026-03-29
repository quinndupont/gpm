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
from scripts.eval.rhyme_analyzer import _rhyme_type
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme
from scripts.eval.rhyme_analyzer import format_analysis_for_prompt
from scripts.eval.rhyme_analyzer import strip_reasoning_blocks

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

    def test_limerick_detected_scheme_flat_no_phantom_stanza(self):
        """Single-stanza AABBA must not be split like ``AABB A`` (old 4-char chunking)."""
        poem = (
            "There once was a baker so fine,\n"
            "Whose bread never rose with design.\n"
            "The yeast was quite bright,\n"
            "But the loaves weren't in sight,\n"
            "And his customers left in decline."
        )
        r = analyze_rhyme(poem, expected_form="limerick")
        assert r["matches_form"] is True
        assert r["detected_scheme"] == "AABBA"
        assert " " not in r["detected_scheme"]

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

    def test_shakespearean_sonnet_stanza_boundaries_detected_scheme(self):
        """Regression: cross-quatrain assonance must not collapse CDCD/EFEF."""
        poem = """In yellow wood where paths diverge and split,
A traveler stands at crossroads, unsure which way.
The sun's last beams cast shadows on the bit
Of road ahead, each turn inviting day.

To leftward leads a lane well-trodden still,
Where whispers of old tales have often rung;
But rightward lures green fields under hill,
With paths less trod and stories yet unsung.

Two roads meet here at twilight's hush deep,
Each leading to unknown destinies.
The choice must be made, for time does sweep
A traveler onward with no ease.

Thus I choose, though the left is worn and tried,
For novelty and chance await the ride."""
        result = analyze_rhyme(poem, expected_form="sonnet", expected_variant="shakespearean")
        flat = result["detected_scheme"].replace(" ", "")
        assert flat == "ABABCDCDEFEFGG"
        assert result["matches_form"] is True

    def test_spenserian_cross_stanza_perfect_b_rhyme(self):
        """B in Q2 must link to Q1 via perfect rhyme, not slant-only."""
        poem = """The sky was dark as deepest night
And morning brought a golden day
When stars gave way to morning sight
Along the road and winding way

We rested in the heat of play
And scanned the ridge for any look
Then turned again upon the stray
That led us where we quickly took"""
        result = analyze_rhyme(poem)
        assert result["detected_scheme"].replace(" ", "") == "ABABBCBC"

    def test_slant_rhyme_tightening_time_mine_vs_deep_ease(self):
        assert _rhyme_type("time", "mine") == "slant"
        assert _rhyme_type("deep", "ease") == "none"
        assert _rhyme_type("deep", "destinies") == "none"

    def test_unicode_em_dash_on_end_word_does_not_break_rhyme(self):
        """Em dash (U+2014) is not in string.punctuation; must still CMU-match."""
        assert _rhyme_type("me", "free\u2014") == "perfect"
        couplet = "The wood was wide, and both roads led to free\u2014\nThe air was clear, and peace returned to me."
        r = analyze_rhyme(couplet)
        assert r["end_words"] == ["free", "me"]
        assert r["detected_scheme"].replace(" ", "") == "AA"

    def test_strip_reasoning_blocks_before_rhyme_analysis(self):
        """``<reasoning>`` prose must not count as poem lines."""
        wrapped = (
            "<reasoning>Line 1 (A): wood\nLine 2 (B): way\nPlanning rhymes…\n</reasoning>\n\n"
            "The cat sat on the mat\nAnd looked up at the bat"
        )
        assert strip_reasoning_blocks(wrapped).strip().startswith("The cat")
        r = analyze_rhyme(wrapped)
        assert r["line_count"] == 2
        assert r["end_words"] == ["mat", "bat"]

    def test_reasoning_closed_with_wrong_think_tag_still_strips(self):
        """Models sometimes close ``<reasoning>`` with ``</think>``."""
        wrapped = (
            "<reasoning>planning\n</think>\n\n"
            "Rose red, violet blue\nSugar sweet, and so are you"
        )
        r = analyze_rhyme(wrapped)
        assert r["line_count"] == 2
        assert "planning" not in " ".join(r["end_words"]).lower()

    def test_multi_pronunciation_on_matches_gone(self):
        """CMU lists *on* as AA1 N and AO1 N; *gone* is AO1 N — must not use only first."""
        assert _rhyme_type("on", "gone") == "perfect"
        assert _rhyme_type("gone", "on") == "perfect"

    def test_stress_insensitive_same_nucleus_grey_yesterday(self):
        """Monosyllable primary vs polysyllable-final secondary stress (EY1 vs EY2)."""
        assert _rhyme_type("grey", "yesterday") == "perfect"
        assert _rhyme_type("gray", "yesterday") == "perfect"

    def test_end_word_when_last_token_is_standalone_em_dash(self):
        """``split()`` yields a final ``—`` token; rhyme end-word must still be *be*."""
        from scripts.eval.rhyme_analyzer import _get_end_word

        assert _get_end_word("To wander thus, though lonely, is to be \u2014  ") == "be"
        couplet = (
            "To wander thus, though lonely, is to be \u2014\n"
            "A traveler bound by no but what's set free."
        )
        r = analyze_rhyme(couplet)
        assert r["line_count"] == 2
        assert r["end_words"] == ["be", "free"]
        assert r["detected_scheme"].replace(" ", "") == "AA"

    def test_quatrain_without_stanza_breaks_still_abab(self):
        # End-words must be CMU ABAB (tin/din/up/cup phonetically reads as AABB).
        poem = """I walked upon the ancient stone
And took the long and winding way
Then found another heavy bone
That led me through the light of day"""
        result = analyze_rhyme(poem, expected_form="quatrain")
        assert result["line_count"] == 4
        assert result["expected_scheme"] == "ABAB"
        flat = result["detected_scheme"].replace(" ", "")
        assert flat == "ABAB"


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
