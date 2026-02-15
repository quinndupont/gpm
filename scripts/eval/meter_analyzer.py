#!/usr/bin/env python3
"""Deterministic scansion analysis using CMU Pronouncing Dictionary.

Detects metrical patterns, measures consistency, identifies strategic variations.
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

import pronouncing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.form_registry import (
    detect_form,
    get_meter,
    get_meter_spec,
    METERS,
)

# ---------------------------------------------------------------------------
# Stress extraction
# ---------------------------------------------------------------------------

# Foot names for common two/three-syllable patterns
FOOT_NAMES = {
    "01": "iamb",
    "10": "trochee",
    "11": "spondee",
    "00": "pyrrhic",
    "001": "anapest",
    "010": "amphibrach",
    "100": "dactyl",
}

# Monosyllabic function words — typically unstressed in English meter
_FUNCTION_WORDS = frozenset({
    # articles
    "a", "an", "the",
    # prepositions
    "at", "by", "for", "from", "in", "of", "on", "to", "up", "with",
    # conjunctions
    "and", "as", "but", "if", "nor", "or", "so", "yet",
    # pronouns (weak forms)
    "he", "her", "him", "his", "i", "it", "its", "me", "my", "she",
    "that", "them", "they", "us", "we", "who", "you", "your",
    # auxiliary/modal (weak)
    "am", "are", "be", "been", "can", "could", "did", "do", "does",
    "had", "has", "have", "is", "may", "might", "must", "shall",
    "should", "was", "were", "will", "would",
    # other
    "all", "each", "no", "not", "some", "than", "this", "too",
})


def _clean_word(word: str) -> str:
    return word.strip().strip(string.punctuation).lower()


def _word_stresses(word: str) -> str | None:
    """Get stress pattern for a word. Returns string of 0s and 1s, or None.

    For monosyllabic function words, returns "0" (unstressed) since CMU dict
    marks all monosyllables as stressed but function words are typically
    unstressed in English meter.
    """
    phones_list = pronouncing.phones_for_word(word)
    if not phones_list:
        return None
    stresses = pronouncing.stresses(phones_list[0])
    # Collapse secondary stress (2) to primary (1)
    stresses = stresses.replace("2", "1")
    # Monosyllabic function words: override to unstressed
    if len(stresses) == 1 and word.lower() in _FUNCTION_WORDS:
        return "0"
    return stresses


def _line_stresses(line: str) -> tuple[str, list[tuple[str, str | None]]]:
    """Get the full stress pattern for a line.

    Returns:
        (stress_string, [(word, word_stress_or_None), ...])
    """
    tokens = line.strip().split()
    word_stresses = []
    full_pattern = []
    for token in tokens:
        w = _clean_word(token)
        if not w or not any(c.isalpha() for c in w):
            continue
        s = _word_stresses(w)
        word_stresses.append((w, s))
        if s is not None:
            full_pattern.append(s)
    return "".join(full_pattern), word_stresses


def _extract_lines(poem: str) -> list[str]:
    """Extract non-empty, non-title lines from a poem."""
    lines = []
    for line in poem.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if all(c in "-—–=* " for c in stripped):
            continue
        lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Foot parsing
# ---------------------------------------------------------------------------

def _parse_feet(stress_pattern: str, foot_size: int = 2) -> list[str]:
    """Split a stress pattern into feet of the given size."""
    feet = []
    for i in range(0, len(stress_pattern), foot_size):
        foot = stress_pattern[i:i + foot_size]
        if foot:
            feet.append(foot)
    return feet


def _classify_foot(foot: str) -> str:
    """Name a foot pattern."""
    return FOOT_NAMES.get(foot, f"({foot})")


def _dominant_foot(all_feet: list[str]) -> tuple[str, float]:
    """Find the most common foot pattern and its frequency."""
    if not all_feet:
        return "", 0.0
    counts: dict[str, int] = {}
    for f in all_feet:
        counts[f] = counts.get(f, 0) + 1
    best = max(counts, key=counts.get)
    return best, counts[best] / len(all_feet)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    poem: str,
    expected_form: str | None = None,
    expected_variant: str | None = None,
) -> dict:
    """Analyze a poem's metrical structure.

    Returns dict with stress patterns, dominant foot, consistency score,
    variations, and form-match info.
    """
    lines = _extract_lines(poem)
    if not lines:
        return {
            "line_stresses": [],
            "dominant_foot": None,
            "dominant_foot_name": None,
            "consistency": 0.0,
            "syllable_counts": [],
            "expected_meter": None,
            "matches_meter": None,
            "variations": [],
        }

    # Get stress pattern for each line
    line_data = []
    all_feet = []
    syllable_counts = []

    # Determine expected foot size from form
    expected_meter_name = None
    expected_foot = None
    expected_feet_per_line = None
    if expected_form:
        expected_meter_name = get_meter(expected_form, expected_variant)
        if expected_meter_name:
            spec = get_meter_spec(expected_meter_name)
            if spec:
                expected_foot = spec["foot"]
                expected_feet_per_line = spec.get("feet_per_line")

    foot_size = len(expected_foot) if expected_foot else 2  # default to disyllabic

    for line in lines:
        pattern, words = _line_stresses(line)
        syllable_counts.append(len(pattern))
        feet = _parse_feet(pattern, foot_size)
        all_feet.extend(feet)
        line_data.append({
            "text": line,
            "stress_pattern": pattern,
            "syllables": len(pattern),
            "feet": feet,
            "feet_named": [_classify_foot(f) for f in feet],
        })

    dominant, consistency = _dominant_foot(all_feet)
    dominant_name = _classify_foot(dominant) if dominant else None

    # Identify variations (feet that differ from expected or dominant)
    check_foot = expected_foot or dominant
    variations = []
    if check_foot:
        for line_idx, ld in enumerate(line_data):
            for foot_idx, foot in enumerate(ld["feet"]):
                if foot != check_foot and len(foot) == len(check_foot):
                    variations.append({
                        "line": line_idx + 1,
                        "foot_position": foot_idx + 1,
                        "expected": _classify_foot(check_foot),
                        "actual": _classify_foot(foot),
                        "pattern": foot,
                    })

    # Check against expected meter
    matches_meter = None
    if expected_foot:
        # Match = dominant foot matches expected AND consistency is reasonable (>50%)
        matches_meter = (dominant == expected_foot and consistency > 0.5)
        # Also check feet per line if specified
        if matches_meter and expected_feet_per_line:
            target_syllables = expected_feet_per_line * foot_size
            # Allow +/- 1 syllable tolerance per line
            on_target = sum(
                1 for sc in syllable_counts
                if abs(sc - target_syllables) <= 1
            )
            if on_target / len(syllable_counts) < 0.5:
                matches_meter = False

    return {
        "line_stresses": line_data,
        "dominant_foot": dominant,
        "dominant_foot_name": dominant_name,
        "consistency": round(consistency, 2),
        "syllable_counts": syllable_counts,
        "avg_syllables_per_line": round(sum(syllable_counts) / len(syllable_counts), 1) if syllable_counts else 0,
        "expected_meter": expected_meter_name,
        "matches_meter": matches_meter,
        "variations": variations,
        "variation_rate": round(len(variations) / len(all_feet), 2) if all_feet else 0.0,
    }


def format_analysis_for_prompt(analysis: dict) -> str:
    """Format meter analysis as a concise string for inclusion in an LLM prompt."""
    parts = []
    if analysis.get("expected_meter"):
        parts.append(f"Expected meter: {analysis['expected_meter']}")
    if analysis.get("dominant_foot_name"):
        parts.append(
            f"Dominant foot: {analysis['dominant_foot_name']} "
            f"({analysis['consistency']:.0%} of feet)"
        )
    if analysis.get("matches_meter") is not None:
        parts.append(f"Matches expected meter: {'yes' if analysis['matches_meter'] else 'no'}")
    parts.append(f"Avg syllables/line: {analysis.get('avg_syllables_per_line', 0)}")

    variations = analysis.get("variations", [])
    if variations:
        # Summarize variation types
        var_counts: dict[str, int] = {}
        for v in variations:
            var_counts[v["actual"]] = var_counts.get(v["actual"], 0) + 1
        var_summary = ", ".join(f"{count} {name}" for name, count in
                                sorted(var_counts.items(), key=lambda x: -x[1]))
        parts.append(f"Variations: {var_summary} ({analysis['variation_rate']:.0%} of feet)")
        # Show first few specific variations
        for v in variations[:4]:
            parts.append(
                f"  Line {v['line']}, foot {v['foot_position']}: "
                f"{v['expected']} -> {v['actual']}"
            )
        if len(variations) > 4:
            parts.append(f"  ... and {len(variations) - 4} more")
    else:
        parts.append("Variations: none (perfectly regular)")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze poem meter/scansion")
    parser.add_argument("--poem", type=str, help="Poem text (inline)")
    parser.add_argument("--file", type=Path, help="File containing poem text")
    parser.add_argument("--form", type=str, help="Expected form (e.g. sonnet)")
    parser.add_argument("--variant", type=str, help="Form variant (e.g. petrarchan)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.file:
        poem = args.file.read_text()
    elif args.poem:
        poem = args.poem
    else:
        print("Provide --poem or --file", file=sys.stderr)
        sys.exit(1)

    form = args.form or detect_form(poem)
    result = analyze(poem, expected_form=form, expected_variant=args.variant)

    if args.json:
        # Trim per-line data for readability
        output = {k: v for k, v in result.items() if k != "line_stresses"}
        output["line_count"] = len(result["line_stresses"])
        print(json.dumps(output, indent=2))
    else:
        print(format_analysis_for_prompt(result))


if __name__ == "__main__":
    main()
