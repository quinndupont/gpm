#!/usr/bin/env python3
"""Deterministic rhyme analysis using CMU Pronouncing Dictionary.

Detects rhyme schemes, validates against expected forms, classifies rhyme quality.
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path

import pronouncing

from scripts.eval.form_registry import detect_form, get_scheme, parse_scheme

# ---------------------------------------------------------------------------
# Phoneme helpers
# ---------------------------------------------------------------------------

def _clean_word(word: str) -> str:
    """Strip punctuation and lowercase."""
    return word.strip().strip(string.punctuation).lower()


def _get_end_word(line: str) -> str:
    """Extract the last word from a line, handling trailing punctuation."""
    tokens = line.strip().split()
    if not tokens:
        return ""
    return _clean_word(tokens[-1])


def _rhyme_suffix(phones: str) -> str:
    """Extract the rhyme suffix: from the last stressed vowel onward.

    E.g. "K AE1 T" -> "AE1 T", "S L IY1 P" -> "IY1 P"
    """
    parts = phones.split()
    # Find last stressed vowel (digit indicates stress: 0, 1, or 2)
    last_stressed = -1
    for i, p in enumerate(parts):
        if p[-1].isdigit():
            last_stressed = i
    if last_stressed == -1:
        # No stress marks found — use entire phoneme string
        return phones
    return " ".join(parts[last_stressed:])


def _get_rhyme_suffix(word: str) -> str | None:
    """Get rhyme suffix for a word. Returns None if not in CMU dict."""
    phones_list = pronouncing.phones_for_word(word)
    if not phones_list:
        return None
    return _rhyme_suffix(phones_list[0])


def _suffix_fallback(word: str) -> str:
    """Fallback for words not in CMU dict: use last 2-3 chars as pseudo-suffix."""
    w = word.lower()
    return w[-3:] if len(w) >= 3 else w


def _rhyme_type(word_a: str, word_b: str, strict: bool = False) -> str:
    """Classify rhyme between two words: 'perfect', 'slant', or 'none'.

    Perfect: identical rhyme suffix (stressed vowel onward).
    Slant: shared final consonant cluster with different vowel, OR same vowel
           with a single consonant difference (e.g. "time"/"mine").
    strict=True: only CMU-verified perfect/identical count; no fallback, no slant.
    """
    if word_a == word_b:
        return "identical"

    suffix_a = _get_rhyme_suffix(word_a)
    suffix_b = _get_rhyme_suffix(word_b)

    # Both in CMU dict — compare phoneme suffixes
    if suffix_a is not None and suffix_b is not None:
        if suffix_a == suffix_b:
            return "perfect"
        if strict:
            return "none"
        parts_a = suffix_a.split()
        parts_b = suffix_b.split()

        # Slant: same vowel + different final consonant(s), OR same vowel + same
        # trailing consonant(s) but different stress. E.g. "time" (AY1 M) / "mine" (AY1 N).
        #
        # IMPORTANT: Pure consonance (same final consonant, different vowels) is NOT
        # a slant rhyme. "coded" (IH0 D) and "deployed" (OY1 D) share final /D/ but
        # have completely different vowels — they do not rhyme at all.
        vowel_a = re.sub(r"\d", "", parts_a[0]) if parts_a[0][-1].isdigit() else None
        vowel_b = re.sub(r"\d", "", parts_b[0]) if parts_b[0][-1].isdigit() else None

        # Must share the same vowel for slant rhyme
        if vowel_a and vowel_b and vowel_a == vowel_b:
            # Same vowel — this is a slant rhyme (assonance)
            # E.g. "time" (AY1 M) / "mine" (AY1 N) — same vowel, different consonant
            # This is the classic definition of slant/near rhyme
            if abs(len(parts_a) - len(parts_b)) <= 1:
                return "slant"

        return "none"

    # Fallback: simple suffix matching for words not in CMU dict.
    # In strict mode, never count fallback — orthographic matches are unreliable.
    if strict:
        return "none"

    # Common suffixes that DON'T indicate rhyme (grammatical endings)
    FALSE_RHYME_SUFFIXES = {"ed", "ing", "ly", "er", "est", "ness", "ment", "tion", "sion"}

    fa = _suffix_fallback(word_a)
    fb = _suffix_fallback(word_b)

    # Don't count matches on common grammatical suffixes — they're false positives
    if fa[-2:] in FALSE_RHYME_SUFFIXES or fb[-2:] in FALSE_RHYME_SUFFIXES:
        if fa[-3:] in FALSE_RHYME_SUFFIXES or fb[-3:] in FALSE_RHYME_SUFFIXES:
            return "none"
        # Only accept if the full 3-char suffix matches AND it's not a common suffix
        if fa == fb and fa not in FALSE_RHYME_SUFFIXES:
            return "slant"  # Demote to slant since orthographic is unreliable
        return "none"

    if fa == fb:
        return "slant"  # Demote to slant — orthographic matches are not reliable "perfect"
    if len(fa) >= 2 and len(fb) >= 2 and fa[-2:] == fb[-2:]:
        return "slant"
    return "none"


# ---------------------------------------------------------------------------
# Line extraction
# ---------------------------------------------------------------------------

def _extract_lines(poem: str) -> list[str]:
    """Extract non-empty, non-title lines from a poem."""
    lines = []
    for line in poem.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip markdown headers (titles)
        if stripped.startswith("#"):
            continue
        # Skip lines that are just punctuation/dashes (dividers)
        if all(c in "-—–=* " for c in stripped):
            continue
        lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Scheme detection
# ---------------------------------------------------------------------------

def _detect_scheme(end_words: list[str], strict: bool = False) -> str:
    """Detect the rhyme scheme from a list of end-words. Returns e.g. 'ABABCDCDEFEF'.
    strict=True: only CMU-verified perfect/identical rhymes count."""
    if not end_words:
        return ""
    labels: list[str] = []
    next_label = 0

    for i, word in enumerate(end_words):
        matched_label = None
        for j in range(i):
            rt = _rhyme_type(word, end_words[j], strict=strict)
            ok = ("perfect", "slant", "identical") if not strict else ("perfect", "identical")
            if rt in ok:
                matched_label = labels[j]
                break
        if matched_label is not None:
            labels.append(matched_label)
        else:
            labels.append(chr(ord("A") + next_label))
            next_label += 1
            if next_label > 25:
                next_label = 0
    return "".join(labels)


def _format_scheme(scheme: str, line_count: int | None = None) -> str:
    """Group scheme into stanza-like chunks for readability.

    If the poem has a known line count we just return it flat.
    Otherwise, try groups of 4.
    """
    if not scheme:
        return ""
    if line_count and len(scheme) == line_count:
        return scheme
    # Group into fours
    groups = [scheme[i:i + 4] for i in range(0, len(scheme), 4)]
    return " ".join(groups)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    poem: str,
    expected_form: str | None = None,
    expected_variant: str | None = None,
) -> dict:
    """Analyze a poem's rhyme structure.

    Args:
        poem: The poem text.
        expected_form: Form name from registry (e.g. 'sonnet'). If None, auto-detect from text.
        expected_variant: Variant name (e.g. 'petrarchan'). Optional.

    Returns:
        Dict with scheme, rhyme pairs, deviations, etc.
    """
    lines = _extract_lines(poem)
    end_words = [_get_end_word(line) for line in lines]

    # Filter out empty end words (blank lines that slipped through)
    valid = [(i, w) for i, w in enumerate(end_words) if w]
    if not valid:
        return {
            "detected_scheme": "",
            "expected_scheme": None,
            "matches_form": None,
            "deviations": [],
            "rhyme_pairs": [],
            "rhyme_density": 0.0,
            "line_count": 0,
            "end_words": [],
        }

    indices, words = zip(*valid)
    words = list(words)
    detected_raw = _detect_scheme(words)
    strict_detected_raw = _detect_scheme(words, strict=True)

    # Build rhyme pairs (permissive: perfect, slant, fallback)
    rhyme_pairs = []
    strict_rhyme_pairs = []
    seen_pairs: set[tuple[int, int]] = set()
    seen_strict: set[tuple[int, int]] = set()
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            rt = _rhyme_type(words[i], words[j])
            if rt in ("perfect", "slant") and (i, j) not in seen_pairs:
                seen_pairs.add((i, j))
                rhyme_pairs.append({
                    "words": [words[i], words[j]],
                    "type": rt,
                    "lines": [indices[i] + 1, indices[j] + 1],
                })
            rt_strict = _rhyme_type(words[i], words[j], strict=True)
            if rt_strict in ("perfect", "identical") and (i, j) not in seen_strict:
                seen_strict.add((i, j))
                strict_rhyme_pairs.append({
                    "words": [words[i], words[j]],
                    "type": rt_strict,
                    "lines": [indices[i] + 1, indices[j] + 1],
                })

    # Rhyme density: fraction of lines participating in at least one rhyme
    rhyming_lines = set()
    for pair in rhyme_pairs:
        for ln in pair["lines"]:
            rhyming_lines.add(ln)
    density = len(rhyming_lines) / len(words) if words else 0.0

    strict_rhyming_lines = set()
    for pair in strict_rhyme_pairs:
        for ln in pair["lines"]:
            strict_rhyming_lines.add(ln)
    strict_density = len(strict_rhyming_lines) / len(words) if words else 0.0

    # Expected scheme from form
    expected_scheme = None
    matches_form = None
    deviations = []

    if expected_form:
        scheme_str = get_scheme(expected_form, expected_variant)
        if scheme_str:
            expected_scheme = scheme_str
            expected_flat = parse_scheme(scheme_str)
            detected_flat = list(detected_raw)

            # Compare up to the shorter length
            compare_len = min(len(expected_flat), len(detected_flat))
            if compare_len > 0:
                matches_form = True
                # Build expected groupings: which positions should share a label
                expected_groups: dict[str, list[int]] = {}
                for idx, lbl in enumerate(expected_flat[:compare_len]):
                    expected_groups.setdefault(lbl, []).append(idx)

                for lbl, positions in expected_groups.items():
                    if len(positions) < 2:
                        continue
                    anchor = positions[0]
                    for pos in positions[1:]:
                        if pos >= len(words):
                            continue
                        rt = _rhyme_type(words[anchor], words[pos])
                        if rt not in ("perfect", "slant", "identical"):
                            matches_form = False
                            deviations.append({
                                "line": pos + 1,
                                "word": words[pos],
                                "expected_rhyme_with": words[anchor],
                                "expected_label": lbl,
                                "actual_type": rt,
                            })

    result = {
        "detected_scheme": _format_scheme(detected_raw),
        "strict_detected_scheme": _format_scheme(strict_detected_raw),
        "expected_scheme": expected_scheme,
        "matches_form": matches_form,
        "deviations": deviations,
        "rhyme_pairs": rhyme_pairs,
        "strict_rhyme_pairs": strict_rhyme_pairs,
        "rhyme_density": round(density, 2),
        "strict_rhyme_density": round(strict_density, 2),
        "line_count": len(words),
        "end_words": words,
    }
    return result


def format_analysis_for_prompt(analysis: dict) -> str:
    """Format analysis results as a concise string for inclusion in an LLM prompt."""
    parts = []
    if analysis.get("expected_scheme"):
        parts.append(f"Expected scheme: {analysis['expected_scheme']}")
    parts.append(f"Detected scheme: {analysis['detected_scheme']}")
    if analysis.get("matches_form") is not None:
        parts.append(f"Matches form: {'yes' if analysis['matches_form'] else 'no'}")
    if analysis.get("deviations"):
        dev_strs = []
        for d in analysis["deviations"]:
            dev_strs.append(
                f"  Line {d['line']} — \"{d['word']}\" does not rhyme with "
                f"\"{d['expected_rhyme_with']}\" (expected {d['expected_label']} pair)"
            )
        parts.append("Deviations:\n" + "\n".join(dev_strs))
    if analysis.get("rhyme_pairs"):
        top = analysis["rhyme_pairs"][:6]
        pair_strs = [
            f"  \"{p['words'][0]}\" / \"{p['words'][1]}\" ({p['type']}, "
            f"lines {p['lines'][0]}&{p['lines'][1]})"
            for p in top
        ]
        parts.append("Rhyme pairs:\n" + "\n".join(pair_strs))
    parts.append(f"Rhyme density: {analysis['rhyme_density']:.0%}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Reward scoring (for REINFORCE training)
# ---------------------------------------------------------------------------

RHYME_SCORE = {"identical": 0.8, "perfect": 1.0, "slant": 0.6, "assonance": 0.3, "none": 0.0}


def _rhyme_score(word_a: str, word_b: str) -> float:
    """Tiered rhyme score between two words (partial credit).

    perfect 1.0 > identical 0.8 > slant 0.6 > assonance 0.3 > none 0.0
    Identical (same word) scores below perfect — repeating end-words is weak craft.
    """
    rt = _rhyme_type(word_a, word_b)
    if rt != "none":
        return RHYME_SCORE[rt]
    suffix_a = _get_rhyme_suffix(word_a)
    suffix_b = _get_rhyme_suffix(word_b)
    if suffix_a is not None and suffix_b is not None:
        tok_a = suffix_a.split()[0]
        tok_b = suffix_b.split()[0]
        vowel_a = re.sub(r"\d", "", tok_a) if tok_a[-1].isdigit() else None
        vowel_b = re.sub(r"\d", "", tok_b) if tok_b[-1].isdigit() else None
        if vowel_a and vowel_b and vowel_a == vowel_b:
            return RHYME_SCORE["assonance"]
    return RHYME_SCORE["none"]


def compute_reward(
    poem: str,
    expected_form: str | None = None,
    expected_variant: str | None = None,
) -> float:
    """Compute a [0, 1] rhyme reward for REINFORCE training.

    Combines three signals:
    1. Pair quality (50%): How well do rhyme pairs phonetically match?
    2. Scheme adherence (30%): Does the poem follow the expected rhyme pattern?
    3. Rhyme density (20%): What fraction of lines participate in rhymes?

    The scheme adherence component is critical for penalizing poems that rhyme
    but in the wrong pattern (e.g., ABCD when ABAB is expected).

    Returns:
        Float in [0, 1]. Higher = better rhyme compliance.
    """
    analysis = analyze(poem, expected_form=expected_form, expected_variant=expected_variant)
    end_words = analysis.get("end_words", [])
    line_count = analysis.get("line_count", 0)

    if len(end_words) < 2:
        return 0.0

    scheme = analysis.get("expected_scheme")
    if scheme:
        flat = parse_scheme(scheme)
    else:
        detected_raw = analysis.get("strict_detected_scheme") or analysis.get("detected_scheme", "")
        flat = list(detected_raw.replace(" ", ""))

    if not flat or len(flat) < 2:
        return analysis.get("strict_rhyme_density", 0.0)

    compare_len = min(len(flat), len(end_words))
    groups: dict[str, list[int]] = {}
    for idx, lbl in enumerate(flat[:compare_len]):
        groups.setdefault(lbl, []).append(idx)

    pair_scores: list[float] = []
    for positions in groups.values():
        if len(positions) < 2:
            continue
        anchor = positions[0]
        for pos in positions[1:]:
            if anchor < len(end_words) and pos < len(end_words):
                pair_scores.append(_rhyme_score(end_words[anchor], end_words[pos]))

    if not pair_scores:
        return analysis.get("strict_rhyme_density", 0.0)

    # Component 1: Pair quality (how phonetically accurate are the rhymes?)
    pair_mean = sum(pair_scores) / len(pair_scores)

    # Component 2: Scheme adherence (does it follow the expected pattern?)
    # This is critical for penalizing wrong rhyme schemes
    matches_form = analysis.get("matches_form")
    deviations = analysis.get("deviations", [])
    deviations_count = len(deviations)

    if matches_form is True:
        scheme_adherence = 1.0
    elif matches_form is False and line_count > 0:
        # Penalize based on fraction of lines that deviate
        # More deviations = lower score
        scheme_adherence = max(0.0, 1.0 - (deviations_count / line_count))
    else:
        # No expected form provided, assume full adherence
        scheme_adherence = 1.0

    # Component 3: Density (what fraction of lines participate in rhymes?)
    density = analysis.get("strict_rhyme_density", 0.0)

    # Weighted combination: 50% pairs, 30% scheme, 20% density
    return 0.5 * pair_mean + 0.3 * scheme_adherence + 0.2 * density


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze poem rhyme structure")
    parser.add_argument("--poem", type=str, help="Poem text (inline)")
    parser.add_argument("--file", type=Path, help="File containing poem text")
    parser.add_argument("--form", type=str, help="Expected form (e.g. sonnet, villanelle)")
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
        print(json.dumps(result, indent=2))
    else:
        print(format_analysis_for_prompt(result))


if __name__ == "__main__":
    main()
