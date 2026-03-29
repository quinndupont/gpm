#!/usr/bin/env python3
"""Deterministic rhyme analysis using CMU Pronouncing Dictionary.

Detects rhyme schemes, validates against expected forms, classifies rhyme quality.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pronouncing

from scripts.eval.form_registry import detect_form, get_scheme, parse_scheme

# Single-coda slant pairs (CMU tokens): same nucleus, one consonant differs by a
# common near-rhyme relation (e.g. M/N). Not all different codas qualify (e.g. P/Z).
_SLANT_SINGLE_CODA_PAIRS: frozenset[frozenset[str]] = frozenset(
    frozenset(p) for p in (
        ("M", "N"),
        ("T", "D"),
        ("K", "G"),
        ("P", "B"),
        ("F", "V"),
        ("S", "Z"),
        ("CH", "JH"),
        ("SH", "ZH"),
        ("DH", "TH"),
    )
)

# Tags whose bodies are model chain-of-thought, not verse (closed and unclosed).
_REASONING_TAGS = ("reasoning", "thinking", "analysis", "think")


def strip_reasoning_blocks(text: str) -> str:
    """Remove model CoT wrappers so rhyme analysis runs only on poem text.

    Strips ``<reasoning>…</reasoning>``, `` <think>…</think>``, ``<thinking>…``, etc.
    Unclosed opening tags remove the remainder of the string (no poem after tag).
    """
    if not text:
        return text
    t = text
    # Tolerate opening <reasoning> closed with think end-tag instead of </reasoning>.
    t = re.sub(
        r"(?i)<reasoning>.*?(?:</reasoning>|</think>)",
        "",
        t,
        flags=re.DOTALL,
    )
    for tag in _REASONING_TAGS:
        t = re.sub(rf"(?i)<{tag}>.*?</{tag}>", "", t, flags=re.DOTALL)
    for tag in _REASONING_TAGS:
        t = re.sub(rf"(?i)<{tag}>.*", "", t, flags=re.DOTALL)
    t = re.sub(r"(?i)^thinking:\s*.*?(?=\n\n|\Z)", "", t, flags=re.DOTALL | re.MULTILINE)
    t = re.sub(r"\n\n\n+", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------------------
# Phoneme helpers
# ---------------------------------------------------------------------------

def _clean_word(word: str) -> str:
    """Strip surrounding punctuation and lowercase.

    Uses Unicode-aware trimming so em dashes, curly quotes, and other marks
    not in ``string.punctuation`` (e.g. ``\\u2014``) do not break CMU lookup.
    """
    w = word.strip().lower()
    if not w:
        return ""
    while w and not w[0].isalpha():
        w = w[1:]
    while w and not w[-1].isalpha():
        w = w[:-1]
    return w


def _get_end_word(line: str) -> str:
    """Extract the last word from a line, handling trailing punctuation.

    Trailing stand-alone dashes or quotes (e.g. ``to be —``) become their own
    tokens after split; skip backward until a token with alphabetic content.
    """
    tokens = line.strip().split()
    if not tokens:
        return ""
    for tok in reversed(tokens):
        w = _clean_word(tok)
        if w:
            return w
    return ""


def _stress_normalize_suffix_tokens(suffix: str) -> str:
    """ARPAbet tokens with stress digits stripped, for rhyme equality.

    CMU assigns different stress to the same nucleus across contexts (e.g. *grey*
    ``EY1`` vs final syllable of *yesterday* ``EY2``). Those are still perfect
    rhymes; comparing raw strings misses them.
    """
    parts = suffix.split()
    out: list[str] = []
    for p in parts:
        if p and p[-1].isdigit():
            out.append(p[:-1])
        else:
            out.append(p)
    return " ".join(out)


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


def _get_rhyme_suffixes(word: str) -> list[str]:
    """Rhyme suffixes for all CMU pronunciations (deduped, order preserved).

    Words like *on* have multiple entries (e.g. ``AA1 N`` vs ``AO1 N``); *gone*
    may only match the second, so rhyme checks must consider every pair.
    """
    w = _clean_word(word)
    if not w:
        return []
    phones_list = pronouncing.phones_for_word(w)
    if not phones_list:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for phones in phones_list:
        suf = _rhyme_suffix(phones)
        if suf not in seen:
            seen.add(suf)
            out.append(suf)
    return out


def _get_rhyme_suffix(word: str) -> str | None:
    """First CMU pronunciation's rhyme suffix, or None if unknown."""
    suf = _get_rhyme_suffixes(word)
    return suf[0] if suf else None


def _cmu_slant_suffix_pair(suffix_a: str, suffix_b: str) -> bool:
    """True if two rhyme-suffix strings form a slant rhyme (CMU rules)."""
    parts_a = suffix_a.split()
    parts_b = suffix_b.split()
    vowel_a = re.sub(r"\d", "", parts_a[0]) if parts_a[0][-1].isdigit() else None
    vowel_b = re.sub(r"\d", "", parts_b[0]) if parts_b[0][-1].isdigit() else None
    if not (vowel_a and vowel_b and vowel_a == vowel_b):
        return False
    tail_a = parts_a[1:]
    tail_b = parts_b[1:]
    if not tail_a or not tail_b:
        return False
    set_a = set(tail_a)
    set_b = set(tail_b)
    if set_a & set_b:
        return True
    if len(tail_a) == 1 and len(tail_b) == 1 and tail_a[0] != tail_b[0]:
        if frozenset((tail_a[0], tail_b[0])) in _SLANT_SINGLE_CODA_PAIRS:
            return True
    return False


def _suffix_fallback(word: str) -> str:
    """Fallback for words not in CMU dict: use last 2-3 chars as pseudo-suffix."""
    w = word.lower()
    return w[-3:] if len(w) >= 3 else w


def _rhyme_type(word_a: str, word_b: str, strict: bool = False) -> str:
    """Classify rhyme between two words: 'perfect', 'slant', or 'none'.

    Perfect: identical rhyme suffix for some CMU pronunciation pair.
    Slant (CMU path): same nucleus vowel and either overlapping post-vowel
    consonants, or a single-coda minimal pair (e.g. M/N as in "time"/"mine").
    Pure assonance (same vowel, unrelated codas) is not slant.
    strict=True: only CMU-verified perfect/identical count; no fallback, no slant.
    """
    wa = _clean_word(word_a)
    wb = _clean_word(word_b)
    if not wa or not wb:
        return "none"
    if wa == wb:
        return "identical"

    suffixes_a = _get_rhyme_suffixes(wa)
    suffixes_b = _get_rhyme_suffixes(wb)

    # Both in CMU dict — compare all pronunciation pairs
    if suffixes_a and suffixes_b:
        for sa in suffixes_a:
            for sb in suffixes_b:
                if sa == sb or _stress_normalize_suffix_tokens(sa) == _stress_normalize_suffix_tokens(sb):
                    return "perfect"
        if strict:
            return "none"
        for sa in suffixes_a:
            for sb in suffixes_b:
                if _cmu_slant_suffix_pair(sa, sb):
                    return "slant"

        return "none"

    # Fallback: simple suffix matching for words not in CMU dict.
    # In strict mode, never count fallback — orthographic matches are unreliable.
    if strict:
        return "none"

    # Common suffixes that DON'T indicate rhyme (grammatical endings)
    FALSE_RHYME_SUFFIXES = {"ed", "ing", "ly", "er", "est", "ness", "ment", "tion", "sion"}

    fa = _suffix_fallback(wa)
    fb = _suffix_fallback(wb)

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

def _extract_lines(poem: str) -> tuple[list[str], list[int]]:
    """Extract non-empty, non-title lines and stanza start indices.

    Stanzas are separated by one or more blank lines (``\\n`` with optional
    whitespace only between paragraph breaks). ``stanza_starts[k]`` is the
    index in the returned ``lines`` list where stanza ``k`` begins; the first
    stanza always starts at 0 when any lines exist.
    """
    lines: list[str] = []
    stanza_starts: list[int] = []
    # Paragraphs: robust blank-line separator (not a single \\n between lines)
    blocks = re.split(r"\n\s*\n+", poem)
    for block in blocks:
        block_lines: list[str] = []
        for raw in block.split("\n"):
            stripped = raw.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if all(c in "-—–=* " for c in stripped):
                continue
            block_lines.append(stripped)
        if not block_lines:
            continue
        stanza_starts.append(len(lines))
        lines.extend(block_lines)
    if lines and not stanza_starts:
        stanza_starts = [0]
    elif lines and stanza_starts[0] != 0:
        stanza_starts.insert(0, 0)
    return lines, stanza_starts


def _compress_stanza_starts_for_valid_words(
    lines: list[str],
    stanza_starts: list[int],
) -> list[int]:
    """Map stanza boundaries from full ``lines`` to indices in the list of non-empty end-words."""
    if not lines:
        return [0]
    start_set = set(stanza_starts)
    compressed: list[int] = []
    out_idx = 0
    for i, line in enumerate(lines):
        w = _get_end_word(line)
        if not w:
            continue
        if i in start_set:
            compressed.append(out_idx)
        out_idx += 1
    if not compressed:
        return [0]
    if compressed[0] != 0:
        compressed.insert(0, 0)
    return sorted(set(compressed))


# ---------------------------------------------------------------------------
# Scheme detection
# ---------------------------------------------------------------------------

def _stanza_index_for_line(line_idx: int, stanza_starts: list[int]) -> int:
    """Which stanza (0-based) contains ``line_idx``."""
    for k in range(len(stanza_starts) - 1, -1, -1):
        if line_idx >= stanza_starts[k]:
            return k
    return 0


def _detect_scheme(
    end_words: list[str],
    stanza_starts: list[int] | None = None,
    strict: bool = False,
) -> str:
    """Detect the rhyme scheme from a list of end-words. Returns e.g. 'ABABCDCDEFEFGG'.

    Scans **backwards** so the nearest rhyming line wins. Within a stanza, perfect,
    slant, and identical matches count; across stanzas, only perfect and identical
    (no slant) so quatrains keep independent rhyme letters unless strongly linked.

    strict=True: only CMU-verified perfect/identical rhymes count everywhere.
    """
    if not end_words:
        return ""
    starts = stanza_starts if stanza_starts else [0]
    labels: list[str] = []
    next_label = 0

    for i, word in enumerate(end_words):
        matched_label = None
        for j in range(i - 1, -1, -1):
            same_stanza = _stanza_index_for_line(i, starts) == _stanza_index_for_line(j, starts)
            rt = _rhyme_type(word, end_words[j], strict=strict)
            if strict:
                ok = ("perfect", "identical")
            elif same_stanza:
                ok = ("perfect", "slant", "identical")
            else:
                ok = ("perfect", "identical")
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


def _format_scheme(scheme: str, stanza_starts: list[int] | None = None) -> str:
    """Group scheme by detected stanzas; single stanza stays flat (matches registry).

    A single block of verse (no ``\\n\\n``) used to be split every four letters for
    display (e.g. ``ABABABCC`` → ``ABAB ABCC``), which looked like a form mismatch
    next to ``ABABABCC`` in :data:`~scripts.eval.form_registry.FORMS` even when
    :func:`analyze` already set ``matches_form`` True.
    """
    if not scheme:
        return ""
    n = len(scheme)
    if stanza_starts and len(stanza_starts) > 1:
        groups: list[str] = []
        for k, start in enumerate(stanza_starts):
            if start >= n:
                break
            end = stanza_starts[k + 1] if k + 1 < len(stanza_starts) else n
            end = min(end, n)
            if start < end:
                groups.append(scheme[start:end])
        if groups:
            return " ".join(groups)
    return scheme


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
    poem = strip_reasoning_blocks(poem)
    lines, stanza_starts_lines = _extract_lines(poem)
    scheme_stanza_starts = _compress_stanza_starts_for_valid_words(lines, stanza_starts_lines)
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
    detected_raw = _detect_scheme(words, stanza_starts=scheme_stanza_starts)
    strict_detected_raw = _detect_scheme(words, stanza_starts=scheme_stanza_starts, strict=True)

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
        "detected_scheme": _format_scheme(detected_raw, stanza_starts=scheme_stanza_starts),
        "strict_detected_scheme": _format_scheme(
            strict_detected_raw, stanza_starts=scheme_stanza_starts
        ),
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
    for sa in _get_rhyme_suffixes(word_a):
        for sb in _get_rhyme_suffixes(word_b):
            tok_a = sa.split()[0]
            tok_b = sb.split()[0]
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
