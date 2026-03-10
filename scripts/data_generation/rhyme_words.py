#!/usr/bin/env python3
"""Select rhyme word sets from CMU pronouncing dictionary for constrained generation.

Groups words by rhyme suffix, filters to poetic vocabulary, and samples
end-words that satisfy a given rhyme scheme.
"""
import random
from collections import defaultdict

import pronouncing

from scripts.eval.form_registry import parse_scheme
from scripts.eval.rhyme_analyzer import _rhyme_suffix

# Min word length, max length, max syllables for end-words (prefer common vocabulary)
MIN_WORD_LEN = 3
MAX_WORD_LEN = 8
MAX_SYLLABLES = 3

# Common poetic end-words: (seed, preferred_rhyme) for high-quality pairs
COMMON_PAIRS: list[tuple[str, str]] = [
    ("night", "light"), ("day", "way"), ("love", "above"), ("heart", "part"),
    ("time", "rhyme"), ("dream", "stream"), ("sea", "free"), ("mind", "find"),
    ("face", "place"), ("hand", "land"), ("eyes", "rise"), ("soul", "whole"),
    ("life", "strife"), ("death", "breath"), ("song", "long"), ("still", "will"),
    ("fall", "all"), ("end", "friend"), ("fire", "desire"), ("rain", "pain"),
    ("dark", "mark"), ("word", "heard"), ("gone", "upon"),
]
COMMON_SEEDS = [p[0] for p in COMMON_PAIRS]


def _is_poetic_word(word: str, phones: str | None = None) -> bool:
    """Filter to words suitable for poetic end-rhymes."""
    w = word.lower().strip()
    if len(w) < MIN_WORD_LEN or len(w) > MAX_WORD_LEN:
        return False
    if "'" in w or any(c.isdigit() for c in w):
        return False
    if phones is None:
        phones_list = pronouncing.phones_for_word(w)
        if not phones_list:
            return False
        phones = phones_list[0]
    try:
        syl = pronouncing.syllable_count(phones)
        if syl < 1 or syl > MAX_SYLLABLES:
            return False
    except Exception:
        return False
    return True


def _build_suffix_index() -> dict[str, list[str]]:
    """Build rhyme-suffix -> words index from CMU dict."""
    pronouncing.phones_for_word("the")  # Trigger CMU dict load
    suffix_to_words: dict[str, list[str]] = defaultdict(list)
    for word, phones in pronouncing.pronunciations:
        if not _is_poetic_word(word, phones):
            continue
        suffix = _rhyme_suffix(phones)
        suffix_to_words[suffix].append(word.lower())
    # Dedupe and keep as list for sampling
    return {s: list(dict.fromkeys(w)) for s, w in suffix_to_words.items()}


_SUFFIX_INDEX: dict[str, list[str]] | None = None


def _get_suffix_index() -> dict[str, list[str]]:
    """Lazy-loaded suffix index."""
    global _SUFFIX_INDEX
    if _SUFFIX_INDEX is None:
        _SUFFIX_INDEX = _build_suffix_index()
    return _SUFFIX_INDEX


def _get_common_rhyme_groups(rng: random.Random) -> list[tuple[str, list[str]]]:
    """Build rhyme groups from curated common pairs. Returns [(suffix, words), ...]."""
    pronouncing.phones_for_word("the")  # Trigger load
    groups: list[tuple[str, list[str]]] = []
    seen_suffixes: set[str] = set()
    for seed, preferred in COMMON_PAIRS:
        if not _is_poetic_word(seed) or not _is_poetic_word(preferred):
            continue
        if seed == preferred:
            continue
        suffix = _get_rhyme_suffix(seed)
        if not suffix or suffix in seen_suffixes:
            continue
        if _get_rhyme_suffix(preferred) != suffix:
            continue
        seen_suffixes.add(suffix)
        rhymes = pronouncing.rhymes(seed)
        rest = [w for w in rhymes if _is_poetic_word(w) and w not in (seed, preferred)]
        rest.sort(key=lambda x: (len(x), x))
        groups.append((suffix, [seed, preferred] + rest[:10]))
    return groups


def _get_rhyme_suffix(word: str) -> str | None:
    """Get rhyme suffix for a word. Returns None if not in CMU dict."""
    phones_list = pronouncing.phones_for_word(word)
    if not phones_list:
        return None
    return _rhyme_suffix(phones_list[0])


def pick_endwords(
    scheme_str: str,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Pick end-words for each rhyme group in the scheme.

    Args:
        scheme_str: Scheme string like "ABAB CDCD EFEF GG"
        seed: Random seed for reproducibility

    Returns:
        Mapping of scheme label to ordered list of words (one per line with that label).
        E.g. {"A": ["night", "light"], "B": ["sea", "free"], ...}
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    scheme_flat = parse_scheme(scheme_str)
    if not scheme_flat:
        return {}

    # Prefer common rhyme groups from seed words (no shuffle - use common first)
    common = _get_common_rhyme_groups(rng)
    index = _get_suffix_index()
    fallback = [(s, w) for s, w in index.items() if len(w) >= 2]
    candidates = common + fallback

    # Count words needed per label
    label_counts: dict[str, int] = {}
    for lbl in scheme_flat:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    used_suffixes: set[str] = set()
    result: dict[str, list[str]] = {}

    for lbl in sorted(label_counts.keys()):
        need = label_counts[lbl]
        available = [
            (suffix, words)
            for suffix, words in candidates
            if suffix not in used_suffixes and len(words) >= need
        ]
        if not available:
            available = [(s, w) for s, w in candidates if len(w) >= need]
        if not available:
            return {}

        common_suffixes = {c[0] for c in common}
        common_avail = [x for x in available if x[0] in common_suffixes]
        pool = common_avail if common_avail else available
        suffix, words = rng.choice(pool)
        used_suffixes.add(suffix)
        if need == 2 and suffix in common_suffixes:
            chosen = list(words[:2])
        elif need <= len(words):
            chosen = rng.sample(words, need)
        else:
            chosen = words[:need]
        result[lbl] = chosen

    return result


def format_endword_constraint(
    scheme_str: str,
    endwords: dict[str, list[str]],
) -> str:
    """Format end-word constraints as a line-by-line instruction block."""
    scheme_flat = parse_scheme(scheme_str)
    if not scheme_flat:
        return ""

    counters: dict[str, int] = {}
    lines: list[str] = []
    for i, lbl in enumerate(scheme_flat):
        idx = counters.get(lbl, 0)
        if lbl not in endwords or idx >= len(endwords[lbl]):
            continue
        word = endwords[lbl][idx]
        lines.append(f"Line {i + 1} must end with '{word}'")
        counters[lbl] = idx + 1

    return "\n".join(lines)
