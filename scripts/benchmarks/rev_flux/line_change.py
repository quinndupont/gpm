"""Compute per-line change percentage between consecutive drafts."""
from __future__ import annotations

import math


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


def _change_pct(a: str, b: str) -> float:
    """Character-level edit distance as percentage of max length. 0 = identical, 100 = fully different."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 100.0
    # Levenshtein distance
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[m][n]
    denom = max(m, n, 1)
    return 100.0 * dist / denom


def line_change_percentages(prev_draft: str, next_draft: str) -> list[float]:
    """
    For each line position, compute percentage change from prev to next.
    Aligns by index; if lengths differ, extra lines are 100% change.
    """
    prev_lines = _lines(prev_draft)
    next_lines = _lines(next_draft)
    max_len = max(len(prev_lines), len(next_lines), 1)
    result = []
    for i in range(max_len):
        p = prev_lines[i] if i < len(prev_lines) else ""
        n = next_lines[i] if i < len(next_lines) else ""
        result.append(_change_pct(p, n))
    return result


def revision_round_changes(revision_history: list[dict]) -> list[list[float]]:
    """
    For each revision round, return per-line change percentages vs previous draft.
    Pipeline appends final draft when it differs from last; so we have full sequence.
    Returns list of lists: round 0 = []; round 1+ = change from prev to curr.
    """
    if not revision_history:
        return []
    rounds = []
    for i in range(len(revision_history)):
        if i == 0:
            rounds.append([])
            continue
        prev = revision_history[i - 1]["draft"]
        curr = revision_history[i]["draft"]
        rounds.append(line_change_percentages(prev, curr))
    return rounds


def aggregate_line_changes(rounds: list[list[float]]) -> list[float]:
    """
    Flatten all per-line changes across rounds into one list for histogram.
    """
    out = []
    for r in rounds:
        out.extend(r)
    return out


def revised_lines_per_round(
    rounds: list[list[float]],
    threshold: float = 0.5,
) -> list[list[tuple[int, float]]]:
    """
    Per round: list of (line_idx, change_pct) for lines that changed (pct > threshold).
    """
    result = []
    for r in rounds:
        result.append([(i, p) for i, p in enumerate(r) if p > threshold])
    return result


def lines_changed_per_round(
    rounds: list[list[float]],
    threshold: float = 0.5,
) -> list[int]:
    """Count of lines changed per round (change_pct > threshold)."""
    return [len([p for p in r if p > threshold]) for r in rounds]


def line_stability_indices(
    revision_history: list[dict],
    threshold: float = 5.0,
) -> list[int]:
    """
    For each line in the final draft, count how many revision rounds it stayed
    unchanged (change_pct < threshold). Higher = more stable.
    """
    rounds = revision_round_changes(revision_history)
    if not revision_history:
        return []
    final_lines = _lines(revision_history[-1]["draft"])
    n_lines = len(final_lines)
    stability = [0] * n_lines
    for r in rounds:
        if not r:
            continue
        for i in range(min(len(r), n_lines)):
            if r[i] < threshold:
                stability[i] += 1
    return stability


def stanza_structure(draft: str) -> list[list[str]]:
    """
    Split poem into stanzas by blank lines. Returns list of stanzas, each a list of lines.
    """
    stanzas: list[list[str]] = []
    current: list[str] = []
    for ln in draft.strip().splitlines():
        if ln.strip():
            current.append(ln.strip())
        elif current:
            stanzas.append(current)
            current = []
    if current:
        stanzas.append(current)
    return stanzas


def stanza_change_map(
    revision_history: list[dict],
) -> tuple[list[list[str]], list[float]]:
    """
    Returns (stanzas, mean_change_per_stanza). Stanzas from final draft.
    Mean change is average of per-line change % for lines in that stanza,
    using the last revision round (prev draft -> final).
    """
    rounds = revision_round_changes(revision_history)
    if not revision_history:
        return [], []
    stanzas = stanza_structure(revision_history[-1]["draft"])
    # Use last round with data: change from prev draft to final
    last_round = next((r for r in reversed(rounds) if r), [])
    if not last_round:
        return stanzas, [0.0] * len(stanzas)
    pos = 0
    stanza_means = []
    for s in stanzas:
        n = len(s)
        if n > 0 and pos + n <= len(last_round):
            stanza_means.append(sum(last_round[pos : pos + n]) / n)
        else:
            stanza_means.append(sum(last_round[pos:]) / max(len(last_round[pos:]), 1) if pos < len(last_round) else 0.0)
        pos += n
    return stanzas, stanza_means


def positional_change_profile(change_pcts: list[float], n_bins: int = 10) -> list[float]:
    """
    Bin lines into n_bins equal positional buckets (0.0-0.1, 0.1-0.2, ..., 0.9-1.0),
    return mean change% per bucket. Normalizes so poems of different lengths are comparable.
    """
    if not change_pcts:
        return [0.0] * n_bins
    n = len(change_pcts)
    buckets: list[list[float]] = [[] for _ in range(n_bins)]
    for i, pct in enumerate(change_pcts):
        bin_idx = min(int(i / n * n_bins), n_bins - 1)
        buckets[bin_idx].append(pct)
    return [sum(b) / len(b) if b else 0.0 for b in buckets]


def revision_coverage(change_pcts: list[float], threshold: float = 0.5) -> float:
    """Fraction of lines with change > threshold. Range [0, 1]."""
    if not change_pcts:
        return 0.0
    changed = sum(1 for p in change_pcts if p > threshold)
    return changed / len(change_pcts)


def head_preservation(
    change_pcts: list[float],
    fraction: float = 0.2,
    threshold: float = 5.0,
) -> float:
    """Fraction of lines in the first fraction of the poem that stayed below threshold. High = preserves opening."""
    if not change_pcts:
        return 1.0
    n = len(change_pcts)
    head_count = max(1, int(n * fraction))
    head_lines = change_pcts[:head_count]
    preserved = sum(1 for p in head_lines if p < threshold)
    return preserved / len(head_lines)


def tail_attention(
    change_pcts: list[float],
    fraction: float = 0.33,
) -> float:
    """Mean change% in the last fraction of lines. High = model reaches deep into the poem."""
    if not change_pcts:
        return 0.0
    n = len(change_pcts)
    tail_count = max(1, int(n * fraction))
    tail_lines = change_pcts[-tail_count:]
    return sum(tail_lines) / len(tail_lines)


def structural_growth(revision_history: list[dict]) -> float:
    """Ratio len(final_lines) / len(initial_lines). >1 = poem grew; <1 = shrank."""
    if not revision_history or len(revision_history) < 2:
        return 1.0
    initial = len(_lines(revision_history[0]["draft"]))
    final = len(_lines(revision_history[-1]["draft"]))
    if initial == 0:
        return 1.0
    return final / initial


def change_entropy(change_pcts: list[float], n_bins: int = 5) -> float:
    """
    Shannon entropy of binned change distribution (bins: 0-20, 20-40, 40-60, 60-80, 80-100).
    Normalized to [0, 1] by dividing by log2(n_bins). High = diverse edits; low = concentrated.
    """
    if not change_pcts:
        return 0.0
    bin_edges = [0, 20, 40, 60, 80, 100]
    counts = [0] * n_bins
    for p in change_pcts:
        p = max(0, min(100, p))
        for b in range(n_bins):
            if b == n_bins - 1:
                if p >= bin_edges[b]:
                    counts[b] += 1
                    break
            elif bin_edges[b] <= p < bin_edges[b + 1]:
                counts[b] += 1
                break
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    max_entropy = math.log2(n_bins)
    return entropy / max_entropy
