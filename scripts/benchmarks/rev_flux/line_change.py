"""Compute per-line change percentage between consecutive drafts."""
from __future__ import annotations


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
