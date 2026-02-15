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
    revision_history[i]["draft"] = draft after round i (initial draft at i=0).
    Returns list of lists: one list per round (round 0 = initial, no change; round 1+ = vs previous).
    """
    if not revision_history:
        return []
    rounds = []
    for i in range(len(revision_history)):
        if i == 0:
            # Initial draft: no prior to compare
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
