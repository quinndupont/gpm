#!/usr/bin/env python3
"""Enumerate data generation outputs for each phase (good/bad, contrastive)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"


@dataclass
class PhaseCount:
    name: str
    path: Path
    total: int
    good: int | None = None
    bad: int | None = None
    contrastive: bool = False
    contrastive_note: str | None = None  # e.g. "good vs bad" or "bad poems"


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _count_critiques_by_source(path: Path) -> tuple[int, int, int]:
    """Returns (total, good, bad)."""
    if not path.exists():
        return 0, 0, 0
    good, bad = 0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            src = obj.get("source", "")
            if src == "good":
                good += 1
            elif src == "bad":
                bad += 1
        except json.JSONDecodeError:
            pass
    return good + bad, good, bad


def get_data_generation_status() -> list[PhaseCount]:
    """Return counts for each data generation phase."""
    phases: list[PhaseCount] = []

    # Phase 1: Hard tasks (Opus)
    p = ANNOTATED / "critiques_seed.jsonl"
    total, good, bad = _count_critiques_by_source(p)
    phases.append(PhaseCount(
        name="critiques_seed",
        path=p,
        total=total,
        good=good,
        bad=bad,
        contrastive=True,
        contrastive_note="good + bad poems",
    ))

    p = ANNOTATED / "comparisons.jsonl"
    n = _count_jsonl(p)
    phases.append(PhaseCount(
        name="comparisons",
        path=p,
        total=n,
        good=None,
        bad=None,
        contrastive=True,
        contrastive_note="good vs bad pairs",
    ))

    p = EDUCATOR_TRAINING / "revision_briefs_seed.jsonl"
    phases.append(PhaseCount(
        name="revision_briefs",
        path=p,
        total=_count_jsonl(p),
    ))

    # Phase 2: Local educator (briefs, autopsies, lessons)
    p = EDUCATOR_TRAINING / "briefs.jsonl"
    phases.append(PhaseCount(name="briefs", path=p, total=_count_jsonl(p)))

    p = ANNOTATED / "autopsies.jsonl"
    phases.append(PhaseCount(
        name="autopsies",
        path=p,
        total=_count_jsonl(p),
        contrastive=True,
        contrastive_note="bad poems",
    ))

    p = EDUCATOR_TRAINING / "lessons.jsonl"
    phases.append(PhaseCount(name="lessons", path=p, total=_count_jsonl(p)))

    p = EDUCATOR_TRAINING / "dialogues.jsonl"
    phases.append(PhaseCount(name="dialogues", path=p, total=_count_jsonl(p)))

    # Phase 3: Poet
    p = POET_TRAINING / "pairs.jsonl"
    phases.append(PhaseCount(name="poet_pairs", path=p, total=_count_jsonl(p)))

    # Rhyme (separate pipeline)
    p = ANNOTATED / "strong_rhyme_poems.jsonl"
    phases.append(PhaseCount(name="strong_rhyme_poems", path=p, total=_count_jsonl(p)))

    return phases


def format_status(phases: list[PhaseCount]) -> str:
    """Format phase counts for display."""
    lines = ["Generated prompts by phase:"]
    for p in phases:
        if p.good is not None and p.bad is not None:
            detail = f" ({p.good} good, {p.bad} bad, contrastive)"
        elif p.contrastive and p.contrastive_note:
            detail = f" (contrastive: {p.contrastive_note})"
        elif p.contrastive:
            detail = " (contrastive)"
        else:
            detail = ""
        lines.append(f"  {p.name}: {p.total}{detail}")
    return "\n".join(lines)
