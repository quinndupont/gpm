#!/usr/bin/env python3
"""Prepare rhyme-focused training data: Claude rhyme_pairs (or strong_rhyme_poems) + 20% general (anti-collapse).

Quality gate: poems are re-validated through the deterministic rhyme analyzer.
Only poems with strict_rhyme_density >= 0.6 are included as positive examples.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
POET_TRAINING = ROOT / "data" / "poet_training"
RHYME_TRAINING = ROOT / "data" / "rhyme_training"

sys.path.insert(0, str(ROOT))
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme
from scripts.eval.form_registry import get_scheme

POET_SYSTEM = """You are a poet. You receive generation briefs and write poems.
You never output instructions, critique, or analysis — only poems.
When a rhyme scheme is specified, every end-word pair must be a true phonetic rhyme."""

POET_USER_SUFFIX = "\n\nWrite the poem. Output ONLY the poem — no title unless it's part of the poem, no commentary."
POET_RHYME_SUFFIX = "\n\nWrite the poem. Output ONLY the poem — no title unless it's part of the poem, no commentary.\nFollow the specified form and rhyme scheme precisely. Plan your end-words before writing each line."

MIN_STRICT_DENSITY = 0.6


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def to_poet_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": POET_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def poem_to_rhyme_example(poem: str, form: str | None = None, brief: str | None = None) -> dict | None:
    """Convert a poem to a rhyme training example if it passes the quality gate."""
    if not poem.strip():
        return None
    analysis = analyze_rhyme(poem, expected_form=form)
    strict_density = analysis.get("strict_rhyme_density", 0)
    if strict_density < MIN_STRICT_DENSITY:
        return None
    scheme = analysis.get("strict_detected_scheme") or analysis.get("detected_scheme", "")
    if brief and brief.strip():
        user = brief.strip()
        if not user.endswith(POET_RHYME_SUFFIX.strip()):
            user = user + "\n\n" + POET_RHYME_SUFFIX.strip()
    else:
        parts = ["Write a poem with strong end rhyme."]
        if scheme:
            parts.append(f"Rhyme scheme: {scheme}.")
        if form:
            scheme_str = get_scheme(form) if form else ""
            if scheme_str:
                parts.append(f"Form: {form}, scheme {scheme_str}.")
        parts.append("Every end-word pair must be a true phonetic rhyme. Follow the form precisely.")
        user = " ".join(parts) + POET_RHYME_SUFFIX
    return to_poet_example(user, poem)


def main():
    parser = argparse.ArgumentParser(description="Prepare rhyme training data (rhyme pairs + 20% general)")
    parser.add_argument("--strong-rhyme", type=Path, default=ANNOTATED / "strong_rhyme_poems.jsonl")
    parser.add_argument("--rhyme-pairs", type=Path, default=POET_TRAINING / "rhyme_pairs.jsonl")
    parser.add_argument("--general", type=Path, default=POET_TRAINING / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RHYME_TRAINING)
    parser.add_argument("--general-frac", type=float, default=0.2, help="Fraction of general poetry (anti-collapse)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    strong = []
    raw_count = 0

    # Source 1: strong_rhyme_poems (curated)
    for rec in load_jsonl(args.strong_rhyme):
        if not rec.get("strong_rhyme", True):
            continue
        raw_count += 1
        ex = poem_to_rhyme_example(rec.get("poem", ""), brief=None)
        if ex:
            strong.append(ex)

    # Source 2: rhyme_pairs (Claude-generated) when strong_rhyme is empty
    if not strong:
        for rec in load_jsonl(args.rhyme_pairs):
            raw_count += 1
            ex = poem_to_rhyme_example(
                rec.get("poem", ""),
                form=rec.get("form"),
                brief=rec.get("brief"),
            )
            if ex:
                strong.append(ex)
        if strong:
            print(f"Using rhyme_pairs: {len(strong)}/{raw_count} passed (strict_density >= {MIN_STRICT_DENSITY})")
    else:
        filtered = raw_count - len(strong)
        print(f"Quality gate: {len(strong)}/{raw_count} poems passed (strict_density >= {MIN_STRICT_DENSITY}), {filtered} filtered out")

    general_raw = load_jsonl(args.general)
    general = [e for e in general_raw if e.get("messages") and len(e["messages"]) >= 3]

    if not strong:
        raise SystemExit(
            f"No rhyme examples. Run generate_rhyme_pairs.py to create {args.rhyme_pairs}, "
            f"or provide {args.strong_rhyme}."
        )

    n_general = max(1, int(len(strong) * args.general_frac / (1 - args.general_frac)))
    n_general = min(n_general, len(general)) if general else 0
    general_sample = random.sample(general, n_general) if general else []

    combined = strong + general_sample
    random.shuffle(combined)

    n_val = max(1, len(combined) // 10)
    train_data = combined[:-n_val]
    valid_data = combined[-n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(args.output_dir / "valid.jsonl", "w") as f:
        for ex in valid_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Rhyme training: {len(train_data)} train, {len(valid_data)} valid")
    print(f"  Rhyme: {len(strong)}, General (anti-collapse): {n_general}")


if __name__ == "__main__":
    main()
