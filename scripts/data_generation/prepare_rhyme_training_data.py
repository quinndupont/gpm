#!/usr/bin/env python3
"""Prepare rhyme-focused training data: 80% strong_rhyme_poems + 20% general poet data (anti-collapse)."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
POET_TRAINING = ROOT / "data" / "poet_training"
RHYME_TRAINING = ROOT / "data" / "rhyme_training"

POET_SYSTEM = """You are a poet. You receive generation briefs and write poems.
You never output instructions, critique, or analysis — only poems."""

POET_USER_SUFFIX = "\n\nWrite the poem. Output ONLY the poem — no title unless it's part of the poem, no commentary."
POET_RHYME_SUFFIX = "\n\nWrite the poem. Output ONLY the poem — no title unless it's part of the poem, no commentary.\nFollow the specified form and rhyme scheme precisely."


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


def strong_rhyme_to_example(rec: dict) -> dict | None:
    poem = rec.get("poem", "").strip()
    if not poem:
        return None
    ra = rec.get("rhyme_analysis", {})
    scheme = ra.get("strict_detected_scheme") or ra.get("detected_scheme", "")
    author = rec.get("author", "")
    title = rec.get("title", "")
    parts = ["Write a poem with strong end rhyme."]
    if scheme:
        parts.append(f"Rhyme scheme: {scheme}.")
    if title:
        parts.append(f"Title: {title}.")
    if author:
        parts.append(f"Style: {author}.")
    parts.append("Follow the form precisely.")
    brief = " ".join(parts) + POET_RHYME_SUFFIX
    return to_poet_example(brief, poem)


def main():
    parser = argparse.ArgumentParser(description="Prepare rhyme training data (80% strong rhyme + 20% general)")
    parser.add_argument("--strong-rhyme", type=Path, default=ANNOTATED / "strong_rhyme_poems.jsonl")
    parser.add_argument("--general", type=Path, default=POET_TRAINING / "train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RHYME_TRAINING)
    parser.add_argument("--general-frac", type=float, default=0.2, help="Fraction of general poetry (anti-collapse)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    strong = []
    for rec in load_jsonl(args.strong_rhyme):
        if not rec.get("strong_rhyme", True):
            continue
        ex = strong_rhyme_to_example(rec)
        if ex:
            strong.append(ex)

    general_raw = load_jsonl(args.general)
    general = [e for e in general_raw if e.get("messages") and len(e["messages"]) >= 3]

    if not strong:
        raise SystemExit(f"No strong rhyme examples from {args.strong_rhyme}")

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
    print(f"  Strong rhyme: {len(strong)}, General (anti-collapse): {n_general}")


if __name__ == "__main__":
    main()
