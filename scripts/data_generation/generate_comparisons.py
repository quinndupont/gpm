#!/usr/bin/env python3
"""T3: Comparative workshop — two poems, same topic."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_GOOD = ROOT / "data" / "raw" / "good"
RAW_BAD = ROOT / "data" / "raw" / "bad"
ANNOTATED = ROOT / "data" / "annotated"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude, get_educator_system_prompt, EDUCATOR_NAME


def load_poems(directory: Path) -> list[dict]:
    poems = []
    for p in directory.glob("**/*.txt"):
        poems.append({"text": p.read_text(), "source": str(p)})
    for p in directory.glob("**/*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                poems.append(json.loads(line))
    return poems


def poem_text(poem: dict) -> str:
    return poem.get("text", poem.get("poem", poem.get("content", "")))


T3_PROMPT = """Here are two poems on a similar topic. Poem A:

---
{poem_a}
---

Poem B:

---
{poem_b}
---

Explain which is stronger and why, in your voice. Be specific about craft choices — line breaks, imagery, sound, structure. Which poem earns its effects? Which reaches for something it doesn't achieve? No scores or rubrics."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ANNOTATED / "comparisons.jsonl")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    args = parser.parse_args()

    system = get_educator_system_prompt()
    good = load_poems(RAW_GOOD)
    bad = load_poems(RAW_BAD)
    pairs = list(zip(good[: len(bad)], bad[: len(good)]))
    if args.limit:
        pairs = pairs[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, (a, b) in enumerate(pairs):
            text_a = poem_text(a)
            text_b = poem_text(b)
            if not text_a.strip() or not text_b.strip():
                continue
            user_msg = T3_PROMPT.format(poem_a=text_a, poem_b=text_b)
            try:
                comparison = call_claude(user_msg, system, model=args.model, max_tokens=1024)
            except Exception as e:
                print(f"Error on pair {i + 1}: {e}", file=sys.stderr)
                comparison = ""
            f.write(json.dumps({"poem_a": a, "poem_b": b, "comparison": comparison}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(pairs)}", file=sys.stderr)

    print(f"Processed {len(pairs)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
