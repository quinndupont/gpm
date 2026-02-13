#!/usr/bin/env python3
"""T1: Workshop critiques — Claude API batch generation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_GOOD = ROOT / "data" / "raw" / "good"
RAW_BAD = ROOT / "data" / "raw" / "bad"
ANNOTATED = ROOT / "data" / "annotated"
PERSONA = ROOT / "persona"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude, get_educator_system_prompt, EDUCATOR_NAME


def load_poems(directory: Path) -> list[dict]:
    """Load poems from .txt or .jsonl files."""
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


T1_PROMPT = """A student has brought this poem to workshop:

---
{poem_text}
---

Give your honest workshop response. Remember:
- Start with what's alive in this poem — the specific moment where the poet's actual attention is on the page
- Then address what isn't working yet, naming the specific type of failure
- Offer at least one concrete direction — not a rewrite, but a question or suggestion that could unlock the next draft
- If the poem is genuinely bad, say so with compassion but without lying
- If the poem is genuinely good, let your enthusiasm show

Respond as {educator_name}. No scores, no rubrics, no bullet points. This is a conversation about a poem."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max poems to process")
    parser.add_argument("--output", type=Path, default=ANNOTATED / "critiques.jsonl")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    args = parser.parse_args()

    system = get_educator_system_prompt()
    poems = load_poems(RAW_GOOD) + load_poems(RAW_BAD)
    if args.limit:
        poems = poems[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, poem in enumerate(poems):
            text = poem_text(poem)
            if not text.strip():
                continue
            user_msg = T1_PROMPT.format(poem_text=text, educator_name=EDUCATOR_NAME)
            try:
                critique = call_claude(user_msg, system, model=args.model, max_tokens=1024)
            except Exception as e:
                print(f"Error on poem {i + 1}: {e}", file=sys.stderr)
                critique = ""
            f.write(json.dumps({"poem": poem, "critique": critique}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(poems)}", file=sys.stderr)

    print(f"Processed {len(poems)} poems -> {args.output}")


if __name__ == "__main__":
    main()
