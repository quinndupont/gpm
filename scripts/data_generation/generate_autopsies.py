#!/usr/bin/env python3
"""T4: Cliché autopsy — Claude dissects bad poems."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_BAD = ROOT / "data" / "raw" / "bad"
ANNOTATED = ROOT / "data" / "annotated"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude, get_educator_system_prompt


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


T4_PROMPT = """Here is a poem from an amateur poet:

---
{bad_poem_text}
---

Perform a "cliché autopsy" — for each clichéd element, explain:
1. WHAT the cliché is (quote it)
2. WHY it became a cliché (what was it before it died?)
3. WHAT the poet was probably reaching for
4. WHAT they could do instead (a direction, not a rewrite)

You're not just saying "this is bad." You're teaching the student to see the living impulse buried under dead language. Every cliché was once a fresh observation. Help them find their way back to the original seeing.

Be honest but not cruel. This poet is trying."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ANNOTATED / "autopsies.jsonl")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    args = parser.parse_args()

    system = get_educator_system_prompt()
    poems = load_poems(RAW_BAD)
    if args.limit:
        poems = poems[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, poem in enumerate(poems):
            text = poem_text(poem)
            if not text.strip():
                continue
            user_msg = T4_PROMPT.format(bad_poem_text=text)
            try:
                autopsy = call_claude(user_msg, system, model=args.model, max_tokens=1024)
            except Exception as e:
                print(f"Error on poem {i + 1}: {e}", file=sys.stderr)
                autopsy = ""
            f.write(json.dumps({"poem": poem, "autopsy": autopsy}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(poems)}", file=sys.stderr)

    print(f"Processed {len(poems)} poems -> {args.output}")


if __name__ == "__main__":
    main()
