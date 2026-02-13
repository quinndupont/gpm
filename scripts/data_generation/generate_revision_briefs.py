#!/usr/bin/env python3
"""Revision briefs — from critique + poem. Anthropic Opus. Hard task: synthesis."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import (
    call_claude,
    get_educator_system_prompt,
    CLAUDE_OPUS_4_6,
    poem_text,
)

REVISION_BRIEF_PROMPT = """Original poem:

---
{poem_text}
---

Your critique:

---
{critique}
---

Construct a revised generation brief for the poet. Compact format (~300 tokens):
- Angle (2-3 sentences)
- Clichés to avoid (5-6 items, one line each)
- Imagery domain (1-2 sentences)
- Form guidance (1-2 sentences)

No rhetorical flourish. Actionable only."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critiques", type=Path, default=ANNOTATED / "critiques_seed.jsonl")
    parser.add_argument("--limit", type=int, default=50, help="Max revision briefs to generate")
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "revision_briefs_seed.jsonl")
    args = parser.parse_args()

    if not args.critiques.exists():
        print(f"Run generate_critiques_seed.py first. Missing: {args.critiques}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for line in args.critiques.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    entries = entries[: args.limit]

    system = get_educator_system_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for i, e in enumerate(entries):
            poem = e.get("poem", {})
            critique = e.get("critique", "")
            text = poem_text(poem)
            if not text.strip() or not critique.strip():
                continue
            print(f"[{i + 1}/{len(entries)}] Revision brief...", flush=True)
            user_msg = REVISION_BRIEF_PROMPT.format(poem_text=text, critique=critique)
            try:
                brief = call_claude(user_msg, system, model=CLAUDE_OPUS_4_6, max_tokens=500, force_anthropic=True)
            except Exception as err:
                print(f"  Error: {err}", file=sys.stderr)
                brief = ""
            f.write(
                json.dumps(
                    {"poem": poem, "critique": critique, "revision_brief": brief}
                )
                + "\n"
            )

    print(f"Done: {len(entries)} revision briefs -> {args.output}")


if __name__ == "__main__":
    main()
