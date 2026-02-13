#!/usr/bin/env python3
"""T5: Revision dialogue — poem + critique + student revision → educator follow-up (what improved, what still needs work)."""
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
    CLAUDE_SONNET_4_5,
    poem_text,
)

# Student persona: simulate a revision attempt (no educator voice)
STUDENT_REVISION_PROMPT = """You are a student poet. You wrote this poem and received this workshop critique.

Original poem:
---
{poem_text}
---

Critique:
---
{critique}
---

Write a revised version of the poem that responds to the critique. Output ONLY the revised poem — no title unless part of the poem, no commentary."""

# T5: Educator follow-up after seeing the revision (plan: 200–300 words)
DIALOGUE_PROMPT = """Original poem:

---
{poem_text}
---

Your earlier critique:

---
{critique}
---

The student has attempted a revision:

---
{revised_poem}
---

Give your follow-up: what improved, what still needs work. Be specific to lines and choices. 200–300 words. In your voice. No rubrics or scores."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critiques", type=Path, default=ANNOTATED / "critiques_seed.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Max dialogues (0 = all)")
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "dialogues.jsonl")
    parser.add_argument("--model", type=str, default=CLAUDE_OPUS_4_6)
    args = parser.parse_args()

    if not args.critiques.exists():
        print(f"Run generate_critiques_seed.py first. Missing: {args.critiques}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for line in args.critiques.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    if args.limit:
        entries = entries[: args.limit]

    system_educator = get_educator_system_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        for i, e in enumerate(entries):
            poem = e.get("poem", {})
            critique = e.get("critique", "")
            text = poem_text(poem)
            if not text.strip() or not critique.strip():
                continue
            print(f"[{i + 1}/{len(entries)}] Revision (student)...", flush=True)
            user_revision = STUDENT_REVISION_PROMPT.format(poem_text=text, critique=critique)
            try:
                revised_poem = call_claude(
                    user_revision,
                    system_message="You are a student poet. Output only the revised poem.",
                    model=CLAUDE_SONNET_4_5,
                    max_tokens=600,
                    force_anthropic=True,
                )
            except Exception as err:
                print(f"  Error: {err}", file=sys.stderr)
                revised_poem = ""
            if not revised_poem.strip():
                continue

            print(f"[{i + 1}/{len(entries)}] Follow-up (educator)...", flush=True)
            user_dialogue = DIALOGUE_PROMPT.format(
                poem_text=text, critique=critique, revised_poem=revised_poem.strip()
            )
            try:
                follow_up = call_claude(
                    user_dialogue,
                    system_message=system_educator,
                    model=args.model,
                    max_tokens=500,
                    force_anthropic=True,
                )
            except Exception as err:
                print(f"  Error: {err}", file=sys.stderr)
                follow_up = ""
            if not follow_up.strip():
                continue

            f.write(
                json.dumps(
                    {
                        "poem": poem,
                        "critique": critique,
                        "revised_poem": revised_poem.strip(),
                        "follow_up": follow_up.strip(),
                    }
                )
                + "\n"
            )

    print(f"Done: {len(entries)} dialogues -> {args.output}")


if __name__ == "__main__":
    main()
