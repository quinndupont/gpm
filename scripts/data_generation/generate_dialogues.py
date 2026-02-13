#!/usr/bin/env python3
"""T5: Revision dialogue — educator follow-up after revision attempt."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude, get_educator_system_prompt


def poem_text(poem: dict) -> str:
    return poem.get("text", poem.get("poem", poem.get("content", ""))) if isinstance(poem, dict) else str(poem)


T5_PROMPT = """A student brought this poem to workshop:

---
{poem_text}
---

You gave them this critique:

---
{critique}
---

They attempted a revision. Here is their revised draft:

---
{revision}
---

Give your follow-up. What improved? What still needs work? Be specific. Offer direction. If they've found something, say so. If they've missed the point, say so with compassion."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critiques", type=Path, default=ANNOTATED / "critiques.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "dialogues.jsonl")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    args = parser.parse_args()

    entries = []
    if args.critiques.exists():
        for line in args.critiques.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    if args.limit:
        entries = entries[: args.limit]

    system = get_educator_system_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, e in enumerate(entries):
            poem = e.get("poem", {})
            critique = e.get("critique", "")
            poem_text_val = poem_text(poem)
            if not poem_text_val.strip() or not critique.strip():
                f.write(json.dumps({"poem": poem, "critique": critique, "revision": "", "follow_up": ""}) + "\n")
                continue
            # First call: generate a plausible revision the student might have made
            revision_prompt = f"""A student received this critique of their poem:

---
{critique}
---

Their original poem:
---
{poem_text_val}
---

Invent a REVISION the student might have made — 2-4 lines changed, addressing at least one thing from the critique. Output ONLY the revised poem, no commentary."""
            try:
                revision = call_claude(revision_prompt, None, model=args.model, max_tokens=512)
            except Exception as err:
                print(f"Error generating revision {i + 1}: {err}", file=sys.stderr)
                revision = poem_text_val
            # Second call: educator follow-up
            user_msg = T5_PROMPT.format(poem_text=poem_text_val, critique=critique, revision=revision)
            try:
                follow_up = call_claude(user_msg, system, model=args.model, max_tokens=1024)
            except Exception as err:
                print(f"Error on follow-up {i + 1}: {err}", file=sys.stderr)
                follow_up = ""
            f.write(
                json.dumps(
                    {"poem": poem, "critique": critique, "revision": revision, "follow_up": follow_up}
                )
                + "\n"
            )
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(entries)}", file=sys.stderr)

    print(f"Processed {len(entries)} entries -> {args.output}")


if __name__ == "__main__":
    main()
