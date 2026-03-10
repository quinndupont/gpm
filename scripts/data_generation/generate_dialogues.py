#!/usr/bin/env python3
"""T5: Revision dialogue — poem + critique + student revision → educator follow-up (what improved, what still needs work)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

from scripts.data_generation.claude_utils import (
    call_claude,
    get_educator_system_prompt,
    CLAUDE_SONNET_4_5,
    poem_text,
)
from models.prompts.loader import render_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critiques", type=Path, default=ANNOTATED / "critiques_seed.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="Max dialogues (0 = all)")
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "dialogues.jsonl")
    parser.add_argument("--replace", action="store_true", help="Overwrite output file (default: append)")
    parser.add_argument("--model", type=str, default=CLAUDE_SONNET_4_5)
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

    with open(args.output, "w" if args.replace else "a") as f:
        for i, e in enumerate(entries):
            poem = e.get("poem", {})
            critique = e.get("critique", "")
            text = poem_text(poem)
            if not text.strip() or not critique.strip():
                continue
            print(f"[{i + 1}/{len(entries)}] Revision (student)...", flush=True)
            user_revision = render_prompt("tuning", "dialogue", template="student_revision", poem_text=text, critique=critique)
            try:
                revised_poem = call_claude(
                    user_revision,
                    system_message="You are a student poet. Output only the revised poem.",
                    model=CLAUDE_SONNET_4_5,
                    max_tokens=600,
                )
            except Exception as err:
                print(f"  Error: {err}", file=sys.stderr)
                revised_poem = ""
            if not revised_poem.strip():
                continue

            print(f"[{i + 1}/{len(entries)}] Follow-up (educator)...", flush=True)
            user_dialogue = render_prompt("tuning", "dialogue", template="dialogue", poem_text=text, critique=critique, revised_poem=revised_poem.strip())
            try:
                follow_up = call_claude(
                    user_dialogue,
                    system_message=system_educator,
                    model=args.model,
                    max_tokens=500,
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
