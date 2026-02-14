#!/usr/bin/env python3
"""Autopsies — cliché dissection on bad poems. Sonnet or local. Contrastive."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import (
    call_claude,
    get_educator_system_prompt,
    CLAUDE_SONNET_4_5,
    load_poems,
    poem_text,
    RAW_BAD,
)

AUTOPSY_PROMPT = """Here is a poem from an amateur poet:

---
{bad_poem_text}
---

Perform a cliché autopsy. For each clichéd element:
1. WHAT the cliché is (quote it)
2. WHY it became a cliché
3. WHAT the poet was probably reaching for
4. WHAT they could do instead (direction, not rewrite)

Structured. Honest but not cruel."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ANNOTATED / "autopsies.jsonl")
    parser.add_argument("--replace", action="store_true", help="Overwrite output file (default: append)")
    parser.add_argument("--model", type=str, default=CLAUDE_SONNET_4_5)
    args = parser.parse_args()

    system = get_educator_system_prompt()
    poems = load_poems(RAW_BAD)
    if args.limit:
        poems = poems[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w" if args.replace else "a") as f:
        for i, poem in enumerate(poems):
            text = poem_text(poem)
            if not text.strip():
                continue
            print(f"[{i + 1}/{len(poems)}] Autopsy...", flush=True)
            user_msg = AUTOPSY_PROMPT.format(bad_poem_text=text)
            try:
                autopsy = call_claude(user_msg, system, model=args.model, max_tokens=600)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                autopsy = ""
            f.write(json.dumps({"poem": poem, "autopsy": autopsy}) + "\n")

    print(f"Done: {len(poems)} autopsies -> {args.output}")


if __name__ == "__main__":
    main()
