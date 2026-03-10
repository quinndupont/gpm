#!/usr/bin/env python3
"""Autopsies — cliché dissection on bad poems. Sonnet or local. Contrastive."""
import argparse
import json
import sys
from pathlib import Path

from models.prompts.loader import render_prompt
from scripts.data_generation.claude_utils import (
    CLAUDE_SONNET_4_5,
    RAW_BAD,
    call_claude,
    get_educator_system_prompt,
    load_poems,
    poem_text,
)

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ANNOTATED / "autopsies.jsonl")
    parser.add_argument(
        "--replace", action="store_true", help="Overwrite output file (default: append)"
    )
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
            user_msg = render_prompt("tuning", "autopsy", bad_poem_text=text)
            try:
                autopsy = call_claude(user_msg, system, model=args.model, max_tokens=600)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                autopsy = ""
            f.write(json.dumps({"poem": poem, "autopsy": autopsy}) + "\n")

    print(f"Done: {len(poems)} autopsies -> {args.output}")


if __name__ == "__main__":
    main()
