#!/usr/bin/env python3
"""Critiques seed — Anthropic Opus. Hard task: discriminating what works vs fails."""
import argparse
import json
import random
import sys
from pathlib import Path

from models.prompts.loader import render_prompt
from scripts.data_generation.claude_utils import (
    CLAUDE_SONNET_4_5,
    RAW_BAD,
    RAW_GOOD,
    call_claude,
    get_educator_system_prompt,
    load_poems,
    poem_text,
)

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-bad", type=int, default=0, help="Max bad poems (0=all)")
    parser.add_argument("--limit-good", type=int, default=200, help="Max good poems to subsample")
    parser.add_argument("--output", type=Path, default=ANNOTATED / "critiques_seed.jsonl")
    parser.add_argument(
        "--replace", action="store_true", help="Overwrite output file (default: append)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system = get_educator_system_prompt()
    bad = load_poems(RAW_BAD)
    good = load_poems(RAW_GOOD)
    if args.limit_bad:
        bad = bad[: args.limit_bad]
    random.seed(args.seed)
    good_subsample = random.sample(good, min(args.limit_good, len(good))) if good else []

    poems = bad + good_subsample
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w" if args.replace else "a") as f:
        for i, poem in enumerate(poems):
            text = poem_text(poem)
            if not text.strip():
                continue
            source = "bad" if i < len(bad) else "good"
            print(f"[{i + 1}/{len(poems)}] Critique ({source})...", flush=True)
            user_msg = render_prompt("tuning", "critique", poem_text=text)
            try:
                critique = call_claude(user_msg, system, model=CLAUDE_SONNET_4_5, max_tokens=600)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                critique = ""
            f.write(json.dumps({"poem": poem, "critique": critique, "source": source}) + "\n")

    print(f"Done: {len(poems)} critiques -> {args.output}")


if __name__ == "__main__":
    main()
