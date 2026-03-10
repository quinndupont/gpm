#!/usr/bin/env python3
"""Comparisons — good vs bad poem pairs. Anthropic Opus. Contrastive learning."""
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
    parser.add_argument("--limit", type=int, default=0, help="Max pairs (0=all bad poems)")
    parser.add_argument("--output", type=Path, default=ANNOTATED / "comparisons.jsonl")
    parser.add_argument(
        "--replace", action="store_true", help="Overwrite output file (default: append)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system = get_educator_system_prompt()
    bad = load_poems(RAW_BAD)
    good = load_poems(RAW_GOOD)
    if not bad or not good:
        print("Need both good and bad poems.", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    n = len(bad) if args.limit == 0 else min(args.limit, len(bad))
    pairs = []
    for i in range(n):
        b = bad[i]
        g = random.choice(good)
        pairs.append((g, b))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w" if args.replace else "a") as f:
        for i, (poem_a, poem_b) in enumerate(pairs):
            text_a = poem_text(poem_a)
            text_b = poem_text(poem_b)
            if not text_a.strip() or not text_b.strip():
                continue
            print(f"[{i + 1}/{len(pairs)}] Comparison...", flush=True)
            user_msg = render_prompt("tuning", "comparison", poem_a=text_a, poem_b=text_b)
            try:
                comparison = call_claude(user_msg, system, model=CLAUDE_SONNET_4_5, max_tokens=600)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                comparison = ""
            f.write(
                json.dumps(
                    {
                        "poem_a": poem_a,
                        "poem_b": poem_b,
                        "comparison": comparison,
                        "stronger": "a",
                    }
                )
                + "\n"
            )

    print(f"Done: {len(pairs)} comparisons -> {args.output}")


if __name__ == "__main__":
    main()
