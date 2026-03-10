#!/usr/bin/env python3
"""Generation briefs — from user requests. Compact format. Sonnet or local."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

from scripts.data_generation.claude_utils import (
    call_claude,
    get_educator_system_prompt,
    CLAUDE_SONNET_4_5,
    load_requests,
    RAW_GOOD,
)
from models.prompts.loader import render_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="File or dir with user requests")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "briefs.jsonl")
    parser.add_argument("--replace", action="store_true", help="Overwrite output file (default: append)")
    parser.add_argument("--model", type=str, default=CLAUDE_SONNET_4_5)
    args = parser.parse_args()

    source = args.input or RAW_GOOD
    requests = load_requests(source) if source and source.exists() else []
    if not requests:
        requests = [
            "Write a poem about winter light",
            "Write a poem about grief",
            "Write a poem about a meal shared with friends",
        ]
    if args.limit:
        requests = requests[: args.limit]

    system = get_educator_system_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w" if args.replace else "a") as f:
        for i, req in enumerate(requests):
            print(f"[{i + 1}/{len(requests)}] Brief: {req[:50]}...", flush=True)
            user_msg = render_prompt("tuning", "brief", user_request=req)
            try:
                brief = call_claude(user_msg, system, model=args.model, max_tokens=500)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                brief = ""
            f.write(json.dumps({"user_request": req, "brief": brief}) + "\n")

    print(f"Done: {len(requests)} briefs -> {args.output}")


if __name__ == "__main__":
    main()
