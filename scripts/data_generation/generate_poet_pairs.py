#!/usr/bin/env python3
"""Reverse briefs for poet — brief → poem pairs via Claude."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude

POET_SYSTEM = """You are a poet. Write with precision, musicality, and originality.
Every word must earn its place. Avoid cliché. Trust the image.
Follow the generation brief exactly — its constraints are generative."""

POET_PROMPT = """Write a poem following this generation brief:

---
{brief}
---

Output ONLY the poem. No commentary."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--briefs", type=Path, default=EDUCATOR_TRAINING / "briefs.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=POET_TRAINING / "pairs.jsonl")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    args = parser.parse_args()

    entries = []
    if args.briefs.exists():
        for line in args.briefs.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    if args.limit:
        entries = entries[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, e in enumerate(entries):
            brief = e.get("brief", e.get("user_request", ""))
            if not brief.strip():
                continue
            user_msg = POET_PROMPT.format(brief=brief)
            try:
                poem = call_claude(user_msg, POET_SYSTEM, model=args.model, max_tokens=1024)
            except Exception as err:
                print(f"Error on brief {i + 1}: {err}", file=sys.stderr)
                poem = ""
            f.write(json.dumps({"brief": brief, "poem": poem, "user_request": e.get("user_request", "")}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(entries)}", file=sys.stderr)

    print(f"Processed {len(entries)} entries -> {args.output}")


if __name__ == "__main__":
    main()
