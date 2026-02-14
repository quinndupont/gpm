#!/usr/bin/env python3
"""Poet pairs — brief → poem via Claude Opus. Poem-only output format."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude, CLAUDE_OPUS_4_6

POET_SYSTEM = """You are a poet. You receive generation briefs and write poems.
You never output instructions, critique, or analysis — only poems."""

# Cap brief at ~300 tokens (~1200 chars) for poet training
BRIEF_CAP = 1200

POET_PROMPT = """Generation brief:

---
{brief}
---

Write the poem. Output ONLY the poem — no title unless it's part of the poem, no commentary."""


def _truncate_brief(brief: str) -> str:
    if len(brief) <= BRIEF_CAP:
        return brief
    return brief[:BRIEF_CAP].rsplit("\n", 1)[0] + "\n\n[truncated]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--briefs", type=Path, default=EDUCATOR_TRAINING / "briefs.jsonl")
    parser.add_argument("--revision-briefs", type=Path, default=EDUCATOR_TRAINING / "revision_briefs_seed.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=POET_TRAINING / "pairs.jsonl")
    parser.add_argument("--replace", action="store_true", help="Overwrite output file (default: append)")
    parser.add_argument("--model", type=str, default=CLAUDE_OPUS_4_6)
    args = parser.parse_args()

    entries = []
    for path in [args.briefs, args.revision_briefs]:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    e = json.loads(line)
                    brief = e.get("brief", e.get("revision_brief", ""))
                    if brief.strip():
                        entries.append({"brief": brief, "user_request": e.get("user_request", "")})
    if args.limit:
        entries = entries[: args.limit]

    if not entries:
        print("No briefs found. Run generate_briefs.py and generate_revision_briefs.py first.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w" if args.replace else "a") as f:
        for i, e in enumerate(entries):
            brief = _truncate_brief(e["brief"])
            print(f"[{i + 1}/{len(entries)}] Poem from brief...", flush=True)
            user_msg = POET_PROMPT.format(brief=brief)
            try:
                poem = call_claude(user_msg, POET_SYSTEM, model=args.model, max_tokens=1024, force_anthropic=True)
            except Exception as err:
                print(f"  Error: {err}", file=sys.stderr)
                poem = ""
            f.write(
                json.dumps(
                    {
                        "brief": brief,
                        "poem": poem,
                        "user_request": e.get("user_request", ""),
                    }
                )
                + "\n"
            )

    print(f"Done: {len(entries)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
