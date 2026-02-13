#!/usr/bin/env python3
"""T2: Generation briefs — Claude API batch generation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_GOOD = ROOT / "data" / "raw" / "good"
RAW_BAD = ROOT / "data" / "raw" / "bad"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude, get_educator_system_prompt


def load_requests(source: Path) -> list[str]:
    """Load user requests from file or directory."""
    requests = []
    if source.is_file():
        if source.suffix == ".jsonl":
            for line in source.read_text().splitlines():
                if line.strip():
                    obj = json.loads(line)
                    requests.append(obj.get("request", obj.get("prompt", str(obj))))
        else:
            requests.extend(source.read_text().strip().split("\n\n"))
    else:
        for p in source.glob("**/*.txt"):
            requests.extend(p.read_text().strip().split("\n\n"))
        for p in source.glob("**/*.jsonl"):
            for line in p.read_text().splitlines():
                if line.strip():
                    obj = json.loads(line)
                    requests.append(obj.get("request", obj.get("prompt", str(obj))))
    return [r.strip() for r in requests if r.strip()]


T2_PROMPT = """A student has asked for help writing a poem. Their request:
"{user_request}"

Construct a GENERATION BRIEF — the assignment you'd give your most talented MFA student. The brief MUST include:

1. A SPECIFIC angle — not the obvious approach to this topic
2. ANTI-CLICHÉ GUIDANCE: At minimum 8 specific phrases, images, and structural moves to AVOID for this topic
3. An UNEXPECTED IMAGERY DOMAIN orthogonal to the topic
4. FORM AND STRUCTURE guidance that serves the content — argue for why this form fits
5. SOUND guidance — specific consonant/vowel textures
6. A STRUCTURAL ARC — where should the poem turn?

Write in your voice, as if excited about this specific poem."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="File or dir with user requests")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "briefs.jsonl")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022")
    args = parser.parse_args()

    source = args.input or RAW_GOOD
    requests = load_requests(source) if source and source.exists() else []
    if not requests:
        # Fallback: use default prompts if no input
        requests = [
            "Write a poem about winter light",
            "Write a poem about grief",
            "Write a poem about a meal shared with friends",
        ]
    if args.limit:
        requests = requests[: args.limit]

    system = get_educator_system_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, req in enumerate(requests):
            user_msg = T2_PROMPT.format(user_request=req)
            try:
                brief = call_claude(user_msg, system, model=args.model, max_tokens=1024)
            except Exception as e:
                print(f"Error on request {i + 1}: {e}", file=sys.stderr)
                brief = ""
            f.write(json.dumps({"user_request": req, "brief": brief}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(requests)}", file=sys.stderr)

    print(f"Processed {len(requests)} requests -> {args.output}")


if __name__ == "__main__":
    main()
