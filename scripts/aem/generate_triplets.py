#!/usr/bin/env python3
"""Generate aesthetic similarity triplets using the Educator model.

Samples poems from the corpus and uses the Educator to determine which of two
candidates is more aesthetically similar to an anchor. Outputs JSONL of
(anchor, positive, negative) for training.

Usage:
  EDUCATOR_URL=http://localhost:8080/v1 python generate_triplets.py --output triplets.jsonl --count 1000
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GOOD_PATH = ROOT / "data" / "raw" / "good" / "combined_clean.json"
BAD_PATH = ROOT / "data" / "raw" / "bad" / "poetrydotcom.json"


def load_poems() -> list[dict]:
    poems = []
    if GOOD_PATH.exists():
        data = json.loads(GOOD_PATH.read_text())
        for p in data:
            if isinstance(p, dict) and p.get("poem"):
                poems.append(p)
    return poems


def load_bad() -> list[dict]:
    poems = []
    if BAD_PATH.exists():
        data = json.loads(BAD_PATH.read_text())
        for p in data:
            if isinstance(p, dict) and p.get("poem"):
                poems.append(p)
    return poems


def get_poem_text(p: dict) -> str:
    """Extract poem text, handling title/author."""
    poem = p.get("poem", "")
    if isinstance(poem, str):
        return poem.strip()
    return ""


def call_educator(anchor: str, candidate_b: str, candidate_c: str, url: str) -> str | None:
    """Call Educator: which of B or C is more aesthetically similar to anchor? Returns 'B' or 'C'."""
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=url, api_key=os.environ.get("EDUCATOR_API_KEY", "not-needed"))
    model = os.environ.get("EDUCATOR_MODEL", "educator")

    prompt = f"""You are a poetry educator. Consider aesthetic similarity: same relationship to image, degree of abstraction, trust in concrete detail, emotional restraint.

Anchor poem A:
{anchor[:1500]}

Which poem is more aesthetically similar to A?

B:
{candidate_b[:800]}

C:
{candidate_c[:800]}

Respond with exactly one word: B or C."""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip().upper()
    if "B" in text and "C" not in text:
        return "B"
    if "C" in text:
        return "C"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "aem_triplets.jsonl")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--url", default=os.environ.get("EDUCATOR_URL", "http://localhost:8080/v1"))
    args = parser.parse_args()

    good = load_poems()
    bad = load_bad()
    if len(good) < 10:
        print("Need at least 10 good poems", file=sys.stderr)
        sys.exit(1)

    good_texts = [get_poem_text(p) for p in good if len(get_poem_text(p)) > 50]
    bad_texts = [get_poem_text(p) for p in bad if len(get_poem_text(p)) > 50]
    random.shuffle(good_texts)
    random.shuffle(bad_texts)

    count = 0
    with open(args.output, "w") as f:
        for i in range(args.count * 2):  # oversample
            if count >= args.count:
                break
            anchor = random.choice(good_texts)
            pos = random.choice(good_texts)
            neg = random.choice(bad_texts) if bad_texts else random.choice(good_texts)
            if anchor == pos or anchor == neg or pos == neg:
                continue
            if len(anchor) < 100 or len(pos) < 100 or len(neg) < 100:
                continue

            result = call_educator(anchor, pos, neg, args.url)
            if result == "B":
                triplet = {"anchor": anchor, "positive": pos, "negative": neg}
            elif result == "C":
                triplet = {"anchor": anchor, "positive": neg, "negative": pos}
            else:
                continue

            f.write(json.dumps(triplet) + "\n")
            count += 1
            if count % 50 == 0:
                print(f"Generated {count} triplets...", file=sys.stderr)

    print(f"Wrote {count} triplets to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
