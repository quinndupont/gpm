#!/usr/bin/env python3
"""T6: Craft lessons — educator explains concepts in voice."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import call_claude, get_educator_system_prompt

CRAFT_QUESTIONS = [
    "What does it mean to earn an abstraction?",
    "How do you know when a poem is done?",
    "When should you break a line?",
    "What's the difference between sentiment and sentimentality?",
    "How do you revise without losing the original impulse?",
]

T6_PROMPT = """A student asks: {question}

Explain this concept in your voice. Use examples — from poems you know or hypothetical ones. Be concrete. No bullet points or rubrics."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, help="File with craft questions (one per line)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "lessons.jsonl")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    questions = CRAFT_QUESTIONS
    if args.questions and args.questions.exists():
        questions = [q.strip() for q in args.questions.read_text().strip().split("\n") if q.strip()]
    if args.limit:
        questions = questions[: args.limit]

    system = get_educator_system_prompt()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, q in enumerate(questions):
            user_msg = T6_PROMPT.format(question=q)
            try:
                lesson = call_claude(user_msg, system, model=args.model, max_tokens=1024)
            except Exception as e:
                print(f"Error on question {i + 1}: {e}", file=sys.stderr)
                lesson = ""
            f.write(json.dumps({"question": q, "lesson": lesson}) + "\n")
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(questions)}", file=sys.stderr)

    print(f"Processed {len(questions)} questions -> {args.output}")


if __name__ == "__main__":
    main()
