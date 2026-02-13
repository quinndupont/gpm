#!/usr/bin/env python3
"""Convert generated outputs to Llama 3 chat format for training."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"
PERSONA = ROOT / "persona"

def _educator_system() -> str:
    p = PERSONA / "persona_condensed.txt"
    return p.read_text().strip() if p.exists() else """Maren is a poetry educator who believes poems are acts of radical attention. She trusts the image to do emotional work; she loves Bishop, Levis, Brigit Pegeen Kelly, Vuong. She thinks most poems are 30% too long. Her concepts: "The specific is the universal," "Trust the image," "Earn your abstractions," "The poem knows more than the poet," "Cut toward the bone." She uses "I" frequently, asks questions as her main pedagogical tool, gets excited about strong images. Direct, warm, never falsely encouraging. Never uses rubrics, scores, or phrases like "Great job!" or "This poem resonates." """


POET_SYSTEM = """You are a poet. Write with precision, musicality, and originality. Every word must earn its place."""


def poem_text(poem) -> str:
    if isinstance(poem, str):
        return poem
    return poem.get("text", poem.get("poem", poem.get("content", ""))) if isinstance(poem, dict) else ""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def to_educator_example(user: str, assistant: str, system: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def to_poet_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": POET_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def collect_educator_examples(system: str) -> list[dict]:
    examples = []

    # critiques: user=poem, assistant=critique
    for e in load_jsonl(ANNOTATED / "critiques.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        if poem.strip() and critique.strip():
            examples.append(to_educator_example(poem, critique, system))

    # briefs: user=request, assistant=brief
    for e in load_jsonl(EDUCATOR_TRAINING / "briefs.jsonl"):
        req = e.get("user_request", "")
        brief = e.get("brief", "")
        if req.strip() and brief.strip():
            examples.append(to_educator_example(req, brief, system))

    # lessons: user=question, assistant=lesson
    for e in load_jsonl(EDUCATOR_TRAINING / "lessons.jsonl"):
        q = e.get("question", "")
        lesson = e.get("lesson", "")
        if q.strip() and lesson.strip():
            examples.append(to_educator_example(q, lesson, system))

    # autopsies: user=poem, assistant=autopsy
    for e in load_jsonl(ANNOTATED / "autopsies.jsonl"):
        poem = poem_text(e.get("poem", {}))
        autopsy = e.get("autopsy", "")
        if poem.strip() and autopsy.strip():
            examples.append(to_educator_example(poem, autopsy, system))

    # comparisons: user=poem_a+poem_b, assistant=comparison
    for e in load_jsonl(ANNOTATED / "comparisons.jsonl"):
        a = poem_text(e.get("poem_a", {}))
        b = poem_text(e.get("poem_b", {}))
        comp = e.get("comparison", "")
        if a.strip() and b.strip() and comp.strip():
            user = f"Poem A:\n---\n{a}\n---\n\nPoem B:\n---\n{b}\n---"
            examples.append(to_educator_example(user, comp, system))

    # dialogues: user=poem+critique+revision, assistant=follow_up
    for e in load_jsonl(EDUCATOR_TRAINING / "dialogues.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        revision = e.get("revision", "")
        follow_up = e.get("follow_up", "")
        if poem.strip() and critique.strip() and revision.strip() and follow_up.strip():
            user = f"Original poem:\n---\n{poem}\n---\n\nYour critique:\n---\n{critique}\n---\n\nTheir revision:\n---\n{revision}\n---"
            examples.append(to_educator_example(user, follow_up, system))

    return examples


def collect_poet_examples() -> list[dict]:
    examples = []
    for e in load_jsonl(POET_TRAINING / "pairs.jsonl"):
        brief = e.get("brief", e.get("user_request", ""))
        poem = e.get("poem", "")
        if brief.strip() and poem.strip():
            examples.append(to_poet_example(brief, poem))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--educator-only", action="store_true")
    parser.add_argument("--poet-only", action="store_true")
    parser.add_argument("--min-samples", type=int, default=0, help="Min samples per split (for quick test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.poet_only:
        system = _educator_system()
        educator = collect_educator_examples(system)
        if args.min_samples and len(educator) < args.min_samples:
            raise SystemExit(
                f"Need at least {args.min_samples} educator examples (got {len(educator)}). "
                "Run generate_briefs, generate_lessons, etc. first."
            )
        random.shuffle(educator)
        n_val = max(1, len(educator) // 10) if len(educator) >= 2 else 0
        train_edu = educator[:-n_val] if n_val else educator
        valid_edu = educator[-n_val:] if n_val else educator[:1]  # at least 1 for valid
        if args.min_samples:
            train_edu = train_edu[: args.min_samples]
            valid_edu = valid_edu[: max(1, min(len(valid_edu), args.min_samples // 5))]

        EDUCATOR_TRAINING.mkdir(parents=True, exist_ok=True)
        with open(EDUCATOR_TRAINING / "train.jsonl", "w") as f:
            for ex in train_edu:
                f.write(json.dumps(ex) + "\n")
        with open(EDUCATOR_TRAINING / "valid.jsonl", "w") as f:
            for ex in valid_edu:
                f.write(json.dumps(ex) + "\n")
        print(f"Educator: {len(train_edu)} train, {len(valid_edu)} valid")

    if not args.educator_only:
        poet = collect_poet_examples()
        if args.min_samples and len(poet) < args.min_samples:
            raise SystemExit(
                f"Need at least {args.min_samples} poet examples (got {len(poet)}). "
                "Run generate_briefs then generate_poet_pairs first."
            )
        random.shuffle(poet)
        n_val = max(1, len(poet) // 10)
        n_val = min(n_val, len(poet) - 1) if len(poet) > 1 else 0
        train_poet = poet[:-n_val] if n_val else poet
        valid_poet = poet[-n_val:] if n_val else poet[:1]
        if args.min_samples:
            train_poet = train_poet[: args.min_samples]
            valid_poet = valid_poet[: max(1, min(len(valid_poet), args.min_samples // 5))]

        POET_TRAINING.mkdir(parents=True, exist_ok=True)
        with open(POET_TRAINING / "train.jsonl", "w") as f:
            for ex in train_poet:
                f.write(json.dumps(ex) + "\n")
        with open(POET_TRAINING / "valid.jsonl", "w") as f:
            for ex in valid_poet:
                f.write(json.dumps(ex) + "\n")
        print(f"Poet: {len(train_poet)} train, {len(valid_poet)} valid")


if __name__ == "__main__":
    main()
