#!/usr/bin/env python3
"""Convert generated outputs to chat format for training. New pipeline."""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"
PERSONA = ROOT / "persona"

import sys
sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import poem_text


def _educator_system() -> str:
    p = PERSONA / "educator_neutral.txt"
    if not p.exists():
        p = PERSONA / "persona_condensed.txt"
    return p.read_text().strip() if p.exists() else "You are a poetry educator. Identify craft issues. Give concrete directions."


POET_SYSTEM = """You are a poet. You receive generation briefs and write poems.
You never output instructions, critique, or analysis — only poems."""

POET_USER_SUFFIX = "\n\nWrite the poem. Output ONLY the poem — no title unless it's part of the poem, no commentary."


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


def collect_educator_seed_examples(system: str) -> list[dict]:
    """Only critiques, comparisons, revision_briefs — for interim educator training."""
    examples = []
    for e in load_jsonl(ANNOTATED / "critiques_seed.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        if poem.strip() and critique.strip():
            examples.append(to_educator_example(poem, critique, system))
    for e in load_jsonl(ANNOTATED / "comparisons.jsonl"):
        a = poem_text(e.get("poem_a", {}))
        b = poem_text(e.get("poem_b", {}))
        comp = e.get("comparison", "")
        if a.strip() and b.strip() and comp.strip():
            user = f"Poem A:\n---\n{a}\n---\n\nPoem B:\n---\n{b}\n---"
            examples.append(to_educator_example(user, comp, system))
    for e in load_jsonl(EDUCATOR_TRAINING / "revision_briefs_seed.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        brief = e.get("revision_brief", "")
        if poem.strip() and critique.strip() and brief.strip():
            user = f"Poem:\n---\n{poem}\n---\n\nCritique:\n---\n{critique}\n---\n\nConstruct a revised generation brief."
            examples.append(to_educator_example(user, brief, system))
    return examples


def collect_educator_examples(system: str) -> list[dict]:
    examples = []

    # critiques_seed: user=poem, assistant=critique
    for e in load_jsonl(ANNOTATED / "critiques_seed.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        if poem.strip() and critique.strip():
            examples.append(to_educator_example(poem, critique, system))

    # comparisons: user=poem_a+poem_b, assistant=comparison
    for e in load_jsonl(ANNOTATED / "comparisons.jsonl"):
        a = poem_text(e.get("poem_a", {}))
        b = poem_text(e.get("poem_b", {}))
        comp = e.get("comparison", "")
        if a.strip() and b.strip() and comp.strip():
            user = f"Poem A:\n---\n{a}\n---\n\nPoem B:\n---\n{b}\n---"
            examples.append(to_educator_example(user, comp, system))

    # revision_briefs_seed: user=poem+critique, assistant=revision_brief
    for e in load_jsonl(EDUCATOR_TRAINING / "revision_briefs_seed.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        brief = e.get("revision_brief", "")
        if poem.strip() and critique.strip() and brief.strip():
            user = f"Poem:\n---\n{poem}\n---\n\nCritique:\n---\n{critique}\n---\n\nConstruct a revised generation brief."
            examples.append(to_educator_example(user, brief, system))

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

    # dialogues (T5): user=poem+critique+revised_poem, assistant=follow_up
    for e in load_jsonl(EDUCATOR_TRAINING / "dialogues.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        revised = e.get("revised_poem", "")
        follow_up = e.get("follow_up", "")
        if poem.strip() and critique.strip() and revised.strip() and follow_up.strip():
            user = (
                f"Original poem:\n---\n{poem}\n---\n\nYour critique:\n---\n{critique}\n---\n\n"
                f"The student's revision:\n---\n{revised}\n---\n\nGive your follow-up: what improved, what still needs work."
            )
            examples.append(to_educator_example(user, follow_up, system))

    # autopsies: user=poem, assistant=autopsy
    for e in load_jsonl(ANNOTATED / "autopsies.jsonl"):
        poem = poem_text(e.get("poem", {}))
        autopsy = e.get("autopsy", "")
        if poem.strip() and autopsy.strip():
            examples.append(to_educator_example(poem, autopsy, system))

    return examples


def collect_poet_examples() -> list[dict]:
    examples = []
    for e in load_jsonl(POET_TRAINING / "pairs.jsonl"):
        brief = e.get("brief", e.get("user_request", ""))
        poem = e.get("poem", "")
        if brief.strip() and poem.strip():
            user = brief + POET_USER_SUFFIX
            examples.append(to_poet_example(user, poem))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--educator-only", action="store_true")
    parser.add_argument("--poet-only", action="store_true")
    parser.add_argument("--interim-educator", action="store_true", help="Only seed data (critiques, comparisons, revision_briefs) for interim educator")
    parser.add_argument("--min-samples", type=int, default=0, help="Min samples per split (for quick test)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.poet_only:
        print("Loading educator examples...", flush=True)
        system = _educator_system()
        educator = collect_educator_seed_examples(system) if args.interim_educator else collect_educator_examples(system)
        if args.min_samples and len(educator) < args.min_samples:
            raise SystemExit(
                f"Need at least {args.min_samples} educator examples (got {len(educator)}). "
                + ("Run generate_critiques_seed, generate_comparisons, generate_revision_briefs first." if args.interim_educator else "Run generate_critiques_seed, generate_comparisons, generate_revision_briefs, generate_briefs, generate_lessons, generate_dialogues, generate_autopsies first.")
            )
        random.shuffle(educator)
        n_val = max(1, len(educator) // 10) if len(educator) >= 2 else 0
        train_edu = educator[:-n_val] if n_val else educator
        valid_edu = educator[-n_val:] if n_val else educator[:1]
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
        print(f"Educator: {len(train_edu)} train, {len(valid_edu)} valid" + (" (interim seed only)" if args.interim_educator else ""))

    if not args.educator_only and not args.interim_educator:
        print("Loading poet examples...", flush=True)
        poet = collect_poet_examples()
        if args.min_samples and len(poet) < args.min_samples:
            raise SystemExit(
                f"Need at least {args.min_samples} poet examples (got {len(poet)}). "
                "Run generate_briefs, generate_revision_briefs, generate_poet_pairs first."
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
