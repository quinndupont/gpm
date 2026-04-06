#!/usr/bin/env python3
"""Convert generated outputs to chat format for training. New pipeline."""
import argparse
import json
import random
from pathlib import Path

from models.prompts.loader import get_persona, get_prompt
from scripts.data_generation.claude_utils import get_educator_system_prompt, poem_text
from scripts.eval.form_registry import get_scheme
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"
RHYME_TRAINING = ROOT / "data" / "rhyme_training"

MIN_STRICT_DENSITY = 0.6


def _educator_system() -> str:
    try:
        return get_educator_system_prompt()
    except Exception:
        return "You are a poetry educator. Identify craft issues. Give concrete directions."




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
            {"role": "system", "content": get_persona("poet")},
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
            user = (
                f"Poem:\n---\n{poem}\n---\n\nCritique:\n---\n{critique}\n---\n\n"
                "Construct a revised generation brief."
            )
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
            user = (
                f"Poem A:\n---\n{a}\n---\n\nPoem B:\n---\n{b}\n---"
            )
            examples.append(to_educator_example(user, comp, system))

    # revision_briefs_seed: user=poem+critique, assistant=revision_brief
    for e in load_jsonl(EDUCATOR_TRAINING / "revision_briefs_seed.jsonl"):
        poem = poem_text(e.get("poem", {}))
        critique = e.get("critique", "")
        brief = e.get("revision_brief", "")
        if poem.strip() and critique.strip() and brief.strip():
            user = (
                f"Poem:\n---\n{poem}\n---\n\nCritique:\n---\n{critique}\n---\n\n"
                "Construct a revised generation brief."
            )
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
                f"The student's revision:\n---\n{revised}\n---\n\n"
                "Give your follow-up: what improved, what still needs work."
            )
            examples.append(to_educator_example(user, follow_up, system))

    # autopsies: user=poem, assistant=autopsy
    for e in load_jsonl(ANNOTATED / "autopsies.jsonl"):
        poem = poem_text(e.get("poem", {}))
        autopsy = e.get("autopsy", "")
        if poem.strip() and autopsy.strip():
            examples.append(to_educator_example(poem, autopsy, system))

    # rhyme critiques: user=poem (with form context), assistant=rhyme-aware critique
    for e in load_jsonl(EDUCATOR_TRAINING / "rhyme_critiques.jsonl"):
        poem = e.get("poem", "")
        critique = e.get("critique", "")
        form = e.get("form", "")
        scheme = e.get("expected_scheme", "")
        if poem.strip() and critique.strip():
            user = poem
            if form and scheme:
                user = f"[Form: {form}, scheme: {scheme}]\n\n{poem}"
            examples.append(to_educator_example(user, critique, system))

    # approval/rejection examples: teach when to say "This poem has found its shape."
    for e in load_jsonl(EDUCATOR_TRAINING / "approval_examples.jsonl"):
        poem = e.get("poem", "")
        critique = e.get("critique", "")
        ra = e.get("rhyme_analysis", {})
        if poem.strip() and critique.strip():
            density_note = f"[strict_rhyme_density: {ra.get('strict_rhyme_density', 'N/A')}]"
            scheme_note = f"[detected_scheme: {ra.get('detected_scheme', 'N/A')}]"
            user = f"{density_note} {scheme_note}\n\n{poem}"
            examples.append(to_educator_example(user, critique, system))

    # Socratic examples with tool calls (multi-turn, pre-formatted)
    examples.extend(collect_socratic_examples())

    return examples


def collect_socratic_examples() -> list[dict]:
    """Load Socratic training examples (multi-turn with tool calls).

    These are already in chat format: {"messages": [...]} with system/user/assistant/tool roles.
    """
    examples = []
    for e in load_jsonl(EDUCATOR_TRAINING / "socratic_examples.jsonl"):
        msgs = e.get("messages", [])
        if len(msgs) >= 3:
            examples.append({"messages": msgs})
    return examples


def _poem_to_rhyme_example(
    poem: str, form: str | None = None, brief: str | None = None,
) -> dict | None:
    """Convert a poem to a rhyme training example if it passes the quality gate."""
    if not poem.strip():
        return None
    analysis = analyze_rhyme(poem, expected_form=form)
    strict_density = analysis.get("strict_rhyme_density", 0)
    if strict_density < MIN_STRICT_DENSITY:
        return None
    scheme = analysis.get("strict_detected_scheme") or analysis.get("detected_scheme", "")
    if brief and brief.strip():
        user = brief.strip()
        rhyme_suffix = get_prompt("tuning", "poet_generation", "rhyme_suffix").strip()
        if not user.endswith(rhyme_suffix):
            user = user + "\n\n" + rhyme_suffix
    else:
        parts = ["Write a poem with strong end rhyme."]
        if scheme:
            parts.append(f"Rhyme scheme: {scheme}.")
        if form:
            scheme_str = get_scheme(form) if form else ""
            if scheme_str:
                parts.append(f"Form: {form}, scheme {scheme_str}.")
        parts.append(
            "Every end-word pair must be a true phonetic rhyme. Follow the form precisely."
        )
        user = " ".join(parts) + get_prompt("tuning", "poet_generation", "rhyme_suffix")
    return to_poet_example(user, poem)


def collect_poet_examples() -> tuple[list[dict], int]:
    """Collect all poet training examples (general + rhyme).

    Returns (examples, n_rhyme) where n_rhyme is the count of rhyme-specific examples.
    """
    examples = []
    n_rhyme = 0

    # Source 1: general pairs (brief -> poem)
    for e in load_jsonl(POET_TRAINING / "pairs.jsonl"):
        brief = e.get("brief", e.get("user_request", ""))
        poem = e.get("poem", "")
        if brief.strip() and poem.strip():
            user = brief + get_prompt("tuning", "poet_generation", "user_suffix")
            examples.append(to_poet_example(user, poem))

    # Source 2: rhyme pairs (brief with form constraint -> poem)
    rhyme_pairs_paths = [
        POET_TRAINING / "rhyme_pairs.jsonl",
        RHYME_TRAINING / "rhyme_pairs.jsonl",
    ]
    seen_poems: set[str] = set()
    for rp_path in rhyme_pairs_paths:
        for e in load_jsonl(rp_path):
            brief = e.get("brief", "")
            poem = e.get("poem", "")
            if brief.strip() and poem.strip() and poem not in seen_poems:
                seen_poems.add(poem)
                user = brief + get_prompt("tuning", "poet_generation", "rhyme_suffix")
                examples.append(to_poet_example(user, poem))
                n_rhyme += 1

    # Source 3: strong-rhyme poems (curated, quality-gated)
    strong_path = ANNOTATED / "strong_rhyme_poems.jsonl"
    raw_strong, passed_strong = 0, 0
    for rec in load_jsonl(strong_path):
        if not rec.get("strong_rhyme", True):
            continue
        poem = rec.get("poem", "")
        if poem in seen_poems:
            continue
        raw_strong += 1
        ex = _poem_to_rhyme_example(poem)
        if ex:
            seen_poems.add(poem)
            examples.append(ex)
            n_rhyme += 1
            passed_strong += 1
    if raw_strong:
        print(
            f"  Strong-rhyme poems: {passed_strong}/{raw_strong} passed "
            f"(strict_density >= {MIN_STRICT_DENSITY})"
        )

    return examples, n_rhyme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--educator-only", action="store_true")
    parser.add_argument("--poet-only", action="store_true")
    parser.add_argument(
        "--interim-educator", action="store_true",
        help="Only seed data (critiques, comparisons, revision_briefs) for interim educator",
    )
    parser.add_argument(
        "--min-samples", type=int, default=0,
        help="Min samples per split (for quick test)",
    )
    parser.add_argument(
        "--quality-gate", action="store_true",
        help="Run quality gate on educator data; fail if pass rate < 90%%",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.poet_only:
        print("Loading educator examples...", flush=True)
        system = _educator_system()
        educator = (
            collect_educator_seed_examples(system)
            if args.interim_educator
            else collect_educator_examples(system)
        )
        if args.min_samples and len(educator) < args.min_samples:
            raise SystemExit(
                f"Need at least {args.min_samples} educator examples (got {len(educator)}). "
                + (
                    "Run generate_critiques_seed, generate_comparisons, "
                    "generate_revision_briefs first."
                    if args.interim_educator
                    else "Run generate_critiques_seed, generate_comparisons, "
                    "generate_revision_briefs, generate_briefs, generate_lessons, "
                    "generate_dialogues, generate_autopsies first."
                )
            )
        random.shuffle(educator)
        n_val = max(1, len(educator) // 10) if len(educator) >= 2 else 0
        train_edu = educator[:-n_val] if n_val else educator
        valid_edu = educator[-n_val:] if n_val else educator[:1]
        if args.min_samples:
            train_edu = train_edu[: args.min_samples]
            valid_edu = valid_edu[: max(1, min(len(valid_edu), args.min_samples // 5))]

        if args.quality_gate:
            from scripts.data_generation.quality_gate import check as quality_gate_check
            REJECTED_DIR = ROOT / "data" / "rejected"
            passed_edu, rejected_edu = [], []
            for ex in train_edu:
                assistant = next(
                    (m["content"] for m in ex.get("messages", []) if m["role"] == "assistant"),
                    "",
                )
                gate_entry = {"critique": assistant}
                ok, reasons = quality_gate_check(gate_entry)
                if ok:
                    passed_edu.append(ex)
                else:
                    ex["_reject_reasons"] = reasons
                    rejected_edu.append(ex)
            pass_rate = len(passed_edu) / len(train_edu) if train_edu else 1.0
            if pass_rate < 0.9:
                REJECTED_DIR.mkdir(parents=True, exist_ok=True)
                with open(REJECTED_DIR / "rejected_educator.jsonl", "w") as f:
                    for e in rejected_edu:
                        f.write(json.dumps(e) + "\n")
                raise SystemExit(
                    f"Quality gate failed: {len(passed_edu)}/{len(train_edu)} passed "
                    f"({pass_rate:.0%}). "
                    f"Need >= 90%%. Rejected examples in data/rejected/rejected_educator.jsonl"
                )
            train_edu = passed_edu

        EDUCATOR_TRAINING.mkdir(parents=True, exist_ok=True)
        with open(EDUCATOR_TRAINING / "train.jsonl", "w") as f:
            for ex in train_edu:
                f.write(json.dumps(ex) + "\n")
        with open(EDUCATOR_TRAINING / "valid.jsonl", "w") as f:
            for ex in valid_edu:
                f.write(json.dumps(ex) + "\n")
        suffix = " (interim seed only)" if args.interim_educator else ""
        print(f"Educator: {len(train_edu)} train, {len(valid_edu)} valid{suffix}")

    if not args.educator_only and not args.interim_educator:
        print("Loading poet examples...", flush=True)
        poet, n_rhyme = collect_poet_examples()
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
        n_general = len(poet) - n_rhyme
        print(f"Poet: {len(train_poet)} train, {len(valid_poet)} valid "
              f"(general: {n_general}, rhyme: {n_rhyme})")


if __name__ == "__main__":
    main()
