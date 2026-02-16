#!/usr/bin/env python3
"""Generate educator training examples that teach approval/rejection behavior.

Two types of examples:
1. APPROVE: poem passes deterministic rhyme analysis → educator critique ends with
   "This poem has found its shape."
2. REJECT: poem fails rhyme analysis → educator critique names the broken rhymes
   and does NOT use the approval phrase.

This teaches the educator to gate approval on actual rhyme correctness.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import (
    call_claude,
    get_educator_system_prompt,
    CLAUDE_SONNET_4_5,
)
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme, format_analysis_for_prompt
from scripts.eval.form_registry import detect_form, is_rhyming_form, get_scheme, form_description

APPROVE_PROMPT = """Here is a poem that successfully follows its intended form.

Poem:
---
{poem}
---

Rhyme analysis (automated, deterministic):
{analysis}

Write a 3-5 sentence craft observation. Note what works well — specific rhyme pairs, line turns, imagery.
You MUST end your response with exactly this sentence: "This poem has found its shape."
Do not use this phrase anywhere else in your response."""

REJECT_PROMPT = """Here is a poem that was intended to follow a specific rhyming form but has rhyme failures.

Poem:
---
{poem}
---

Form requested: {form_desc}
Expected rhyme scheme: {expected_scheme}

Rhyme analysis (automated, deterministic):
{analysis}

Write a 3-5 sentence craft observation. Name the specific end-words that fail to rhyme and which lines they are on. Offer direction for how to fix them.
Do NOT end with "This poem has found its shape." — the rhymes are not correct yet."""


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate approval/rejection training examples")
    parser.add_argument("--strong-rhyme", type=Path, default=ANNOTATED / "strong_rhyme_poems.jsonl")
    parser.add_argument("--rhyme-critiques", type=Path, default=EDUCATOR_TRAINING / "rhyme_critiques.jsonl")
    parser.add_argument("--output", type=Path, default=EDUCATOR_TRAINING / "approval_examples.jsonl")
    parser.add_argument("--approve-count", type=int, default=80, help="Number of approval examples")
    parser.add_argument("--reject-count", type=int, default=40, help="Number of rejection examples")
    parser.add_argument("--model", type=str, default=CLAUDE_SONNET_4_5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    system = get_educator_system_prompt()
    poems = load_jsonl(args.strong_rhyme)
    if not poems:
        # Fallback: use rhyme_critiques (Claude-generated) when strong_rhyme is deleted
        poems = load_jsonl(args.rhyme_critiques)
        if poems:
            print("Using rhyme_critiques (run generate_rhyme_pairs first)")
    if not poems:
        raise SystemExit(
            f"No poems. Run generate_rhyme_pairs.py to create {args.rhyme_critiques}, "
            f"or provide {args.strong_rhyme}."
        )

    random.shuffle(poems)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_approve = 0
    total_reject = 0

    with open(args.output, "w" if args.replace else "a") as f:
        # APPROVE examples: poems with high strict_rhyme_density
        for rec in poems:
            if total_approve >= args.approve_count:
                break
            poem = rec.get("poem", "").strip()
            if not poem:
                continue
            analysis = analyze_rhyme(poem)
            if analysis.get("strict_rhyme_density", 0) < 0.7:
                continue

            analysis_text = format_analysis_for_prompt(analysis)
            prompt = APPROVE_PROMPT.format(poem=poem, analysis=analysis_text)

            try:
                critique = call_claude(prompt, system, model=args.model, max_tokens=300)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                continue

            # Verify the response actually ends with the approval phrase
            if not critique.strip().endswith("This poem has found its shape."):
                # Fix it: append the phrase
                critique = critique.strip() + "\n\nThis poem has found its shape."

            f.write(json.dumps({
                "type": "approval",
                "poem": poem,
                "critique": critique.strip(),
                "rhyme_analysis": {
                    "strict_rhyme_density": analysis["strict_rhyme_density"],
                    "rhyme_density": analysis["rhyme_density"],
                    "detected_scheme": analysis["detected_scheme"],
                },
            }) + "\n")
            total_approve += 1
            print(f"[approve {total_approve}/{args.approve_count}] density={analysis['strict_rhyme_density']}", flush=True)

        # REJECT examples: poems with moderate density (some rhymes but not all correct)
        for rec in poems:
            if total_reject >= args.reject_count:
                break
            poem = rec.get("poem", "").strip()
            if not poem:
                continue
            analysis = analyze_rhyme(poem)
            density = analysis.get("strict_rhyme_density", 0)
            # Target poems that partially rhyme but have gaps
            if density >= 0.7 or density < 0.2:
                continue
            # Need deviations or low density to make a convincing reject
            if not analysis.get("deviations") and density >= 0.5:
                continue

            analysis_text = format_analysis_for_prompt(analysis)
            form = detect_form(poem)
            form_desc = form_description(form) if form else "rhyming poem"
            scheme = get_scheme(form) if form else analysis.get("detected_scheme", "")

            prompt = REJECT_PROMPT.format(
                poem=poem,
                form_desc=form_desc,
                expected_scheme=scheme or "N/A",
                analysis=analysis_text,
            )

            try:
                critique = call_claude(prompt, system, model=args.model, max_tokens=300)
            except Exception as e:
                print(f"  Error: {e}", file=sys.stderr)
                continue

            # Verify the response does NOT end with approval
            if critique.strip().endswith("This poem has found its shape."):
                critique = critique.strip().rsplit("This poem has found its shape.", 1)[0].strip()

            f.write(json.dumps({
                "type": "rejection",
                "poem": poem,
                "critique": critique.strip(),
                "rhyme_analysis": {
                    "strict_rhyme_density": analysis["strict_rhyme_density"],
                    "rhyme_density": analysis["rhyme_density"],
                    "detected_scheme": analysis["detected_scheme"],
                },
            }) + "\n")
            total_reject += 1
            print(f"[reject {total_reject}/{args.reject_count}] density={density}", flush=True)

    print(f"\nDone: {total_approve} approval + {total_reject} rejection = {total_approve + total_reject} total")
    print(f"  {args.output}")


if __name__ == "__main__":
    main()
