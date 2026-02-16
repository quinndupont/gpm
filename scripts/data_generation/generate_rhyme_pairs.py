#!/usr/bin/env python3
"""Generate rhyming-form training data — briefs, poems, and rhyme-aware critiques.

Produces poet training pairs (brief + poem) for rhyming forms and
educator training examples (poem + rhyme critique).

Rhyme validation is deterministic via CMU dict; only the poem generation
and critique narration require LLM calls.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POET_TRAINING = ROOT / "data" / "poet_training"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import (
    call_claude,
    get_educator_system_prompt,
    CLAUDE_OPUS_4_6,
    CLAUDE_SONNET_4_5,
    load_requests,
    RAW_GOOD,
)
from scripts.eval.form_registry import (
    FORMS,
    RHYMING_FORMS,
    form_description,
    get_scheme,
    is_metered_form,
)
from scripts.eval.rhyme_analyzer import analyze, format_analysis_for_prompt
from scripts.eval.meter_analyzer import (
    analyze as analyze_meter,
    format_analysis_for_prompt as format_meter_for_prompt,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BRIEF_PROMPT = """A student has asked for help writing a poem. Their request:
"{user_request}"

Construct a COMPACT generation brief (~300 tokens max). Include:
1. Angle — 2-3 sentences, not the obvious approach
2. Clichés to avoid — 5-6 specific phrases/images for this topic
3. Imagery domain — 1-2 sentences, unexpected
4. Form guidance — the poem MUST be a {form_desc}. State the form, line count, and rhyme scheme explicitly.

No rhetorical flourish. Actionable only."""

POET_SYSTEM = """You are a poet. You receive generation briefs and write poems.
You never output instructions, critique, or analysis — only poems."""

POET_PROMPT = """Generation brief:

---
{brief}
---

Write the poem. Output ONLY the poem — no title unless it's part of the poem, no commentary.
Follow the specified form and rhyme scheme precisely."""

CRITIQUE_PROMPT = """Here is a poem written in a specific form, along with automated analysis.

Poem:
---
{poem}
---

Form requested: {form_name}
Expected rhyme scheme: {expected_scheme}

{analysis_block}

Write a 3-5 sentence craft observation about how this poem handles (or fails to handle) its form. Cover rhyme and meter where applicable. Be specific — name the words and lines. Note where rhymes land well, where meter holds or breaks for emphasis, and where control slips. No scores, no rubrics."""

# ---------------------------------------------------------------------------
# Topics — used when no external request source is available
# ---------------------------------------------------------------------------

FALLBACK_TOPICS = [
    "a childhood kitchen",
    "watching someone sleep",
    "the last day of winter",
    "a letter never sent",
    "the sound of a train at night",
    "losing a language",
    "a dog who waits",
    "the smell of rain on concrete",
    "a broken clock",
    "first light through hospital blinds",
    "a musician's hands",
    "the weight of a key",
    "swimming in the dark",
    "my grandfather's tools",
    "the color blue",
]

# Forms to generate training data for (skip free_verse)
TARGET_FORMS = ["sonnet", "villanelle", "limerick", "couplets", "tercets", "ballad"]


def main():
    parser = argparse.ArgumentParser(description="Generate rhyming-form training data")
    parser.add_argument("--topics-per-form", type=int, default=10,
                        help="Number of topics per form")
    parser.add_argument("--poet-output", type=Path,
                        default=POET_TRAINING / "rhyme_pairs.jsonl")
    parser.add_argument("--educator-output", type=Path,
                        default=EDUCATOR_TRAINING / "rhyme_critiques.jsonl")
    parser.add_argument("--brief-model", type=str, default=CLAUDE_SONNET_4_5,
                        help="Model for brief generation (cheap)")
    parser.add_argument("--poem-model", type=str, default=CLAUDE_OPUS_4_6,
                        help="Model for poem generation (needs quality)")
    parser.add_argument("--critique-model", type=str, default=CLAUDE_SONNET_4_5,
                        help="Model for critique narration (cheap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true", help="Overwrite output files (default: append)")
    parser.add_argument("--forms", type=str, nargs="*", default=TARGET_FORMS,
                        help="Which forms to generate for")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load topics from existing good poems, fall back to built-in list
    topics = load_requests(RAW_GOOD) if RAW_GOOD.exists() else []
    if not topics:
        topics = FALLBACK_TOPICS
    random.shuffle(topics)

    educator_system = get_educator_system_prompt()

    args.poet_output.parent.mkdir(parents=True, exist_ok=True)
    args.educator_output.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.replace else "a"
    poet_f = open(args.poet_output, mode)
    edu_f = open(args.educator_output, mode)

    total = 0
    for form_name in args.forms:
        if form_name not in RHYMING_FORMS:
            print(f"Skipping {form_name} (not a rhyming form)", file=sys.stderr)
            continue

        desc = form_description(form_name)
        scheme = get_scheme(form_name)
        form_topics = topics[: args.topics_per_form]

        for i, topic in enumerate(form_topics):
            label = f"[{form_name} {i + 1}/{len(form_topics)}]"
            print(f"{label} Topic: {topic[:50]}...", flush=True)

            # Step A: Generate brief
            brief_msg = BRIEF_PROMPT.format(
                user_request=topic, form_desc=desc,
            )
            try:
                brief = call_claude(brief_msg, educator_system,
                                    model=args.brief_model, max_tokens=500)
            except Exception as e:
                print(f"  {label} Brief error: {e}", file=sys.stderr)
                continue
            if not brief.strip():
                continue

            # Step B: Generate poem
            poem_msg = POET_PROMPT.format(brief=brief)
            try:
                poem = call_claude(poem_msg, POET_SYSTEM,
                                   model=args.poem_model, max_tokens=1024)
            except Exception as e:
                print(f"  {label} Poem error: {e}", file=sys.stderr)
                continue
            if not poem.strip():
                continue

            # Step C: Validate with rhyme and meter analyzers
            rhyme_result = analyze(poem, expected_form=form_name)
            analysis_parts = [format_analysis_for_prompt(rhyme_result)]

            meter_result = None
            if is_metered_form(form_name):
                meter_result = analyze_meter(poem, expected_form=form_name)
                analysis_parts.append(format_meter_for_prompt(meter_result))

            analysis_block = "\n\n".join(analysis_parts)

            # Write poet pair regardless of validation (the model needs
            # to see the form constraint even if execution isn't perfect)
            poet_f.write(json.dumps({
                "brief": brief,
                "poem": poem,
                "user_request": topic,
                "form": form_name,
            }) + "\n")

            # Step D: Generate educator critique informed by analyzers
            critique_msg = CRITIQUE_PROMPT.format(
                poem=poem,
                form_name=desc,
                expected_scheme=scheme,
                analysis_block=analysis_block,
            )
            try:
                critique = call_claude(critique_msg, educator_system,
                                       model=args.critique_model, max_tokens=400)
            except Exception as e:
                print(f"  {label} Critique error: {e}", file=sys.stderr)
                critique = ""

            if critique.strip():
                edu_entry = {
                    "poem": poem,
                    "form": form_name,
                    "expected_scheme": scheme,
                    "rhyme_analysis": rhyme_result,
                    "critique": critique,
                }
                if meter_result:
                    edu_entry["meter_analysis"] = {
                        "dominant_foot": meter_result.get("dominant_foot_name"),
                        "consistency": meter_result.get("consistency"),
                        "matches_meter": meter_result.get("matches_meter"),
                        "variation_rate": meter_result.get("variation_rate"),
                    }
                edu_f.write(json.dumps(edu_entry) + "\n")

            total += 1
            matches = rhyme_result.get("matches_form")
            status = "pass" if matches else ("fail" if matches is False else "n/a")
            meter_status = ""
            if meter_result:
                m = meter_result.get("matches_meter")
                meter_status = f" | meter={'pass' if m else 'fail'} ({meter_result['consistency']:.0%})"
            print(f"  {label} Rhyme: {status} | density={rhyme_result['rhyme_density']:.0%}{meter_status}",
                  flush=True)

    poet_f.close()
    edu_f.close()
    print(f"\nDone: {total} pairs")
    print(f"  Poet:     {args.poet_output}")
    print(f"  Educator: {args.educator_output}")


if __name__ == "__main__":
    main()
