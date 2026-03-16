#!/usr/bin/env python3
"""Generate rhyming-form training data — briefs, poems, and rhyme-aware critiques.

Produces poet training pairs (brief + poem) for rhyming forms and
educator training examples (poem + rhyme critique).

Rhyme validation is deterministic via CMU dict; only the poem generation
and critique narration require LLM calls.

Supports both Claude API (Bedrock/Anthropic) and local models (Ollama).
"""
import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from models.prompts.loader import get_persona, render_prompt
from scripts.data_generation.claude_utils import (
    CLAUDE_OPUS_4_6,
    CLAUDE_SONNET_4_5,
    RAW_GOOD,
    call_claude,
    get_educator_system_prompt,
    load_requests,
)
from scripts.data_generation.rhyme_words import format_endword_constraint, pick_endwords
from scripts.eval.form_registry import (
    RHYMING_FORMS,
    form_description,
    get_line_count,
    get_scheme,
    is_metered_form,
    parse_scheme,
)
from scripts.eval.meter_analyzer import analyze as analyze_meter
from scripts.eval.meter_analyzer import (
    format_analysis_for_prompt as format_meter_for_prompt,
)
from scripts.eval.rhyme_analyzer import analyze, format_analysis_for_prompt

ROOT = Path(__file__).resolve().parents[2]
RHYME_TRAINING = ROOT / "data" / "rhyme_training"

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
# Ordered by complexity for curriculum learning: simple → complex
TARGET_FORMS = ["couplets", "quatrain", "limerick", "ballad", "tercets", "sonnet", "villanelle"]

# Local model presets (Ollama model names)
# Run `ollama list` to see available models
LOCAL_MODEL_PRESETS = {
    "llama3.1-8b": "llama3.1:8b",
    "llama3.1-8b-vanilla": "llama3.1:8b",
    "qwen2.5-7b": "qwen2.5:7b-instruct",
    "qwen3-8b": "qwen3:8b",
    "qwen3.5-9b": "qwen3.5-9b-claude-opus:9b-q4",
    "command-r7b": "command-r7b:7b",
}


@dataclass
class RunStats:
    """Track generation statistics for diagnostics."""
    total_generated: int = 0
    total_written: int = 0
    total_skipped: int = 0

    # Rhyme quality
    perfect_matches: int = 0
    scheme_failures: int = 0
    density_failures: int = 0

    # Per-form tracking
    form_stats: dict = field(default_factory=dict)

    # Timing
    total_tokens: int = 0
    total_gen_time: float = 0.0

    def add_result(self, form: str, matches_form: bool, density: float,
                   written: bool, tokens: int = 0, gen_time: float = 0.0):
        self.total_generated += 1
        if written:
            self.total_written += 1
        else:
            self.total_skipped += 1

        if matches_form:
            self.perfect_matches += 1
        elif matches_form is False:
            self.scheme_failures += 1

        self.total_tokens += tokens
        self.total_gen_time += gen_time

        if form not in self.form_stats:
            self.form_stats[form] = {
                "generated": 0, "written": 0, "matches_form": 0,
                "densities": [], "tokens": 0, "time": 0.0
            }
        fs = self.form_stats[form]
        fs["generated"] += 1
        if written:
            fs["written"] += 1
        if matches_form:
            fs["matches_form"] += 1
        fs["densities"].append(density)
        fs["tokens"] += tokens
        fs["time"] += gen_time

    def print_summary(self):
        """Print comprehensive run diagnostics."""
        print("\n" + "=" * 70)
        print("RUN DIAGNOSTICS")
        print("=" * 70)

        # Overall stats
        print(f"\n{'OVERALL STATS':^70}")
        print("-" * 70)
        print(f"  Total generated:  {self.total_generated}")
        print(f"  Written to file:  {self.total_written} ({self.total_written/max(1,self.total_generated):.1%})")
        print(f"  Skipped:          {self.total_skipped}")
        print(f"  Perfect schemes:  {self.perfect_matches} ({self.perfect_matches/max(1,self.total_generated):.1%})")
        print(f"  Scheme failures:  {self.scheme_failures}")

        # Performance metrics
        if self.total_gen_time > 0:
            print(f"\n{'PERFORMANCE':^70}")
            print("-" * 70)
            toks_per_sec = self.total_tokens / self.total_gen_time if self.total_gen_time > 0 else 0
            print(f"  Total tokens:     {self.total_tokens:,}")
            print(f"  Total time:       {self.total_gen_time:.1f}s")
            print(f"  Throughput:       {toks_per_sec:.1f} tok/s")

        # Per-form breakdown
        print(f"\n{'PER-FORM BREAKDOWN':^70}")
        print("-" * 70)
        print(f"  {'Form':<15} {'Gen':>6} {'Write':>6} {'Match%':>8} {'Density':>10} {'tok/s':>8}")
        print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*8}")

        for form, fs in sorted(self.form_stats.items()):
            gen = fs["generated"]
            written = fs["written"]
            match_pct = fs["matches_form"] / max(1, gen) * 100
            avg_density = sum(fs["densities"]) / max(1, len(fs["densities"]))
            toks = fs["tokens"] / max(0.01, fs["time"]) if fs["time"] > 0 else 0
            print(f"  {form:<15} {gen:>6} {written:>6} {match_pct:>7.1f}% {avg_density:>9.1%} {toks:>7.1f}")

        # Quality recommendation
        print(f"\n{'QUALITY ASSESSMENT':^70}")
        print("-" * 70)
        match_rate = self.perfect_matches / max(1, self.total_generated)
        if match_rate >= 0.8:
            print("  ✓ Excellent: 80%+ scheme match rate — high-quality training data")
        elif match_rate >= 0.6:
            print("  ⚠ Good: 60-80% scheme match — consider more retries or stricter filtering")
        else:
            print("  ✗ Poor: <60% scheme match — increase retries, check model capability")

        print("=" * 70 + "\n")


def call_local_model(
    user_message: str,
    system_message: str | None = None,
    model: str = "llama3.1:8b-instruct-q8_0",
    max_tokens: int = 1024,
    verbose: bool = False,
) -> tuple[str, int, float]:
    """Call local model via Ollama. Returns (text, token_count, gen_time)."""
    try:
        from ollama import chat
    except ImportError:
        raise RuntimeError("pip install ollama required for local models")

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    start = time.perf_counter()
    try:
        response = chat(
            model=model,
            messages=messages,
            options={"num_predict": max_tokens, "temperature": 0.7},
        )
        elapsed = time.perf_counter() - start

        # Extract response text
        if hasattr(response, "message"):
            text = getattr(response.message, "content", "") or ""
        else:
            text = response.get("message", {}).get("content", "") or ""

        # Extract token count from response if available
        tokens = 0
        if hasattr(response, "eval_count"):
            tokens = response.eval_count
        elif isinstance(response, dict):
            tokens = response.get("eval_count", 0) or len(text.split())
        else:
            tokens = len(text.split())  # Rough estimate

        return text, tokens, elapsed

    except Exception as e:
        raise RuntimeError(f"Ollama error ({model}): {e}") from e


def call_model(
    prompt: str,
    system: str,
    model: str,
    max_tokens: int,
    use_local: bool,
) -> tuple[str, int, float]:
    """Call model (local or Claude). Returns (text, tokens, time)."""
    if use_local:
        # Resolve model preset
        resolved_model = LOCAL_MODEL_PRESETS.get(model, model)
        return call_local_model(prompt, system, resolved_model, max_tokens)
    else:
        start = time.perf_counter()
        text = call_claude(prompt, system, model=model, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start
        tokens = len(text.split())  # Rough estimate for Claude
        return text, tokens, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Generate rhyming-form training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with local Llama 3.1 8B (recommended for high-quality rhymes)
  python generate_rhyme_pairs.py --local --poem-model llama3.1-8b-vanilla

  # Generate with Claude API (Bedrock)
  python generate_rhyme_pairs.py --poem-model claude-opus-4-6

  # Strict filtering for training data (recommended)
  python generate_rhyme_pairs.py --local --require-form-match --min-density 0.7 --retries 3

Quality Filtering Recommendations:
  For POET training data, use strict settings to ensure only correct examples:
    --require-form-match (default: True)
    --min-density 0.7 (reject poems with <70% strict rhyme density)
    --retries 3 (retry with fresh end-words on scheme failure)
        """,
    )
    parser.add_argument("--topics-per-form", type=int, default=10,
                        help="Number of topics per form")
    parser.add_argument("--poet-output", type=Path,
                        default=RHYME_TRAINING / "rhyme_pairs.jsonl")
    parser.add_argument("--educator-output", type=Path,
                        default=RHYME_TRAINING / "rhyme_critiques.jsonl")

    # Model selection
    parser.add_argument("--local", action="store_true",
                        help="Use local Ollama model instead of Claude API")
    parser.add_argument("--poem-model", type=str, default=None,
                        help="Model for poem generation. Local presets: llama3.1-8b-vanilla, qwen2.5-7b")
    parser.add_argument("--brief-model", type=str, default=CLAUDE_SONNET_4_5,
                        help="Model for brief generation (Claude API)")
    parser.add_argument("--critique-model", type=str, default=CLAUDE_SONNET_4_5,
                        help="Model for critique narration (Claude API)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace", action="store_true", help="Overwrite output files (default: append)"
    )
    parser.add_argument("--forms", type=str, nargs="*", default=TARGET_FORMS,
                        help="Which forms to generate for")
    parser.add_argument("--constrained", dest="constrained", action="store_true",
                        default=True, help="Use CMU-seeded end-words (default)")
    parser.add_argument("--no-constrained", dest="constrained", action="store_false",
                        help="Unconstrained generation")
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries with fresh end-words on rhyme fail (default: 3)")

    # Quality filtering
    parser.add_argument("--require-form-match", dest="require_form_match", action="store_true",
                        default=True, help="Only write poems that match expected rhyme scheme (default: True)")
    parser.add_argument("--no-require-form-match", dest="require_form_match", action="store_false",
                        help="Write all poems regardless of scheme match")
    parser.add_argument("--min-density", type=float, default=0.6,
                        help="Minimum strict rhyme density to include poem (default: 0.6)")

    args = parser.parse_args()

    # Set default poem model based on --local flag
    if args.poem_model is None:
        args.poem_model = "llama3.1-8b-vanilla" if args.local else CLAUDE_OPUS_4_6

    random.seed(args.seed)
    stats = RunStats()

    # Load topics from existing good poems, fall back to built-in list
    topics = load_requests(RAW_GOOD) if RAW_GOOD.exists() else []
    if not topics:
        topics = FALLBACK_TOPICS
    random.shuffle(topics)

    educator_system = get_educator_system_prompt()
    poet_system = get_persona("poet")

    args.poet_output.parent.mkdir(parents=True, exist_ok=True)
    args.educator_output.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if args.replace else "a"
    poet_f = open(args.poet_output, mode)
    edu_f = open(args.educator_output, mode)

    resolved_model = LOCAL_MODEL_PRESETS.get(args.poem_model, args.poem_model) if args.local else args.poem_model
    print(f"Generating rhyme pairs")
    print(f"  Mode: {'LOCAL' if args.local else 'CLAUDE API'}")
    print(f"  Model: {resolved_model}")
    print(f"  Forms: {', '.join(args.forms)}")
    print(f"  Topics per form: {args.topics_per_form}")
    print(f"  Quality gates: require_form_match={args.require_form_match}, min_density={args.min_density}")
    print()

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
            brief_msg = render_prompt(
                "tuning", "rhyme_pairs", template="brief",
                user_request=topic, form_desc=desc,
            )
            try:
                brief_model = args.poem_model if args.local else args.brief_model
                brief, _, _ = call_model(
                    brief_msg, educator_system,
                    brief_model, 500, args.local,
                )
            except Exception as e:
                print(f"  {label} Brief error: {e}", file=sys.stderr)
                continue
            if not brief.strip():
                continue

            # Step B: Generate poem (with optional retries in constrained mode)
            poem = ""
            rhyme_result = None
            gen_tokens = 0
            gen_time = 0.0
            brief_for_poet = brief

            if args.constrained and scheme:
                brief_for_poet = (
                    brief.rstrip()
                    + "\n\nEnd-words are fixed; write lines that conclude with the specified words."
                )

            for attempt in range(args.retries + 1):
                if args.constrained and scheme:
                    endwords = pick_endwords(scheme, seed=args.seed + i * 1000 + attempt)
                    endword_block = format_endword_constraint(scheme, endwords)
                    if not endword_block:
                        print(f"  {label} No end-words for scheme, skipping", file=sys.stderr)
                        break
                    line_count = get_line_count(form_name) or len(parse_scheme(scheme))
                    poem_msg = render_prompt(
                        "tuning", "rhyme_pairs", template="poet_constrained",
                        brief=brief_for_poet, endword_block=endword_block,
                        line_count=line_count,
                    )
                else:
                    poem_msg = render_prompt("tuning", "rhyme_pairs", template="poet", brief=brief)

                try:
                    poem, gen_tokens, gen_time = call_model(
                        poem_msg, poet_system, args.poem_model, 1024, args.local
                    )
                except Exception as e:
                    print(f"  {label} Poem error: {e}", file=sys.stderr)
                    break

                if not poem.strip():
                    break

                rhyme_result = analyze(poem, expected_form=form_name)

                # Check if we should retry
                if rhyme_result.get("matches_form"):
                    break  # Success!
                elif attempt < args.retries:
                    print(f"  {label} Attempt {attempt + 1} failed scheme, retrying...",
                          file=sys.stderr)

            if not poem.strip() or rhyme_result is None:
                continue

            # Step C: Validate with rhyme and meter analyzers
            analysis_parts = [format_analysis_for_prompt(rhyme_result)]

            meter_result = None
            if is_metered_form(form_name):
                meter_result = analyze_meter(poem, expected_form=form_name)
                analysis_parts.append(format_meter_for_prompt(meter_result))

            analysis_block = "\n\n".join(analysis_parts)

            # Check quality gates before writing poet training data
            matches_form = rhyme_result.get("matches_form")
            rhyme_density = rhyme_result.get("strict_rhyme_density", 0.0)

            should_write_poet = True
            skip_reason = None

            if args.require_form_match and matches_form is False:
                should_write_poet = False
                detected = rhyme_result.get("detected_scheme", "?")[:20]
                skip_reason = f"wrong scheme ({detected})"

            if rhyme_density < args.min_density:
                should_write_poet = False
                skip_reason = f"low density ({rhyme_density:.0%} < {args.min_density:.0%})"

            # Track stats
            stats.add_result(
                form_name, matches_form, rhyme_density,
                should_write_poet, gen_tokens, gen_time
            )

            if should_write_poet:
                poet_f.write(json.dumps({
                    "brief": brief,
                    "poem": poem,
                    "user_request": topic,
                    "form": form_name,
                }) + "\n")
            else:
                print(f"  {label} Skipping poet pair: {skip_reason}", file=sys.stderr)

            # Step D: Generate educator critique
            critique_msg = render_prompt(
                "tuning", "rhyme_pairs", template="critique",
                poem=poem, form_name=desc, expected_scheme=scheme,
                analysis_block=analysis_block,
            )
            try:
                critique_model = args.poem_model if args.local else args.critique_model
                critique, _, _ = call_model(
                    critique_msg, educator_system,
                    critique_model, 400, args.local,
                )
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

            # Print per-item status
            status = "pass" if matches_form else ("fail" if matches_form is False else "n/a")
            written_status = "WROTE" if should_write_poet else "SKIP"
            meter_status = ""
            if meter_result:
                m = meter_result.get("matches_meter")
                m_cons = meter_result["consistency"]
                meter_status = f" | meter={'pass' if m else 'fail'} ({m_cons:.0%})"
            toks_sec = gen_tokens / max(0.01, gen_time)
            print(
                f"  {label} [{written_status}] Rhyme: {status} | "
                f"density={rhyme_result['strict_rhyme_density']:.0%}{meter_status} | "
                f"{toks_sec:.0f} tok/s",
                flush=True,
            )

    poet_f.close()
    edu_f.close()

    print(f"\nOutput files:")
    print(f"  Poet:     {args.poet_output}")
    print(f"  Educator: {args.educator_output}")

    # Print run diagnostics
    stats.print_summary()


if __name__ == "__main__":
    main()
