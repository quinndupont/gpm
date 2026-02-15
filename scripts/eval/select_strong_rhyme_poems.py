#!/usr/bin/env python3
"""Run deterministic rhyme analysis on all good poems; save those with strong rhyme schemes for fine-tuning."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data_generation.claude_utils import load_poems, poem_text, RAW_GOOD
from scripts.eval.rhyme_analyzer import analyze

ANNOTATED = ROOT / "data" / "annotated"
DEFAULT_OUT = ANNOTATED / "strong_rhyme_poems.jsonl"


def _scheme_has_repetition(detected_scheme: str) -> bool:
    """True if scheme has repeated labels (not every line unique)."""
    flat = detected_scheme.replace(" ", "")
    return len(flat) >= 2 and len(set(flat)) < len(flat)


def is_strong_rhyme(
    analysis: dict,
    min_strict_density: float = 0.85,
    min_lines: int = 6,
    min_perfect_pairs: int = 3,
) -> tuple[bool, str | None]:
    """Poem has a strong rhyme scheme: CMU-verified perfect rhymes, high density, clear pattern.
    Returns (passed, rejection_reason)."""
    if analysis["line_count"] < min_lines:
        return False, f"line_count={analysis['line_count']}<{min_lines}"
    strict_density = analysis.get("strict_rhyme_density", 0.0)
    if strict_density < min_strict_density:
        return False, f"strict_density={strict_density:.2f}<{min_strict_density}"
    strict_pairs = analysis.get("strict_rhyme_pairs", [])
    if len(strict_pairs) < min_perfect_pairs:
        return False, f"strict_pairs={len(strict_pairs)}<{min_perfect_pairs}"
    strict_scheme = analysis.get("strict_detected_scheme", "")
    if not _scheme_has_repetition(strict_scheme):
        return False, f"no_scheme_repetition ({strict_scheme})"
    return True, None


def main():
    parser = argparse.ArgumentParser(description="Select good poems with strong rhyme schemes for fine-tuning")
    parser.add_argument("--input", type=Path, default=RAW_GOOD, help="Directory of good poems")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT, help="Output JSONL path")
    parser.add_argument("--min-strict-density", type=float, default=0.85, help="Min strict rhyme density (CMU perfect only, 0–1)")
    parser.add_argument("--min-lines", type=int, default=6, help="Min line count")
    parser.add_argument("--min-perfect-pairs", type=int, default=3, help="Min CMU-verified perfect rhyme pairs")
    parser.add_argument("--all", action="store_true", help="Write all poems with their analysis; only strong ones get strong_rhyme=True")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress and rejection reasons")
    parser.add_argument("--progress", type=int, default=5000, help="Print progress every N poems (0=off)")
    args = parser.parse_args()

    poems = load_poems(args.input)
    if not poems:
        print("No poems found in", args.input, file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(poems)} poems from {args.input}", file=sys.stderr)
    print(f"Criteria: strict_density>={args.min_strict_density}, min_lines>={args.min_lines}, min_pairs>={args.min_perfect_pairs}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    strong_count = 0
    perfect_count = 0
    reject_reasons: dict[str, int] = {}
    empty_count = 0

    with open(args.output, "w") as f:
        for i, rec in enumerate(poems):
            if args.progress and (i + 1) % args.progress == 0:
                print(f"  ... {i + 1}/{len(poems)} ({strong_count} strong so far)", file=sys.stderr)

            text = poem_text(rec)
            if not text.strip():
                empty_count += 1
                continue
            analysis = analyze(text)
            strong, reason = is_strong_rhyme(
                analysis,
                min_strict_density=args.min_strict_density,
                min_lines=args.min_lines,
                min_perfect_pairs=args.min_perfect_pairs,
            )
            if strong:
                strong_count += 1
                if args.verbose:
                    title = rec.get("title", "(untitled)")
                    print(f"  PASS {title}: strict={analysis.get('strict_rhyme_density', 0):.2f} pairs={len(analysis.get('strict_rhyme_pairs', []))} scheme={analysis.get('strict_detected_scheme', '')}", file=sys.stderr)
            else:
                key = (reason or "unknown").split("=")[0].split("(")[0].strip() or "unknown"
                reject_reasons[key] = reject_reasons.get(key, 0) + 1
                if args.verbose and strong_count + sum(reject_reasons.values()) <= 20:
                    print(f"  REJECT {rec.get('title', '(untitled)')}: {reason}", file=sys.stderr)

            if analysis.get("matches_form") is True:
                perfect_count += 1

            if strong or args.all:
                out = {
                    "author": rec.get("author", ""),
                    "title": rec.get("title", ""),
                    "poem": text,
                    "rhyme_analysis": {
                        "detected_scheme": analysis["detected_scheme"],
                        "strict_detected_scheme": analysis.get("strict_detected_scheme", ""),
                        "rhyme_density": analysis["rhyme_density"],
                        "strict_rhyme_density": analysis.get("strict_rhyme_density", 0),
                        "line_count": analysis["line_count"],
                        "rhyme_pairs_count": len(analysis["rhyme_pairs"]),
                        "strict_rhyme_pairs_count": len(analysis.get("strict_rhyme_pairs", [])),
                    },
                    "strong_rhyme": strong,
                }
                if analysis.get("expected_scheme"):
                    out["rhyme_analysis"]["expected_scheme"] = analysis["expected_scheme"]
                    out["rhyme_analysis"]["matches_form"] = analysis["matches_form"]
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

    processed = len(poems) - empty_count
    print(f"Processed {processed} poems → {strong_count} strong rhyme (saved to {args.output})")
    if empty_count:
        print(f"  Skipped {empty_count} empty poems", file=sys.stderr)
    if perfect_count:
        print(f"  {perfect_count} matched an expected form (when form was provided)", file=sys.stderr)
    if reject_reasons:
        print("  Rejection breakdown:", file=sys.stderr)
        for k, v in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
