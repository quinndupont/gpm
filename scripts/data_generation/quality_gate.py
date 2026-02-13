#!/usr/bin/env python3
"""S2.2 Quality gates — filter training examples before inclusion."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANTI_LLM = ROOT / "persona" / "anti_llm_isms.txt"

BANNED_PHRASES = [
    "delve into", "dive into", "unpack",
    "rich tapestry", "tapestry of",
    "it's worth noting", "it bears mentioning",
    "a testament to", "resonates deeply",
    "at its core", "strikes a balance", "in terms of",
    "great job!", "this is a solid effort",
    "nice use of imagery", "this poem resonates",
    "consider perhaps exploring",
]
RUBRIC_PATTERN = re.compile(r"\b\d+/10\b|\b\d+/\d+\b|bullet|checklist|•\s+[A-Z]")
GENERIC_VERBS = ("explores", "examines", "captures", "demonstrates", "highlights")


def fails_voice_consistency(text: str) -> bool:
    """Would the educator say this?"""
    return any(
        p in text.lower() for p in ["nice use of imagery", "great job!", "solid effort"]
    )


def fails_anti_rubric(text: str) -> bool:
    """Contains numbered scoring or checklist format?"""
    return bool(RUBRIC_PATTERN.search(text)) or "•" in text[:500]


def fails_llm_ism(text: str) -> bool:
    """Contains banned phrases?"""
    lower = text.lower()
    return any(p in lower for p in BANNED_PHRASES)


def fails_specificity(text: str) -> bool:
    """No line-specific references?"""
    return not re.search(r"(line \d+|stanza \d+|"\w[^"]{5,}")", text, re.I)


def check(entry: dict) -> tuple[bool, list[str]]:
    """Return (passes, list of failure reasons)."""
    text = entry.get("critique", entry.get("lesson", entry.get("response", "")))
    if not text:
        return False, ["empty response"]
    failures = []
    if fails_voice_consistency(text):
        failures.append("voice_consistency")
    if fails_anti_rubric(text):
        failures.append("anti_rubric")
    if fails_llm_ism(text):
        failures.append("llm_ism")
    if fails_specificity(text) and "critique" in str(entry).lower():
        failures.append("specificity")
    return len(failures) == 0, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reject-log", type=Path)
    args = parser.parse_args()

    passed = []
    rejected = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        ok, reasons = check(entry)
        if ok:
            passed.append(entry)
        else:
            entry["_reject_reasons"] = reasons
            rejected.append(entry)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for e in passed:
                f.write(json.dumps(e) + "\n")
    if args.reject_log:
        args.reject_log.parent.mkdir(parents=True, exist_ok=True)
        with open(args.reject_log, "w") as f:
            for e in rejected:
                f.write(json.dumps(e) + "\n")

    print(f"Passed: {len(passed)}, Rejected: {len(rejected)}")


if __name__ == "__main__":
    main()
