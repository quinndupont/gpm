#!/usr/bin/env python3
"""S6.1 Voice consistency evaluation — post-training."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERSONA = ROOT / "persona" / "pedagogy_design_doc.md"
ANTI_LLM = ROOT / "persona" / "anti_llm_isms.txt"

BANNED = [
    "delve into", "dive into", "unpack", "rich tapestry", "it's worth noting",
    "a testament to", "resonates deeply", "at its core", "strikes a balance",
    "great job!", "solid effort", "nice use of imagery", "this poem resonates",
]


def check_persona_fidelity(output: str, persona: str) -> bool:
    """Simplified: would human recognize same character?"""
    return True  # Manual eval


def check_anti_llm(output: str) -> list[str]:
    """Find banned phrases in output."""
    found = [p for p in BANNED if p in output.lower()]
    return found


def check_specificity(output: str) -> int:
    """Count line-specific references."""
    import re
    return len(re.findall(r"line \d+|stanza \d+|"[^"]{10,}"", output, re.I))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+", help="JSONL with model outputs")
    args = parser.parse_args()

    total = 0
    llm_violations = 0
    for p in args.inputs:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            total += 1
            obj = json.loads(line)
            text = obj.get("output", obj.get("response", obj.get("content", "")))
            bad = check_anti_llm(text)
            if bad:
                llm_violations += 1
                print(f"LLM-ism in {p}: {bad}")

    print(f"Total: {total}, LLM violations: {llm_violations}")


if __name__ == "__main__":
    main()
