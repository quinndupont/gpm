#!/usr/bin/env python3
"""
Clean reasoning text from poems and re-run rhyme analysis.
"""
import json
import re
from pathlib import Path
from collections import defaultdict
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme

DATA_DIR = Path("data/rhyme_bench")


def extract_poem(text: str) -> str:
    """Extract the actual poem from text containing reasoning."""
    # Remove <reasoning> tags and their content
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    text = re.sub(r'<analysis>.*?</analysis>', '', text, flags=re.DOTALL)

    # Remove common prefix patterns
    text = re.sub(r'^(Here\'s|Let me write|I\'ll write|This (?:is|would be)|I need to write).*?:\s*\n', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove bold headers like "**Poem:**" or "**Sonnet:**"
    text = re.sub(r'\*\*[^*]+\*\*:\s*\n', '', text)

    # Remove "Now critique" and everything after it (revision instructions)
    text = re.sub(r'\n\nNow (?:critique|revise|improve).*', '', text, flags=re.IGNORECASE)

    # Remove "Wait, but" and reasoning continuations
    text = re.sub(r'\n\nWait,.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n\nActually,.*$', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove trailing explanation text (common pattern)
    text = re.sub(r'\n\n(?:The (?:first|poem|sonnet|verse)|This (?:follows|meets|adheres)).*$', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Clean up multiple blank lines
    text = re.sub(r'\n\n\n+', '\n\n', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def has_reasoning(text: str) -> bool:
    """Check if text contains reasoning patterns."""
    patterns = [
        r'<reasoning>',
        r'<thinking>',
        r'<analysis>',
        r'^Let me (?:write|create|generate)',
        r'^I (?:need to |\'ll |would )?(?:write|create|generate)',
        r'^Here\'s',
        r'^\*\*[^*]+\*\*:',
        r'\n\nNow (?:critique|revise|improve)',
        r'\n\nWait,',
        r'\n\nActually,',
    ]

    for pattern in patterns:
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            return True
    return False


def main():
    json_files = sorted(DATA_DIR.glob("rhyme_*.json"))

    affected_by_model = defaultdict(list)
    cleaned_count = 0
    analysis_changed = 0

    print(f"Scanning {len(json_files)} poems...\n")

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        poem = data.get("final_poem", "")
        if not poem:
            continue

        model = data.get("model_id", "unknown")

        if has_reasoning(poem):
            # Extract clean poem
            clean_poem = extract_poem(poem)
            affected_by_model[model].append({
                "file": json_file.name,
                "original": poem,
                "cleaned": clean_poem,
                "data": data
            })

            # Re-analyze the cleaned poem
            old_analysis = data.get("rhyme_analysis", {})
            form = data.get("form", "unknown")
            variant = data.get("variant")

            new_analysis = analyze_rhyme(clean_poem, expected_form=form, expected_variant=variant)
            new_analysis_dict = {
                "strict_rhyme_density": new_analysis.get("strict_rhyme_density", 0),
                "rhyme_density": new_analysis.get("rhyme_density", 0),
                "matches_form": new_analysis.get("matches_form"),
                "deviations_count": len(new_analysis.get("deviations", [])),
                "strict_rhyme_pairs": len(new_analysis.get("strict_rhyme_pairs", [])),
                "rhyme_pairs": len(new_analysis.get("rhyme_pairs", [])),
                "line_count": new_analysis.get("line_count", 0),
                "detected_scheme": new_analysis.get("detected_scheme", ""),
                "expected_scheme": new_analysis.get("expected_scheme"),
            }

            # Update the JSON file
            data["final_poem"] = clean_poem
            data["rhyme_analysis"] = new_analysis_dict
            data["_cleaned"] = True  # Mark as cleaned

            with open(json_file, "w") as f:
                json.dump(data, f, indent=2)

            cleaned_count += 1

            # Check if analysis changed
            if old_analysis != new_analysis_dict:
                analysis_changed += 1

    # Print summary
    print(f"\n✓ Cleaning Complete")
    print(f"=" * 60)
    print(f"Total poems cleaned: {cleaned_count}")
    print(f"Analyses updated: {analysis_changed}\n")

    if affected_by_model:
        print("Breakdown by model:")
        for model in sorted(affected_by_model.keys(), key=lambda m: -len(affected_by_model[m])):
            count = len(affected_by_model[model])
            print(f"  {model}: {count}")

        # Show samples
        print(f"\n\nSample cleaning (before/after):")
        for model in list(affected_by_model.keys())[:2]:
            item = affected_by_model[model][0]
            print(f"\n{model} - {item['file']}")
            print("-" * 60)
            print("BEFORE (first 250 chars):")
            print(item['original'][:250])
            print("\nAFTER:")
            print(item['cleaned'][:250])
            print("-" * 60)


if __name__ == "__main__":
    main()
