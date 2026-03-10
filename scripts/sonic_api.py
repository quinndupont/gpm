#!/usr/bin/env python3
"""FastAPI service for sonic (rhyme + phoneme) analysis of poems."""
import json
import re
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import pronouncing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import rhyme analyzer
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme

app = FastAPI(title="GPM Sonic API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VOWELS = frozenset(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()
)


def _clean_word(word: str) -> str:
    return word.strip().strip(string.punctuation).lower()


def _extract_lines(poem: str) -> list[str]:
    lines = []
    for line in poem.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if all(c in "-—–=* " for c in stripped):
            continue
        lines.append(stripped)
    return lines


def analyze_line_phonemes(line: str) -> dict:
    """Per-line phoneme analysis: vowel density, consonance, stress, syllables."""
    words = [_clean_word(w) for w in line.split() if _clean_word(w)]
    if not words:
        return {
            "phonemes": "",
            "vowelDensity": 0.0,
            "consonanceScore": 0.0,
            "stressPattern": "",
            "syllableCount": 0,
        }

    all_phones: list[str] = []
    stress_pattern: list[str] = []
    syllable_count = 0

    for word in words:
        phones_list = pronouncing.phones_for_word(word)
        if phones_list:
            phones = phones_list[0].split()
            all_phones.extend(phones)
            for p in phones:
                if p[-1].isdigit():
                    stress_pattern.append(p[-1])
            syllable_count += pronouncing.syllable_count(word)
        else:
            syllable_count += max(1, len(word) // 3)

    if not all_phones:
        return {
            "phonemes": "",
            "vowelDensity": 0.0,
            "consonanceScore": 0.0,
            "stressPattern": "",
            "syllableCount": syllable_count,
        }

    vowel_count = sum(1 for p in all_phones if p.rstrip("012") in VOWELS)
    vowel_density = vowel_count / len(all_phones)

    # Consonance: count repeated consonants (simplified)
    consonants = [p for p in all_phones if p.rstrip("012") not in VOWELS]
    consonance_score = 0.0
    if len(consonants) >= 2:
        seen: set[str] = set()
        for c in consonants:
            base = c.rstrip("012")
            if base in seen:
                consonance_score += 0.2
            seen.add(base)
        consonance_score = min(1.0, consonance_score)

    return {
        "phonemes": " ".join(all_phones),
        "vowelDensity": round(vowel_density, 3),
        "consonanceScore": round(consonance_score, 3),
        "stressPattern": "".join(stress_pattern),
        "syllableCount": syllable_count,
    }


class SonicRequest(BaseModel):
    poem: str


@app.post("/analyze/sonic")
def analyze_sonic(req: SonicRequest):
    poem = req.poem or ""
    rhyme = analyze_rhyme(poem)
    lines = _extract_lines(poem)
    line_analyses = [analyze_line_phonemes(line) for line in lines]

    return {
        "rhymeAnalysis": {
            "detectedScheme": rhyme.get("detected_scheme", ""),
            "strictDetectedScheme": rhyme.get("strict_detected_scheme", ""),
            "rhymeDensity": rhyme.get("rhyme_density", 0),
            "rhymePairs": rhyme.get("rhyme_pairs", []),
            "lineCount": rhyme.get("line_count", 0),
        },
        "lines": line_analyses,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
