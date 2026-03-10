#!/usr/bin/env python3
"""Load poem data from all corpus sources and normalize to unified records."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_GOOD = ROOT / "data" / "raw" / "good"
RAW_BAD = ROOT / "data" / "raw" / "bad"
ANNOTATED = ROOT / "data" / "annotated"
POET_TRAINING = ROOT / "data" / "poet_training"
REV_FLUX = ROOT / "data" / "rev_flux"

SOURCE_TYPE_GOOD = "good"
SOURCE_TYPE_BAD = "bad"
SOURCE_TYPE_ANNOTATED = "annotated"
SOURCE_TYPE_SYNTHETIC = "synthetic"


def _poem_text(obj: dict) -> str:
    return obj.get("poem", obj.get("text", obj.get("content", ""))) or ""


def _line_count(text: str) -> int:
    return len([l for l in text.splitlines() if l.strip()]) if text else 0


def _load_raw(source_type: str, directory: Path) -> list[dict]:
    """Load from raw good/bad via claude_utils."""
    sys_path = list(__import__("sys").path)
    __import__("sys").path.insert(0, str(ROOT))
    from scripts.data_generation.claude_utils import load_poems
    __import__("sys").path[:] = sys_path

    poems = load_poems(directory)
    records = []
    for p in poems:
        text = _poem_text(p)
        if not text.strip():
            continue
        records.append({
            "source_path": str(directory),
            "source_type": source_type,
            "author": p.get("author", ""),
            "title": p.get("title", ""),
            "poem": text,
            "line_count": _line_count(text),
            "char_count": len(text),
        })
    return records


def _load_annotated(path: Path) -> list[dict]:
    """Load critiques_seed.jsonl and similar; extract poem from nested structure."""
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        poem_obj = obj.get("poem")
        if not poem_obj or not isinstance(poem_obj, dict):
            continue
        text = _poem_text(poem_obj)
        if not text.strip():
            continue
        source = obj.get("source", "")
        source_type = SOURCE_TYPE_ANNOTATED
        if source in ("good", "bad"):
            source_type = source
        records.append({
            "source_path": str(path),
            "source_type": source_type,
            "author": poem_obj.get("author", ""),
            "title": poem_obj.get("title", ""),
            "poem": text,
            "line_count": _line_count(text),
            "char_count": len(text),
        })
    return records


def _load_poet_training(path: Path) -> list[dict]:
    """Load pairs.jsonl, train.jsonl, valid.jsonl; poem only. Author = poet_training:<stem> for source distinction."""
    records = []
    if not path.exists():
        return records
    stem = path.stem
    author = f"poet_training:{stem}"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = obj.get("poem", "")
        if not text.strip():
            continue
        title = obj.get("user_request", obj.get("brief", ""))[:80] if obj.get("user_request") or obj.get("brief") else ""
        records.append({
            "source_path": str(path),
            "source_type": SOURCE_TYPE_SYNTHETIC,
            "author": author,
            "title": title,
            "poem": text,
            "line_count": _line_count(text),
            "char_count": len(text),
        })
    return records


def _load_rev_flux(directory: Path) -> list[dict]:
    """Load rev_flux/*.json final_poem outputs. Author = model_id for source distinction."""
    records = []
    if not directory.exists():
        return records
    for p in directory.glob("*.json"):
        if p.name == "summary.json":
            continue
        try:
            obj = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        text = obj.get("final_poem", "")
        if not text.strip():
            continue
        author = obj.get("model_id") or obj.get("metadata", {}).get("model_poet", "rev_flux:unknown")
        category = obj.get("category", "")
        prompt_idx = obj.get("prompt_idx", "")
        title = f"{category}_{prompt_idx}" if category or prompt_idx != "" else ""
        records.append({
            "source_path": str(p),
            "source_type": SOURCE_TYPE_SYNTHETIC,
            "author": author,
            "title": title,
            "poem": text,
            "line_count": _line_count(text),
            "char_count": len(text),
        })
    return records


def load_corpus(
    sources: list[str] | None = None,
    include_rev_flux: bool = True,
) -> list[dict]:
    """
    Load all poem data into unified records.
    sources: optional list of 'raw_good','raw_bad','annotated','poet_training','rev_flux'.
    If None, load all except rev_flux unless include_rev_flux=True.
    """
    all_sources = sources or ["raw_good", "raw_bad", "annotated", "poet_training"]
    if include_rev_flux:
        all_sources = list(all_sources) + ["rev_flux"]

    records = []
    if "raw_good" in all_sources:
        records.extend(_load_raw(SOURCE_TYPE_GOOD, RAW_GOOD))
    if "raw_bad" in all_sources:
        records.extend(_load_raw(SOURCE_TYPE_BAD, RAW_BAD))
    if "annotated" in all_sources:
        records.extend(_load_annotated(ANNOTATED / "critiques_seed.jsonl"))
    if "poet_training" in all_sources:
        for name in ("pairs.jsonl", "train.jsonl", "valid.jsonl", "rhyme_pairs.jsonl"):
            records.extend(_load_poet_training(POET_TRAINING / name))
    if "rev_flux" in all_sources:
        records.extend(_load_rev_flux(REV_FLUX))
    return records
