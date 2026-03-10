#!/usr/bin/env python3
"""Rule-based and optional LLM checks for poem records."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Poem-specific boilerplate (in body or title)
BOILERPLATE_PHRASES = [
    "about this poem",
    "submitted by",
    "copyright",
    "all rights reserved",
    "©",
    "published in",
    "first appeared in",
    "reprinted with permission",
]

# Placeholder author/title values
UNKNOWN_AUTHOR = frozenset({"unknown", "anonymous", "n/a", "na", ""})
UNKNOWN_TITLE = frozenset({"unknown", "untitled", "n/a", "na", ""})


def check_record(
    record: dict,
    min_lines: int = 2,
    max_lines: int = 500,
    min_chars: int = 20,
    max_chars: int = 100_000,
) -> list[str]:
    """Return list of flag names for this record."""
    flags = []
    author = (record.get("author") or "").strip()
    title = (record.get("title") or "").strip()
    poem = record.get("poem", "")

    if not title:
        flags.append("missing_title")
    elif title.lower() in UNKNOWN_TITLE:
        flags.append("unknown_title")

    if not author:
        flags.append("missing_author")
    elif author.lower() in UNKNOWN_AUTHOR:
        flags.append("unknown_author")

    line_count = record.get("line_count", 0)
    char_count = record.get("char_count", len(poem))
    if line_count < min_lines:
        flags.append("too_short")
    elif line_count > max_lines:
        flags.append("too_long")
    if char_count < min_chars:
        flags.append("too_short")
    elif char_count > max_chars:
        flags.append("too_long")

    lower_poem = poem.lower()
    lower_title = title.lower()
    for phrase in BOILERPLATE_PHRASES:
        if phrase in lower_poem or phrase in lower_title:
            flags.append("boilerplate")
            break

    return flags


def check_corpus(
    records: list[dict],
    min_lines: int = 2,
    max_lines: int = 500,
    min_chars: int = 20,
    max_chars: int = 100_000,
) -> list[tuple[dict, list[str]]]:
    """Return list of (record, flags) for each record."""
    return [
        (r, check_record(r, min_lines, max_lines, min_chars, max_chars))
        for r in records
    ]


def llm_check_placeholder(
    model_path: str | Path,
    title: str,
    author: str,
    max_tokens: int = 32,
) -> bool:
    """
    Use local GGUF to classify if title/author is placeholder/boilerplate.
    Returns True if placeholder detected.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("pip install llama-cpp-python")

    system = "Reply with exactly YES or NO. Is the given poem title or author a placeholder, boilerplate, or generic (e.g. Unknown, Anonymous, Untitled)?"
    user = f"Title: {title}\nAuthor: {author}"
    llm = Llama(
        model_path=str(model_path),
        n_ctx=512,
        n_gpu_layers=-1,
        n_threads=4,
        use_mmap=True,
        verbose=False,
    )
    r = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>", "\n"],
    )
    text = (r["choices"][0]["message"]["content"] or "").strip().upper()
    return "YES" in text[:10]
