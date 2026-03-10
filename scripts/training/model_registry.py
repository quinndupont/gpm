#!/usr/bin/env python3
"""Single source of truth for model metadata. Loads config/model_registry.yaml."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "config" / "model_registry.yaml"
if not REGISTRY_PATH.exists():
    REGISTRY_PATH = Path("/config/model_registry.yaml")  # Modal image

# Union of common stop tokens for fallback when model unknown
DEFAULT_STOP_TOKENS = [
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|END_OF_TURN|>",
    "</s>",
]

_registry: list[dict] | None = None


def _load_registry() -> list[dict]:
    global _registry
    if _registry is None:
        import yaml
        with open(REGISTRY_PATH) as f:
            data = yaml.safe_load(f)
        _registry = data.get("models", [])
    return _registry


def short_to_hf(short_name: str) -> str | None:
    """Map short name (e.g. qwen2.5-7b) to HuggingFace ID."""
    for m in _load_registry():
        if m.get("short_name") == short_name:
            return m["hf_id"]
    return None


def hf_to_short(hf_id: str) -> str:
    """Map HuggingFace ID to short name for filenames."""
    for m in _load_registry():
        if m.get("hf_id") == hf_id:
            return m["short_name"]
    base = hf_id.split("/")[-1].lower().replace("_", "-")
    base = re.sub(r"-instruct$", "", base)
    base = re.sub(r"-chat$", "", base)
    return base


def stop_tokens_for(hf_id: str | None = None, short_name: str | None = None) -> list[str]:
    """Return stop tokens for a model. Prefer hf_id or short_name lookup; fallback to union."""
    if hf_id:
        for m in _load_registry():
            if m.get("hf_id") == hf_id:
                return m.get("stop_tokens", DEFAULT_STOP_TOKENS)
    if short_name:
        hf = short_to_hf(short_name)
        if hf:
            return stop_tokens_for(hf_id=hf)
    return DEFAULT_STOP_TOKENS


def get_model_entry(hf_id: str | None = None, short_name: str | None = None) -> dict | None:
    """Return full registry entry for a model, or None if not found."""
    if hf_id:
        for m in _load_registry():
            if m.get("hf_id") == hf_id:
                return m
    if short_name:
        hf = short_to_hf(short_name)
        if hf:
            return get_model_entry(hf_id=hf)
    return None


def all_short_names() -> list[str]:
    """Return all short names in the registry."""
    return [m["short_name"] for m in _load_registry() if m.get("short_name")]


def ollama_tag_to_short(tag: str) -> str | None:
    """Map Ollama model tag (e.g. qwen2.5:7b-instruct) to registry short_name."""
    tag_norm = tag.lower().replace(":", "-").replace("_", "-")
    for short in all_short_names():
        # qwen2.5-7b matches qwen2.5-7b-instruct (tag_norm)
        if short in tag_norm:
            return short
        parts = short.split("-")
        if len(parts) >= 2 and all(p in tag_norm for p in parts):
            return short
    if "command" in tag_norm or "cohere" in tag_norm:
        return "command-r7b"
    if "qwen" in tag_norm:
        return "qwen2.5-7b" if "7b" in tag_norm else ("qwen2.5-14b" if "14b" in tag_norm else "qwen2.5-32b")
    if "llama3.1" in tag_norm:
        return "llama3.1-8b" if "8b" in tag_norm else ("llama3.1-14b" if "14b" in tag_norm else "llama3.1-32b")
    if "llama3.2" in tag_norm:
        return "llama3.2-3b" if "3b" in tag_norm else "llama3.2-8b"
    return None
