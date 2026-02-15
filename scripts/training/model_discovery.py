#!/usr/bin/env python3
"""Discover trained models (educator, poet) on disk and infer base model."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
ADAPTERS_DIR = ROOT / "adapters"

# Map short names (from filenames or GGUF basename) to HuggingFace IDs
SHORT_NAME_TO_HF: dict[str, str] = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-14b": "meta-llama/Llama-3.1-14B-Instruct",
    "llama3.1-32b": "meta-llama/Llama-3.1-32B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2-8b": "meta-llama/Llama-3.2-8B-Instruct",
    "deepseek-v2-lite-7b": "deepseek-ai/DeepSeek-V2-Lite-7B-Instruct",
    "deepseek-v2-lite-16b": "deepseek-ai/DeepSeek-V2-Lite-16B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "glm4-9b": "THUDM/glm-4-9b-chat-hf",
}


# Reverse: HuggingFace ID to short name for filename generation
def _hf_to_short_name(hf_id: str) -> str:
    for short, hf in SHORT_NAME_TO_HF.items():
        if hf == hf_id:
            return short
    # Fallback: derive from HF path (e.g. Qwen/Qwen2.5-7B-Instruct -> qwen2.5-7b)
    base = hf_id.split("/")[-1].lower()
    base = re.sub(r"-instruct$", "", base)
    base = re.sub(r"-chat$", "", base)
    return base.replace("_", "-")


@dataclass
class DiscoveredModel:
    path: str
    task: str
    base_model_hf_id: str
    quant: str | None
    source: Literal["local_gguf", "local_adapter", "modal_checkpoint"]


def _parse_gguf_filename(name: str) -> tuple[str | None, str | None, str | None]:
    """Parse {base}-{task}-{quant}.gguf. Returns (base, task, quant) or (None, None, None)."""
    if not name.endswith(".gguf"):
        return None, None, None
    stem = name[:-5]
    # Quant patterns: Q4_K_M, Q5_K_M, Q3_K_M, Q4_K_S, etc.
    quant_match = re.search(r"-(Q[0-9]_K_[A-Z]+)$", stem, re.IGNORECASE)
    quant = quant_match.group(1) if quant_match else None
    if quant:
        stem = stem[: quant_match.start()]
    parts = stem.split("-")
    if len(parts) < 2:
        return None, None, None
    # Task: educator, poet, poet_rhyme, or educator-interim
    if len(parts) >= 3 and parts[-2] == "educator" and parts[-1] == "interim":
        task = "educator-interim"
        base = "-".join(parts[:-2])
    else:
        task = parts[-1]
        base = "-".join(parts[:-1])
    return base, task, quant


def _infer_hf_from_short(base: str) -> str:
    """Map short base name to HuggingFace ID."""
    base_lower = base.lower()
    for short, hf in SHORT_NAME_TO_HF.items():
        if short in base_lower or base_lower in short:
            return hf
    return base


def _read_gguf_metadata(path: Path) -> dict[str, str]:
    """Read GGUF metadata if gguf package available."""

    try:
        import gguf
    except ImportError:
        return {}

    try:
        reader = gguf.GGUFReader(str(path), "r")
        out = {}
        for key in ("general.basename", "general.size_label", "general.name"):
            field = reader.get_field(key)
            if field is not None:
                val = field.contents()
                if val is not None:
                    out[key] = str(val)
        return out
    except Exception:
        return {}


def discover_local_gguf() -> list[DiscoveredModel]:
    """Scan models/*.gguf and infer base model, task, quant."""
    if not MODELS_DIR.exists():
        return []
    results: list[DiscoveredModel] = []
    for f in MODELS_DIR.glob("*.gguf"):
        meta = _read_gguf_metadata(f)
        base = None
        task = None
        quant = None
        if meta.get("general.basename"):
            base = meta["general.basename"].replace(" ", "-").lower()
            if meta.get("general.size_label"):
                base = f"{base}-{meta['general.size_label']}"

        if base is None:
            base, task, quant = _parse_gguf_filename(f.name)

        if base and task:
            hf_id = _infer_hf_from_short(base)
            results.append(
                DiscoveredModel(
                    path=str(f),
                    task=task,
                    base_model_hf_id=hf_id,
                    quant=quant,
                    source="local_gguf",
                )
            )
    return results


def discover_local_adapters() -> list[DiscoveredModel]:
    """Scan adapters/*/adapter_config.json for base model and task."""
    if not ADAPTERS_DIR.exists():
        return []
    results: list[DiscoveredModel] = []
    for subdir in ADAPTERS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        config_path = subdir / "adapter_config.json"
        if not config_path.exists():
            continue
        try:
            cfg = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        hf_id = cfg.get("base_model_name_or_path") or cfg.get("model")
        if not hf_id:
            continue
        task = subdir.name
        results.append(
            DiscoveredModel(
                path=str(subdir),
                task=task,
                base_model_hf_id=hf_id,
                quant=None,
                source="local_adapter",
            )
        )
    return results


def discover_all(include_modal: bool = False) -> list[DiscoveredModel]:
    """Discover all trained models (local + optional Modal)."""
    results = discover_local_gguf() + discover_local_adapters()
    if include_modal:
        results.extend(discover_modal_checkpoints())
    return results


def discover_modal_checkpoints() -> list[DiscoveredModel]:
    """Discover checkpoints on Modal volume. Requires modal run."""
    list_script = ROOT / "scripts" / "modal" / "list_checkpoints.py"
    if not list_script.exists():
        return []
    try:
        out = subprocess.run(
            [sys.executable, "-m", "modal", "run", str(list_script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode != 0:
            return []
        data = json.loads(out.stdout.strip())
        return [DiscoveredModel(**d) for d in data]
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, TypeError):
        return []


def discover_by_task(task: str, include_modal: bool = False) -> list[DiscoveredModel]:
    """Return discovered models for a given task (educator, poet, poet_rhyme)."""
    all_ = discover_all(include_modal=include_modal)
    if task == "educator":
        return [m for m in all_ if m.task in ("educator", "educator-interim")]
    if task == "poet":
        return [m for m in all_ if m.task in ("poet", "poet_rhyme")]
    return [m for m in all_ if m.task == task]


def hf_to_short_name(hf_id: str) -> str:
    """Map HuggingFace ID to short name for filenames."""
    return _hf_to_short_name(hf_id)
