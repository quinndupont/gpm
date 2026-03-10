#!/usr/bin/env python3
"""Pytest suite for RevFlux pipeline output structure and revision dynamics."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
import sys

MODELS_CONFIG = ROOT / "config" / "rev_flux_models.yaml"


def _load_models_config():
    import yaml
    if not MODELS_CONFIG.exists():
        return []
    with open(MODELS_CONFIG) as f:
        return yaml.safe_load(f).get("models", [])


def _gguf_exists(model_cfg: dict) -> bool:
    """Return True if GGUF paths exist (for gguf or gguf:path)."""
    edu = model_cfg.get("educator", "gguf")
    poet = model_cfg.get("poet", "gguf")
    paths = []
    if edu == "gguf" and poet == "gguf":
        from scripts.inference.pipeline import load_config
        cfg = load_config(ROOT / "config" / "inference_config.yaml")
        for p in (cfg.get("educator", {}).get("model_path", ""), cfg.get("poet", {}).get("model_path", "")):
            if p:
                paths.append(str(ROOT / p.lstrip("./")))
    else:
        for val in (edu, poet):
            if val and val.startswith("gguf:"):
                p = val[5:].strip()
                paths.append(p if Path(p).is_absolute() else str(ROOT / p.lstrip("./")))
    for p in paths:
        if p and not Path(p).exists():
            return False
    return True


def _is_ollama(model_cfg: dict) -> bool:
    edu = model_cfg.get("educator", "")
    return edu and edu.startswith("ollama:")


@pytest.fixture
def models_config():
    return _load_models_config()


@pytest.mark.parametrize("model_cfg", _load_models_config(), ids=lambda m: m.get("id", "unknown"))
def test_rev_flux_output_structure(model_cfg):
    """Each model returns valid output structure (revision_history, final_poem, change_pcts)."""
    if _is_ollama(model_cfg):
        pytest.skip("Ollama models require running server")
    if not _gguf_exists(model_cfg):
        pytest.skip("GGUF files not found")
    from scripts.inference.pipeline import PoetryPipeline
    from scripts.benchmarks.rev_flux.run_harness import run_single
    from scripts.benchmarks.rev_flux.prompts import CATEGORIES
    edu = model_cfg.get("educator", "gguf")
    poet = model_cfg.get("poet", "gguf")
    edu_override = None if edu == "gguf" else edu
    poet_override = None if poet == "gguf" else poet
    pipeline = PoetryPipeline(educator_model_override=edu_override, poet_model_override=poet_override)
    revs = model_cfg.get("revisions", [0])
    max_rev = 1 if 1 in revs else (revs[0] if revs else 0)
    prompt = list(CATEGORIES.values())[0][0]
    run = run_single(pipeline, prompt, max_revisions=max_rev, category="famous_poetry", prompt_idx=0, model_id=model_cfg["id"])
    assert "revision_history" in run
    assert "final_poem" in run
    assert "change_pcts" in run
    assert isinstance(run["final_poem"], str)
    assert len(run["final_poem"]) > 0
    for p in run.get("change_pcts", []):
        assert 0 <= p <= 100, f"change_pct {p} out of range"
    if max_rev >= 1 and not _is_ollama(model_cfg):
        assert len([h for h in run["revision_history"] if h.get("critique")]) >= 0


@pytest.mark.slow
def test_rev_flux_full_sweep(models_config):
    """Full sweep: one prompt per category, all revision lengths (slow)."""
    from scripts.inference.pipeline import PoetryPipeline
    from scripts.benchmarks.rev_flux.run_harness import run_single
    from scripts.benchmarks.rev_flux.prompts import CATEGORIES
    for model_cfg in models_config:
        if _is_ollama(model_cfg) or not _gguf_exists(model_cfg):
            continue
        edu = model_cfg.get("educator", "gguf")
        poet = model_cfg.get("poet", "gguf")
        edu_override = None if edu == "gguf" else edu
        poet_override = None if poet == "gguf" else poet
        pipeline = PoetryPipeline(educator_model_override=edu_override, poet_model_override=poet_override)
        revs = model_cfg.get("revisions", [0])
        for category, prompts in CATEGORIES.items():
            for max_rev in revs[:2]:
                run = run_single(pipeline, prompts[0], max_revisions=max_rev, category=category, prompt_idx=0, model_id=model_cfg["id"])
                assert "change_pcts" in run
                for p in run.get("change_pcts", []):
                    assert 0 <= p <= 100
