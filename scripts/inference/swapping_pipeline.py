#!/usr/bin/env python3
"""
SwappingPipeline — Option C: load one model at a time for 32B poet.
Use when 14B poet quality is insufficient and 32B doesn't fit alongside educator.
"""
import gc
from pathlib import Path

from models.prompts.loader import get_persona, render_prompt

from .pipeline import PoetryPipeline, load_config

ROOT = Path(__file__).resolve().parents[2]


def _merge_llama_kwargs(section: dict, overrides: dict | None) -> dict:
    """Build kwargs for Llama() from YAML educator/poet section + optional overrides."""
    base = {
        "n_ctx": section.get("n_ctx", 8192),
        "n_gpu_layers": section.get("n_gpu_layers", -1),
        "n_threads": section.get("n_threads", 8),
        "use_mmap": section.get("use_mmap", True),
        "use_mlock": section.get("use_mlock", False),
        "verbose": False,
    }
    if overrides:
        for k, v in overrides.items():
            if k in base and v is not None:
                base[k] = v
    return base


class SwappingPipeline(PoetryPipeline):
    """Single loaded GGUF at a time; compatible with PoetryPipeline.generate()."""

    def __init__(
        self,
        config_path: Path | None = None,
        educator_model_override: str | None = None,
        poet_model_override: str | None = None,
        *,
        educator_load_overrides: dict | None = None,
        poet_load_overrides: dict | None = None,
    ):
        super().__init__(config_path, educator_model_override, poet_model_override)
        path = config_path or ROOT / "config" / "inference_config.yaml"
        raw = load_config(path)
        edu_sec = raw.get("educator") or {}
        poet_sec = raw.get("poet") or {}
        self._llama_kwargs_educator = _merge_llama_kwargs(edu_sec, educator_load_overrides)
        self._llama_kwargs_poet = _merge_llama_kwargs(poet_sec, poet_load_overrides)
        self.active_model = None
        self.active_role = None

    def _load_models(self):
        """SwappingPipeline uses _load(role) instead of dual resident models."""
        pass

    def _load(self, role: str):
        if self.active_role == role:
            return
        if self.active_model is not None:
            del self.active_model
            gc.collect()
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")
        path = (
            self.config.educator_model_path
            if role == "educator"
            else self.config.poet_model_path
        )
        if self.educator_model_override and self.educator_model_override.startswith("gguf:"):
            p = self.educator_model_override[5:].strip()
            if role == "educator":
                path = p if Path(p).is_absolute() else str(ROOT / p.lstrip("./"))
        if self.poet_model_override and self.poet_model_override.startswith("gguf:"):
            p = self.poet_model_override[5:].strip()
            if role == "poet":
                path = p if Path(p).is_absolute() else str(ROOT / p.lstrip("./"))
        kwargs = (
            self._llama_kwargs_educator if role == "educator" else self._llama_kwargs_poet
        ).copy()
        kwargs["model_path"] = path
        self.active_model = Llama(**kwargs)
        self.active_role = role

    def _educator_generate(self, prompt: str, task: str = "critique") -> str:
        if self.educator_model_override and self.educator_model_override.startswith(
            ("ollama:", "bedrock:"),
        ):
            return PoetryPipeline._educator_generate(self, prompt, task)
        self._load("educator")
        params = {
            "brief": {"temperature": 0.4, "max_tokens": 800},
            "critique": {"temperature": 0.3, "max_tokens": 600},
            "revision_brief": {"temperature": 0.4, "max_tokens": 600},
            "final_note": {"temperature": 0.3, "max_tokens": 400},
            "summarize": {"temperature": 0.2, "max_tokens": 300},
            "poet_instructions": {"temperature": 0.2, "max_tokens": 250},
            "diagnostic": {"temperature": 0.2, "max_tokens": 220},
        }[task]
        r = self.active_model.create_chat_completion(
            messages=[
                {"role": "system", "content": self.educator_system},
                {"role": "user", "content": prompt},
            ],
            **params,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=self._get_stop_tokens("educator"),
        )
        return r["choices"][0]["message"]["content"]

    def _poet_generate(
        self,
        prompt: str,
        is_revision: bool = False,
        revision_context: dict | None = None,
    ) -> str:
        if self.poet_model_override and self.poet_model_override.startswith(
            ("ollama:", "bedrock:"),
        ):
            return PoetryPipeline._poet_generate(
                self, prompt, is_revision, revision_context,
            )
        temp = 0.75 if is_revision else 0.8
        system = get_persona("poet")

        if is_revision and revision_context:
            rhyme_ctx = self._format_rhyme_deviations(
                revision_context["draft"],
                revision_context.get("brief", ""),
            )
            poet_prompt = render_prompt(
                "inference", "poet_self_revision",
                brief=revision_context.get("brief", ""),
                draft=revision_context["draft"],
                critique=revision_context["critique"],
                rhyme_ctx=rhyme_ctx,
            )
        elif is_revision:
            poet_prompt = prompt
        else:
            poet_prompt = self._build_poet_prompt(prompt)

        self._load("poet")
        r = self.active_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": poet_prompt},
            ],
            temperature=temp,
            top_p=0.95,
            repeat_penalty=1.15,
            max_tokens=4096,
            stop=self._get_stop_tokens("poet"),
        )
        output = r["choices"][0]["message"]["content"]
        return self._clean_poet_output(output)
