#!/usr/bin/env python3
"""
SwappingPipeline — Option C: load one model at a time for 32B poet.
Use when 14B poet quality is insufficient and 32B doesn't fit alongside educator.
"""
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERSONA_PATH = ROOT / "persona" / "educator_neutral.txt"
PERSONA_FALLBACK = ROOT / "persona" / "persona_condensed.txt"

from .pipeline import PoetryPipeline, Config


class SwappingPipeline(PoetryPipeline):
    def __init__(self, config_path: Path = None):
        import yaml
        path = config_path or ROOT / "config" / "inference_config.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.config = Config(cfg)
        self.config.educator_model_path = str(ROOT / self.config.educator_model_path.lstrip("./"))
        self.config.poet_model_path = str(ROOT / self.config.poet_model_path.lstrip("./"))
        p = PERSONA_PATH if PERSONA_PATH.exists() else PERSONA_FALLBACK
        self.config.educator_persona_condensed = p.read_text().strip() if p.exists() else ""
        self.active_model = None
        self.active_role = None
        self.educator_system = self.config.educator_persona_condensed
        self.max_revisions = self.config.max_revisions
        self.user_profile = self.config.user_style_profile
        self.max_revisions = self.config.max_revisions
        self.user_profile = self.config.user_style_profile
        self._load("educator")

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
        path = self.config.educator_model_path if role == "educator" else self.config.poet_model_path
        n_ctx = self.config.educator_ctx if role == "educator" else self.config.poet_ctx
        self.active_model = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            n_threads=8,
            use_mmap=True,
            verbose=False,
        )
        self.active_role = role

    def _educator_generate(self, prompt: str, task: str = "critique") -> str:
        self._load("educator")
        params = {
            "brief": {"temperature": 0.4, "max_tokens": 800},
            "critique": {"temperature": 0.3, "max_tokens": 600},
            "revision_brief": {"temperature": 0.4, "max_tokens": 600},
            "final_note": {"temperature": 0.3, "max_tokens": 400},
        }[task]
        r = self.active_model.create_chat_completion(
            messages=[
                {"role": "system", "content": self.educator_system},
                {"role": "user", "content": prompt},
            ],
            **params,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return r["choices"][0]["message"]["content"]

    def _poet_generate(self, prompt: str, is_revision: bool = False) -> str:
        self._load("poet")
        temp = 0.75 if is_revision else 0.8
        poet_prompt = prompt if is_revision else self._build_poet_prompt(prompt)
        r = self.active_model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a poet. You receive generation briefs and write poems. You never output instructions, critique, or analysis — only poems.",
                },
                {"role": "user", "content": poet_prompt},
            ],
            temperature=temp,
            top_p=0.95,
            repeat_penalty=1.15,
            max_tokens=4096,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return r["choices"][0]["message"]["content"]
