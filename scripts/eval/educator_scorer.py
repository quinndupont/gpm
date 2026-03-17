#!/usr/bin/env python3
"""Multi-dimensional poem scoring using trained educator model."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.prompts.loader import get_persona, render_prompt
from scripts.eval.form_registry import is_rhyming_form
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme, format_analysis_for_prompt

DEFAULT_WEIGHTS = {
    "imagery_clarity": 0.20,
    "originality": 0.20,
    "sentiment_authenticity": 0.20,
    "form_adherence": 0.25,
    "voice_distinctiveness": 0.15,
}

NEUTRAL_FALLBACK = {
    "imagery_clarity": 0.5,
    "originality": 0.5,
    "sentiment_authenticity": 0.5,
    "form_adherence": 0.5,
    "voice_distinctiveness": 0.5,
}


class EducatorScorer:
    """Score poems using trained educator model."""

    def __init__(self, model_path: Path):
        """Initialize educator model for scoring."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=-1,
            n_threads=8,
            use_mmap=True,
            verbose=False,
        )
        self.system = get_persona("educator_neutral")

    def score_poem(
        self,
        poem: str,
        brief: str,
        expected_form: str | None = None,
    ) -> dict[str, float]:
        """Score poem on multiple dimensions.

        Returns:
            Dict of dimension -> [0,1] normalized score.
        """
        form_ctx = ""
        if expected_form and is_rhyming_form(expected_form):
            rhyme = analyze_rhyme(poem, expected_form=expected_form)
            form_ctx = f"\n\nRhyme analysis (automated):\n{format_analysis_for_prompt(rhyme)}\n"

        user_prompt = render_prompt(
            "inference",
            "poem_scoring",
            brief=brief,
            poem=poem,
            form_ctx=form_ctx,
        )

        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|endoftext|>"],
        )

        output_text = response["choices"][0]["message"]["content"]

        try:
            scores = json.loads(output_text)
            normalized = {
                dim: (score - 1) / 4
                for dim, score in scores.items()
            }
            return normalized
        except json.JSONDecodeError:
            print(f"Failed to parse educator scores: {output_text}", file=sys.stderr)
            return NEUTRAL_FALLBACK.copy()

    def compute_aggregate_reward(
        self,
        scores: dict[str, float],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Aggregate dimensional scores into single reward in [0, 1]."""
        if weights is None:
            weights = DEFAULT_WEIGHTS
        return sum(
            scores.get(dim, 0.5) * weight for dim, weight in weights.items()
        )


def load_educator_scorer(
    model_name: str = "llama3.1-8b-educator-Q4_K_M.gguf",
) -> EducatorScorer:
    """Load educator model for scoring."""
    model_path = ROOT / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Educator model not found: {model_path}\n"
            "Download from training checkpoint or export GGUF."
        )
    return EducatorScorer(model_path)
