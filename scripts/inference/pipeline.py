#!/usr/bin/env python3
"""
Poetry Generation Pipeline — llama.cpp Metal Backend. S4.3
Requires: pip install llama-cpp-python
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "inference_config.yaml"
PERSONA_PATH = ROOT / "persona" / "persona_condensed.txt"


def load_config(path: Path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class Config:
    """Config object for PoetryPipeline."""

    def __init__(self, yaml_config: dict):
        edu = yaml_config.get("educator", {})
        poet = yaml_config.get("poet", {})
        self.educator_model_path = edu.get("model_path", "./models/llama3.1-14b-educator-Q4_K_M.gguf")
        self.educator_ctx = edu.get("n_ctx", 4096)
        self.poet_model_path = poet.get("model_path", "./models/llama3.1-14b-poet-Q4_K_M.gguf")
        self.poet_ctx = poet.get("n_ctx", 2048)
        self.max_revisions = yaml_config.get("max_revisions", 3)
        self.educator_persona_condensed = PERSONA_PATH.read_text() if PERSONA_PATH.exists() else ""
        self.user_style_profile = None


class PoetryPipeline:
    def __init__(self, config_path: Path = None):
        path = config_path or CONFIG_PATH
        cfg = load_config(path)
        # Resolve paths relative to project root
        self.config = Config(cfg)
        self.config.educator_model_path = str(ROOT / self.config.educator_model_path.lstrip("./"))
        self.config.poet_model_path = str(ROOT / self.config.poet_model_path.lstrip("./"))
        self.educator = None
        self.poet = None
        self.educator_system = self.config.educator_persona_condensed
        self.max_revisions = self.config.max_revisions
        self.user_profile = self.config.user_style_profile

    def _load_models(self):
        if self.educator is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")
        self.educator = Llama(
            model_path=self.config.educator_model_path,
            n_ctx=self.config.educator_ctx,
            n_gpu_layers=-1,
            n_threads=8,
            use_mmap=True,
            verbose=False,
        )
        self.poet = Llama(
            model_path=self.config.poet_model_path,
            n_ctx=self.config.poet_ctx,
            n_gpu_layers=-1,
            n_threads=8,
            use_mmap=True,
            verbose=False,
        )

    def _educator_generate(self, prompt: str, task: str = "critique") -> str:
        params = {
            "brief": {"temperature": 0.4, "max_tokens": 800},
            "critique": {"temperature": 0.3, "max_tokens": 600},
            "revision_brief": {"temperature": 0.4, "max_tokens": 600},
            "final_note": {"temperature": 0.3, "max_tokens": 400},
        }[task]
        self._load_models()
        r = self.educator.create_chat_completion(
            messages=[
                {"role": "system", "content": self.educator_system},
                {"role": "user", "content": prompt},
            ],
            **params,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|eot_id|>", "</s>"],
        )
        return r["choices"][0]["message"]["content"]

    def _poet_generate(self, prompt: str, is_revision: bool = False) -> str:
        temp = 0.75 if is_revision else 0.8
        self._load_models()
        r = self.poet.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a poet. Write with precision, musicality, and originality. Every word must earn its place.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temp,
            top_p=0.95,
            repeat_penalty=1.15,
            max_tokens=500,
            stop=["<|eot_id|>", "</s>"],
        )
        return r["choices"][0]["message"]["content"]

    def _educator_approves(self, critique: str) -> bool:
        signals = ["found its shape", "this is ready", "let this one go", "this poem is done", "nothing left to cut"]
        return any(s in critique.lower() for s in signals)

    def _build_brief_prompt(self, user_request: str) -> str:
        style_ctx = ""
        if self.user_profile:
            style_ctx = f"\n\nThis poet's style profile:\n{self.user_profile}\nGuide toward this sensibility without mere imitation.\n"
        return (
            f'A poet has asked for help. Their request:\n\n"{user_request}"\n\n'
            "Construct a generation brief. Include:\n"
            "- Your specific angle (not the obvious one)\n"
            "- At least 8 specific clichés to avoid\n"
            "- An unexpected imagery domain\n"
            "- Form/structure guidance (argued)\n"
            "- Sound/rhythm guidance\n"
            "- Structural arc\n"
            f"{style_ctx}"
            "Write as you would — in your voice, about this specific poem."
        )

    def _build_critique_prompt(self, draft: str, brief: str, history: list) -> str:
        history_ctx = ""
        if history:
            prev = history[-1]
            history_ctx = f"\n\nPrevious draft and your critique:\n---\n{prev['draft']}\n---\nYour notes:\n{prev['critique']}\n\nThis is the revision.\n"
        return (
            f"Generation brief:\n---\n{brief}\n---\n\nDraft:\n---\n{draft}\n---\n"
            f"{history_ctx}"
            "Give your workshop response. Start with what's alive. Then what isn't working — name the failure type. Offer direction.\n\n"
            "If the poem has found its shape, say so — use 'this poem has found its shape.'"
        )

    def _build_revision_brief_prompt(self, draft: str, critique: str, brief: str) -> str:
        return (
            f"Original brief:\n---\n{brief}\n---\n\n"
            f"Current draft:\n---\n{draft}\n---\n\n"
            f"Your critique:\n---\n{critique}\n---\n\n"
            "Construct a revised generation brief that addresses your critique while preserving what's working."
        )

    def _build_final_note_prompt(self, final_draft: str, brief: str) -> str:
        return (
            f"Final poem:\n---\n{final_draft}\n---\n\n"
            "Write a brief note about what makes this poem work. What's the strongest moment? "
            "What craft choice pays off? What should the poet learn from this about their instincts?\n\n"
            "Keep it to 3-5 sentences."
        )

    def generate(self, user_request: str) -> dict:
        brief = self._educator_generate(self._build_brief_prompt(user_request), task="brief")
        draft = self._poet_generate(brief)
        revision_history = []
        for i in range(self.max_revisions):
            critique = self._educator_generate(
                self._build_critique_prompt(draft, brief, revision_history),
                task="critique",
            )
            revision_history.append({"draft": draft, "critique": critique, "iteration": i})
            if self._educator_approves(critique):
                break
            revision_brief = self._educator_generate(
                self._build_revision_brief_prompt(draft, critique, brief),
                task="revision_brief",
            )
            draft = self._poet_generate(revision_brief, is_revision=True)
        final_note = self._educator_generate(
            self._build_final_note_prompt(draft, brief),
            task="final_note",
        )
        return {
            "final_poem": draft,
            "educator_note": final_note,
            "generation_brief": brief,
            "revision_history": revision_history,
            "metadata": {
                "revisions": len(revision_history),
                "model_educator": "llama3.1-14b-educator-Q4_K_M",
                "model_poet": "llama3.1-14b-poet-Q4_K_M",
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=str, help="User's poem request")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    pipeline = PoetryPipeline(args.config)
    result = pipeline.generate(args.request)
    print("FINAL POEM:\n")
    print(result["final_poem"])
    print("\nEDUCATOR NOTE:\n")
    print(result["educator_note"])


if __name__ == "__main__":
    main()
