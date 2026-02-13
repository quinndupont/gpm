#!/usr/bin/env python3
"""
Poetry Generation Pipeline — llama.cpp Metal Backend. S4.3
Requires: pip install llama-cpp-python
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "inference_config.yaml"
PERSONA_PATH = ROOT / "persona" / "educator_neutral.txt"
PERSONA_FALLBACK = ROOT / "persona" / "persona_condensed.txt"
BRIEF_CAP = 1200  # ~300 tokens for poet input
REVISION_CTX_CAP = 2400  # draft + critique + brief for poet revision input


def load_config(path: Path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class Config:
    """Config object for PoetryPipeline."""

    def __init__(self, yaml_config: dict):
        edu = yaml_config.get("educator", {})
        poet = yaml_config.get("poet", {})
        self.educator_model_path = edu.get("model_path", "./models/qwen2.5-7b-educator-Q4_K_M.gguf")
        self.educator_ctx = edu.get("n_ctx", 4096)
        self.poet_model_path = poet.get("model_path", "./models/qwen2.5-7b-poet-Q4_K_M.gguf")
        self.poet_ctx = poet.get("n_ctx", 2048)
        self.max_revisions = yaml_config.get("max_revisions", 3)
        p = PERSONA_PATH if PERSONA_PATH.exists() else PERSONA_FALLBACK
        self.educator_persona_condensed = p.read_text().strip() if p.exists() else ""
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
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        return r["choices"][0]["message"]["content"]

    def _build_poet_prompt(self, brief: str) -> str:
        if len(brief) > BRIEF_CAP:
            brief = brief[:BRIEF_CAP].rsplit("\n", 1)[0] + "\n\n[truncated]"
        return brief + "\n\nOutput ONLY the poem. Do not repeat the brief. Do not add commentary."

    def _build_poet_revision_prompt(
        self, draft: str, critique: str, revision_brief: str, user_input: str = ""
    ) -> str:
        user_ctx = ""
        if user_input.strip():
            user_ctx = f"\n\nPoet's additional direction:\n---\n{user_input.strip()}\n---\n\n"
        ctx = (
            f"Previous draft:\n---\n{draft}\n---\n\n"
            f"Educator critique:\n---\n{critique}\n---\n\n"
            f"Revision directions:\n---\n{revision_brief}\n---\n"
            f"{user_ctx}"
            "Revise the poem according to the critique and directions. Output ONLY the revised poem."
        )
        if len(ctx) > REVISION_CTX_CAP:
            ctx = ctx[:REVISION_CTX_CAP].rsplit("\n", 1)[0] + "\n\n[truncated]"
        return ctx

    def _poet_generate(self, prompt: str, is_revision: bool = False) -> str:
        temp = 0.75 if is_revision else 0.8
        self._load_models()
        poet_prompt = prompt if is_revision else self._build_poet_prompt(prompt)
        r = self.poet.create_chat_completion(
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

    def _educator_approves(self, critique: str) -> bool:
        signals = ["found its shape", "this is ready", "let this one go", "this poem is done", "nothing left to cut"]
        return any(s in critique.lower() for s in signals)

    def _build_brief_prompt(self, user_request: str) -> str:
        style_ctx = ""
        if self.user_profile:
            style_ctx = f"\n\nThis poet's style profile:\n{self.user_profile}\n"
        return (
            f'A poet has asked for help. Their request:\n\n"{user_request}"\n\n'
            "Construct a COMPACT generation brief (~300 tokens max). Include:\n"
            "- Angle (2-3 sentences, not the obvious approach)\n"
            "- Clichés to avoid (5-6 specific phrases for this topic)\n"
            "- Imagery domain (1-2 sentences, unexpected)\n"
            "- Form guidance (1-2 sentences)\n"
            f"{style_ctx}"
            "No rhetorical flourish. Actionable only."
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

    def _build_revision_brief_prompt(
        self,
        draft: str,
        critique: str,
        brief: str,
        revision_history: list,
        user_input: str = "",
    ) -> str:
        history_ctx = ""
        prev = revision_history[:-1]  # exclude current (draft/critique passed separately)
        if prev:
            # Include last 2 revision cycles to stay within context
            prev = prev[-2:] if len(prev) > 2 else prev
            parts = [f"Draft:\n{h['draft']}\nYour critique:\n{h['critique']}" for h in prev]
            history_ctx = (
                "\n\nPrevious revision(s) and your critiques:\n---\n"
                + "\n---\n".join(parts)
                + "\n---\n\nThis is the current draft. Ensure your brief targets what still needs fixing."
            )
        user_ctx = ""
        if user_input.strip():
            user_ctx = (
                f"\n\nPoet's additional direction (incorporate into your brief):\n---\n"
                f"{user_input.strip()}\n---\n\n"
            )
        return (
            f"Original brief:\n---\n{brief}\n---\n\n"
            f"Current draft:\n---\n{draft}\n---\n\n"
            f"Your critique:\n---\n{critique}\n---\n"
            f"{history_ctx}"
            f"{user_ctx}"
            "Construct a COMPACT revised generation brief (~300 tokens). "
            "Each critique point must map to a concrete revision direction. "
            "The poet receives: this draft, your critique, and your brief — ensure your brief translates critique into actionable directions. "
            "Angle, clichés to avoid, imagery domain, form. No flourish."
        )

    def _build_final_note_prompt(self, final_draft: str, brief: str) -> str:
        return (
            f"Final poem:\n---\n{final_draft}\n---\n\n"
            "Write a brief note about what makes this poem work. What's the strongest moment? "
            "What craft choice pays off? What should the poet learn from this about their instincts?\n\n"
            "Keep it to 3-5 sentences."
        )

    def generate(
        self,
        user_request: str,
        max_revisions: int = None,
        verbose: bool = False,
        interactive: bool = False,
    ) -> dict:
        revisions = max_revisions if max_revisions is not None else self.max_revisions

        def out(agent: str, label: str, text: str):
            if verbose:
                sep = "─" * 60
                body = text.strip() if text else "(no output)"
                print(f"\n{sep}\n[{agent}] {label}\n{sep}\n{body}\n", flush=True)

        if verbose:
            print("\n→ Educator: generating brief...", flush=True)
        brief = self._educator_generate(self._build_brief_prompt(user_request), task="brief")
        out("EDUCATOR", "Generation brief → Poet", brief)

        if verbose:
            print("→ Poet: generating initial draft...", flush=True)
        draft = self._poet_generate(brief)
        out("POET", "Initial draft → Educator", draft)

        revision_history = []
        for i in range(revisions):
            if verbose:
                print(f"→ Educator: critiquing (revision {i + 1})...", flush=True)
            critique = self._educator_generate(
                self._build_critique_prompt(draft, brief, revision_history),
                task="critique",
            )
            revision_history.append({"draft": draft, "critique": critique, "iteration": i})
            out("EDUCATOR", f"Critique (revision {i + 1}) → Poet", critique)

            if self._educator_approves(critique):
                if verbose:
                    print("\n  ✓ Educator approved — poem complete.\n", flush=True)
                break

            user_input = ""
            if interactive:
                raw = input("\nYour direction for revision (Enter to skip): ").strip()
                user_input = raw

            if verbose:
                print(f"→ Educator: building revision brief...", flush=True)
            revision_brief = self._educator_generate(
                self._build_revision_brief_prompt(
                    draft, critique, brief, revision_history[:-1], user_input
                ),
                task="revision_brief",
            )
            out("EDUCATOR", f"Revision brief → Poet", revision_brief)

            if verbose:
                print(f"→ Poet: generating revision {i + 1}...", flush=True)
            revision_prompt = self._build_poet_revision_prompt(
                draft, critique, revision_brief, user_input
            )
            draft = self._poet_generate(revision_prompt, is_revision=True)
            out("POET", f"Revision {i + 1} draft → Educator", draft)

        return {
            "final_poem": draft,
            "generation_brief": brief,
            "revision_history": revision_history,
            "metadata": {
                "revisions": len(revision_history),
                "model_educator": "qwen2.5-7b-educator-Q4_K_M",
                "model_poet": "qwen2.5-7b-poet-Q4_K_M",
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", nargs="?", type=str, help="User's poem request (omit for interactive)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--max-revisions", type=int, choices=[1, 2, 3, 4, 5], help="Max revisions (non-interactive)")
    args = parser.parse_args()

    pipeline = PoetryPipeline(args.config)
    verbose = True

    if args.request:
        max_rev = args.max_revisions if args.max_revisions is not None else pipeline.max_revisions
        result = pipeline.generate(
            args.request, max_revisions=max_rev, verbose=verbose, interactive=False
        )
        print("\n" + "═" * 60 + "\nFINAL POEM\n" + "═" * 60 + "\n")
        print(result["final_poem"])
        return

    # Interactive mode (single run)
    REV_OPTS = "1, 2, 3, 4, 5"
    max_rev = 2
    request = input("Your poem request: ").strip()
    if not request:
        return
    rev_input = input(f"Max revisions [{REV_OPTS}] (Enter={max_rev}): ").strip() or str(max_rev)
    try:
        r = int(rev_input)
        if 1 <= r <= 5:
            max_rev = r
    except ValueError:
        pass
    result = pipeline.generate(
        request, max_revisions=max_rev, verbose=verbose, interactive=True
    )
    print("\n" + "═" * 60 + "\nFINAL POEM\n" + "═" * 60 + "\n")
    print(result["final_poem"])


if __name__ == "__main__":
    main()
