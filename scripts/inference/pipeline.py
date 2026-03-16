#!/usr/bin/env python3
"""
Poetry Generation Pipeline — llama.cpp Metal Backend. S4.3
Requires: pip install llama-cpp-python
"""
import argparse
import re
import sys
from pathlib import Path

# Patch Jinja2 to support break tag in GGUF chat templates
# Many GGUF models use {% break %} which is not valid Jinja2 syntax
# This patch MUST be applied before llama_cpp is used anywhere
try:
    from jinja2 import nodes
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    import jinja2 as jinja2_module

    class BreakExtension(Extension):
        """Custom Jinja2 extension to handle {% break %} tags."""
        tags = {"break"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            # Create a Break node that acts like loop control
            return nodes.Break().set_lineno(lineno)

    # Patch ImmutableSandboxedEnvironment to always include BreakExtension
    _original_sandboxed_env_init = ImmutableSandboxedEnvironment.__init__

    def _patched_sandboxed_env_init(self, *args, **kwargs):
        """Patched ImmutableSandboxedEnvironment that always includes BreakExtension."""
        extensions = kwargs.get('extensions', [])
        if extensions is None:
            extensions = []
        elif not isinstance(extensions, list):
            extensions = list(extensions)
        else:
            extensions = extensions.copy()

        # Add BreakExtension if not present
        if BreakExtension not in extensions:
            extensions.append(BreakExtension)

        kwargs['extensions'] = extensions
        _original_sandboxed_env_init(self, *args, **kwargs)

    # Apply the patch to ImmutableSandboxedEnvironment
    ImmutableSandboxedEnvironment.__init__ = _patched_sandboxed_env_init
    jinja2_module.sandbox.ImmutableSandboxedEnvironment.__init__ = _patched_sandboxed_env_init

    print("Successfully patched Jinja2 for break tag support", file=sys.stderr)

except Exception as e:
    print(f"Warning: Failed to patch Jinja2 for break tag support: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)

from models.prompts.loader import get_persona, render_prompt
from scripts.eval.form_registry import (
    detect_form,
    get_scheme,
    is_metered_form,
    is_rhyming_form,
)
from scripts.eval.meter_analyzer import analyze as analyze_meter
from scripts.eval.meter_analyzer import format_analysis_for_prompt as format_meter_for_prompt
from scripts.eval.rhyme_analyzer import analyze as analyze_rhyme
from scripts.eval.rhyme_analyzer import format_analysis_for_prompt
from scripts.training.model_registry import (
    DEFAULT_STOP_TOKENS,
    ollama_tag_to_short,
    stop_tokens_for,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "inference_config.yaml"
BRIEF_CAP = 1200  # ~300 tokens for poet input


def load_config(path: Path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _infer_short_from_gguf_path(path: str) -> str | None:
    """Infer registry short_name from GGUF path (e.g. .../llama3.1-8b-educator-Q4_K_M.gguf)."""
    from scripts.training.model_registry import all_short_names
    stem = Path(path).stem
    for short in all_short_names():
        if short in stem:
            return short
    return None


class Config:
    """Config object for PoetryPipeline."""

    def __init__(self, yaml_config: dict):
        edu = yaml_config.get("educator", {})
        poet = yaml_config.get("poet", {})
        self.educator_model_path = edu.get(
            "model_path", "./models/qwen2.5-7b-educator-Q4_K_M.gguf",
        )
        self.educator_ctx = edu.get("n_ctx", 4096)
        self.poet_model_path = poet.get("model_path", "./models/qwen2.5-7b-poet-Q4_K_M.gguf")
        self.poet_ctx = poet.get("n_ctx", 2048)
        self.max_revisions = yaml_config.get("max_revisions", 3)
        self.revision_mode = yaml_config.get("revision_mode", "srpo")  # "srpo" or "educator"
        self.educator_stop = (
            edu.get("generation_brief", {}).get("stop")
            or edu.get("critique", {}).get("stop")
            or DEFAULT_STOP_TOKENS
        )
        self.poet_stop = (
            poet.get("generation", {}).get("stop")
            or poet.get("revision", {}).get("stop")
            or DEFAULT_STOP_TOKENS
        )
        try:
            self.educator_persona_condensed = get_persona("educator_neutral")
        except FileNotFoundError:
            self.educator_persona_condensed = get_persona("educator_condensed")
        self.user_style_profile = None


class PoetryPipeline:
    def __init__(
        self,
        config_path: Path = None,
        educator_model_override: str = None,
        poet_model_override: str = None,
    ):
        path = config_path or CONFIG_PATH
        cfg = load_config(path)
        # Resolve paths relative to project root
        self.config = Config(cfg)
        self.config.educator_model_path = str(
            ROOT / self.config.educator_model_path.lstrip("./"),
        )
        self.config.poet_model_path = str(
            ROOT / self.config.poet_model_path.lstrip("./"),
        )
        self.educator = None
        self.poet = None
        self.educator_system = self.config.educator_persona_condensed
        self.max_revisions = self.config.max_revisions
        self.revision_mode = self.config.revision_mode
        self.user_profile = self.config.user_style_profile
        self.educator_model_override = educator_model_override
        self.poet_model_override = poet_model_override

    def _get_stop_tokens(self, role: str) -> list[str]:
        """Return stop tokens for educator or poet."""
        override = self.educator_model_override if role == "educator" else self.poet_model_override
        if override and override.startswith("ollama:"):
            tag = override[7:]
            short = ollama_tag_to_short(tag)
            return stop_tokens_for(short_name=short) if short else DEFAULT_STOP_TOKENS
        if override and override.startswith("gguf:"):
            path = override[5:]
            short = _infer_short_from_gguf_path(path)
            return stop_tokens_for(short_name=short) if short else DEFAULT_STOP_TOKENS
        if override and override.startswith("bedrock:"):
            # Bedrock/Claude models handle stop sequences internally
            return []
        path = (
            self.config.educator_model_path
            if role == "educator"
            else self.config.poet_model_path
        )
        short = _infer_short_from_gguf_path(path)
        if short:
            return stop_tokens_for(short_name=short)
        return self.config.educator_stop if role == "educator" else self.config.poet_stop

    def _ollama_chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.4,
        max_tokens: int = 800,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> str:
        """Call Ollama. model is the Ollama model name (e.g. qwen2.5:7b-instruct)."""
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("pip install ollama. Run: ollama pull " + model)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        opts = {
            "temperature": temperature, "num_predict": max_tokens,
            "top_p": top_p, "repeat_penalty": repeat_penalty,
        }
        r = chat(model=model, messages=messages, options=opts)
        if hasattr(r, "message"):
            return getattr(r.message, "content", "") or ""
        return r.get("message", {}).get("content", "") or ""

    def _bedrock_chat(
        self,
        model_id: str,
        system: str,
        user: str,
        temperature: float = 0.4,
        max_tokens: int = 800,
        top_p: float = 0.9,
    ) -> str:
        """Call AWS Bedrock. model_id is the Bedrock inference profile ID (e.g. us.anthropic.claude-3-5-sonnet-20241022-v2:0)."""
        try:
            import boto3
        except ImportError:
            raise ImportError("pip install boto3")

        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

        # Build Bedrock request based on model type
        import json

        if "anthropic" in model_id:
            # Claude models use Messages API format
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": user}],
                "system": system,
            }
            # Claude 4+ models don't support both temperature and top_p
            # Only add top_p for Claude 3.x models
            if "claude-3" in model_id or "claude-sonnet-3" in model_id or "claude-opus-3" in model_id:
                body["top_p"] = top_p
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            return result["content"][0]["text"]

        elif "meta" in model_id or "llama" in model_id:
            # Meta Llama models use text generation format
            prompt = f"{system}\n\n{user}"
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            return result.get("generation", result.get("text", ""))

        elif "qwen" in model_id:
            # Qwen models use Messages API format similar to Claude
            body = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
            }
            response = bedrock.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
            )
            result = json.loads(response["body"].read())
            # Try different response formats
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            elif "content" in result:
                return result["content"][0]["text"] if isinstance(result["content"], list) else result["content"]
            else:
                return result.get("text", result.get("generation", ""))

        else:
            raise ValueError(f"Unsupported Bedrock model: {model_id}")

    def _load_models(self):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")
        use_edu_gguf = not (
            self.educator_model_override
            and (self.educator_model_override.startswith("ollama:") or self.educator_model_override.startswith("bedrock:"))
        )
        use_poet_gguf = not (
            self.poet_model_override
            and (self.poet_model_override.startswith("ollama:") or self.poet_model_override.startswith("bedrock:"))
        )
        edu_path = self.config.educator_model_path
        poet_path = self.config.poet_model_path
        if self.educator_model_override and self.educator_model_override.startswith("gguf:"):
            p = self.educator_model_override[5:].strip()
            edu_path = p if Path(p).is_absolute() else str(ROOT / p.lstrip("./"))
        if self.poet_model_override and self.poet_model_override.startswith("gguf:"):
            p = self.poet_model_override[5:].strip()
            poet_path = p if Path(p).is_absolute() else str(ROOT / p.lstrip("./"))
        if use_edu_gguf and self.educator is None:
            self.educator = Llama(
                model_path=edu_path,
                n_ctx=self.config.educator_ctx,
                n_gpu_layers=-1,
                n_threads=8,
                use_mmap=True,
                verbose=False,
            )
        if use_poet_gguf and self.poet is None:
            self.poet = Llama(
                model_path=poet_path,
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
            "summarize": {"temperature": 0.2, "max_tokens": 300},
            "poet_instructions": {"temperature": 0.2, "max_tokens": 250},
        }[task]
        if self.educator_model_override and self.educator_model_override.startswith("ollama:"):
            model = self.educator_model_override[7:]  # strip "ollama:"
            return self._ollama_chat(
                model, self.educator_system, prompt,
                temperature=params["temperature"], max_tokens=params["max_tokens"],
            )
        if self.educator_model_override and self.educator_model_override.startswith("bedrock:"):
            model_id = self.educator_model_override[8:]  # strip "bedrock:"
            return self._bedrock_chat(
                model_id, self.educator_system, prompt,
                temperature=params["temperature"], max_tokens=params["max_tokens"],
            )
        self._load_models()
        r = self.educator.create_chat_completion(
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

    def _clean_poet_output(self, text: str) -> str:
        """Remove thinking tags and extract only the poem content.

        Some models (like Qwen with Claude Opus) output thinking process in <think> tags.
        This method removes those tags and any content within them.
        """
        if not text:
            return text

        import re

        # Handle properly closed <think>...</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Handle unclosed <think> tags - remove everything from <think> onwards
        # This is common with some models that start thinking but never close the tag
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)

        # Also handle thinking: prefix patterns (some models use this)
        cleaned = re.sub(r'(?i)^thinking:.*?(?=\n\n|\Z)', '', cleaned, flags=re.DOTALL | re.MULTILINE)

        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()

        # If nothing remains after cleaning, return a placeholder
        # This indicates the model only output thinking with no actual poem
        if not cleaned:
            cleaned = "[Model output only thinking text, no poem generated]"

        return cleaned

    def _build_poet_prompt(self, brief: str) -> str:
        if len(brief) > BRIEF_CAP:
            brief = brief[:BRIEF_CAP].rsplit("\n", 1)[0] + "\n\n[truncated]"
        detected = detect_form(brief)
        scheme_reminder = ""
        if detected and is_rhyming_form(detected):
            scheme = get_scheme(detected)
            if scheme:
                scheme_reminder = (
                    f"\n\nIMPORTANT: This poem must follow the {detected} rhyme scheme: {scheme}. "
                    "Every end-word pair must be a true phonetic rhyme. "
                    "Plan your end-words before writing each line."
                )
        return render_prompt(
            "inference", "poet_generation", brief=brief, scheme_reminder=scheme_reminder,
        )

    def _build_poet_revision_instructions(
        self,
        draft: str,
        critique: str,
        revision_brief: str,
        revision_history: list,
        user_input: str = "",
        verbose: bool = False,
        past_summary: str = None,
    ) -> str:
        """Use educator to produce compact revision instructions for the poet (~150 words)."""
        past_ctx = past_summary if past_summary else (
            self._summarize_critique_history(revision_history) if revision_history else ""
        )
        user_ctx = f"\n\nPoet's direction: {user_input.strip()}\n" if user_input.strip() else ""
        draft_label = "Current draft (poet's latest revision)" if past_ctx else "Draft"
        past_section = (
            f"Past revision rounds (poet revised after each; draft above is result):\n"
            f"---\n{past_ctx}\n---\n"
            if past_ctx else ""
        )
        draft_trunc = draft[:1200] + ("..." if len(draft) > 1200 else "")
        prompt = render_prompt(
            "inference", "poet_revision_instructions",
            draft_label=draft_label, draft=draft_trunc, critique=critique,
            revision_brief=revision_brief, past_section=past_section, user_ctx=user_ctx,
        )
        return self._educator_generate(prompt, task="poet_instructions")

    def _build_poet_revision_prompt(
        self,
        draft: str,
        critique: str,
        revision_brief: str,
        revision_history: list,
        user_input: str = "",
        verbose: bool = False,
        past_summary: str = None,
        brief: str = "",
    ) -> str:
        """Build poet revision prompt. Uses educator to summarize context so we never truncate."""
        if verbose:
            print("→ Educator: building compact revision instructions for poet...", flush=True)
        instructions = self._build_poet_revision_instructions(
            draft, critique, revision_brief, revision_history, user_input, verbose, past_summary
        )
        user_ctx = ""
        if user_input.strip():
            user_ctx = f"\n\nPoet's additional direction:\n---\n{user_input.strip()}\n---\n\n"

        # Inject deterministic rhyme deviations so poet knows exactly what to fix
        rhyme_ctx = ""
        rhyme = self._rhyme_gate(draft, brief)
        if rhyme and rhyme.get("deviations"):
            dev_lines = []
            for d in rhyme["deviations"]:
                dev_lines.append(
                    f'- Line {d["line"]}: "{d["word"]}" does not rhyme with '
                    f'"{d["expected_rhyme_with"]}" (needs {d["expected_label"]} rhyme)'
                )
            rhyme_ctx = (
                "\n\nRhyme errors (fix these end-words):\n"
                + "\n".join(dev_lines) + "\n"
            )

        return render_prompt(
            "inference", "poet_revision",
            draft=draft, instructions=instructions,
            rhyme_ctx=rhyme_ctx, user_ctx=user_ctx,
        )

    def _format_rhyme_deviations(self, draft: str, brief: str) -> str:
        """Format rhyme deviations for poet self-revision prompt."""
        rhyme = self._rhyme_gate(draft, brief)
        if not rhyme or not rhyme.get("deviations"):
            return ""
        dev_lines = []
        for d in rhyme["deviations"]:
            dev_lines.append(
                f'- Line {d["line"]}: "{d["word"]}" does not rhyme with '
                f'"{d["expected_rhyme_with"]}" (needs {d["expected_label"]} rhyme)'
            )
        return "\n\nRhyme errors to fix:\n" + "\n".join(dev_lines) + "\n"

    def _poet_generate(
        self,
        prompt: str,
        is_revision: bool = False,
        revision_context: dict | None = None,
    ) -> str:
        """Generate or revise a poem.

        For SRPO mode, revision_context contains:
          - brief: the original generation brief
          - draft: the previous draft
          - critique: the educator's critique
        """
        temp = 0.75 if is_revision else 0.8
        system = get_persona("poet")

        if is_revision and revision_context:
            # SRPO path: poet self-revises using learned capability
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
            # Legacy path: prompt is already built (educator-driven revision)
            poet_prompt = prompt
        else:
            # Generation path
            poet_prompt = self._build_poet_prompt(prompt)

        output = ""
        if self.poet_model_override and self.poet_model_override.startswith("ollama:"):
            model = self.poet_model_override[7:]
            output = self._ollama_chat(
                model, system, poet_prompt,
                temperature=temp, max_tokens=4096, top_p=0.95, repeat_penalty=1.15,
            )
        elif self.poet_model_override and self.poet_model_override.startswith("bedrock:"):
            model_id = self.poet_model_override[8:]
            output = self._bedrock_chat(
                model_id, system, poet_prompt,
                temperature=temp, max_tokens=4096, top_p=0.95,
            )
        else:
            self._load_models()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": poet_prompt},
            ]
            r = self.poet.create_chat_completion(
                messages=messages,
                temperature=temp,
                top_p=0.95,
                repeat_penalty=1.15,
                max_tokens=4096,
                stop=self._get_stop_tokens("poet"),
            )
            output = r["choices"][0]["message"]["content"]

        # Clean thinking tags and extract only the poem
        return self._clean_poet_output(output)

    def _summarize_critique_history(self, revision_history: list) -> str:
        """Compress past draft+critique pairs. Each round: poet revised after critique."""
        if not revision_history:
            return ""
        prev = revision_history[-2:] if len(revision_history) > 2 else revision_history
        parts = []
        for i, h in enumerate(prev, start=1):
            d = h["draft"][:800] + ("..." if len(h["draft"]) > 800 else "")
            c = h["critique"][:500] + ("..." if len(h["critique"]) > 500 else "")
            label = "Initial draft" if i == 1 else f"Round {i} (poet's revision)"
            parts.append(f"{label}:\n{d}\nYour critique:\n{c}")
        combined = "\n---\n".join(parts)
        prompt = (
            f"This is a revision chain. After each round the poet revised. "
            f"Summarize into 3-5 bullet points: what was wrong, what direction was given. "
            f"Note which issues the poet addressed in later drafts. No preamble.\n\n{combined}"
        )
        return self._educator_generate(prompt, task="summarize")

    def _summarize_critique(self, critique: str) -> str:
        """Compress a single critique when space is limited."""
        if len(critique) <= 400:
            return critique
        prompt = (
            f"Compress this workshop critique to ~100 words. "
            f"Keep: failure types, specific directions. No preamble.\n\n{critique}"
        )
        return self._educator_generate(prompt, task="summarize")

    def _rhyme_gate(self, draft: str, brief: str) -> dict | None:
        """Run deterministic rhyme analysis if the brief specifies a rhyming form.

        Returns the analysis dict if the form requires rhyming, None otherwise.
        """
        detected = detect_form(brief)
        if not detected or not is_rhyming_form(detected):
            return None
        return analyze_rhyme(draft, expected_form=detected)

    def _educator_approves(self, critique: str, draft: str = "", brief: str = "") -> bool:
        """
        Detect conclusive approval. Two conditions must be met:
        1. The educator must end with an approval phrase (not mid-critique).
        2. If the form requires rhyming, the deterministic rhyme analysis must
           confirm the form matches (no unresolved deviations).
        """
        c = critique.lower().strip()
        closing = c[-250:] if len(c) > 250 else c
        patterns = [
            r"this poem has found its shape\.\s*$",
            r"this is ready\.\s*$",
            r"let this one go\.\s*$",
            r"this poem is done\.\s*$",
            r"nothing left to cut\.\s*$",
        ]
        text_approved = any(re.search(p, closing) for p in patterns)
        if not text_approved:
            return False

        # Deterministic rhyme gate: override educator if form doesn't match
        rhyme = self._rhyme_gate(draft, brief)
        if rhyme is not None:
            if rhyme.get("matches_form") is False:
                return False
            if rhyme.get("strict_rhyme_density", 0) < 0.5:
                return False
        return True

    def _build_brief_prompt(self, user_request: str) -> str:
        style_ctx = ""
        if self.user_profile:
            style_ctx = f"\n\nThis poet's style profile:\n{self.user_profile}\n"
        return render_prompt("inference", "brief", user_request=user_request, style_ctx=style_ctx)

    def _build_critique_prompt(self, draft: str, brief: str, history: list) -> str:
        history_ctx = ""
        if history:
            prev = history[-1]
            history_ctx = (
            f"\n\nPrevious draft and your critique:\n---\n{prev['draft']}\n---\n"
            f"Your notes:\n{prev['critique']}\n\nThis is the revision.\n"
        )

        # If the brief specifies a formal form, run deterministic analysis
        # and inject the results so the educator has concrete data to work with.
        form_ctx = ""
        detected = detect_form(brief)
        if detected:
            form_parts = []
            if is_rhyming_form(detected):
                rhyme = analyze_rhyme(draft, expected_form=detected)
                form_parts.append(
                    f"Rhyme analysis (automated):\n{format_analysis_for_prompt(rhyme)}"
                )
            if is_metered_form(detected):
                meter = analyze_meter(draft, expected_form=detected)
                form_parts.append(
                    f"Meter analysis (automated):\n{format_meter_for_prompt(meter)}"
                )
            if form_parts:
                form_ctx = (
                    "\n\n" + "\n\n".join(form_parts) + "\n\n"
                    "Address form adherence in your critique — rhyme and meter where applicable. "
                    "Name specific lines and words.\n"
                )

        return render_prompt(
            "inference", "critique",
            brief=brief, draft=draft, history_ctx=history_ctx, form_ctx=form_ctx,
        )

    def _build_revision_brief_prompt(
        self,
        draft: str,
        critique: str,
        brief: str,
        revision_history: list,
        user_input: str = "",
        verbose: bool = False,
        past_summary: str = None,
    ) -> str:
        history_ctx = ""
        prev = revision_history
        if prev:
            summary = (
                past_summary
                if past_summary is not None
                else self._summarize_critique_history(prev)
            )
            history_ctx = (
                f"\n\nSummary of previous critiques:\n---\n{summary}\n---\n\n"
                "This is the current draft. Ensure your brief targets what still needs fixing."
            )
        user_ctx = ""
        if user_input.strip():
            user_ctx = (
                f"\n\nPoet's additional direction (incorporate into your brief):\n---\n"
                f"{user_input.strip()}\n---\n\n"
            )
        return render_prompt(
            "inference", "revision_brief",
            brief=brief, draft=draft, critique=critique,
            history_ctx=history_ctx, user_ctx=user_ctx,
        )

    def _build_final_note_prompt(self, final_draft: str, brief: str) -> str:
        return render_prompt("inference", "final_note", final_draft=final_draft)

    def generate(
        self,
        user_request: str,
        max_revisions: int = None,
        min_revisions: int = None,
        verbose: bool = False,
        interactive: bool = False,
    ) -> dict:
        import time
        t_start = time.perf_counter()
        revisions = max_revisions if max_revisions is not None else self.max_revisions

        def out(agent: str, label: str, text: str):
            if verbose:
                sep = "─" * 60
                body = text.strip() if text else "(no output)"
                print(f"\n{sep}\n[{agent}] {label}\n{sep}\n{body}\n", flush=True)

        # 0 rev = poet only, no educator
        if revisions == 0:
            if verbose:
                print("→ Poet: generating (no educator)...", flush=True)
            poet_prompt = (
                f"Write a poem. Request: {user_request}\n\n"
                "Output ONLY the poem. Do not add commentary."
            )
            draft = self._poet_generate(poet_prompt, is_revision=False)
            t1 = time.perf_counter()
            t_total = t1 - t_start
            poet_name = self.poet_model_override or "qwen2.5-7b-poet-Q4_K_M"

            # Estimate token count (rough heuristic: ~4 chars per token)
            estimated_tokens = len(draft) // 4
            tokens_per_sec = round(estimated_tokens / t_total, 1) if t_total > 0 else 0

            return {
                "final_poem": draft,
                "generation_brief": None,
                "revision_history": [],
                "metadata": {
                    "revisions": 0,
                    "approved": False,
                    "approved_at_round": None,
                    "model_educator": "N/A (poet only)",
                    "model_poet": poet_name,
                    "perf_total_sec": round(t_total, 2),
                    "perf_tokens_per_sec": tokens_per_sec,
                    "perf_estimated_tokens": estimated_tokens,
                    "perf_first_token_ms": None,
                    "perf_avg_token_ms": None,
                },
            }

        if verbose:
            print("\n→ Educator: generating brief...", flush=True)
        brief = self._educator_generate(self._build_brief_prompt(user_request), task="brief")
        out("EDUCATOR", "Generation brief → Poet", brief)

        if verbose:
            print("→ Poet: generating initial draft...", flush=True)
        draft = self._poet_generate(brief)
        out("POET", "Initial draft → Educator", draft)

        revision_history = []
        approved_at_round = None
        for i in range(revisions):
            if verbose:
                print(f"→ Educator: critiquing (revision {i + 1})...", flush=True)
            critique = self._educator_generate(
                self._build_critique_prompt(draft, brief, revision_history),
                task="critique",
            )
            revision_history.append({"draft": draft, "critique": critique, "iteration": i})
            out("EDUCATOR", f"Critique (revision {i + 1}) → Poet", critique)

            honor_approval = (
                self._educator_approves(critique, draft=draft, brief=brief)
                and (min_revisions is None or i >= min_revisions)
            )
            if honor_approval:
                approved_at_round = len(revision_history)  # 1-indexed
                if verbose:
                    print("\n  ✓ Educator approved — poem complete.\n", flush=True)
                break

            user_input = ""
            if interactive:
                raw = input("\nYour direction for revision (Enter to skip): ").strip()
                user_input = raw

            # SRPO mode: Poet self-revises using (draft + critique) directly
            if self.revision_mode == "srpo":
                if verbose:
                    print(f"→ Poet: self-revising (revision {i + 1})...", flush=True)
                draft = self._poet_generate(
                    prompt=None,
                    is_revision=True,
                    revision_context={
                        "brief": brief,
                        "draft": draft,
                        "critique": critique,
                    },
                )
                out("POET", f"Self-revision {i + 1} → Educator", draft)
            else:
                # Educator mode: Educator generates revision brief + instructions
                prev_history = revision_history[:-1]
                if prev_history and verbose:
                    print("→ Educator: summarizing past advice...", flush=True)
                past_summary = self._summarize_critique_history(prev_history) if prev_history else ""

                if verbose:
                    print("→ Educator: building revision brief...", flush=True)
                revision_brief = self._educator_generate(
                    self._build_revision_brief_prompt(
                        draft, critique, brief, prev_history, user_input, verbose, past_summary
                    ),
                    task="revision_brief",
                )
                out("EDUCATOR", "Revision brief → Poet", revision_brief)

                if verbose:
                    print(f"→ Poet: generating revision {i + 1}...", flush=True)
                revision_prompt = self._build_poet_revision_prompt(
                    draft, critique, revision_brief, prev_history,
                    user_input, verbose, past_summary, brief=brief,
                )
                draft = self._poet_generate(revision_prompt, is_revision=True)
                out("POET", f"Revision {i + 1} draft → Educator", draft)

        # Append final draft so revision_round_changes has full sequence
        if revision_history and revision_history[-1]["draft"] != draft:
            revision_history.append({"draft": draft, "critique": None, "iteration": "final"})

        t_total = time.perf_counter() - t_start
        edu_name = self.educator_model_override or "qwen2.5-7b-educator-Q4_K_M"
        poet_name = self.poet_model_override or "qwen2.5-7b-poet-Q4_K_M"

        # Estimate token count (rough heuristic: ~4 chars per token)
        total_text = draft + brief
        for h in revision_history:
            if h.get("draft"):
                total_text += h["draft"]
            if h.get("critique"):
                total_text += h["critique"]
        estimated_tokens = len(total_text) // 4
        tokens_per_sec = round(estimated_tokens / t_total, 1) if t_total > 0 else 0

        return {
            "final_poem": draft,
            "generation_brief": brief,
            "revision_history": revision_history,
            "metadata": {
                "revisions": len([h for h in revision_history if h.get("critique")]),
                "approved": approved_at_round is not None,
                "approved_at_round": approved_at_round,
                "model_educator": edu_name,
                "model_poet": poet_name,
                "perf_total_sec": round(t_total, 2),
                "perf_tokens_per_sec": tokens_per_sec,
                "perf_estimated_tokens": estimated_tokens,
                "perf_first_token_ms": None,
                "perf_avg_token_ms": None,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "request", nargs="?", type=str,
        help="User's poem request (omit for interactive)",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--max-revisions", type=int, choices=[0, 1, 2, 3, 4, 5],
        help="Max revisions (0=poet only)",
    )
    parser.add_argument(
        "--reinforce", action="store_true",
        help="Run REINFORCE training stage on poet (optional)",
    )
    args = parser.parse_args()

    if args.reinforce:
        import subprocess
        modal_script = str(ROOT / "scripts" / "modal" / "modal_app.py")
        subprocess.run(
            [sys.executable, modal_script, "--reinforce"],
            cwd=str(ROOT), check=True,
        )
        return

    pipeline = PoetryPipeline(args.config)
    verbose = True

    if args.request:
        max_rev = args.max_revisions if args.max_revisions is not None else pipeline.max_revisions
        result = pipeline.generate(
            args.request,
            max_revisions=max_rev,
            verbose=verbose,
            interactive=False,
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
