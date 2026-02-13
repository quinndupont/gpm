"""Shared Claude API helpers for data generation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PERSONA = ROOT / "persona"
EDUCATOR_NAME = "Maren"


def load_env():
    """Load .env from project root."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def get_persona_doc() -> str:
    return (PERSONA / "pedagogy_design_doc.md").read_text()


def get_persona_condensed() -> str:
    return (PERSONA / "persona_condensed.txt").read_text()


def get_anti_llm() -> str:
    return (PERSONA / "anti_llm_isms.txt").read_text()


def get_educator_system_prompt() -> str:
    """Full system prompt for educator tasks."""
    persona = get_persona_doc()
    anti_llm = get_anti_llm()
    return f"""You are {EDUCATOR_NAME}, a poetry educator with the following characteristics:

{persona}

You are now responding to a student's work. Stay in character.
Your voice, opinions, and teaching approach must be consistent with the persona defined above.

CRITICAL: You are not a rubric. You are a person who has spent their life reading and writing poetry and who cares deeply about helping others find their voice. Respond as that person, not as an evaluation system.

{anti_llm}"""


def call_claude(
    user_message: str,
    system_message: str | None = None,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 4096,
) -> str:
    """Call Claude API. Loads ANTHROPIC_API_KEY from env (or .env)."""
    load_env()
    import os
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env or environment")

    client = Anthropic(api_key=api_key)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system_message:
        kwargs["system"] = system_message

    response = client.messages.create(**kwargs)
    return response.content[0].text
