"""Shared Claude API helpers for data generation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_GOOD = ROOT / "data" / "raw" / "good"
RAW_BAD = ROOT / "data" / "raw" / "bad"

# Model choice: Opus 4.6 for hardest tasks, Sonnet 4.5 for easier
CLAUDE_OPUS_4_6 = "claude-opus-4-6"
CLAUDE_SONNET_4_5 = "claude-sonnet-4-5"

QUOTA_FILE = ROOT / "data" / ".llm_quota.json"
PERSONA = ROOT / "persona"
EDUCATOR_NEUTRAL = PERSONA / "educator_neutral.txt"


def load_env():
    """Load .env from project root."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def get_persona_condensed() -> str:
    p = PERSONA / "persona_condensed.txt"
    return p.read_text().strip() if p.exists() else ""


def get_educator_system_prompt() -> str:
    """System prompt for educator tasks. Neutral, issue-focused."""
    p = EDUCATOR_NEUTRAL if EDUCATOR_NEUTRAL.exists() else PERSONA / "persona_condensed.txt"
    return p.read_text().strip() if p.exists() else "You are a poetry educator. Identify craft issues. Give concrete directions."


def load_poems(directory: Path) -> list[dict]:
    """Load poems from directory. Supports standardized {author, title, poem} in .json, .jsonl; .txt as {poem}."""
    poems = []
    if not directory.exists():
        return poems
    for p in directory.glob("**/*.json"):
        data = json.loads(p.read_text())
        items = data if isinstance(data, list) else [data]
        for obj in items:
            if isinstance(obj, dict) and obj.get("poem"):
                poems.append(obj)
            elif isinstance(obj, dict) and (obj.get("text") or obj.get("content")):
                poems.append({"author": obj.get("author", ""), "title": obj.get("title", ""), "poem": obj.get("text") or obj.get("content", "")})
    for p in directory.glob("**/*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict) and (obj.get("poem") or obj.get("text") or obj.get("content")):
                    poems.append({"author": obj.get("author", ""), "title": obj.get("title", ""), "poem": obj.get("poem") or obj.get("text") or obj.get("content", "")})
    for p in directory.glob("**/*.txt"):
        poems.append({"author": "", "title": p.stem, "poem": p.read_text(), "source": str(p)})
    return poems


def poem_text(poem) -> str:
    """Extract poem text from standardized {author, title, poem} or legacy keys."""
    if isinstance(poem, str):
        return poem
    return poem.get("poem", poem.get("text", poem.get("content", ""))) if isinstance(poem, dict) else ""


def load_requests(source: Path) -> list[str]:
    """Load user requests from file or directory. For {author, title, poem} uses title as request."""
    requests = []
    if not source.exists():
        return requests
    if source.is_file():
        if source.suffix == ".json":
            data = json.loads(source.read_text())
            items = data if isinstance(data, list) else [data]
            for obj in items:
                if isinstance(obj, dict):
                    r = obj.get("request") or obj.get("prompt") or obj.get("title") or (obj.get("poem", "")[:80] + "..." if len(obj.get("poem", "")) > 80 else obj.get("poem", ""))
                    if r:
                        requests.append(r if isinstance(r, str) else str(r))
        elif source.suffix == ".jsonl":
            for line in source.read_text().splitlines():
                if line.strip():
                    obj = json.loads(line)
                    r = obj.get("request", obj.get("prompt", obj.get("title", str(obj))))
                    requests.append(r.strip() if isinstance(r, str) else str(r))
        else:
            requests.extend(source.read_text().strip().split("\n\n"))
    else:
        for p in source.glob("**/*.json"):
            data = json.loads(p.read_text())
            items = data if isinstance(data, list) else [data]
            for obj in items:
                if isinstance(obj, dict) and obj.get("poem"):
                    r = obj.get("request") or obj.get("prompt") or obj.get("title") or (obj["poem"][:80] + "..." if len(obj.get("poem", "")) > 80 else obj.get("poem", ""))
                    if r:
                        requests.append(r if isinstance(r, str) else str(r))
        for p in source.glob("**/*.jsonl"):
            for line in p.read_text().splitlines():
                if line.strip():
                    obj = json.loads(line)
                    r = obj.get("request", obj.get("prompt", obj.get("title", str(obj))))
                    requests.append(r.strip() if isinstance(r, str) else str(r))
        for p in source.glob("**/*.txt"):
            requests.extend(p.read_text().strip().split("\n\n"))
    return [r.strip() for r in requests if r and str(r).strip()]


def _load_quota_config():
    cfg_path = ROOT / "config" / "data_generation.yaml"
    if not cfg_path.exists():
        return {"opus_max": 50, "sonnet_max": 400}, {"backend": "ollama", "model": "qwen2.5:7b-instruct"}
    import yaml
    data = yaml.safe_load(cfg_path.read_text()) or {}
    quotas = data.get("quotas", {})
    local = data.get("local", {})
    return quotas, local


def _load_quota_state() -> dict:
    if not QUOTA_FILE.exists():
        return {"opus_used": 0, "sonnet_used": 0}
    try:
        return json.loads(QUOTA_FILE.read_text())
    except Exception:
        return {"opus_used": 0, "sonnet_used": 0}


def _save_quota_state(state: dict):
    QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUOTA_FILE.write_text(json.dumps(state))


def _select_model(requested: str, quotas: dict, state: dict, force_anthropic: bool = False) -> tuple[str, str]:
    """Return (actual_model, provider) where provider is 'claude' or 'local'."""
    import os
    if force_anthropic or os.environ.get("FORCE_ANTHROPIC"):
        if requested.startswith("claude-"):
            return requested, "claude"
        return CLAUDE_OPUS_4_6, "claude"
    if os.environ.get("DISABLE_LLM_QUOTA"):
        return requested, "claude" if requested.startswith("claude-") else "local"

    opus_max = quotas.get("opus_max", 50)
    sonnet_max = quotas.get("sonnet_max", 400)
    opus_used = state.get("opus_used", 0)
    sonnet_used = state.get("sonnet_used", 0)

    if requested == CLAUDE_OPUS_4_6:
        if opus_used < opus_max:
            return CLAUDE_OPUS_4_6, "claude"
        if sonnet_used < sonnet_max:
            return CLAUDE_SONNET_4_5, "claude"
        return "local", "local"
    if requested == CLAUDE_SONNET_4_5:
        if sonnet_used < sonnet_max:
            return CLAUDE_SONNET_4_5, "claude"
        return "local", "local"
    return requested, "claude" if requested.startswith("claude-") else "local"


def _call_local(user_message: str, system_message: str | None, max_tokens: int, local_cfg: dict) -> str:
    """Call local model via Ollama."""
    model = local_cfg.get("model", "qwen2.5:7b-instruct")
    try:
        from ollama import chat
    except ImportError:
        raise RuntimeError("pip install ollama. Run: ollama pull " + model)

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    try:
        response = chat(model=model, messages=messages, options={"num_predict": max_tokens})
        if hasattr(response, "message"):
            return getattr(response.message, "content", "") or ""
        return response.get("message", {}).get("content", "") or ""
    except Exception as e:
        raise RuntimeError(f"Ollama error (is 'ollama pull {model}' done?): {e}") from e


def call_claude(
    user_message: str,
    system_message: str | None = None,
    model: str = CLAUDE_SONNET_4_5,
    max_tokens: int = 4096,
    fallback_model: str | None = CLAUDE_SONNET_4_5,
    force_anthropic: bool = False,
) -> str:
    """Call LLM. If force_anthropic=True, always use Anthropic (never local)."""
    load_env()
    import os

    quotas, local_cfg = _load_quota_config()
    state = _load_quota_state()
    actual_model, provider = _select_model(model, quotas, state, force_anthropic)

    if provider == "local":
        if force_anthropic:
            raise RuntimeError("force_anthropic=True but quota exhausted. Use Anthropic API for hard tasks.")
        print(f"  [local/{local_cfg.get('model', 'ollama')}]", file=sys.stderr, flush=True)
        return _call_local(user_message, system_message, max_tokens, local_cfg)

    from anthropic import Anthropic, APIStatusError

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in .env or environment")

    client = Anthropic(api_key=api_key)
    kwargs = {
        "model": actual_model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": user_message}],
    }
    if system_message:
        kwargs["system"] = system_message

    try:
        response = client.messages.create(**kwargs)
        text = response.content[0].text
        if actual_model == CLAUDE_OPUS_4_6:
            state["opus_used"] = state.get("opus_used", 0) + 1
        else:
            state["sonnet_used"] = state.get("sonnet_used", 0) + 1
        _save_quota_state(state)
        return text
    except APIStatusError as e:
        if e.status_code == 529 and fallback_model and fallback_model != actual_model:
            print(f"  [529 over capacity, falling back to {fallback_model}]", file=sys.stderr, flush=True)
            kwargs["model"] = fallback_model
            response = client.messages.create(**kwargs)
            text = response.content[0].text
            state["sonnet_used"] = state.get("sonnet_used", 0) + 1
            _save_quota_state(state)
            return text
        raise
