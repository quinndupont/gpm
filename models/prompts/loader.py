"""Load prompts, personas, and tool schemas from JSON. Paths relative to this module's directory."""
import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_persona_cache: dict[str, str] = {}
_prompt_cache: dict[tuple[str, str], dict] = {}
_tool_cache: dict[str, dict] = {}


def get_persona(persona_id: str) -> str:
    """Return persona text by ID. Cached."""
    if persona_id in _persona_cache:
        return _persona_cache[persona_id]
    path = _PROMPTS_DIR / "personas" / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Persona not found: {persona_id} ({path})")
    data = json.loads(path.read_text())
    text = data.get("text", "").strip()
    _persona_cache[persona_id] = text
    return text


def get_prompt(category: str, prompt_id: str, template: str = "default") -> str:
    """Return raw template string. Cached."""
    key = (category, prompt_id)
    if key not in _prompt_cache:
        path = _PROMPTS_DIR / category / f"{prompt_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {category}/{prompt_id} ({path})")
        _prompt_cache[key] = json.loads(path.read_text())
    data = _prompt_cache[key]
    templates = data.get("templates", {})
    if template not in templates:
        avail = list(templates)
        raise KeyError(f"Template '{template}' not in {category}/{prompt_id}. Available: {avail}")
    return templates[template]


def render_prompt(
    category: str,
    prompt_id: str,
    template: str = "default",
    **kwargs,
) -> str:
    """Return template with variables filled via str.format(**kwargs)."""
    tpl = get_prompt(category, prompt_id, template)
    return tpl.format(**kwargs)


def get_tool(tool_id: str) -> dict:
    """Return tool schema dict by ID (e.g. 'request_poem'). Cached."""
    if tool_id in _tool_cache:
        return _tool_cache[tool_id]
    path = _PROMPTS_DIR / "tools" / f"{tool_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Tool schema not found: {tool_id} ({path})")
    data = json.loads(path.read_text())
    _tool_cache[tool_id] = data
    return data


def list_tools() -> list[dict]:
    """Return all tool schemas from the tools/ directory."""
    tools_dir = _PROMPTS_DIR / "tools"
    if not tools_dir.exists():
        return []
    schemas = []
    for p in sorted(tools_dir.glob("*.json")):
        try:
            schemas.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return schemas


def get_prompt_config(category: str, prompt_id: str) -> dict:
    """Return full prompt JSON for inspection."""
    key = (category, prompt_id)
    if key not in _prompt_cache:
        path = _PROMPTS_DIR / category / f"{prompt_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {category}/{prompt_id} ({path})")
        _prompt_cache[key] = json.loads(path.read_text())
    return _prompt_cache[key].copy()
