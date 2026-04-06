#!/usr/bin/env python3
"""Modal ASGI inference server for educator + poet models.

Serves both GGUFs from a single FastAPI app with OpenAI-compatible endpoints.
The educator model can call `request_poem` to invoke the poet mid-conversation.

Deploy:  modal deploy scripts/modal/serve_inference.py
Test:    curl -X POST https://<app-url>/v1/chat/completions -H 'Content-Type: application/json' \
           -d '{"model":"educator","messages":[{"role":"user","content":"Critique this draft..."}]}'
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import modal

_ROOT = Path(__file__).resolve().parents[2]
_INFERENCE_CFG = _ROOT / "config" / "inference_config.yaml"
_PROMPTS_DIR = _ROOT / "models" / "prompts"
_TOOLS_DIR = _PROMPTS_DIR / "tools"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "cmake")
    .pip_install(
        "llama-cpp-python>=0.2.70",
        "fastapi>=0.110",
        "uvicorn>=0.29",
        "pyyaml>=6.0",
    )
    .add_local_dir(str(_PROMPTS_DIR), "/opt/gpm/models/prompts")
    .add_local_file(str(_INFERENCE_CFG), "/opt/gpm/config/inference_config.yaml")
)

app = modal.App("poetry-inference")
gguf_vol = modal.Volume.from_name("poetry-gguf", create_if_missing=True)

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
MAX_TOOL_ROUNDS = 3


def _load_tool_schema() -> dict:
    path = Path("/opt/gpm/models/prompts/tools/request_poem.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _load_persona(persona_id: str) -> str:
    path = Path(f"/opt/gpm/models/prompts/personas/{persona_id}.json")
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    return data.get("text", "").strip()


def _list_gguf_files() -> list[str]:
    vol = Path("/vol/gguf")
    if not vol.exists():
        return []
    return sorted(p.name for p in vol.glob("*.gguf"))


def _resolve_model_path(model: str) -> str | None:
    """Map a model alias ('educator', 'poet') or filename to a /vol/gguf path."""
    vol = Path("/vol/gguf")
    if (vol / model).exists():
        return str(vol / model)
    for f in vol.glob("*.gguf"):
        if model in f.stem:
            return str(f)
    return None


@app.cls(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={"/vol/gguf": gguf_vol},
    allow_concurrent_inputs=8,
    container_idle_timeout=300,
    keep_warm=1,
)
class InferenceService:
    """Holds loaded Llama models and serves requests."""

    @modal.enter()
    def setup(self):
        from llama_cpp import Llama

        gguf_vol.reload()
        self._models: dict[str, Llama] = {}
        self._gguf_files = _list_gguf_files()
        self._tool_schema = _load_tool_schema()

        educator_persona = _load_persona("educator_neutral")
        if not educator_persona:
            educator_persona = _load_persona("educator_condensed")
        self._personas = {
            "educator": educator_persona,
            "poet": _load_persona("poet"),
        }

        for fname in self._gguf_files:
            path = f"/vol/gguf/{fname}"
            key = self._model_key(fname)
            n_ctx = 4096 if "educator" in fname else 2048
            print(f"Loading {fname} (key={key}, n_ctx={n_ctx})...")
            self._models[key] = Llama(
                model_path=path,
                n_ctx=n_ctx,
                n_gpu_layers=-1,
                n_threads=4,
                use_mmap=True,
                verbose=False,
            )
        print(f"Loaded {len(self._models)} models: {list(self._models.keys())}")

    @staticmethod
    def _model_key(filename: str) -> str:
        low = filename.lower()
        if "educator" in low:
            return "educator"
        if "poet" in low:
            return "poet"
        return Path(filename).stem

    def _get_model(self, name: str):
        from llama_cpp import Llama

        if name in self._models:
            return self._models[name]
        path = _resolve_model_path(name)
        if path:
            key = self._model_key(Path(path).name)
            if key in self._models:
                return self._models[key]
        return None

    def _chat_completion(
        self,
        model_key: str,
        messages: list[dict],
        temperature: float = 0.4,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        stop: list[str] | None = None,
    ) -> str:
        llm = self._get_model(model_key)
        if llm is None:
            raise ValueError(f"Model not loaded: {model_key}")
        stops = stop or ["<|im_end|>", "<|endoftext|>"]
        r = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repeat_penalty=1.1,
            stop=stops,
        )
        return r["choices"][0]["message"]["content"]

    def _poet_generate(self, brief: str) -> str:
        """Internal poet call for tool-call fulfillment."""
        system = self._personas.get("poet", "")
        prompt = f"Write a poem based on this brief. Output ONLY the poem.\n\n{brief}"
        return self._chat_completion(
            "poet",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=4096,
            top_p=0.95,
        )

    def _fulfill_tool_calls(self, text: str) -> tuple[str, list[dict]]:
        """Parse <tool_call> tags, execute request_poem, return (clean_text, tool_results)."""
        results = []
        for match in TOOL_CALL_RE.finditer(text):
            try:
                call = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            name = call.get("name", "")
            if name != "request_poem":
                continue
            args = call.get("arguments", call)
            brief = args.get("brief", "")
            if not brief:
                continue
            poem = self._poet_generate(brief)
            results.append({
                "name": "request_poem",
                "arguments": args,
                "result": poem,
            })
        clean = TOOL_CALL_RE.sub("", text).strip()
        return clean, results

    @modal.method()
    def chat_completions(self, body: dict) -> dict:
        model_name = body.get("model", "educator")
        messages = list(body.get("messages", []))
        temperature = body.get("temperature", 0.4)
        max_tokens = body.get("max_tokens", 2048)
        top_p = body.get("top_p", 0.9)
        tools = body.get("tools")
        stop = body.get("stop")

        role = "educator" if "educator" in model_name else "poet"

        if not any(m["role"] == "system" for m in messages):
            persona = self._personas.get(role, "")
            if persona:
                messages.insert(0, {"role": "system", "content": persona})

        if role == "educator" and tools and self._tool_schema:
            tool_desc = json.dumps([self._tool_schema], indent=2)
            sys_msg = next((m for m in messages if m["role"] == "system"), None)
            if sys_msg:
                sys_msg["content"] += (
                    f"\n\n# Tools\n\nYou have access to these tools:\n{tool_desc}\n\n"
                    "To use a tool, output:\n<tool_call>\n"
                    '{"name": "request_poem", "arguments": {"brief": "...", "purpose": "..."}}\n'
                    "</tool_call>"
                )

        t0 = time.perf_counter()

        all_tool_calls = []
        for _round in range(MAX_TOOL_ROUNDS + 1):
            raw = self._chat_completion(
                role, messages, temperature, max_tokens, top_p, stop,
            )
            if role != "educator" or not TOOL_CALL_RE.search(raw):
                break
            prefix, tool_results = self._fulfill_tool_calls(raw)
            if not tool_results:
                raw = prefix
                break
            all_tool_calls.extend(tool_results)
            if prefix:
                messages.append({"role": "assistant", "content": prefix})
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "content": json.dumps({
                        "name": tr["name"],
                        "result": tr["result"],
                    }),
                })
        else:
            pass

        elapsed = time.perf_counter() - t0
        resp_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        result = {
            "id": resp_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": raw},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
                "completion_tokens": len(raw) // 4,
                "total_tokens": (sum(len(m.get("content", "")) // 4 for m in messages)
                                 + len(raw) // 4),
            },
            "_meta": {
                "elapsed_sec": round(elapsed, 3),
                "tool_calls": all_tool_calls if all_tool_calls else None,
            },
        }
        return result

    @modal.method()
    def list_models(self) -> dict:
        models = []
        for fname in self._gguf_files:
            key = self._model_key(fname)
            models.append({
                "id": key,
                "object": "model",
                "owned_by": "gpm",
                "filename": fname,
            })
        return {"object": "list", "data": models}


@app.function(image=image, volumes={"/vol/gguf": gguf_vol})
@modal.asgi_app()
def inference_app():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    web = FastAPI(title="GPM Poetry Inference", version="1.0.0")
    svc = InferenceService()

    @web.get("/v1/models")
    async def list_models():
        return svc.list_models.remote()

    @web.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if not body.get("messages"):
            raise HTTPException(400, "messages required")
        return svc.chat_completions.remote(body)

    @web.get("/health")
    async def health():
        return {"status": "ok"}

    return web
