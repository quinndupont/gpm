#!/usr/bin/env python3
"""
Serve the trained GPM poetry generator as an Ollama-compatible API.

Supports:
  - GET /api/tags     — list models (returns "gpm")
  - POST /api/generate — single prompt (legacy)
  - POST /api/chat    — full conversation with context, optional streaming

  python serve_gpm.py          # default port 11435
  OLLAMA_HOST=http://localhost:11435 ollama run gpm "Write a villanelle about autumn."
"""
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ADAPTER_PATH = Path(__file__).parent / "models" / "adapters" / "gpm_lora"
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
MODEL_NAME = "gpm"
DEFAULT_PORT = 11435

SYSTEM_PROMPT = """You are Quinn's poetry voice. Write poetry that respects readers as philosophical equals. Warm but never condescending, with thematic weight beneath surface accessibility. Favor Anglo-Saxon base with strategic Latinate elevation. Active voice, present tense, verb-heavy over adjective-heavy. Create sonic inevitability through internal rhyme, consonance, and assonance as structure, not decoration. Vary sentence length for rhythmic contrast. Use strategic enjambment for emphasis and double meanings. Endings should surprise -- deep unintuitive jumps, not tidy resolutions. Operate on multiple registers: accessible to children, layered for adults. Models: Frost's depth beneath simplicity, Milne's whimsy with weight, Poe's technical mastery, Silverstein's accessible sophistication, Lear's musical invention."""

_model = _tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        if not (ADAPTER_PATH / "adapters.safetensors").exists():
            raise FileNotFoundError(
                f"No adapter at {ADAPTER_PATH}. Train first: python orchestrator.py --phase train"
            )
        from mlx_lm import load

        _model, _tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER_PATH))
    return _model, _tokenizer


def _messages_with_system(messages: list, system: str | None) -> list:
    """Prepend system prompt if not already present."""
    if not messages:
        return [{"role": "system", "content": system or SYSTEM_PROMPT}]
    if messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system or SYSTEM_PROMPT}] + messages


def generate(user_prompt: str, max_tokens: int = 512) -> str:
    """Single-prompt generation (legacy)."""
    from mlx_lm import generate as mlx_generate

    model, tokenizer = get_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def chat(
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 512,
    stream: bool = False,
):
    """
    Full conversation: build prompt from message history and generate next reply.
    messages: list of {"role": "user"|"assistant"|"system", "content": "..."}
    If stream=True, yields text chunks. Otherwise returns full response string.
    """
    model, tokenizer = get_model()
    full = _messages_with_system(messages, system)
    prompt = tokenizer.apply_chat_template(
        full, tokenize=False, add_generation_prompt=True
    )

    if stream:
        try:
            from mlx_lm import stream_generate
            for part in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
                text = part.text if hasattr(part, "text") else str(part)
                if text:
                    yield text
        except ImportError:
            from mlx_lm import generate as mlx_generate
            full_text = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
            for i in range(0, len(full_text), 32):
                yield full_text[i : i + 32]
    else:
        from mlx_lm import generate as mlx_generate
        return mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


class OllamaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path != "/api/tags":
            self.send_error(404)
            return
        payload = {"models": [{"name": MODEL_NAME}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/generate", "/api/chat"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body) if body else {}
        except Exception as e:
            self.send_error(400, str(e))
            return

        options = data.get("options") or {}
        max_tokens = options.get("num_predict", 512)

        if path == "/api/generate":
            prompt = data.get("prompt", "")
            if not prompt:
                self.send_error(400, "Missing prompt")
                return
            try:
                text = generate(prompt, max_tokens=max_tokens)
            except Exception as e:
                self.send_error(500, str(e))
                return
            payload = {"model": MODEL_NAME, "response": text, "done": True}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())
            return

        # /api/chat — full conversation with context
        messages = data.get("messages") or []
        stream = data.get("stream", True)
        extra_system = None
        if messages and messages[0].get("role") == "system":
            extra_system = messages[0].get("content", "")
            messages = messages[1:]
        # Always use GPM poetry persona; append any client system (e.g. RAG context)
        system = SYSTEM_PROMPT + ("\n\n" + extra_system if extra_system else "")

        if not messages:
            self.send_error(400, "Missing or empty messages")
            return

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                for chunk in chat(messages, system=system, max_tokens=max_tokens, stream=True):
                    line = json.dumps({"message": {"content": chunk}, "done": False}) + "\n"
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                self.wfile.write(json.dumps({"message": {"content": ""}, "done": True}).encode() + b"\n")
            else:
                full = chat(messages, system=system, max_tokens=max_tokens, stream=False)
                payload = {"message": {"content": full}, "done": True}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())
        except Exception as e:
            self.send_error(500, str(e))
            return

    def log_message(self, format, *args):
        print(args[0] if args else format, file=sys.stderr)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    print(f"Loading GPM adapter from {ADAPTER_PATH}...", flush=True)
    get_model()
    print(f"GPM API at http://127.0.0.1:{port}", flush=True)
    server = HTTPServer(("127.0.0.1", port), OllamaHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
