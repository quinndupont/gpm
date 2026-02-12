#!/usr/bin/env python3
"""
Serve the trained GPM poetry generator as an Ollama-compatible API.

  python serve_gpm.py          # default port 11435
  OLLAMA_HOST=http://localhost:11435 python -c "
    import ollama
    r = ollama.generate(model='gpm', prompt='Write a villanelle about autumn.')
    print(r.get('response', ''))
  "
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


def generate(user_prompt: str, max_tokens: int = 512) -> str:
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


class OllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body) if body else {}
        except Exception as e:
            self.send_error(400, str(e))
            return
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        options = data.get("options") or {}
        max_tokens = options.get("num_predict", 512)
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

    def log_message(self, format, *args):
        print(args[0] if args else format, file=sys.stderr)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    print(f"Loading GPM adapter from {ADAPTER_PATH}...", flush=True)
    get_model()
    print(f"Ollama-compatible API at http://127.0.0.1:{port}", flush=True)
    print(f"Use: OLLAMA_HOST=http://127.0.0.1:{port} ollama run gpm 'Write a sonnet about the sea'", flush=True)
    print("Or: OLLAMA_HOST=http://127.0.0.1:{port} python -c \"import ollama; print(ollama.generate(model='gpm', prompt='Write a haiku about winter').get('response'))\"", flush=True)
    server = HTTPServer(("127.0.0.1", port), OllamaHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
