#!/usr/bin/env python3
"""
Reference chat server for the trained educator model. POST /api/chat with JSON
{messages: [{role, content}, ...]} for streaming chat completion. Any client can use it.
Run from repo root: python serve_gpm.py [port]
Requires: pip install llama-cpp-python pyyaml
"""
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "inference_config.yaml"

_llama = None


def load_config():
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_llama():
    global _llama
    if _llama is None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")
        cfg = load_config()
        edu = cfg.get("educator", {})
        path = edu.get("model_path", "./models/qwen2.5-7b-educator-Q4_K_M.gguf")
        model_path = str(ROOT / path.lstrip("./"))
        _llama = Llama(
            model_path=model_path,
            n_ctx=edu.get("n_ctx", 32768),
            n_gpu_layers=edu.get("n_gpu_layers", -1),
            n_threads=edu.get("n_threads", 8),
            use_mmap=edu.get("use_mmap", True),
            verbose=False,
        )
    return _llama


class GPMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(args[0] if args else format)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_response(404)
            self.end_headers()
            return
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Invalid JSON"}')
            return
        messages = data.get("messages", [])
        if not messages:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"messages required"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            model = get_llama()
            for chunk in model.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=0.4,
                max_tokens=2048,
                stop=["<|im_end|>", "<|endoftext|>"],
            ):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    line = json.dumps({"message": {"content": content}}) + "\n"
                    encoded = line.encode("utf-8")
                    self.wfile.write(f"{len(encoded):x}\r\n".encode())
                    self.wfile.write(encoded)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
        except Exception as e:
            err = json.dumps({"error": str(e)}) + "\n"
            encoded = err.encode("utf-8")
            self.wfile.write(f"{len(encoded):x}\r\n".encode())
            self.wfile.write(encoded)
            self.wfile.write(b"\r\n")
        self.wfile.write(b"0\r\n\r\n")


def main():
    port = 11435
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    server = HTTPServer(("127.0.0.1", port), GPMHandler)
    print(f"GPM server http://127.0.0.1:{port} (POST /api/chat)")
    server.serve_forever()


if __name__ == "__main__":
    main()
