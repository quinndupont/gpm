#!/usr/bin/env python3
"""
Download models for inference testing.
- Ollama: pulls vanilla models for RevFlux (qwen2.5:7b-instruct, llama3.1:8b, command-r:7b)
- GGUF: downloads base model quantizations from HuggingFace for local llama.cpp inference
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

OLLAMA_MODELS = [
    "qwen2.5:7b-instruct",
    "llama3.1:8b",
    "command-r7b:7b",
]

# HuggingFace repo, local output name
# Llama: bartowski converts from meta-llama/Llama-3.1-8B-Instruct (Meta does not publish GGUF)
GGUF_DOWNLOADS = [
    ("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-Q4_K_M.gguf"),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "llama3.1-8b-instruct-Q4_K_M.gguf"),
    ("dranger003/c4ai-command-r7b-12-2024-GGUF", "command-r7b-instruct-Q4_K_M.gguf"),
]


def pull_ollama():
    """Pull Ollama models for vanilla RevFlux testing."""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ollama not found. Install: https://ollama.com")
        return False
    for model in OLLAMA_MODELS:
        print(f"Pulling {model}...")
        r = subprocess.run(["ollama", "pull", model], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"  Failed to pull {model}")
            return False
    print("Ollama models ready.")
    return True


def download_gguf():
    """Download GGUF files from HuggingFace to models/."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("pip install huggingface_hub")
        return False
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo, local_name in GGUF_DOWNLOADS:
        out = MODELS_DIR / local_name
        if out.exists():
            print(f"  {local_name} exists, skip")
            continue
        try:
            files = list_repo_files(repo)
            # Prefer single-file; split files need both parts (skip for simplicity)
            def single_gguf(f):
                return f.endswith(".gguf") and "-00001-of-" not in f and "-00002-of-" not in f
            candidates = [f for f in files if "q4_k_m" in f.lower() and single_gguf(f)]
            if not candidates:
                candidates = [f for f in files if "q5_k_m" in f.lower() and single_gguf(f)]
            if not candidates:
                candidates = [f for f in files if "q4_k" in f.lower() and single_gguf(f)]
            if not candidates:
                candidates = [f for f in files if "q3_k_m" in f.lower() and single_gguf(f)]
            if not candidates:
                candidates = [f for f in files if "q4_k" in f.lower() and single_gguf(f)]
            if not candidates:
                print(f"  No Q4_K_M/Q3_K_M/Q4_K.gguf in {repo}")
                continue
            filename = candidates[0]
            print(f"Downloading {repo} {filename} -> {local_name}...")
            path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(MODELS_DIR),
                local_dir_use_symlinks=False,
            )
            if path:
                src = Path(path)
                if src.name != local_name and src.exists():
                    src.rename(out)
        except Exception as e:
            print(f"  Failed: {e}")
            print(
                "  Note: Llama may need HuggingFace login + Meta license: "
                "huggingface-cli login",
            )
            return False
    print("GGUF models ready.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Download models for inference testing")
    ap.add_argument("--ollama", action="store_true", help="Pull Ollama models (vanilla RevFlux)")
    ap.add_argument("--gguf", action="store_true", help="Download GGUF from HuggingFace")
    ap.add_argument("--all", action="store_true", help="Ollama + GGUF")
    args = ap.parse_args()
    if not (args.ollama or args.gguf or args.all):
        ap.print_help()
        print("\nExamples:")
        print("  python scripts/download_models.py --ollama   # Vanilla RevFlux (Ollama)")
        print("  python scripts/download_models.py --gguf     # Base GGUF for local inference")
        print("  python scripts/download_models.py --all      # Both")
        return
    ok = True
    if args.ollama or args.all:
        ok = pull_ollama() and ok
    if args.gguf or args.all:
        ok = download_gguf() and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
