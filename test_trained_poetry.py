#!/usr/bin/env python3
"""Minimal test: load the trained GPM poetry generator and generate a poem."""
import sys
from pathlib import Path

SYSTEM_PROMPT = """You are a gifted poet who writes original, evocative poetry. You write in various forms and styles, from formal sonnets to experimental free verse."""


def main():
    adapter_path = Path("models/adapters/gpm_lora")
    if not adapter_path.exists() or not (adapter_path / "adapters.safetensors").exists():
        print("No trained adapter at models/adapters/gpm_lora. Train first: python orchestrator.py --phase train")
        return

    from mlx_lm import load, generate

    print("Loading base model + adapter...", flush=True)
    model, tokenizer = load(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        adapter_path=str(adapter_path),
    )

    user_prompt = "Write a sonnet about artificial intelligence."
    print(f"Prompt: {user_prompt}\n", flush=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=True)
    out = response if isinstance(response, str) else getattr(response, "text", str(response))
    print(out or "(no output)", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
