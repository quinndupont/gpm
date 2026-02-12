#!/usr/bin/env python3
"""Test the trained GPM poetry generator."""
from pathlib import Path


def main():
    adapter_path = Path("models/adapters/gpm_lora")
    if not adapter_path.exists() or not (adapter_path / "adapters.safetensors").exists():
        print("No trained adapter found. Run the full pipeline first:")
        print("  python orchestrator.py --phase full")
        return

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("Install mlx-lm: pip install mlx mlx-lm")
        return

    SYSTEM_PROMPT = """You are a gifted poet who writes original, evocative poetry. You write in various forms and styles, from formal sonnets to experimental free verse."""

    print("Loading model...")
    model, tokenizer = load(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        adapter_path=str(adapter_path),
    )

    prompt = "Write a villanelle about autumn that captures the bittersweet feeling of seasonal change."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print(f"Prompt: {prompt}\n")
    response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=512)
    print(response)


if __name__ == "__main__":
    main()
