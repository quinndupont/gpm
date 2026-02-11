#!/usr/bin/env python3
"""Test the trained GPM model."""
from pathlib import Path


def main():
    adapter_path = Path("models/adapters/gpm_lora")
    if not adapter_path.exists():
        print("No trained adapter found. Run the full pipeline first:")
        print("  python orchestrator.py --phase full")
        return

    try:
        from mlx_lm import load, generate
    except ImportError:
        print("Install mlx-lm: pip install mlx mlx-lm")
        return

    print("Loading model...")
    model, tokenizer = load(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        adapter_path=str(adapter_path)
    )

    test_poem = """
The fog comes
on little cat feet.
It sits looking
over harbor and city
on silent haunches
and then moves on.
"""
    prompt = f"Analyze this poem:\n\n{test_poem}"
    response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
    print(response)


if __name__ == "__main__":
    main()
