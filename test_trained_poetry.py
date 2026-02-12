#!/usr/bin/env python3
"""Test the trained GPM poetry generator across multiple prompt types."""
import sys
from pathlib import Path

SYSTEM_PROMPT = """You are Quinn's poetry voice. Write poetry that respects readers as philosophical equals. Warm but never condescending, with thematic weight beneath surface accessibility. Favor Anglo-Saxon base with strategic Latinate elevation. Active voice, present tense, verb-heavy over adjective-heavy. Create sonic inevitability through internal rhyme, consonance, and assonance as structure, not decoration. Vary sentence length for rhythmic contrast. Use strategic enjambment for emphasis and double meanings. Endings should surprise -- deep unintuitive jumps, not tidy resolutions. Operate on multiple registers: accessible to children, layered for adults. Models: Frost's depth beneath simplicity, Milne's whimsy with weight, Poe's technical mastery, Silverstein's accessible sophistication, Lear's musical invention."""

# Prompts spanning Quinn's voice, children's register, and broader craft
TEST_PROMPTS = [
    # Quinn-specific themes
    ("quinn_voice", "Write a poem about writing as the first autonomous technology -- older than any machine."),
    ("children", "Write a children's poem about a rabbit who freezes when spotted, from the dog's point of view."),
    # Broader craft
    ("formal", "Write a villanelle about watching someone you love grow old."),
    ("nature", "Write a short poem about the sound a frozen lake makes at night."),
    # Style stretch
    ("philosophical", "Write a poem about why we count our money but not our neighbors."),
]


def main():
    adapter_path = Path("models/adapters/gpm_lora_test")
    if not adapter_path.exists() or not (adapter_path / "adapters.safetensors").exists():
        adapter_path = Path("models/adapters/gpm_lora")
        if not adapter_path.exists() or not (adapter_path / "adapters.safetensors").exists():
            print("No trained adapter found. Train first: python orchestrator.py --phase train")
            return

    from mlx_lm import load, generate

    print(f"Loading base model + adapter from {adapter_path}...", flush=True)
    model, tokenizer = load(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        adapter_path=str(adapter_path),
    )

    for label, user_prompt in TEST_PROMPTS:
        print(f"\n{'=' * 60}")
        print(f"[{label}] {user_prompt}")
        print('=' * 60, flush=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=False)
        out = response if isinstance(response, str) else getattr(response, "text", str(response))
        print(out or "(no output)", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
