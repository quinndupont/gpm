#!/usr/bin/env python3
"""Test the trained GPM poetry generator (full pipeline adapter)."""
from pathlib import Path

SYSTEM_PROMPT = """You are Quinn's poetry voice. Write poetry that respects readers as philosophical equals. Warm but never condescending, with thematic weight beneath surface accessibility. Favor Anglo-Saxon base with strategic Latinate elevation. Active voice, present tense, verb-heavy over adjective-heavy. Create sonic inevitability through internal rhyme, consonance, and assonance as structure, not decoration. Vary sentence length for rhythmic contrast. Use strategic enjambment for emphasis and double meanings. Endings should surprise -- deep unintuitive jumps, not tidy resolutions. Operate on multiple registers: accessible to children, layered for adults. Models: Frost's depth beneath simplicity, Milne's whimsy with weight, Poe's technical mastery, Silverstein's accessible sophistication, Lear's musical invention."""

TEST_PROMPTS = [
    ("quinn_voice", "Write a poem about how aesthetics is not decoration but a system that moves beyond its source."),
    ("children", "Write a children's poem about a hamster who plays a tiny instrument in the forest."),
    ("formal", "Write a sonnet about the first frost of the year."),
    ("broad", "Write a poem about sharing a meal with someone you haven't seen in years."),
]


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

    print(f"Loading model from {adapter_path}...")
    model, tokenizer = load(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        adapter_path=str(adapter_path),
    )

    for label, user_prompt in TEST_PROMPTS:
        print(f"\n{'=' * 60}")
        print(f"[{label}] {user_prompt}")
        print('=' * 60)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=512)
        out = response if isinstance(response, str) else getattr(response, "text", str(response))
        print(out or "(no output)")


if __name__ == "__main__":
    main()
