#!/usr/bin/env python3
"""Generate Socratic educator training examples with request_poem tool calls.

Produces multi-turn conversations where the educator:
- Asks leading questions instead of giving answers
- Suggests partial lines (never more than two consecutive)
- Calls request_poem to show illustrative examples via the poet
- Explicitly refuses to write complete poems

Output: data/educator_training/socratic_examples.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

from models.prompts.loader import get_persona, get_tool
from scripts.data_generation.claude_utils import (
    CLAUDE_SONNET_4_5,
    call_claude,
    get_educator_system_prompt,
    load_poems,
    poem_text,
)

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
POET_TRAINING = ROOT / "data" / "poet_training"

SCENARIOS = [
    {
        "id": "form_explanation",
        "desc": "Student asks how a poetic form works",
        "user_templates": [
            "How does a {form} work? Can you show me one?",
            "I want to write a {form} but I've never tried. What should I know?",
            "What makes a good {form}? I'm confused about the structure.",
            "Can you explain the {form} form and show me an example?",
        ],
        "forms": [
            "sonnet", "villanelle", "ghazal", "pantoum", "sestina",
            "limerick", "haiku", "ballad", "ode", "terza rima",
        ],
        "needs_tool": True,
    },
    {
        "id": "technique_demo",
        "desc": "Student wants to see a technique in action",
        "user_templates": [
            "I keep hearing about {technique} but I don't really get it. Can you show me?",
            "How do I use {technique} effectively in a poem?",
            "What does good {technique} look like? I think my poems are flat.",
            "My workshop says I need to work on {technique}. Help?",
        ],
        "techniques": [
            "enjambment", "volta", "slant rhyme", "caesura",
            "anaphora", "concrete imagery", "synecdoche",
            "end-stopped vs run-on lines", "internal rhyme",
        ],
        "needs_tool": True,
    },
    {
        "id": "critique_socratic",
        "desc": "Student submits a poem — educator critiques via questions",
        "needs_tool": False,
    },
    {
        "id": "write_refusal",
        "desc": "Student asks educator to write a poem — educator refuses and coaches",
        "user_templates": [
            "Write me a {form} about {topic}.",
            "Can you write a poem about {topic} for me?",
            "I need a {form} about {topic} by tomorrow. Just write it.",
            "Generate a poem about {topic}.",
        ],
        "forms": ["sonnet", "villanelle", "haiku", "ballad", "free verse poem"],
        "topics": [
            "autumn", "lost love", "the sea", "childhood memories",
            "a city at night", "grief", "spring rain", "solitude",
        ],
        "needs_tool": True,
    },
    {
        "id": "revision_coaching",
        "desc": "Student revises and educator coaches with questions about specific lines",
        "needs_tool": False,
    },
]


def _build_tool_call_block(brief: str, purpose: str) -> str:
    """Format a tool call in Qwen3 chat template style."""
    call = {"name": "request_poem", "arguments": {"brief": brief, "purpose": purpose}}
    return f"<tool_call>\n{json.dumps(call)}\n</tool_call>"


def _generate_form_explanation(scenario: dict, model: str) -> dict | None:
    """Generate a form-explanation example with tool call."""
    system = get_educator_system_prompt()
    tool_schema = get_tool("request_poem")
    form = random.choice(scenario["forms"])
    user_msg = random.choice(scenario["user_templates"]).format(form=form)

    prompt = (
        f"You are the poetry educator described in your system prompt. A student says:\n\n"
        f'"{user_msg}"\n\n'
        f"Respond in character. You MUST:\n"
        f"1. Explain the {form} form using Socratic questions (e.g. 'What do you notice about...')\n"
        f"2. Include a tool call to request_poem to show an example. Format the tool call as:\n"
        f"<tool_call>\n"
        f'{{"name": "request_poem", "arguments": {{"brief": "...", "purpose": "..."}}}}\n'
        f"</tool_call>\n"
        f"3. NEVER write a complete poem yourself — suggest at most one or two lines\n"
        f"4. After the tool result, discuss what the example demonstrates\n\n"
        f"Write the FULL assistant response (including the <tool_call> block). "
        f"Keep it under 300 words excluding the tool call."
    )

    try:
        raw = call_claude(prompt, system, model=model, max_tokens=1200)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    # Extract the tool call to generate the poem separately
    import re
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.DOTALL)
    if not tc_match:
        return None

    try:
        tc_data = json.loads(tc_match.group(1))
    except json.JSONDecodeError:
        return None

    brief = tc_data.get("arguments", {}).get("brief", "")
    purpose = tc_data.get("arguments", {}).get("purpose", "")
    if not brief:
        return None

    # Generate the poem via Claude (standing in for the poet model)
    poem_prompt = (
        f"Write a poem based on this brief. Output ONLY the poem, no commentary.\n\n{brief}"
    )
    try:
        poem = call_claude(poem_prompt, get_persona("poet"), model=model, max_tokens=2048)
    except Exception as e:
        print(f"  Error generating poem: {e}", file=sys.stderr)
        return None

    # Split raw into pre-tool and post-tool parts
    pre_tool = raw[:tc_match.start()].strip()
    post_tool = raw[tc_match.end():].strip()

    # Build the multi-turn conversation
    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": user_msg})

    tool_call_block = _build_tool_call_block(brief, purpose)
    assistant_with_call = f"{pre_tool}\n\n{tool_call_block}" if pre_tool else tool_call_block
    messages.append({"role": "assistant", "content": assistant_with_call})

    messages.append({
        "role": "tool",
        "content": json.dumps({"name": "request_poem", "result": poem.strip()}),
    })

    if post_tool:
        messages.append({"role": "assistant", "content": post_tool})
    else:
        # Generate a follow-up that contextualizes the example
        followup_prompt = (
            f"You just showed this {form} example to a student:\n\n{poem.strip()}\n\n"
            f"Write a brief follow-up (2-4 sentences) pointing out what to notice in the "
            f"example and asking a Socratic question about it. Do NOT write more poetry."
        )
        try:
            followup = call_claude(followup_prompt, system, model=model, max_tokens=300)
            messages.append({"role": "assistant", "content": followup.strip()})
        except Exception:
            messages.append({"role": "assistant", "content": (
                f"Notice how the example handles the {form} constraints. "
                f"What patterns do you see in the end-words? "
                f"Try writing your own — start with just the first stanza."
            )})

    return {"messages": messages, "scenario": "form_explanation", "form": form}


def _generate_technique_demo(scenario: dict, model: str) -> dict | None:
    """Generate a technique demonstration with tool call."""
    system = get_educator_system_prompt()
    technique = random.choice(scenario["techniques"])
    user_msg = random.choice(scenario["user_templates"]).format(technique=technique)

    prompt = (
        f"You are the poetry educator. A student says:\n\n"
        f'"{user_msg}"\n\n'
        f"Respond in character. You MUST:\n"
        f"1. Explain {technique} with a question or two\n"
        f"2. You may suggest a single illustrative line fragment\n"
        f"3. Include a tool call to request_poem to generate an example that showcases {technique}.\n"
        f"   Format: <tool_call>\n"
        f'   {{"name": "request_poem", "arguments": {{"brief": "...", "purpose": "demonstrate {technique}"}}}}\n'
        f"   </tool_call>\n"
        f"4. After the tool result, point out exactly where {technique} appears in the example\n\n"
        f"Write the FULL assistant response (including <tool_call>). Under 250 words."
    )

    try:
        raw = call_claude(prompt, system, model=model, max_tokens=1000)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    import re
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.DOTALL)
    if not tc_match:
        return None
    try:
        tc_data = json.loads(tc_match.group(1))
    except json.JSONDecodeError:
        return None

    brief = tc_data.get("arguments", {}).get("brief", "")
    purpose = tc_data.get("arguments", {}).get("purpose", f"demonstrate {technique}")
    if not brief:
        return None

    poem_prompt = f"Write a poem based on this brief. Output ONLY the poem.\n\n{brief}"
    try:
        poem = call_claude(poem_prompt, get_persona("poet"), model=model, max_tokens=2048)
    except Exception as e:
        print(f"  Error generating poem: {e}", file=sys.stderr)
        return None

    pre_tool = raw[:tc_match.start()].strip()
    post_tool = raw[tc_match.end():].strip()

    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": user_msg})

    tool_call_block = _build_tool_call_block(brief, purpose)
    assistant_with_call = f"{pre_tool}\n\n{tool_call_block}" if pre_tool else tool_call_block
    messages.append({"role": "assistant", "content": assistant_with_call})

    messages.append({
        "role": "tool",
        "content": json.dumps({"name": "request_poem", "result": poem.strip()}),
    })

    if post_tool:
        messages.append({"role": "assistant", "content": post_tool})
    else:
        followup_prompt = (
            f"You showed this poem to demonstrate {technique}:\n\n{poem.strip()}\n\n"
            f"Point out exactly where {technique} appears (cite lines). "
            f"Ask the student a question about it. 2-3 sentences, no poetry."
        )
        try:
            followup = call_claude(followup_prompt, system, model=model, max_tokens=300)
            messages.append({"role": "assistant", "content": followup.strip()})
        except Exception:
            messages.append({"role": "assistant", "content": (
                f"Look at how {technique} works in the example above. "
                f"Can you identify the specific lines where it's most effective? "
                f"Try using it in your next draft."
            )})

    return {"messages": messages, "scenario": "technique_demo", "technique": technique}


def _generate_critique_socratic(model: str) -> dict | None:
    """Generate a Socratic critique (no tool call — pure questioning)."""
    system = get_educator_system_prompt()
    poems = load_poems(ROOT / "data" / "raw" / "good")
    if not poems:
        poems = load_poems(ANNOTATED)
    if not poems:
        return None

    poem_rec = random.choice(poems)
    poem = poem_text(poem_rec).strip()
    if not poem or len(poem) < 40:
        return None

    prompt = (
        f"You are the poetry educator. A student submits this poem for feedback:\n\n"
        f"---\n{poem}\n---\n\n"
        f"Respond using the Socratic method. You MUST:\n"
        f"1. Name 1-2 specific strengths (briefly)\n"
        f"2. Ask 2-3 targeted questions that lead the student to discover weaknesses "
        f"   (e.g. 'What happens to the rhythm in line 4?' or 'Does the final image earn its place?')\n"
        f"3. Suggest a direction using a question, not a command "
        f"   (e.g. 'What if line 7 ended with something harder than \"gently\"?')\n"
        f"4. You may quote a single phrase as alternative, but NEVER rewrite a full line\n"
        f"5. Do NOT write any poetry or provide a revised version\n\n"
        f"Under 200 words."
    )

    try:
        critique = call_claude(prompt, system, model=model, max_tokens=600)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": poem},
        {"role": "assistant", "content": critique.strip()},
    ]
    return {"messages": messages, "scenario": "critique_socratic"}


def _generate_write_refusal(scenario: dict, model: str) -> dict | None:
    """Generate an example where the educator refuses to write but offers coaching + example."""
    system = get_educator_system_prompt()
    form = random.choice(scenario["forms"])
    topic = random.choice(scenario["topics"])
    user_msg = random.choice(scenario["user_templates"]).format(form=form, topic=topic)

    prompt = (
        f"You are the poetry educator. A student says:\n\n"
        f'"{user_msg}"\n\n'
        f"Respond in character. You MUST:\n"
        f"1. Politely but firmly decline to write the poem for them\n"
        f"2. Explain briefly why writing it themselves matters\n"
        f"3. Give 2-3 concrete starting suggestions (questions or partial phrases)\n"
        f"4. Include a tool call to request_poem to show what a {form} about {topic} might look like.\n"
        f"   Format: <tool_call>\n"
        f'   {{"name": "request_poem", "arguments": {{"brief": "...", "purpose": "illustrate {form} structure for student reference"}}}}\n'
        f"   </tool_call>\n"
        f"5. After the tool result, frame it as reference material, not a solution\n\n"
        f"Write the FULL response. Under 250 words."
    )

    try:
        raw = call_claude(prompt, system, model=model, max_tokens=1000)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    import re
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.DOTALL)
    if not tc_match:
        return None
    try:
        tc_data = json.loads(tc_match.group(1))
    except json.JSONDecodeError:
        return None

    brief = tc_data.get("arguments", {}).get("brief", "")
    purpose = tc_data.get("arguments", {}).get("purpose", "")
    if not brief:
        return None

    poem_prompt = f"Write a poem based on this brief. Output ONLY the poem.\n\n{brief}"
    try:
        poem = call_claude(poem_prompt, get_persona("poet"), model=model, max_tokens=2048)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    pre_tool = raw[:tc_match.start()].strip()
    post_tool = raw[tc_match.end():].strip()

    messages = [{"role": "system", "content": system}]
    messages.append({"role": "user", "content": user_msg})

    tool_call_block = _build_tool_call_block(brief, purpose)
    assistant_with_call = f"{pre_tool}\n\n{tool_call_block}" if pre_tool else tool_call_block
    messages.append({"role": "assistant", "content": assistant_with_call})

    messages.append({
        "role": "tool",
        "content": json.dumps({"name": "request_poem", "result": poem.strip()}),
    })

    if post_tool:
        messages.append({"role": "assistant", "content": post_tool})
    else:
        messages.append({"role": "assistant", "content": (
            f"That's one way a {form} about {topic} could work — study its structure, "
            f"but your version should come from your own images and voice. "
            f"Start with the first stanza. What image comes to mind first?"
        )})

    return {"messages": messages, "scenario": "write_refusal", "form": form, "topic": topic}


def _generate_revision_coaching(model: str) -> dict | None:
    """Generate Socratic revision coaching (no tool call)."""
    system = get_educator_system_prompt()
    poems = load_poems(ROOT / "data" / "raw" / "good")
    if not poems:
        poems = load_poems(ANNOTATED)
    if not poems:
        return None

    poem_rec = random.choice(poems)
    poem = poem_text(poem_rec).strip()
    if not poem or len(poem) < 40:
        return None

    prompt = (
        f"You are the poetry educator. A student has revised their poem and wants feedback "
        f"on whether it improved. Here is the current version:\n\n---\n{poem}\n---\n\n"
        f"Respond using Socratic method:\n"
        f"1. Acknowledge 1 specific improvement (even if you have to invent what was worse before)\n"
        f"2. Ask 2 questions that push deeper — about word choice, line breaks, or imagery\n"
        f"3. Suggest ONE specific thing to try next, framed as a question "
        f"   (e.g. 'What if you cut the last two lines and ended on the image in line 8?')\n"
        f"4. NEVER rewrite lines for them. You may quote a single word or short phrase.\n\n"
        f"Under 180 words."
    )

    try:
        response = call_claude(prompt, system, model=model, max_tokens=500)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None

    user_msg = f"I revised my poem. How is this version?\n\n{poem}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": response.strip()},
    ]
    return {"messages": messages, "scenario": "revision_coaching"}


GENERATORS = {
    "form_explanation": _generate_form_explanation,
    "technique_demo": _generate_technique_demo,
    "critique_socratic": lambda _s, model: _generate_critique_socratic(model),
    "write_refusal": _generate_write_refusal,
    "revision_coaching": lambda _s, model: _generate_revision_coaching(model),
}


def main():
    parser = argparse.ArgumentParser(description="Generate Socratic educator training data")
    parser.add_argument(
        "--output", type=Path,
        default=EDUCATOR_TRAINING / "socratic_examples.jsonl",
    )
    parser.add_argument("--count", type=int, default=120, help="Total examples to generate")
    parser.add_argument("--model", type=str, default=CLAUDE_SONNET_4_5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Distribute across scenarios
    scenario_map = {s["id"]: s for s in SCENARIOS}
    weights = {
        "form_explanation": 0.25,
        "technique_demo": 0.25,
        "critique_socratic": 0.20,
        "write_refusal": 0.15,
        "revision_coaching": 0.15,
    }

    counts = {}
    remaining = args.count
    for sid, w in weights.items():
        n = int(args.count * w)
        counts[sid] = n
        remaining -= n
    # Distribute remainder
    for sid in list(counts):
        if remaining <= 0:
            break
        counts[sid] += 1
        remaining -= 1

    total = 0
    with open(args.output, "w" if args.replace else "a") as f:
        for scenario_id, target in counts.items():
            scenario = scenario_map.get(scenario_id, {})
            gen_fn = GENERATORS[scenario_id]
            generated = 0
            attempts = 0
            max_attempts = target * 3

            while generated < target and attempts < max_attempts:
                attempts += 1
                result = gen_fn(scenario, args.model)
                if result is None:
                    continue
                f.write(json.dumps(result) + "\n")
                f.flush()
                generated += 1
                total += 1
                print(
                    f"[{scenario_id} {generated}/{target}] total={total}/{args.count}",
                    flush=True,
                )

            if generated < target:
                print(
                    f"  Warning: only generated {generated}/{target} for {scenario_id}",
                    file=sys.stderr,
                )

    print(f"\nDone: {total} Socratic examples → {args.output}")


if __name__ == "__main__":
    main()
