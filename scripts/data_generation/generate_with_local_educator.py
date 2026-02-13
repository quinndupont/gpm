#!/usr/bin/env python3
"""Generate briefs, autopsies, lessons using the local interim educator model (llama.cpp)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANNOTATED = ROOT / "data" / "annotated"
EDUCATOR_TRAINING = ROOT / "data" / "educator_training"
MODELS = ROOT / "models"

sys.path.insert(0, str(ROOT))
from scripts.data_generation.claude_utils import (
    load_poems,
    load_requests,
    poem_text,
    RAW_GOOD,
    RAW_BAD,
)
from scripts.data_generation.generate_briefs import BRIEF_PROMPT
from scripts.data_generation.generate_autopsies import AUTOPSY_PROMPT
from scripts.data_generation.generate_lessons import LESSON_PROMPT, CRAFT_QUESTIONS

PERSONA = ROOT / "persona" / "educator_neutral.txt"


def _load_educator(model_path: Path):
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("pip install llama-cpp-python")
    system = PERSONA.read_text().strip() if PERSONA.exists() else ""
    return Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=-1,
        n_threads=8,
        use_mmap=True,
        verbose=False,
    ), system


def _generate(llm, system: str, user: str, max_tokens: int = 600, temp: float = 0.4) -> str:
    r = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        max_tokens=max_tokens,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    return r["choices"][0]["message"]["content"]


def generate_briefs(llm, system: str, requests: list, output: Path, limit: int):
    output.parent.mkdir(parents=True, exist_ok=True)
    requests = requests[:limit] if limit else requests
    with open(output, "w") as f:
        for i, req in enumerate(requests):
            print(f"[{i + 1}/{len(requests)}] Brief: {req[:50]}...", flush=True)
            user = BRIEF_PROMPT.format(user_request=req)
            brief = _generate(llm, system, user, max_tokens=500)
            f.write(json.dumps({"user_request": req, "brief": brief}) + "\n")
    print(f"Done: {len(requests)} briefs -> {output}")


def generate_autopsies(llm, system: str, output: Path, limit: int):
    poems = load_poems(RAW_BAD)
    poems = poems[:limit] if limit else poems
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for i, poem in enumerate(poems):
            text = poem_text(poem)
            if not text.strip():
                continue
            print(f"[{i + 1}/{len(poems)}] Autopsy...", flush=True)
            user = AUTOPSY_PROMPT.format(bad_poem_text=text)
            autopsy = _generate(llm, system, user, max_tokens=600)
            f.write(json.dumps({"poem": poem, "autopsy": autopsy}) + "\n")
    print(f"Done: {len(poems)} autopsies -> {output}")


def generate_lessons(llm, system: str, output: Path, limit: int):
    questions = CRAFT_QUESTIONS[:limit] if limit else CRAFT_QUESTIONS
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for i, q in enumerate(questions):
            print(f"[{i + 1}/{len(questions)}] Lesson: {q[:40]}...", flush=True)
            user = LESSON_PROMPT.format(question=q)
            lesson = _generate(llm, system, user, max_tokens=600)
            f.write(json.dumps({"question": q, "lesson": lesson}) + "\n")
    print(f"Done: {len(questions)} lessons -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODELS / "qwen2.5-7b-educator-interim-Q4_K_M.gguf")
    parser.add_argument("--briefs", action="store_true", help="Generate briefs")
    parser.add_argument("--autopsies", action="store_true", help="Generate autopsies")
    parser.add_argument("--lessons", action="store_true", help="Generate lessons")
    parser.add_argument("--all", action="store_true", help="Generate all (briefs, autopsies, lessons)")
    parser.add_argument("--limit-briefs", type=int, default=200)
    parser.add_argument("--limit-autopsies", type=int, default=0)
    parser.add_argument("--limit-lessons", type=int, default=10)
    parser.add_argument("--input", type=Path, default=RAW_GOOD, help="Source for brief requests")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Interim educator model not found: {args.model}", file=sys.stderr)
        print("Run the workflow through Step 6 (download interim educator) first.", file=sys.stderr)
        sys.exit(1)

    do_all = args.all or (not args.briefs and not args.autopsies and not args.lessons)
    if do_all:
        args.briefs = args.autopsies = args.lessons = True

    print("Loading interim educator model...", flush=True)
    llm, system = _load_educator(args.model)

    if args.briefs:
        requests = load_requests(args.input) if args.input.exists() else []
        if not requests:
            requests = ["Write a poem about winter light", "Write a poem about grief", "Write a poem about a meal shared with friends"]
        generate_briefs(llm, system, requests, EDUCATOR_TRAINING / "briefs.jsonl", args.limit_briefs)
    if args.autopsies:
        generate_autopsies(llm, system, ANNOTATED / "autopsies.jsonl", args.limit_autopsies if args.limit_autopsies else None)
    if args.lessons:
        generate_lessons(llm, system, EDUCATOR_TRAINING / "lessons.jsonl", args.limit_lessons)


if __name__ == "__main__":
    main()
