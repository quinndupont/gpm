#!/usr/bin/env python3
"""
Test annotation quality across models and CAT criteria.
Each criterion uses a different poem batch (different poem pairs for comparison).
Saves input/prompts/results for manual evaluation and displays NLP/performance stats.
"""
import json
import random
import re
import time
from pathlib import Path
from datetime import datetime

import ollama
import yaml

MODELS = ["llama3.2:3b"]
BATCH_SIZE = 15
OUTPUT_DIR = Path("data/annotation_test")
RATE_LIMIT_DELAY = 0.5


def load_config():
    with open("config/gpm_config.yaml") as f:
        return yaml.safe_load(f)


def load_cat_criteria():
    with open("config/prompts/cat_criteria.yaml") as f:
        return yaml.safe_load(f)


def load_corpus(corpus_path: str = "data/processed/gpm_corpus.jsonl"):
    poems = []
    with open(corpus_path) as f:
        for line in f:
            poems.append(json.loads(line))
    return poems


def filter_poems(poems, min_len=150, max_len=8000, min_lines=4):
    filtered = []
    for p in poems:
        text = p.get("text", "")
        if min_len <= len(text) <= max_len and text.count("\n") >= min_lines - 1:
            filtered.append(p)
    return filtered


def create_cat_prompt(poems: list, criterion: str, cat_config: dict) -> str:
    """Build CAT prompt for criterion and poem batch."""
    instruction = cat_config["criteria"][criterion]["instruction"]
    output_format = cat_config.get("output_format", "")

    prompt = f"Below is the collection of {len(poems)} poems. {instruction}\n\n{output_format}\n\nPOEMS:\n\n"
    for i, poem in enumerate(poems, 1):
        prompt += f'--- POEM {i}: "{poem["title"]}" by {poem["author"]} ---\n'
        text = poem["text"]
        if len(text) > 1500:
            text = text[:1500] + "\n[...truncated...]"
        prompt += text + "\n\n"
    return prompt


def annotate_batch(poems: list, criterion: str, model: str, config: dict, cat_config: dict) -> dict:
    prompt = create_cat_prompt(poems, criterion, cat_config)
    t0 = time.perf_counter()
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": cat_config.get("temperature", 1.0),
                "num_predict": config.get("ollama", {}).get("max_tokens", 2000),
                "top_p": 0.9,
            },
        )
        elapsed = time.perf_counter() - t0
        raw_text = response["response"]
        return {
            "raw_response": raw_text,
            "error": None,
            "elapsed_sec": elapsed,
            "eval_count": response.get("eval_count"),
            "eval_duration_ns": response.get("eval_duration"),
        }
    except Exception as e:
        return {
            "raw_response": None,
            "error": str(e),
            "elapsed_sec": time.perf_counter() - t0,
            "eval_count": None,
            "eval_duration_ns": None,
        }


def compute_nlp_stats(text: str) -> dict:
    if not text:
        return {"chars": 0, "words": 0, "sentences": 0, "paragraphs": 0}
    words = len(text.split())
    sentences = len(re.findall(r"[.!?]+", text)) or 1
    paragraphs = len([p for p in text.split("\n\n") if p.strip()]) or 1
    return {
        "chars": len(text),
        "words": words,
        "sentences": sentences,
        "paragraphs": paragraphs,
    }


def parse_ranked_scores(raw: str) -> dict:
    """Extract author - title : score from ranked list."""
    scores = {}
    pattern = r"\d+\.\s*(.+?)\s*-\s*(.+?)\s*:\s*(\d)"
    for m in re.finditer(pattern, raw):
        author, title = m.group(1).strip(), m.group(2).strip()
        try:
            s = int(m.group(3))
            if 1 <= s <= 5:
                scores[(author, title)] = s
        except ValueError:
            pass
    return scores


def run_test():
    config = load_config()
    cat_config = load_cat_criteria()
    criteria = list(cat_config.get("criteria", {}).keys())
    poems = load_corpus()
    eligible = filter_poems(poems)

    # Each criterion gets a different poem batch (different poem pairs for comparison)
    num_batches = len(criteria)
    needed = num_batches * BATCH_SIZE
    if len(eligible) < needed:
        raise SystemExit(f"Need {needed} eligible poems, found {len(eligible)}")

    random.seed(42)
    shuffled = eligible.copy()
    random.shuffle(shuffled)

    poem_batches = []
    for i in range(num_batches):
        batch = shuffled[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        poem_batches.append(batch)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir()

    results = []
    print("\n" + "═" * 60)
    print("  GPM Annotation Model Test (CAT Multi-Criterion)")
    print("═" * 60)
    print(f"  Criteria: {criteria}")
    print(f"  Models: {MODELS}")
    print(f"  Batch size: {BATCH_SIZE}  |  Different poem batch per criterion")
    print("═" * 60 + "\n")

    for c_idx, criterion in enumerate(criteria):
        batch_poems = poem_batches[c_idx]
        for model in MODELS:
            print(f"  [{criterion}] {model} ({len(batch_poems)} poems)")
            result = annotate_batch(batch_poems, criterion, model, config, cat_config)
            record = {
                "criterion": criterion,
                "model": model,
                "poem_batch": [{"author": p["author"], "title": p["title"], "original_id": p.get("original_id"), "text": p["text"]} for p in batch_poems],
                "prompt": create_cat_prompt(batch_poems, criterion, cat_config),
                "raw_response": result["raw_response"],
                "error": result["error"],
                "elapsed_sec": result.get("elapsed_sec"),
                "eval_count": result.get("eval_count"),
                "eval_duration_ns": result.get("eval_duration_ns"),
            }
            record["response_stats"] = compute_nlp_stats(result["raw_response"] or "")
            record["parsed_scores"] = parse_ranked_scores(result["raw_response"] or "") if isinstance(result.get("raw_response"), str) else {}
            results.append(record)
            time.sleep(RATE_LIMIT_DELAY)

    out_jsonl = run_dir / "annotations.jsonl"
    with open(out_jsonl, "w") as f:
        for r in results:
            r_copy = {k: v for k, v in r.items() if k != "parsed_scores"}
            r_copy["parsed_scores"] = [list(k) + [v] for k, v in r.get("parsed_scores", {}).items()]
            f.write(json.dumps(r_copy) + "\n")

    poems_out = run_dir / "poem_batches.json"
    with open(poems_out, "w") as f:
        json.dump(
            {c: [{"author": p["author"], "title": p["title"], "original_id": p.get("original_id")} for p in poem_batches[i]] for i, c in enumerate(criteria)},
            f,
            indent=2,
        )

    print_stats(results, run_dir, corpus_size=len(eligible), criteria=criteria)
    print(f"\n  Output: {run_dir}")
    print(f"  - annotations.jsonl  (prompts, responses, per criterion/model)")
    print(f"  - poem_batches.json   (which poems per criterion)")
    print(f"  - stats.txt           (NLP + performance statistics)")
    return run_dir


def extract_ratings(raw: str, criterion: str) -> dict:
    """Extract scores from ranked list for stats."""
    scores = parse_ranked_scores(raw or "")
    return {"scores": list(scores.values()), "criterion": criterion}


def format_duration(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.1f}h"


def print_stats(results: list, run_dir: Path, corpus_size: int = 0, criteria: list = None):
    criteria = criteria or []
    by_model = {}
    for r in results:
        m = r["model"]
        if m not in by_model:
            by_model[m] = {"responses": [], "errors": 0, "ratings": [], "elapsed": [], "eval_count": [], "eval_duration_ns": []}
        if r["error"]:
            by_model[m]["errors"] += 1
        else:
            by_model[m]["responses"].append(r["response_stats"])
            by_model[m]["ratings"].append(extract_ratings(r["raw_response"] or "", r.get("criterion", "")))
        if r.get("elapsed_sec") is not None:
            by_model[m]["elapsed"].append(r["elapsed_sec"])
        if r.get("eval_count") is not None:
            by_model[m]["eval_count"].append(r["eval_count"])
        if r.get("eval_duration_ns") is not None:
            by_model[m]["eval_duration_ns"].append(r["eval_duration_ns"])

    num_calls = len(results)
    batch_size = 15
    avg_sec_per_call = sum(r.get("elapsed_sec") or 0 for r in results) / num_calls if num_calls else 0

    stats_lines = []
    stats_lines.append("═" * 70)
    stats_lines.append("  PERFORMANCE BY MODEL")
    stats_lines.append("═" * 70)
    for model in MODELS:
        data = by_model.get(model, {"elapsed": [], "eval_count": [], "eval_duration_ns": [], "errors": 0})
        if data["elapsed"]:
            total_sec = sum(data["elapsed"])
            avg_sec = total_sec / len(data["elapsed"])
            total_tokens = sum(data["eval_count"]) if data["eval_count"] else 0
            total_ns = sum(data["eval_duration_ns"]) if data["eval_duration_ns"] else 0
            ms_per_token = (total_ns / 1e6) / total_tokens if total_tokens else 0
            stats_lines.append(f"\n  {model}")
            stats_lines.append("  " + "─" * 60)
            stats_lines.append(f"  Total time: {format_duration(total_sec)}")
            stats_lines.append(f"  Avg per batch: {avg_sec:.2f}s")
            stats_lines.append(f"  Tokens generated: {total_tokens:,}  |  Time per token: {ms_per_token:.2f} ms")
            if corpus_size:
                num_batches_full = (corpus_size + batch_size - 1) // batch_size
                est_sec = num_batches_full * len(criteria) * avg_sec
                stats_lines.append(f"  Est. full corpus ({corpus_size:,} poems, 5 criteria): {format_duration(est_sec)}")

    stats_lines.append("\n" + "═" * 70)
    stats_lines.append("  NLP STATISTICS BY MODEL")
    stats_lines.append("═" * 70)

    for model in MODELS:
        data = by_model.get(model, {"responses": [], "errors": 0, "ratings": []})
        stats_lines.append(f"\n  {model}")
        stats_lines.append("  " + "─" * 60)
        if data["errors"]:
            stats_lines.append(f"  ✗ Errors: {data['errors']}")
        if data["responses"]:
            chars = [x["chars"] for x in data["responses"]]
            words = [x["words"] for x in data["responses"]]
            stats_lines.append(f"  Response length — chars: {min(chars):,}–{max(chars):,} (avg {sum(chars)/len(chars):,.0f})")
            stats_lines.append(f"  Response length — words: {min(words):,}–{max(words):,} (avg {sum(words)/len(words):,.0f})")
            all_scores = []
            for r in data["ratings"]:
                all_scores.extend(r.get("scores", []))
            if all_scores:
                stats_lines.append(f"  Extracted scores (1–5): avg {sum(all_scores)/len(all_scores):.2f}  (n={len(all_scores)})")

    stats_lines.append("\n" + "═" * 70)
    stats_lines.append("  AGGREGATE")
    stats_lines.append("═" * 70)
    all_resp = [r["response_stats"] for r in results if not r["error"]]
    if all_resp:
        chars = [x["chars"] for x in all_resp]
        words = [x["words"] for x in all_resp]
        stats_lines.append(f"  Total successful: {len(all_resp)}  |  Avg chars: {sum(chars)/len(chars):,.0f}  |  Avg words: {sum(words)/len(words):,.0f}")
    stats_lines.append(f"  Criteria: {criteria}  |  Different poem batch per criterion")

    stats_txt = "\n".join(stats_lines)
    (run_dir / "stats.txt").write_text(stats_txt)

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        perf_table = Table(title="Performance by Model", show_header=True, header_style="bold green")
        perf_table.add_column("Model", style="dim")
        perf_table.add_column("Total time", justify="right")
        perf_table.add_column("ms/token", justify="right")
        perf_table.add_column("Est. full corpus", justify="right")

        for model in MODELS:
            data = by_model.get(model, {"elapsed": [], "eval_count": [], "eval_duration_ns": [], "errors": 0})
            if data["elapsed"]:
                total_sec = sum(data["elapsed"])
                avg_sec = total_sec / len(data["elapsed"])
                total_tokens = sum(data["eval_count"]) if data["eval_count"] else 0
                total_ns = sum(data["eval_duration_ns"]) if data["eval_duration_ns"] else 0
                ms_per_token = (total_ns / 1e6) / total_tokens if total_tokens else 0
                num_batches_full = (corpus_size + batch_size - 1) // batch_size if corpus_size else 0
                est_corpus = format_duration(num_batches_full * len(criteria) * avg_sec) if corpus_size else "—"
                perf_table.add_row(model, format_duration(total_sec), f"{ms_per_token:.2f}", est_corpus)
            else:
                perf_table.add_row(model, "—", "—", "—")

        nlp_table = Table(title="NLP Statistics by Model", show_header=True, header_style="bold cyan")
        nlp_table.add_column("Model", style="dim")
        nlp_table.add_column("Chars (avg)", justify="right")
        nlp_table.add_column("Words (avg)", justify="right")
        nlp_table.add_column("Score avg", justify="right")
        nlp_table.add_column("Errors", justify="right")

        for model in MODELS:
            data = by_model.get(model, {"responses": [], "errors": 0, "ratings": []})
            if data["responses"]:
                chars = [x["chars"] for x in data["responses"]]
                words = [x["words"] for x in data["responses"]]
                all_scores = []
                for r in data["ratings"]:
                    all_scores.extend(r.get("scores", []))
                score_avg = f"{sum(all_scores)/len(all_scores):.2f}" if all_scores else "—"
                nlp_table.add_row(
                    model,
                    f"{sum(chars)/len(chars):,.0f}",
                    f"{sum(words)/len(words):,.0f}",
                    score_avg,
                    str(data["errors"]) if data["errors"] else "0",
                )
            else:
                nlp_table.add_row(model, "—", "—", "—", str(data.get("errors", 0)))

        console.print()
        console.print(Panel(perf_table, title="Performance", border_style="green"))
        console.print(Panel(nlp_table, title="NLP Statistics", border_style="cyan"))
        console.print(stats_txt)
    except ImportError:
        print(stats_txt)


if __name__ == "__main__":
    try:
        ollama.list()
    except Exception as e:
        print(f"Error: Ollama not available. Run 'ollama serve' first.\n{e}")
        exit(1)
    run_test()
