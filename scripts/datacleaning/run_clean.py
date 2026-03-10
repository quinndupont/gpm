#!/usr/bin/env python3
"""CLI for poem datacleaning and corpus overview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys_path = list(__import__("sys").path)

from scripts.datacleaning.corpus_loader import load_corpus
from scripts.datacleaning.checks import check_corpus
from scripts.datacleaning.report import (
    build_summary,
    sample_by_type,
    sample_by_flag,
    render_terminal,
    render_rich,
    write_markdown_report,
)

__import__("sys").path[:] = sys_path


def _load_config(path: Path | None) -> dict:
    if path and path.exists():
        try:
            import yaml
            return yaml.safe_load(path.read_text()) or {}
        except ImportError:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Poem datacleaning and corpus overview")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "datacleaning.yaml")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["raw_good", "raw_bad", "annotated", "poet_training", "rev_flux"],
        default=None,
        help="Sources to load (default: all except rev_flux unless --rev-flux)",
    )
    parser.add_argument("--rev-flux", action="store_true", help="Include rev_flux in corpus")
    parser.add_argument("--no-rev-flux", action="store_true", help="Exclude rev_flux (default)")
    parser.add_argument("--use-llm", action="store_true", help="Run optional LLM check for placeholder title/author")
    parser.add_argument("--model", type=Path, default=None, help="GGUF path for LLM (default: from inference_config)")
    parser.add_argument("--output-report", type=Path, help="Write Markdown report to file")
    parser.add_argument("--output-cleaned", type=Path, help="Write cleaned JSONL (records passing all checks)")
    parser.add_argument("--limit", type=int, default=None, help="Limit records for testing")
    parser.add_argument("--no-rich", action="store_true", help="Use plain text instead of rich tables")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    include_rev_flux = args.rev_flux if args.rev_flux else (not args.no_rev_flux and cfg.get("include_rev_flux", False))

    records = load_corpus(sources=args.sources, include_rev_flux=include_rev_flux)
    if args.limit:
        records = records[: args.limit]

    min_lines = cfg.get("min_lines", 2)
    max_lines = cfg.get("max_lines", 500)
    min_chars = cfg.get("min_chars", 20)
    max_chars = cfg.get("max_chars", 100_000)

    checked = check_corpus(records, min_lines, max_lines, min_chars, max_chars)

    if args.use_llm:
        model_path = args.model or (ROOT / "models" / "qwen2.5-7b-educator-Q4_K_M.gguf")
        if not model_path.exists():
            try:
                import yaml
                inf_cfg = yaml.safe_load((ROOT / "config" / "inference_config.yaml").read_text())
                model_path = Path(inf_cfg.get("educator", {}).get("model_path", str(model_path)))
                model_path = ROOT / str(model_path).lstrip("./")
            except Exception:
                pass
        if model_path.exists():
            from scripts.datacleaning.checks import llm_check_placeholder
            updated = []
            for rec, flags in checked:
                if rec.get("title") or rec.get("author"):
                    try:
                        if llm_check_placeholder(model_path, rec.get("title", ""), rec.get("author", "")):
                            flags = list(flags) + ["llm_placeholder"]
                    except Exception as e:
                        print(f"LLM check failed: {e}", flush=True)
                updated.append((rec, flags))
            checked = updated

    summary = build_summary(checked)
    samples_by_type = sample_by_type(checked, n=5)
    samples_by_flag = sample_by_flag(checked, n=5)

    if args.no_rich:
        print(render_terminal(summary, samples_by_type, samples_by_flag))
    else:
        try:
            render_rich(summary, samples_by_type, samples_by_flag)
        except ImportError:
            print(render_terminal(summary, samples_by_type, samples_by_flag))

    if args.output_report:
        write_markdown_report(args.output_report, summary, samples_by_type, samples_by_flag)
        print(f"Report written to {args.output_report}")

    if args.output_cleaned:
        passed = [rec for rec, flags in checked if not flags]
        args.output_cleaned.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_cleaned, "w") as f:
            for rec in passed:
                f.write(json.dumps(rec) + "\n")
        print(f"Cleaned {len(passed)} records -> {args.output_cleaned}")


if __name__ == "__main__":
    main()
