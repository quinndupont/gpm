#!/usr/bin/env python3
"""Build summary stats and at-a-glance report from checked records."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def build_summary(checked: list[tuple[dict, list[str]]]) -> dict:
    """Aggregate counts by source_type, by flag, by synthetic source, and flags per source."""
    by_source = defaultdict(int)
    by_flag = defaultdict(int)
    by_synthetic_source = defaultdict(int)
    flags_per_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record, flags in checked:
        st = record.get("source_type", "unknown")
        by_source[st] += 1
        if st == "synthetic":
            author = record.get("author", "unknown")
            by_synthetic_source[author] += 1
        for f in flags:
            by_flag[f] += 1
            flags_per_source[st][f] += 1
    return {
        "by_source": dict(by_source),
        "by_flag": dict(by_flag),
        "by_synthetic_source": dict(by_synthetic_source),
        "flags_per_source": {k: dict(v) for k, v in flags_per_source.items()},
        "total": len(checked),
    }


def sample_by_type(checked: list[tuple[dict, list[str]]], n: int = 5) -> dict[str, list]:
    """Sample up to n records per source_type."""
    by_type: dict[str, list] = defaultdict(list)
    for record, flags in checked:
        st = record.get("source_type", "unknown")
        if len(by_type[st]) < n:
            by_type[st].append((record, flags))
    return dict(by_type)


def sample_by_flag(checked: list[tuple[dict, list[str]]], n: int = 5) -> dict[str, list]:
    """Sample up to n records per flag (only flagged records)."""
    by_flag: dict[str, list] = defaultdict(list)
    for record, flags in checked:
        for f in flags:
            if len(by_flag[f]) < n:
                by_flag[f].append((record, flags))
    return dict(by_flag)


def render_terminal(summary: dict, samples_by_type: dict, samples_by_flag: dict) -> str:
    """Render report to terminal (rich table if available, else plain text)."""
    lines = []
    lines.append("=== Corpus Summary ===")
    lines.append(f"Total records: {summary['total']}")
    lines.append("")
    lines.append("By source_type (good | bad | synthetic):")
    for k, v in sorted(summary["by_source"].items()):
        lines.append(f"  {k}: {v}")
    if summary.get("by_synthetic_source"):
        lines.append("")
        lines.append("Synthetic by model/source:")
        for k, v in sorted(summary["by_synthetic_source"].items(), key=lambda x: -x[1]):
            lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("By flag:")
    for k, v in sorted(summary["by_flag"].items()):
        lines.append(f"  {k}: {v}")
    if summary.get("flags_per_source"):
        lines.append("")
        lines.append("Flags per source (gaps by type):")
        for st in sorted(summary["flags_per_source"].keys()):
            lines.append(f"  [{st}]")
            for flag, cnt in sorted(summary["flags_per_source"][st].items()):
                lines.append(f"    {flag}: {cnt}")
    lines.append("")
    lines.append("=== Sample by source_type ===")
    for st, items in sorted(samples_by_type.items()):
        lines.append(f"\n[{st}]")
        for i, (rec, flags) in enumerate(items, 1):
            title = (rec.get("title") or "(no title)")[:50]
            author = (rec.get("author") or "(no author)")[:30]
            flag_str = ",".join(flags) if flags else "ok"
            lines.append(f"  {i}. {title} | {author} | {flag_str}")
    lines.append("")
    lines.append("=== Sample by flag ===")
    for flag, items in sorted(samples_by_flag.items()):
        lines.append(f"\n[{flag}]")
        for i, (rec, flist) in enumerate(items, 1):
            title = (rec.get("title") or "(no title)")[:50]
            author = (rec.get("author") or "(no author)")[:30]
            lines.append(f"  {i}. {title} | {author}")
    return "\n".join(lines)


def render_rich(summary: dict, samples_by_type: dict, samples_by_flag: dict) -> None:
    """Print report using rich tables."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold]Corpus Summary[/bold]")
    console.print(f"Total records: {summary['total']}")

    t1 = Table(title="By source_type (good | bad | synthetic)")
    t1.add_column("source_type", style="cyan")
    t1.add_column("count", justify="right")
    for k, v in sorted(summary["by_source"].items()):
        t1.add_row(k, str(v))
    console.print(t1)

    if summary.get("by_synthetic_source"):
        t1b = Table(title="Synthetic by model/source")
        t1b.add_column("model/source", style="cyan")
        t1b.add_column("count", justify="right")
        for k, v in sorted(summary["by_synthetic_source"].items(), key=lambda x: -x[1]):
            t1b.add_row(k, str(v))
        console.print(t1b)

    t2 = Table(title="By flag")
    t2.add_column("flag", style="cyan")
    t2.add_column("count", justify="right")
    for k, v in sorted(summary["by_flag"].items()):
        t2.add_row(k, str(v))
    console.print(t2)

    if summary.get("flags_per_source"):
        t3 = Table(title="Flags per source (gaps by type)")
        t3.add_column("source", style="cyan")
        t3.add_column("flag", style="yellow")
        t3.add_column("count", justify="right")
        for st in sorted(summary["flags_per_source"].keys()):
            for flag, cnt in sorted(summary["flags_per_source"][st].items()):
                t3.add_row(st, flag, str(cnt))
        console.print(t3)

    console.print("\n[bold]Sample by source_type[/bold]")
    for st, items in sorted(samples_by_type.items()):
        t = Table(title=st)
        t.add_column("title", max_width=50)
        t.add_column("author", max_width=30)
        t.add_column("flags")
        for rec, flags in items:
            t.add_row(
                (rec.get("title") or "(no title)")[:50],
                (rec.get("author") or "(no author)")[:30],
                ",".join(flags) if flags else "ok",
            )
        console.print(t)

    console.print("\n[bold]Sample by flag[/bold]")
    for flag, items in sorted(samples_by_flag.items()):
        t = Table(title=flag)
        t.add_column("title", max_width=50)
        t.add_column("author", max_width=30)
        for rec, _ in items:
            t.add_row(
                (rec.get("title") or "(no title)")[:50],
                (rec.get("author") or "(no author)")[:30],
            )
        console.print(t)


def write_markdown_report(
    path: Path,
    summary: dict,
    samples_by_type: dict,
    samples_by_flag: dict,
) -> None:
    """Write Markdown report to file."""
    lines = ["# Poem Corpus Report", ""]
    lines.append("## Summary")
    lines.append(f"- **Total records:** {summary['total']}")
    lines.append("")
    lines.append("### By source_type (good | bad | synthetic)")
    lines.append("| source_type | count |")
    lines.append("|------------|------|")
    for k, v in sorted(summary["by_source"].items()):
        lines.append(f"| {k} | {v} |")
    if summary.get("by_synthetic_source"):
        lines.append("")
        lines.append("### Synthetic by model/source")
        lines.append("| model/source | count |")
        lines.append("|--------------|------|")
        for k, v in sorted(summary["by_synthetic_source"].items(), key=lambda x: -x[1]):
            lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### By flag")
    lines.append("| flag | count |")
    lines.append("|------|------|")
    for k, v in sorted(summary["by_flag"].items()):
        lines.append(f"| {k} | {v} |")
    if summary.get("flags_per_source"):
        lines.append("")
        lines.append("### Flags per source (gaps by type)")
        lines.append("| source | flag | count |")
        lines.append("|--------|------|------|")
        for st in sorted(summary["flags_per_source"].keys()):
            for flag, cnt in sorted(summary["flags_per_source"][st].items()):
                lines.append(f"| {st} | {flag} | {cnt} |")
    lines.append("")
    lines.append("## Sample by source_type")
    for st, items in sorted(samples_by_type.items()):
        lines.append(f"### {st}")
        for rec, flags in items:
            title = (rec.get("title") or "(no title)")[:60]
            author = (rec.get("author") or "(no author)")[:40]
            flag_str = ", ".join(flags) if flags else "ok"
            lines.append(f"- **{title}** | {author} | `{flag_str}`")
        lines.append("")
    lines.append("## Sample by flag")
    for flag, items in sorted(samples_by_flag.items()):
        lines.append(f"### {flag}")
        for rec, _ in items:
            title = (rec.get("title") or "(no title)")[:60]
            author = (rec.get("author") or "(no author)")[:40]
            lines.append(f"- **{title}** | {author}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
