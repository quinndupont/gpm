# RevFlux Benchmark

**RevFlux** measures how much each line changes during the model inference revision stage. We visualize per-line change percentage as vertical bars (bar chart) or as a histogram of change magnitudes. The benchmark includes a test harness for running many revision cycles across different prompt categories and revision lengths.

## Pedagogical Strategy: Process, Not Outcome

We evaluate **the process**, not the outcome. RevFlux does not judge whether the final poem is good. It measures:

- **How much** the model revises (line-level churn)
- **Where** change concentrates (which lines get rewritten)
- **How** revision dynamics vary by prompt type and cycle count

This reflects a pedagogical stance: the value of the system lies in the revision loop itself—the educator’s critique, the poet’s response, the iterative refinement. We want to understand whether that process behaves differently under different inputs, not whether the outputs are “poetically successful.”

## Prompt Categories

| Category | Purpose |
|----------|---------|
| `famous_poetry` | Describes canonical poems without quoting them. Tests whether the model converges toward something resembling the famous original or diverges. |
| `short_generic` | Minimal, generic prompts (e.g., “a poem about rain”). Tests behavior under low constraint. |
| `cliche` | Holiday cards, gift cards, sappy romance. Tests behavior when the prompt is already cliché-heavy. |

## Usage

### Run harness

```bash
# Full run: all categories, revision lengths 1–4, all models (trained + vanilla Ollama)
python scripts/benchmarks/rev_flux/run_harness.py

# Limit prompts, specific revision lengths, specific models
python scripts/benchmarks/rev_flux/run_harness.py --categories famous_poetry --max-revisions 2 3 --limit 2 --models trained qwen2.5-7b

# List available models (from config/rev_flux_models.yaml)
python scripts/benchmarks/rev_flux/run_harness.py --list-models

# Dry run to see config
python scripts/benchmarks/rev_flux/run_harness.py --dry-run
```

Models are defined in `config/rev_flux_models.yaml`. Run `ollama pull <model>` before using vanilla models.

Output: `data/rev_flux/*.json` per run, plus `summary.json`. Each run includes `metadata.approved` and `metadata.approved_at_round` (1-indexed) for approval timing.

```bash
# Run harness and generate all plots
python scripts/benchmarks/rev_flux/run_harness.py --visualize
```

### Visualize

```bash
# Histogram of change percentages
python scripts/benchmarks/rev_flux/visualize.py data/rev_flux/famous_poetry_0_rev3.json -o hist.png

# Bar chart (each bar = one line’s change %)
python scripts/benchmarks/rev_flux/visualize.py data/rev_flux/famous_poetry_0_rev3.json -o bars.png --bars

# From summary (aggregate across all runs)
python scripts/benchmarks/rev_flux/visualize.py data/rev_flux/summary.json -o aggregate.png --title "RevFlux: Aggregate Line Change"
```

### Harness visualizations

`visualize_harness.py` produces:

| Plot | Description |
|------|-------------|
| **Stanza structure map** | 2D blocks per stanza, colored by mean line change % |
| **Line stability index** | Bar chart: rounds each line stayed unchanged |
| **Per-category comparison** | Box plot of change % by prompt category |
| **Revision-length comparison** | Box plot of change % by max revisions |
| **Approval timing** | Distribution of which round educator approved (or max reached) |
| **Model comparison** | Box plot of change % by model (trained vs vanilla Ollama) |

```bash
python scripts/benchmarks/rev_flux/visualize_harness.py data/rev_flux -o data/rev_flux/plots
# Single run: stanza + stability only
python scripts/benchmarks/rev_flux/visualize_harness.py --single data/rev_flux/famous_poetry_0_rev3.json -o data/rev_flux/plots
```

### Batch visualize

```bash
./scripts/benchmarks/rev_flux/visualize_all.sh [data_dir] [output_dir]
# Default: data/rev_flux -> data/rev_flux/plots
```

Requires: `matplotlib`
