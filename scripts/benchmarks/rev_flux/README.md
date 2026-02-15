# RevFlux Benchmark

**RevFlux** measures how much each line changes during the model inference revision stage. We visualize per-line change percentage as vertical bars (bar chart) or as a histogram of change magnitudes. The benchmark includes a test harness for running many revision cycles across different prompt categories and revision lengths.

## Revision scope

- **Trained**: max_revisions 0, 1, 3, 5. Rev 0 = poet only (no educator). Rev 1+ = full educator loop.
- **Vanilla**: max_revisions 0 only (poet-only output for comparisons).
- RevFlux revision analysis (line change, stability, stanza map) applies only to trained runs with rev ≥ 1.

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

# Short test (trained only, 1 prompt per category)
python scripts/benchmarks/rev_flux/run_harness.py --test --models trained

# Full: trained (0,1,3,5 rev) + vanilla llama3.2 + qwen3 (0 rev)
python scripts/benchmarks/rev_flux/run_harness.py --models trained llama3.2-3b qwen3-8b

# List available models (from config/rev_flux_models.yaml)
python scripts/benchmarks/rev_flux/run_harness.py --list-models

# Dry run to see config
python scripts/benchmarks/rev_flux/run_harness.py --dry-run
```

Models are defined in `config/rev_flux_models.yaml`. Run `ollama pull <model>` before using vanilla models.

Output: `data/rev_flux/*.json` per run, plus `summary.json`. Trained: `*_rev0_*.json` (poet only) through `*_rev5_*.json`. Vanilla: `*_rev0_*.json`. Metadata includes `perf_total_sec`, `lines_changed_per_round`, `revised_lines_per_round`.

```bash
# Run harness and generate all plots
python scripts/benchmarks/rev_flux/run_harness.py --visualize
```

### Visualize

```bash
# Histogram of change percentages
python scripts/benchmarks/rev_flux/visualize.py data/rev_flux/famous_poetry_0_rev3_trained.json -o hist.png

# Bar chart (each bar = one line’s change %)
python scripts/benchmarks/rev_flux/visualize.py data/rev_flux/famous_poetry_0_rev3_trained.json -o bars.png --bars

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
python scripts/benchmarks/rev_flux/visualize_harness.py --single data/rev_flux/famous_poetry_0_rev3_trained.json -o data/rev_flux/plots
```

### Batch visualize

```bash
./scripts/benchmarks/rev_flux/visualize_all.sh [data_dir] [output_dir]
# Default: data/rev_flux -> data/rev_flux/plots
```

Requires: `matplotlib`
