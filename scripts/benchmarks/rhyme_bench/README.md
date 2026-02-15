# Rhyme Benchmark

Tests rhyming capability: prompts explicitly request rhyming forms (sonnet, villanelle, limerick, quatrain, couplets). Outputs are analyzed with `rhyme_analyzer` (CMU Pronouncing Dictionary) for strict rhyme density, form adherence, and deviations.

## Metrics

| Metric | Description |
|--------|-------------|
| `strict_rhyme_density` | Fraction of lines in perfect CMU-verified rhymes |
| `rhyme_density` | Fraction including slant rhymes |
| `matches_form` | Whether detected scheme matches expected (e.g. ABAB CDCD EFEF GG) |
| `deviations_count` | Lines that should rhyme but don't |
| `strict_rhyme_pairs` | Count of perfect rhyme pairs |

## Usage

```bash
# Full run: all 12 prompts, trained model (poet only)
python scripts/benchmarks/rhyme_bench/run_bench.py

# Short test: 2 prompts, trained only
python scripts/benchmarks/rhyme_bench/run_bench.py --test

# Specific prompts (by index)
python scripts/benchmarks/rhyme_bench/run_bench.py --prompts 0 1 2

# With revision cycles (educator feedback)
python scripts/benchmarks/rhyme_bench/run_bench.py --max-revisions 1

# Compare trained vs vanilla
python scripts/benchmarks/rhyme_bench/run_bench.py --models trained qwen2.5-7b

# List models
python scripts/benchmarks/rhyme_bench/run_bench.py --list-models
```

Output: `data/rhyme_bench/*.json` per run, plus `summary.json` with `matches_form_rate`, `mean_strict_rhyme_density`.

### Visualize

```bash
# Per-model strict rhyme density (box plot) + form adherence rate (bar chart)
uv run --with matplotlib python3 scripts/benchmarks/rhyme_bench/visualize.py data/rhyme_bench -o data/rhyme_bench/plots
```

Produces `model_comparison.png` (strict rhyme density by model) and `matches_form_rate.png` (form adherence rate by model). Requires `matplotlib`.

Models from `config/rev_flux_models.yaml`. Requires `pronouncing` for rhyme analysis.
