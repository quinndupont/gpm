# Rhyme Benchmark Suite

Comprehensive rhyming capability testing and diagnostic analysis for poetry generation models.

## Overview

The rhyme benchmark suite evaluates a model's ability to generate poems that adhere to specific rhyme schemes across different poetic forms. It provides both quantitative metrics and qualitative diagnostic analysis to identify specific failure patterns and guide model improvement.

**Key Features**:
- **Standard Benchmarking**: Measure rhyme density and form adherence across models
- **Diagnostic Analysis**: Categorize failures and generate actionable insights
- **Visualization**: Charts and heatmaps for quick pattern recognition
- **Extensible**: Easy to add new forms, metrics, and thresholds

**Supported Forms**: Sonnet (4 variants), Villanelle, Limerick, Quatrain, Couplets, Ballad, and more

## Metrics Reference

### Core Metrics

| Metric | Range | Description | Interpretation |
|--------|-------|-------------|----------------|
| `strict_rhyme_density` | 0.0-1.0 | Fraction of lines in perfect CMU-verified rhymes | >0.7 excellent, 0.5-0.7 good, <0.5 needs work |
| `rhyme_density` | 0.0-1.0 | Fraction including slant rhymes | Should be >0.8 for rhyming forms |
| `matches_form` | boolean | Whether detected scheme matches expected pattern | Critical for form adherence |
| `deviations_count` | integer | Lines that should rhyme but don't | Lower is better; 0 is perfect |
| `strict_rhyme_pairs` | integer | Count of perfect rhyme pairs (CMU-verified) | Higher is better for strict forms |
| `rhyme_pairs` | integer | Total rhyme pairs (perfect + slant) | Indicates overall rhyming attempt |
| `line_count` | integer | Number of lines in the poem | Should match expected for the form |
| `detected_scheme` | string | Actual rhyme scheme (e.g., "ABAB CDCD") | Compare to expected_scheme |
| `expected_scheme` | string | Target rhyme scheme for the form | Gold standard for comparison |

### Diagnostic Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `failure_rate_by_form` | Fraction of failures per poetic form | Identify which forms need more training data |
| `near_miss_ratio` | Slant rhymes where perfect expected | Indicates training data quality issues |
| `severity_score` | Weighted failure severity (0.0-1.0) | Prioritize high-impact fixes |
| `category_distribution` | Breakdown by failure type | Guide specific improvements |

### Failure Categories

1. **Scheme Violation**: Wrong rhyme pattern (e.g., ABAB → ABAC)
   - **Cause**: Model doesn't understand the target scheme
   - **Fix**: Add more examples with correct scheme adherence

2. **Near-Miss**: Slant rhymes where perfect rhymes expected (>30% slant ratio)
   - **Cause**: Training data contains too many slant rhymes
   - **Fix**: Filter training data for perfect rhymes only

3. **Density Issue**: Too few lines rhyme (strict_density < 0.7)
   - **Cause**: Weak rhyming signal in training
   - **Fix**: Use SRPO with rhyme-focused trajectories

4. **Form Confusion**: Wrong form detected or complete scheme mismatch
   - **Cause**: Model can't recognize or generate the requested form
   - **Fix**: Add more examples of this specific form

5. **Line Count Error**: Wrong number of lines (beyond ±2 tolerance)
   - **Cause**: Model doesn't understand form structure
   - **Fix**: Add structured examples showing correct line counts

6. **No Rhyme Detected**: Catastrophic failure, almost no rhymes (density < 0.1)
   - **Cause**: Complete breakdown in rhyme generation
   - **Fix**: Review model architecture and basic rhyme training

## Usage Guide

### Interactive (default)

Run with **no arguments** to configure the benchmark in the terminal: study/ablation, output directory, models, prompt subset, max revisions, diagnostic mode, form filters, and verbose logging. You get a confirmation step before any generation runs.

```bash
python scripts/benchmarks/rhyme_bench/run_bench.py
```

### Batch / CI (`--non-interactive`)

Scripts and CI must pass **`--non-interactive`**; command-line flags are ignored without it (the interactive wizard is the default human path).

```bash
# Quick test (2 prompts, trained model only)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --test --output-dir data/rhyme_bench/studies/baseline_default

# Full suite: all prompts, all models from config/rev_flux_models.yaml
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --output-dir data/rhyme_bench/studies/baseline_default

# Specific models
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --models trained qwen2.5-7b --output-dir data/rhyme_bench/studies/baseline_default

# Specific prompts
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --prompts 0 1 2 3 4 --output-dir data/rhyme_bench/studies/baseline_default

# Revision cycles (educator + poet)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --max-revisions 2 --output-dir data/rhyme_bench/studies/baseline_default

# Ablation study (backward prompt or CMU two-pass)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --study ablate_backward --output-dir data/rhyme_bench/studies/ablate_backward
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --study ablate_cmu_two_pass --output-dir data/rhyme_bench/studies/ablate_cmu_two_pass

# Optional JSON config (merged with CLI; CLI overrides for overlapping fields)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --bench-config path/to/bench.json --output-dir data/rhyme_bench/studies/baseline_default

# Exclude educator fine-tuned models
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --exclude-educator-finetuned --output-dir data/rhyme_bench/studies/baseline_default
```

**Summary fields:** Each `summary.json` includes `study_id`, `bench_config` (resolved options), and the usual aggregate metrics so runs are self-describing.

### Studies (ablation metadata)

Version-controlled **info cards** live under [`studies/`](studies/): `baseline_default`, `ablate_backward`, `ablate_cmu_two_pass`. Each folder has `CARD.yaml` (machine-readable) and a short `README.md`.

### Quick Start (visualizations)

```bash
python scripts/benchmarks/rhyme_bench/visualize.py
```

### Standard Benchmark

Run the full benchmark suite to measure overall rhyme performance (use `--non-interactive` as above). Examples in other sections of this README that omit `--non-interactive` should be read as **interactive** (run with no args) or updated to add `--non-interactive` for automation.

**Output**:
- Individual run files: `data/rhyme_bench/studies/<study_id>/rhyme_{form}_{idx}_{model}_{timestamp}.json`
- Run summaries: `data/rhyme_bench/studies/<study_id>/summary_{timestamp}.json` (each run)
- Default study folder: `data/rhyme_bench/studies/baseline_default/` (contains `summary.json` for the latest default run)

### Data Accumulation

**Automatic accumulation:** Rhyme bench preserves data across multiple runs using timestamped filenames. This allows you to gradually expand your dataset by testing new prompts, models, or configurations over time.

**Filename format:**
- Individual runs: `rhyme_{form}_{idx}_{model}_{YYYYMMDD_HHMMSS}.json`
- Run summaries: `summary_{YYYYMMDD_HHMMSS}.json`
- Example: `rhyme_sonnet_0_claude-opus-4_20260314_141530.json`

**Key behaviors:**
- Each benchmark run creates new files with unique timestamps
- Files are never overwritten - dataset grows over time
- `summary.json` always reflects the latest run (for backward compatibility)
- Visualizations and diagnostics automatically aggregate across ALL runs
- Legacy files without timestamps coexist with new timestamped files

**Building a dataset:**
```bash
# Run 1: Test initial set of models (--non-interactive required for flags)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --models trained-llama3.1-8b qwen2.5-7b-vanilla --output-dir data/rhyme_bench/studies/baseline_default

# Run 2: Add frontier models (data accumulates)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --models claude-sonnet-4 claude-opus-4 --output-dir data/rhyme_bench/studies/baseline_default

# Run 3: Test with different prompts (data continues to accumulate)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --prompts 0 1 2 --models llama4-maverick --output-dir data/rhyme_bench/studies/baseline_default

# Result: All runs preserved in data/rhyme_bench/
# Visualizations include ALL accumulated data
```

**Cleanup strategies:**

Delete runs older than N days:
```bash
# Remove runs older than 30 days
find data/rhyme_bench -name "rhyme_*_*.json" -mtime +30 -delete
find data/rhyme_bench -name "summary_*.json" -mtime +30 -delete
```

Delete specific date range:
```bash
# Delete all runs from March 14, 2026
find data/rhyme_bench -name "*_20260314_*.json" -delete
```

Keep only recent runs:
```bash
# List all timestamped files sorted by date, then delete older ones
# Keep last 100 files
ls -t data/rhyme_bench/rhyme_*_*.json | tail -n +101 | xargs rm -f
```

Clean everything and start fresh:
```bash
# WARNING: Deletes ALL benchmark data
rm -f data/rhyme_bench/rhyme_*.json
rm -f data/rhyme_bench/summary*.json
rm -f data/rhyme_bench/diagnostic*.json
rm -f data/rhyme_bench/diagnostic*.md
```

**Disk space monitoring:**
```bash
# Check total size of accumulated data
du -sh data/rhyme_bench

# Count total runs
ls data/rhyme_bench/rhyme_*.json | wc -l
```

### Diagnostic Analysis

Run diagnostic mode to identify specific failure patterns and get actionable recommendations:

```bash
# Full diagnostic analysis
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --diagnostic --output-dir data/rhyme_bench/studies/baseline_default

# Diagnostic on specific forms
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --diagnostic --forms sonnet villanelle --output-dir data/rhyme_bench/studies/baseline_default

# Diagnostic test run (fast)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --test --diagnostic --output-dir data/rhyme_bench/studies/baseline_default
```

**Output**:
- Detailed report: `data/rhyme_bench/diagnostic_report.json`
- Human-readable summary: `data/rhyme_bench/diagnostic_summary.md`
- Visualizations: `data/rhyme_bench/studies/<study_id>/plots/failure_breakdown.png`, etc.

**Example diagnostic_summary.md**:
```markdown
# Rhyme Benchmark Diagnostic Report

Total Runs: 120
Failures: 34 (28.3%)
Mean Severity: 0.42

## Failure Breakdown
| Category | Count | % of Failures | Mean Severity |
|----------|-------|---------------|---------------|
| Scheme Violation | 15 | 44.1% | 0.72 |
| Near Miss | 8 | 23.5% | 0.45 |
...

## Performance by Model

### Fine-Tuned Models

#### trained-llama3.1-8b
- **Success rate**: 85.0% (17/20)
- **Failures**: 3
- **Mean severity**: 0.32
- **Primary issues**: scheme_violation, near_miss

### Vanilla Baselines

#### llama3.1-8b-vanilla
- **Success rate**: 65.0% (13/20)
- **Failures**: 7
- **Mean severity**: 0.48
- **Primary issues**: scheme_violation, density_issue, near_miss

### Frontier Models

#### claude-sonnet-4
- **Success rate**: 92.0% (23/25)
- **Failures**: 2
- **Mean severity**: 0.25
- **Primary issues**: near_miss

## Actionable Insights
### 🔴 Priority 1: Sonnet Final Couplet
Issue: 8 sonnet failures (scheme_violation)
Severity: 0.72 (affects 8 runs)
Recommendation: Review training data for proper couplet patterns
```

### Command-Line Reference (requires `--non-interactive`)

Run `python scripts/benchmarks/rhyme_bench/run_bench.py --help` for interactive-mode help, or `python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --help` for the full batch parser. Typical flags:

- `--non-interactive` — required for any CLI flags (scripts/CI)
- `--study {baseline_default,ablate_backward,ablate_cmu_two_pass}` — ablation / condition
- `--bench-config PATH` — JSON file with the same fields as `bench_config` in `summary.json`
- `--prompts`, `--max-revisions`, `--test`, `--models`, `--list-models`, `--output-dir`, `--verbose`, `--diagnostic`, `--forms`, `--exclude-educator-finetuned`

## Diagnostic Analysis

### Understanding Failure Categories

Each failure is assigned to a primary category based on its most critical issue. The categorization uses configurable thresholds (see Configuration section).

**Categorization Logic**:
1. **No rhyme detected** (highest priority): rhyme_density < 0.1
2. **Form confusion**: >50% scheme deviation from expected
3. **Line count error**: line count outside ±2 of expected
4. **Scheme violation**: 20-50% scheme deviation
5. **Near-miss**: >30% of rhyme pairs are slant instead of perfect
6. **Density issue**: strict_rhyme_density < 0.7 (default fallback)

**Severity Scoring**:
- Weighted combination of:
  - Deviation count (30%)
  - Scheme match distance (40%)
  - Density shortfall (30%)
- Boosted for catastrophic categories (form confusion, no rhyme)
- Range: 0.0 (minor) to 1.0 (critical)

### Interpreting Diagnostic Reports

**Key Questions to Ask**:

1. **Which forms are failing most?**
   - Check `by_form` section for failure rates per form
   - High failure rate (>50%) indicates form-specific issues

2. **What are the primary issues?**
   - Check `failure_breakdown` for category distribution
   - Many scheme violations? → Training data quality
   - Many near-misses? → Slant rhyme filtering needed

3. **Where to focus effort?**
   - Check `insights` section for prioritized recommendations
   - Priority 1-3 items have highest impact (severity × count)

4. **Are failures improving over time?**
   - Compare `diagnostic_report.json` across training runs
   - Track mean_severity and failure_rate trends

### Interpreting Per-Model Performance

**Understanding Model Categories**:

- **Fine-Tuned Models** (`trained-*`): Models trained on poetry-specific data
  - Compare these against vanilla baselines to measure training effectiveness
  - Look for improvements in success rate and reduced severity

- **Vanilla Baselines** (`*-vanilla`): Untrained base models
  - Establish performance floor without fine-tuning
  - Useful for A/B testing training strategies

- **Frontier Models** (no prefix/suffix): State-of-the-art commercial models
  - Claude, GPT-4, etc. via Bedrock API
  - Useful as performance ceiling for comparison

**Key Metrics to Compare**:

1. **Success Rate**: Higher is better
   - Trained models should exceed vanilla baselines
   - Gap indicates training effectiveness

2. **Mean Severity**: Lower is better
   - Even when failures occur, they should be less severe
   - High severity suggests fundamental misunderstanding

3. **Primary Issues**: Category patterns
   - Different models may struggle with different aspects
   - Trained models should have fewer "form_confusion" issues
   - Vanilla models often show more "no_rhyme_detected" failures

**Example Interpretation**:

If `trained-llama3.1-8b` shows:
- Success rate: 75% vs vanilla baseline: 55% → +20% improvement from training
- Mean severity: 0.35 vs vanilla: 0.52 → Failures are less critical
- Primary issues: `near_miss` vs vanilla: `scheme_violation, form_confusion`
  → Training improved form understanding but rhyme quality needs work

**Action Items Based on Model Performance**:

- **Low success rate on trained model**: Review training data quality
- **Similar performance to vanilla**: Training not effective, revisit approach
- **High severity on trained model**: Add more challenging examples to training
- **Frontier models outperform by >30%**: Consider different base model or more training data

### Using Insights for Model Improvement

**Workflow**:

1. **Run diagnostic**: `python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --diagnostic --output-dir data/rhyme_bench/studies/baseline_default`

2. **Review insights**: `cat data/rhyme_bench/diagnostic_summary.md`

3. **Apply recommendations**:
   - **High scheme violations?** → Filter training data, add correct examples
   - **High near-miss ratio?** → Remove slant rhyme examples from training
   - **Form-specific failures?** → Generate more data for that form
   - **Low density across all forms?** → Use SRPO with rhyme trajectories

4. **Re-train model** with improvements

5. **Re-run benchmark**: Compare new `diagnostic_report.json` to baseline

6. **Iterate** until target metrics achieved

**Example Improvement Cycle**:
```bash
# Baseline
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --diagnostic --output-dir data/rhyme_bench/studies/baseline_default
# → Result: 35% scheme violations on sonnets

# Fix: Add 500 more perfect-rhyme sonnet examples to training data
# Re-train model

# Test improvement
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --diagnostic --forms sonnet --output-dir data/rhyme_bench/studies/baseline_default
# → Result: 15% scheme violations on sonnets (20% reduction!)
```

## Visualization Guide

### Standard Plots

**Model Comparison** (`model_comparison.png`):
- Box plot of strict_rhyme_density by model
- Identify best-performing models at a glance

**Form Adherence Rate** (`matches_form_rate.png`):
- Bar chart of form-matching success rate
- Color-coded: green (≥50%), orange (25-50%), red (<25%)

### Diagnostic Plots

**Failure Breakdown** (`failure_breakdown.png`):
- Bar chart of failure counts by category
- Color-coded by severity: red (high), orange (medium), green (low)
- Quickly identify most common failure types

**Severity Heatmap** (`severity_heatmap.png`):
- Form × Category heatmap with mean severity values
- Darker colors = higher severity
- Identify form-category combinations needing attention

**Near-Miss Analysis** (`near_miss_analysis.png`):
- Scatter plot: perfect rhyme pairs (x) vs slant pairs (y)
- Points above diagonal = more slant than perfect
- Color-coded by form
- Identify models producing too many slant rhymes

### Model Performance Plots

**Model Performance Comparison** (`model_performance_comparison.png`):
- Bar chart of success rates by model
- Grouped by type: Fine-Tuned (blue), Vanilla (gray), Frontier (red)
- Sorted by success rate within each group
- Quickly identify best and worst performing models

**Trained vs Vanilla Comparison** (`trained_vs_vanilla.png`):
- Side-by-side comparison of paired trained/vanilla models
- Left panel: Success rate comparison
- Right panel: Mean severity comparison (lower is better)
- Directly measure training effectiveness
- Only generated when paired models exist (e.g., `trained-llama3.1-8b` and `llama3.1-8b-vanilla`)

**Model Severity Comparison** (`model_severity_comparison.png`):
- Bar chart of mean severity by model (lower is better)
- Sorted by severity (ascending)
- Color-coded by model type
- Threshold lines at 0.4 (low severity) and 0.7 (high severity)
- Identify which models have the most/least critical failures

**Model Performance Dimensions** (`model_performance_dimensions.png`):
- Stacked bar chart showing failure category distribution per model
- Each color represents a different failure category
- Sorted by total failures (descending)
- Identify which models struggle with which specific issues

### Generating Visualizations

```bash
# Generate all plots (standard + diagnostic if available)
python scripts/benchmarks/rhyme_bench/visualize.py

# Custom output directory
python scripts/benchmarks/rhyme_bench/visualize.py -o /path/to/output

# Custom data directory
python scripts/benchmarks/rhyme_bench/visualize.py /path/to/data/rhyme_bench/studies/baseline_default

# With title prefix
python scripts/benchmarks/rhyme_bench/visualize.py --title "Model v2.0"
```

**Output**: All plots saved next to the run data (default: `data/rhyme_bench/studies/baseline_default/plots/`)

## Configuration

### Threshold Tuning

Default thresholds are defined in `scripts/benchmarks/rhyme_bench/diagnostic.py`:

```python
DEFAULT_THRESHOLDS = {
    "strict_density_min": 0.7,  # Below this = density_issue
    "near_miss_slant_ratio": 0.3,  # If >30% slant = near_miss
    "scheme_deviation_tolerance": 0.2,  # >20% deviation = violation
    "line_count_tolerance": 2,  # ±2 lines acceptable
}
```

**When to Adjust**:

- **Stricter standards**: Decrease `strict_density_min` to 0.8 if targeting publication-quality poetry
- **More permissive**: Increase `near_miss_slant_ratio` to 0.5 if slant rhymes are acceptable for your use case
- **Form-specific**: Different forms may need different tolerances (modify in code)

**Applying Custom Thresholds**:
```python
from scripts.benchmarks.rhyme_bench.diagnostic import DiagnosticAnalyzer

custom_thresholds = {
    "strict_density_min": 0.8,  # Stricter
    "scheme_deviation_tolerance": 0.1,  # More strict on scheme
}

analyzer = DiagnosticAnalyzer(threshold_config=custom_thresholds)
```

### Model Configuration

Models are defined in `config/rev_flux_models.yaml`:

```yaml
models:
  - id: trained
    label: "Trained (GGUF)"
    educator: "gguf"  # Use default from inference_config.yaml
    poet: "gguf"

  - id: qwen2.5-7b
    label: "Qwen 2.5 7B (Ollama)"
    educator: "ollama:qwen2.5:7b"
    poet: "ollama:qwen2.5:7b"

  - id: claude-3-5-sonnet-v2
    label: "Claude 3.5 Sonnet v2 (Bedrock)"
    educator: "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"
    poet: "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0"
```

**Model Backends**:
- `gguf` or `gguf:./path/to/model.gguf` - Local GGUF files via llama.cpp
- `ollama:model-name` - Ollama models (requires `ollama pull model-name`)
- `bedrock:model-id` - AWS Bedrock models (requires AWS credentials and `pip install boto3`)

**Bedrock Setup**:
```bash
# Install boto3
pip install boto3

# Configure AWS credentials
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region (us-east-1)

# Run benchmark with Bedrock models
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --models claude-3-5-sonnet-v2 --output-dir data/rhyme_bench/studies/baseline_default
```

Add new models to `rev_flux_models.yaml` to include them in benchmarking.

## Integration

### Continuous Integration

Add to your CI pipeline to catch rhyme regressions:

```yaml
# .github/workflows/test.yml
- name: Run rhyme benchmark
  run: |
    python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --test --diagnostic --output-dir data/rhyme_bench/studies/baseline_default

- name: Check rhyme thresholds
  run: pytest tests/test_rhyme_bench.py -m data -v
```

### Training Pipeline Integration

Run benchmarks after each training stage:

```bash
# After Stage 1 (SFT)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --diagnostic --output-dir data/rhyme_bench/stage1

# After Stage 2 (SRPO)
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --diagnostic --output-dir data/rhyme_bench/stage2

# Compare
python -c "
import json
with open('data/rhyme_bench/stage1/summary.json') as f:
    s1 = json.load(f)
with open('data/rhyme_bench/stage2/summary.json') as f:
    s2 = json.load(f)
print(f'Density improvement: {s1[\"mean_strict_rhyme_density\"]} → {s2[\"mean_strict_rhyme_density\"]}')
"
```

### Automated Regression Detection

```bash
# Save baseline
cp data/rhyme_bench/summary.json data/rhyme_bench/baseline.json

# After changes, compare
python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --output-dir data/rhyme_bench/studies/baseline_default
python -c "
import json
with open('data/rhyme_bench/baseline.json') as f:
    baseline = json.load(f)
with open('data/rhyme_bench/summary.json') as f:
    current = json.load(f)

baseline_density = baseline['mean_strict_rhyme_density']
current_density = current['mean_strict_rhyme_density']

if current_density < baseline_density - 0.1:
    print('REGRESSION: Rhyme density dropped by >0.1')
    exit(1)
else:
    print(f'OK: Density {baseline_density} → {current_density}')
"
```

## Development

### Running Tests

```bash
# All rhyme benchmark tests
pytest tests/test_rhyme_bench.py tests/test_rhyme_diagnostic.py -v

# Evaluation tests only (fast, no data required)
pytest tests/test_rhyme_bench.py tests/test_rhyme_diagnostic.py -m eval -v

# Data tests (require benchmark data)
pytest tests/test_rhyme_bench.py tests/test_rhyme_diagnostic.py -m data -v

# Specific test class
pytest tests/test_rhyme_diagnostic.py::TestFailureCategorization -v
```

### Adding New Failure Categories

1. **Add to enum** in `diagnostic.py`:
   ```python
   class FailureCategory(Enum):
       ...
       NEW_CATEGORY = "new_category"
   ```

2. **Add detection logic** in `DiagnosticAnalyzer.categorize_failure()`:
   ```python
   if some_condition:
       return FailureCategory.NEW_CATEGORY
   ```

3. **Add recommendation** in `DiagnosticReport._get_recommendation()`:
   ```python
   elif category == FailureCategory.NEW_CATEGORY.value:
       return "Recommendation text"
   ```

4. **Write tests** in `test_rhyme_diagnostic.py`:
   ```python
   def test_new_category_detected(self):
       # Test case for new category
   ```

### Extending Visualizations

Add new plot functions to `visualize.py`:

```python
def plot_custom_analysis(runs: list[dict], out_path: Path) -> None:
    """Custom visualization."""
    plt = _ensure_matplotlib()
    # ... plotting code ...
    fig.savefig(out_path, dpi=150)
    plt.close()
```

Then add to `main()`:
```python
plot_custom_analysis(runs, out_dir / "custom.png")
```

### Project Structure

```
scripts/benchmarks/rhyme_bench/
├── run_bench.py          # Main benchmark harness
├── prompts.py            # Test prompts (12 forms)
├── diagnostic.py         # Failure analysis module
├── visualize.py          # Plotting functions
└── README.md             # This file

data/rhyme_bench/
├── README.md
└── studies/
    └── baseline_default/   # Default output dir (and other study_id folders)
        ├── CARD.yaml
        ├── rhyme_*.json
        ├── summary.json
        ├── diagnostic_report.json
        ├── diagnostic_summary.md
        └── plots/
            ├── model_comparison.png
            ├── matches_form_rate.png
            └── ...

tests/
├── test_rhyme_bench.py      # Benchmark tests
└── test_rhyme_diagnostic.py # Diagnostic tests
```

## Troubleshooting

### No failures detected but model is clearly failing
- Check threshold configuration - they may be too permissive
- Verify rhyme_analyzer is working: `pytest tests/test_eval.py -k rhyme -v`

### Diagnostic report shows unexpected categories
- Review categorization logic in `DiagnosticAnalyzer.categorize_failure()`
- Check if thresholds need tuning for your use case

### Visualizations not generating
- Ensure matplotlib is installed: `pip install matplotlib`
- Check for `diagnostic_report.json` in data directory
- Run with verbose: `python -c "import matplotlib; print(matplotlib.__version__)"`

### Tests failing on fresh install
- Data tests require benchmark data: `pytest -m "not data"`
- Or generate data first: `python scripts/benchmarks/rhyme_bench/run_bench.py --non-interactive --test --output-dir data/rhyme_bench/studies/baseline_default`

## References

- Rhyme analyzer implementation: `scripts/eval/rhyme_analyzer.py`
- Form definitions: `scripts/eval/form_registry.py`
- Inference pipeline: `scripts/inference/pipeline.py`
- Training configs: `config/poet_training.yaml`, `config/srpo_training.yaml`

## License

Part of the GPM (Generative Poetry Model) project.
