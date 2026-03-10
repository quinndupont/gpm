#!/bin/bash
# Batch visualize RevFlux outputs (model comparisons only). Run from project root.
# Uses: uv run --with matplotlib python (or set PY=python3)
DATA=${1:-data/rev_flux}
OUT=${2:-data/rev_flux/plots}
mkdir -p "$OUT"
RUN_PY=${RUN_PY:-"uv run --with matplotlib python"}

# Remove legacy aggregate/per-run plots (no longer generated)
rm -f "$OUT/aggregate_hist.png" "$OUT/aggregate_bars.png" \
  "$OUT"/*_hist.png "$OUT"/*_bars.png \
  "$OUT"/stanza_*.png "$OUT"/stability_*.png

# Harness comparison plots (category, revision length, model, approval timing)
$RUN_PY scripts/benchmarks/rev_flux/visualize_harness.py "$DATA" -o "$OUT" --comparison-only

# Quantitative revision dashboard
$RUN_PY scripts/benchmarks/rev_flux/visualize_dashboard.py "$DATA" -o "$OUT"

echo "Plots saved to $OUT"
