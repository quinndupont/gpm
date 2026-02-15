#!/bin/bash
# Batch visualize RevFlux outputs. Run from project root.
# Uses: uv run --with matplotlib python (or set PY=python3)
DATA=${1:-data/rev_flux}
OUT=${2:-data/rev_flux/plots}
mkdir -p "$OUT"
RUN_PY=${RUN_PY:-"uv run --with matplotlib python"}

# Aggregate histogram (revised lines only)
if [[ -f "$DATA/summary.json" ]]; then
  $RUN_PY scripts/benchmarks/rev_flux/visualize.py "$DATA/summary.json" -o "$OUT/aggregate_hist.png" --title "RevFlux: Aggregate (revised only)"
  $RUN_PY scripts/benchmarks/rev_flux/visualize.py "$DATA/summary.json" -o "$OUT/aggregate_bars.png" --bars --title "RevFlux: Aggregate (revised only)"
fi

# Harness visualizations
$RUN_PY scripts/benchmarks/rev_flux/visualize_harness.py "$DATA" -o "$OUT"

# Per-run histogram + bars (revised lines only)
for f in "$DATA"/*_rev*.json; do
  [[ -f "$f" ]] || continue
  base=$(basename "$f" .json)
  $RUN_PY scripts/benchmarks/rev_flux/visualize.py "$f" -o "$OUT/${base}_hist.png" --title "RevFlux: $base"
  $RUN_PY scripts/benchmarks/rev_flux/visualize.py "$f" -o "$OUT/${base}_bars.png" --bars --title "RevFlux: $base"
done

echo "Plots saved to $OUT"
