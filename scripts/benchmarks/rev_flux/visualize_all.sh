#!/bin/bash
# Batch visualize RevFlux outputs. Run from project root.
DATA=${1:-data/rev_flux}
OUT=${2:-data/rev_flux/plots}
mkdir -p "$OUT"

# Aggregate histogram
if [[ -f "$DATA/summary.json" ]]; then
  python3 scripts/benchmarks/rev_flux/visualize.py "$DATA/summary.json" -o "$OUT/aggregate_hist.png" --title "RevFlux: Aggregate Line Change"
  python3 scripts/benchmarks/rev_flux/visualize.py "$DATA/summary.json" -o "$OUT/aggregate_bars.png" --bars --title "RevFlux: Aggregate Per-Line (first 200)"
fi

# Harness visualizations (category, revision-length, approval, stanza, stability)
python3 scripts/benchmarks/rev_flux/visualize_harness.py "$DATA" -o "$OUT"

# Per-run histogram + bars
for f in "$DATA"/*_rev*.json; do
  [[ -f "$f" ]] || continue
  [[ "$(basename "$f")" == "summary.json" ]] && continue
  base=$(basename "$f" .json)
  python3 scripts/benchmarks/rev_flux/visualize.py "$f" -o "$OUT/${base}_hist.png" --title "RevFlux: $base"
  python3 scripts/benchmarks/rev_flux/visualize.py "$f" -o "$OUT/${base}_bars.png" --bars --title "RevFlux: $base"
done

echo "Plots saved to $OUT"
