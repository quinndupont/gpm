#!/bin/bash
# Batch visualize RevFlux outputs. Run from project root.
DATA=${1:-data/rev_flux}
OUT=${2:-data/rev_flux/plots}
mkdir -p "$OUT"

# Aggregate histogram
if [[ -f "$DATA/summary.json" ]]; then
  python scripts/benchmarks/rev_flux/visualize.py "$DATA/summary.json" -o "$OUT/aggregate_hist.png" --title "RevFlux: Aggregate Line Change"
  python scripts/benchmarks/rev_flux/visualize.py "$DATA/summary.json" -o "$OUT/aggregate_bars.png" --bars --title "RevFlux: Aggregate Per-Line (first 200)"
fi

# Per-run samples
for f in "$DATA"/*_rev*.json; do
  [[ -f "$f" ]] || continue
  [[ "$(basename "$f")" == "summary.json" ]] && continue
  base=$(basename "$f" .json)
  python scripts/benchmarks/rev_flux/visualize.py "$f" -o "$OUT/${base}_hist.png" --title "RevFlux: $base"
  python scripts/benchmarks/rev_flux/visualize.py "$f" -o "$OUT/${base}_bars.png" --bars --title "RevFlux: $base"
done

echo "Plots saved to $OUT"
