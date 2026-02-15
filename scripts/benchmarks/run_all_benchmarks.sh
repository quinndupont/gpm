#!/bin/bash
# Run RevFlux (revision dynamics) + Rhyme (rhyming form adherence) benchmarks.
set -e
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
RUN="uv run python3"

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }

TEST=false
VISUALIZE=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --test) TEST=true; shift ;;
    --visualize) VISUALIZE=true; shift ;;
    *) echo "Usage: $0 [--test] [--visualize]"; exit 1 ;;
  esac
done

REV_FLUX_DIR=${REV_FLUX_DIR:-$ROOT/data/rev_flux}
RHYME_DIR=${RHYME_DIR:-$ROOT/data/rhyme_bench}

log "=== RevFlux benchmark ==="
if $TEST; then
  $RUN scripts/benchmarks/rev_flux/run_harness.py --test --output-dir "$REV_FLUX_DIR"
else
  $RUN scripts/benchmarks/rev_flux/run_harness.py --output-dir "$REV_FLUX_DIR" ${VISUALIZE:+--visualize}
fi

log "=== Rhyme benchmark ==="
if $TEST; then
  $RUN scripts/benchmarks/rhyme_bench/run_bench.py --test --output-dir "$RHYME_DIR"
else
  $RUN scripts/benchmarks/rhyme_bench/run_bench.py --output-dir "$RHYME_DIR"
fi

log "=== Rhyme visualization ==="
uv run --with matplotlib python3 scripts/benchmarks/rhyme_bench/visualize.py "$RHYME_DIR" -o "$RHYME_DIR/plots" 2>/dev/null || log "Skipping rhyme plots (matplotlib?)"

log "=== Benchmarks complete ==="
log "RevFlux: $REV_FLUX_DIR/summary.json"
log "Rhyme:   $RHYME_DIR/summary.json"
