#!/bin/bash
# Full workflow: Anthropic hard tasks → train interim educator → local educator generates rest → train final both
# Hard tasks (Opus, force Anthropic): critiques_seed, comparisons, revision_briefs
# Interim educator: trained on seed only, used locally for briefs, autopsies, lessons
# Poet pairs: Opus (force Anthropic)
set -e
cd "$(dirname "$0")/.."

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
run() { log ">>> $*"; "$@"; }

CRITIQUES_GOOD=200
REVISION_BRIEFS=50
BRIEFS=200
LESSONS=10

SKIP_GEN=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-generation) SKIP_GEN=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$SKIP_GEN" == true ]]; then
  log "Skipping data generation. Using existing data."
  if [[ ! -f data/educator_training/train.jsonl || ! -f data/poet_training/train.jsonl ]]; then
    log "ERROR: train.jsonl not found. Run without --skip-generation first."
    exit 1
  fi
  run python3 scripts/modal/upload_data.py
  log "=== Train final educator + poet ==="
  run modal run scripts/modal/train_educator.py
  run modal run scripts/modal/train_poet.py
  log "=== Export + download final models ==="
  run modal run scripts/modal/export_gguf.py::export_educator
  run modal run scripts/modal/export_gguf.py::export_poet
  mkdir -p models
  modal volume get --force poetry-gguf qwen2.5-7b-educator-Q4_K_M.gguf models/ || true
  modal volume get --force poetry-gguf qwen2.5-7b-poet-Q4_K_M.gguf models/ || true
  log "Done."
  exit 0
fi

log "=== Step 1: Hard tasks (Anthropic Opus, no local fallback) ==="
run python3 scripts/data_generation/generate_critiques_seed.py --limit-good $CRITIQUES_GOOD
run python3 scripts/data_generation/generate_comparisons.py
run python3 scripts/data_generation/generate_revision_briefs.py --limit $REVISION_BRIEFS

log "=== Step 2: Prepare interim educator training data (seed only) ==="
run python3 scripts/data_generation/prepare_training_data.py --interim-educator --educator-only

log "=== Step 3: Upload interim educator data ==="
run python3 scripts/modal/upload_data.py

log "=== Step 4: Train interim educator ==="
run modal run scripts/modal/train_educator.py --num-epochs-override 2

log "=== Step 5: Export interim educator GGUF ==="
run modal run scripts/modal/export_gguf.py::export_educator_interim

log "=== Step 6: Download interim educator ==="
mkdir -p models
modal volume get --force poetry-gguf qwen2.5-7b-educator-interim-Q4_K_M.gguf models/ || { log "Interim educator GGUF not found."; exit 1; }

log "=== Step 7: Local educator generates briefs, autopsies, lessons ==="
run python3 scripts/data_generation/generate_with_local_educator.py --all --limit-briefs $BRIEFS --limit-lessons $LESSONS

log "=== Step 8: Poet pairs (Anthropic Opus) ==="
run python3 scripts/data_generation/generate_poet_pairs.py

log "=== Step 9: Prepare full training data ==="
run python3 scripts/data_generation/prepare_training_data.py

log "=== Step 10: Upload full data ==="
run python3 scripts/modal/upload_data.py

log "=== Step 11: Train final educator + poet ==="
run modal run scripts/modal/train_educator.py
run modal run scripts/modal/train_poet.py

log "=== Step 12: Export final models ==="
run modal run scripts/modal/export_gguf.py::export_educator
run modal run scripts/modal/export_gguf.py::export_poet

log "=== Step 13: Download final models ==="
modal volume get --force poetry-gguf qwen2.5-7b-educator-Q4_K_M.gguf models/ || { log "Educator GGUF not found."; exit 1; }
modal volume get --force poetry-gguf qwen2.5-7b-poet-Q4_K_M.gguf models/ || { log "Poet GGUF not found."; exit 1; }

log "Done. Test: python scripts/inference/pipeline.py \"Write a poem about winter light\""
