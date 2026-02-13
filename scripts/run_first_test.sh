#!/bin/bash
# First test: minimal data, interim educator flow, 1 epoch each
set -e
cd "$(dirname "$0")/.."

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
run() { log ">>> $*"; "$@"; }

PREPARED=data/educator_training/train.jsonl

if [[ -f "$PREPARED" && -f data/poet_training/train.jsonl ]]; then
  log "Prepared data exists, skipping to final train."
  run python3 scripts/modal/upload_data.py
  run modal run scripts/modal/train_educator.py --num-epochs-override 1
  run modal run scripts/modal/train_poet.py --num-epochs-override 1
  run modal run scripts/modal/export_gguf.py::export_educator
  run modal run scripts/modal/export_gguf.py::export_poet
  mkdir -p models
  modal volume get --force poetry-gguf qwen2.5-7b-educator-Q4_K_M.gguf models/ || true
  modal volume get --force poetry-gguf qwen2.5-7b-poet-Q4_K_M.gguf models/ || true
else
  log "=== Step 1: Hard tasks (Anthropic Opus) ==="
  run python3 scripts/data_generation/generate_critiques_seed.py --limit-bad 5 --limit-good 5
  run python3 scripts/data_generation/generate_comparisons.py --limit 5
  run python3 scripts/data_generation/generate_revision_briefs.py --limit 5

  log "=== Step 2: Prepare interim educator (seed only) ==="
  run python3 scripts/data_generation/prepare_training_data.py --interim-educator --educator-only --min-samples 5

  log "=== Step 3: Upload + train interim educator ==="
  run python3 scripts/modal/upload_data.py
  run modal run scripts/modal/train_educator.py --num-epochs-override 1

  log "=== Step 4: Export + download interim educator ==="
  run modal run scripts/modal/export_gguf.py::export_educator_interim
  mkdir -p models
  modal volume get --force poetry-gguf qwen2.5-7b-educator-interim-Q4_K_M.gguf models/ || { log "Interim educator not found."; exit 1; }

  log "=== Step 5: Local educator generates briefs, autopsies, lessons ==="
  run python3 scripts/data_generation/generate_with_local_educator.py --all --limit-briefs 5 --limit-autopsies 5 --limit-lessons 5

  log "=== Step 6: Poet pairs (Opus) ==="
  run python3 scripts/data_generation/generate_poet_pairs.py --limit 10

  log "=== Step 7: Prepare full training data ==="
  run python3 scripts/data_generation/prepare_training_data.py --min-samples 5

  log "=== Step 8: Upload + train final both ==="
  run python3 scripts/modal/upload_data.py
  run modal run scripts/modal/train_educator.py --num-epochs-override 1
  run modal run scripts/modal/train_poet.py --num-epochs-override 1

  log "=== Step 9: Export + download final models ==="
  run modal run scripts/modal/export_gguf.py::export_educator
  run modal run scripts/modal/export_gguf.py::export_poet
  modal volume get --force poetry-gguf qwen2.5-7b-educator-Q4_K_M.gguf models/ || true
  modal volume get --force poetry-gguf qwen2.5-7b-poet-Q4_K_M.gguf models/ || true
fi

log "=== Test inference ==="
run python3 scripts/inference/pipeline.py "Write a poem about winter light"
log "Done."
