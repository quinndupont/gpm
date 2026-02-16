#!/bin/bash
# Full workflow: train educator + poet. By default skips generation (uses existing data).
# Quotas: 50 Opus (rhyme poems only), 150 Sonnet; beyond that → local.
# Run scripts/run_generate_data.sh first to generate all datasets.
set -e
cd "$(dirname "$0")/.."

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
run() { log ">>> $*"; "$@"; }

CRITIQUES_GOOD=200
REVISION_BRIEFS=50
BRIEFS=200
LESSONS=10

SKIP_GEN=true
GEN_ONLY=false
WITH_GEN=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --with-generation) WITH_GEN=true; SKIP_GEN=false; shift ;;
    --skip-generation) SKIP_GEN=true; shift ;;
    --generation-only) GEN_ONLY=true; SKIP_GEN=false; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$GEN_ONLY" == true ]]; then
  log "=== Generation only (stop before final training) ==="
  run python3 scripts/data_generation/generate_critiques_seed.py --limit-good $CRITIQUES_GOOD
  run python3 scripts/data_generation/generate_comparisons.py
  run python3 scripts/data_generation/generate_revision_briefs.py --limit $REVISION_BRIEFS
  run python3 scripts/data_generation/prepare_training_data.py --interim-educator --educator-only
  run python3 scripts/modal/upload_data.py
  run modal run scripts/modal/train_educator.py --num-epochs-override 2
  run modal run scripts/modal/export_gguf.py::export_educator_interim
  mkdir -p models
  modal volume get --force poetry-gguf qwen2.5-7b-educator-interim-Q4_K_M.gguf models/ || { log "Interim educator GGUF not found."; exit 1; }
  run python3 scripts/data_generation/generate_with_local_educator.py --all --limit-briefs $BRIEFS --limit-lessons $LESSONS
  run python3 scripts/data_generation/generate_poet_pairs.py
  run python3 scripts/data_generation/generate_dialogues.py
  run python3 scripts/data_generation/generate_rhyme_pairs.py --replace
  run python3 scripts/data_generation/generate_approval_examples.py --replace
  run python3 scripts/data_generation/prepare_training_data.py
  run python3 scripts/data_generation/prepare_rhyme_training_data.py
  run python3 scripts/modal/upload_data.py
  log "Generation done. Run training: ./scripts/run_full_workflow.sh"
  exit 0
fi

if [[ "$SKIP_GEN" == true ]]; then
  log "Skipping data generation. Using existing data."
  if [[ ! -f data/educator_training/train.jsonl || ! -f data/poet_training/train.jsonl ]]; then
    log "ERROR: train.jsonl not found. Run: ./scripts/run_generate_data.sh"
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

log "=== Step 1: Seed data (Sonnet, local fallback) ==="
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

log "=== Step 8: Poet pairs (Sonnet, local fallback) ==="
run python3 scripts/data_generation/generate_poet_pairs.py

log "=== Step 9: Dialogues (Sonnet, local fallback) ==="
run python3 scripts/data_generation/generate_dialogues.py

log "=== Step 10: Rhyme pairs (Opus for poems) + approval examples ==="
run python3 scripts/data_generation/generate_rhyme_pairs.py --replace
run python3 scripts/data_generation/generate_approval_examples.py --replace

log "=== Step 11: Prepare full training data ==="
run python3 scripts/data_generation/prepare_training_data.py
run python3 scripts/data_generation/prepare_rhyme_training_data.py

log "=== Step 12: Upload full data ==="
run python3 scripts/modal/upload_data.py

  log "=== Step 13: Train final educator + poet + poet_rhyme ==="
  run modal run scripts/modal/train_educator.py
  run modal run scripts/modal/train_poet.py
  run modal run scripts/modal/train_rhyme_poet.py

  log "=== Step 14: Export final models ==="
  run modal run scripts/modal/export_gguf.py::export_educator
  run modal run scripts/modal/export_gguf.py::export_poet
  run modal run scripts/modal/export_gguf.py::export_poet_rhyme

  log "=== Step 15: Download final models ==="
  modal volume get --force poetry-gguf llama3.1-8b-educator-Q4_K_M.gguf models/ || { log "Educator GGUF not found."; exit 1; }
  modal volume get --force poetry-gguf llama3.1-8b-poet-Q4_K_M.gguf models/ || { log "Poet GGUF not found."; exit 1; }
  modal volume get --force poetry-gguf llama3.1-8b-poet_rhyme-Q4_K_M.gguf models/ || { log "Poet rhyme GGUF not found."; exit 1; }

log "Done. Test: python scripts/inference/pipeline.py \"Write a poem about winter light\""
