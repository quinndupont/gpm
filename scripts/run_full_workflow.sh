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
QUALITY_GATE=false
BACKEND=modal
BASE_MODEL=
while [[ $# -gt 0 ]]; do
  case $1 in
    --with-generation) WITH_GEN=true; SKIP_GEN=false; shift ;;
    --skip-generation) SKIP_GEN=true; shift ;;
    --generation-only) GEN_ONLY=true; SKIP_GEN=false; shift ;;
    --quality-gate) QUALITY_GATE=true; shift ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
BASE_MODEL_ARGS=()
[[ -n "$BASE_MODEL" ]] && BASE_MODEL_ARGS=(--base-model "$BASE_MODEL")

if [[ "$GEN_ONLY" == true ]]; then
  log "=== Generation only (stop before final training) ==="
  run python3 scripts/data_generation/generate_critiques_seed.py --limit-good $CRITIQUES_GOOD
  run python3 scripts/data_generation/generate_comparisons.py
  run python3 scripts/data_generation/generate_revision_briefs.py --limit $REVISION_BRIEFS
  run python3 scripts/data_generation/prepare_training_data.py --interim-educator --educator-only
  if [[ "$BACKEND" == "sagemaker" ]]; then
    run python3 scripts/sagemaker/upload_to_s3.py
    run python3 scripts/sagemaker/train_sagemaker.py --task educator --num-epochs-override 2 "${BASE_MODEL_ARGS[@]}"
    run python3 scripts/sagemaker/export_gguf_sagemaker.py --task educator-interim
    mkdir -p models
    run python3 scripts/sagemaker/download_models.py llama3.1-8b-educator-interim-Q4_K_M.gguf || { log "Interim educator GGUF not found."; exit 1; }
  else
    run python3 scripts/modal/upload_data.py
    run modal run scripts/modal/train_educator.py --num-epochs-override 2 "${BASE_MODEL_ARGS[@]}"
    run modal run scripts/modal/export_gguf.py::export_educator_interim
    mkdir -p models
    modal volume get --force poetry-gguf qwen2.5-7b-educator-interim-Q4_K_M.gguf models/ || modal volume get --force poetry-gguf llama3.1-8b-educator-interim-Q4_K_M.gguf models/ || { log "Interim educator GGUF not found."; exit 1; }
  fi
  run python3 scripts/data_generation/generate_with_local_educator.py --all --limit-briefs $BRIEFS --limit-lessons $LESSONS
  run python3 scripts/data_generation/generate_poet_pairs.py
  run python3 scripts/data_generation/generate_dialogues.py
  run python3 scripts/data_generation/generate_rhyme_pairs.py --replace
  run python3 scripts/data_generation/generate_approval_examples.py --replace
  run python3 scripts/data_generation/prepare_training_data.py $([[ "$QUALITY_GATE" == true ]] && echo --quality-gate)
  run python3 scripts/data_generation/prepare_rhyme_training_data.py
  if [[ "$BACKEND" == "sagemaker" ]]; then
    run python3 scripts/sagemaker/upload_to_s3.py
  else
    run python3 scripts/modal/upload_data.py
  fi
  log "Generation done. Run training: ./scripts/run_full_workflow.sh"
  exit 0
fi

if [[ "$SKIP_GEN" == true ]]; then
  log "Skipping data generation. Using existing data."
  if [[ ! -f data/educator_training/train.jsonl || ! -f data/poet_training/train.jsonl ]]; then
    log "ERROR: train.jsonl not found. Run: ./scripts/run_generate_data.sh"
    exit 1
  fi
  if [[ "$BACKEND" == "sagemaker" ]]; then
    run python3 scripts/sagemaker/upload_to_s3.py
    log "=== Train final educator + poet ==="
    run python3 scripts/sagemaker/train_sagemaker.py --task educator "${BASE_MODEL_ARGS[@]}"
    run python3 scripts/sagemaker/train_sagemaker.py --task poet "${BASE_MODEL_ARGS[@]}"
    log "=== Export + download final models ==="
    run python3 scripts/sagemaker/export_gguf_sagemaker.py --task educator
    run python3 scripts/sagemaker/export_gguf_sagemaker.py --task poet
    mkdir -p models
    run python3 scripts/sagemaker/download_models.py
  else
    run python3 scripts/modal/upload_data.py
    log "=== Train final educator + poet ==="
    run modal run scripts/modal/train_educator.py "${BASE_MODEL_ARGS[@]}"
    run modal run scripts/modal/train_poet.py "${BASE_MODEL_ARGS[@]}"
    log "=== Export + download final models ==="
    run modal run scripts/modal/export_gguf.py::export_educator
    run modal run scripts/modal/export_gguf.py::export_poet
    mkdir -p models
    modal volume get --force poetry-gguf qwen2.5-7b-educator-Q4_K_M.gguf models/ || modal volume get --force poetry-gguf llama3.1-8b-educator-Q4_K_M.gguf models/ || true
    modal volume get --force poetry-gguf qwen2.5-7b-poet-Q4_K_M.gguf models/ || modal volume get --force poetry-gguf llama3.1-8b-poet-Q4_K_M.gguf models/ || true
  fi
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
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/upload_to_s3.py
else
  run python3 scripts/modal/upload_data.py
fi

log "=== Step 4: Train interim educator ==="
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/train_sagemaker.py --task educator --num-epochs-override 2 "${BASE_MODEL_ARGS[@]}"
else
  run modal run scripts/modal/train_educator.py --num-epochs-override 2 "${BASE_MODEL_ARGS[@]}"
fi

log "=== Step 5: Export interim educator GGUF ==="
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/export_gguf_sagemaker.py --task educator-interim
else
  run modal run scripts/modal/export_gguf.py::export_educator_interim
fi

log "=== Step 6: Download interim educator ==="
mkdir -p models
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/download_models.py llama3.1-8b-educator-interim-Q4_K_M.gguf || { log "Interim educator GGUF not found."; exit 1; }
else
  modal volume get --force poetry-gguf qwen2.5-7b-educator-interim-Q4_K_M.gguf models/ || modal volume get --force poetry-gguf llama3.1-8b-educator-interim-Q4_K_M.gguf models/ || { log "Interim educator GGUF not found."; exit 1; }
fi

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
run python3 scripts/data_generation/prepare_training_data.py $([[ "$QUALITY_GATE" == true ]] && echo --quality-gate)
run python3 scripts/data_generation/prepare_rhyme_training_data.py

log "=== Step 12: Upload full data ==="
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/upload_to_s3.py
else
  run python3 scripts/modal/upload_data.py
fi

log "=== Step 13: Train final educator + poet + poet_rhyme ==="
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/train_sagemaker.py --task educator "${BASE_MODEL_ARGS[@]}"
  run python3 scripts/sagemaker/train_sagemaker.py --task poet "${BASE_MODEL_ARGS[@]}"
  run python3 scripts/sagemaker/train_sagemaker.py --task rhyme "${BASE_MODEL_ARGS[@]}"
else
  run modal run scripts/modal/train_educator.py "${BASE_MODEL_ARGS[@]}"
  run modal run scripts/modal/train_poet.py "${BASE_MODEL_ARGS[@]}"
  run modal run scripts/modal/train_rhyme_poet.py "${BASE_MODEL_ARGS[@]}"
fi

log "=== Step 14: Export final models ==="
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/export_gguf_sagemaker.py --task educator
  run python3 scripts/sagemaker/export_gguf_sagemaker.py --task poet
  run python3 scripts/sagemaker/export_gguf_sagemaker.py --task poet_rhyme
else
  run modal run scripts/modal/export_gguf.py::export_educator
  run modal run scripts/modal/export_gguf.py::export_poet
  run modal run scripts/modal/export_gguf.py::export_poet_rhyme
fi

log "=== Step 15: Download final models ==="
if [[ "$BACKEND" == "sagemaker" ]]; then
  run python3 scripts/sagemaker/download_models.py
else
  modal volume get --force poetry-gguf llama3.1-8b-educator-Q4_K_M.gguf models/ || { log "Educator GGUF not found."; exit 1; }
  modal volume get --force poetry-gguf llama3.1-8b-poet-Q4_K_M.gguf models/ || { log "Poet GGUF not found."; exit 1; }
  modal volume get --force poetry-gguf llama3.1-8b-poet_rhyme-Q4_K_M.gguf models/ || { log "Poet rhyme GGUF not found."; exit 1; }
fi

log "Done. Test: python scripts/inference/pipeline.py \"Write a poem about winter light\""
