#!/bin/bash
# Generate all datasets (Claude Opus/Sonnet only). Uses 50 Opus (rhyme poems), 150 Sonnet; rest → local.
# Run from repo root: ./scripts/run_generate_data.sh
set -e
cd "$(dirname "$0")/.."

log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
run() { log ">>> $*"; "$@"; }

CRITIQUES_GOOD=200
REVISION_BRIEFS=50
BRIEFS=200
LESSONS=10

log "=== Step 1: Seed data (Claude Sonnet) ==="
run python3 scripts/data_generation/generate_critiques_seed.py --limit-good $CRITIQUES_GOOD
run python3 scripts/data_generation/generate_comparisons.py
run python3 scripts/data_generation/generate_revision_briefs.py --limit $REVISION_BRIEFS

log "=== Step 2: Briefs, autopsies, lessons (Claude Sonnet) ==="
run python3 scripts/data_generation/generate_briefs.py --replace
run python3 scripts/data_generation/generate_autopsies.py --replace
run python3 scripts/data_generation/generate_lessons.py --replace

log "=== Step 3: Poet pairs (Claude Sonnet) ==="
run python3 scripts/data_generation/generate_poet_pairs.py --replace

log "=== Step 4: Dialogues (Claude Sonnet) ==="
run python3 scripts/data_generation/generate_dialogues.py --replace

log "=== Step 5: Rhyme pairs (Opus poems, Sonnet brief/critique) ==="
run python3 scripts/data_generation/generate_rhyme_pairs.py --replace

log "=== Step 6: Approval examples (Claude Sonnet) ==="
run python3 scripts/data_generation/generate_approval_examples.py --replace

log "=== Step 7: Prepare full training data ==="
run python3 scripts/data_generation/prepare_training_data.py
run python3 scripts/data_generation/prepare_rhyme_training_data.py

log "=== Step 8: Upload to Modal ==="
run python3 scripts/modal/upload_data.py

log "Done. Run full pipeline: ./scripts/run_full_workflow.sh"
