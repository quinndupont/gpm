#!/bin/bash
# First test: minimal data, educator only, 1 epoch
set -e
cd "$(dirname "$0")/.."

echo "1. Generate minimal data (briefs + lessons)"
python3 scripts/data_generation/generate_briefs.py --limit 3
python3 scripts/data_generation/generate_lessons.py --limit 3

echo "2. Prepare training data"
python3 scripts/data_generation/prepare_training_data.py --educator-only --min-samples 5

echo "3. Upload to Modal"
python3 scripts/modal/upload_data.py

echo "4. Train educator (1 epoch)"
modal run scripts/modal/train_educator.py --num-epochs-override 1

echo "5. Export to GGUF"
modal run scripts/modal/export_gguf.py

echo "6. Download GGUF"
modal volume get poetry-gguf llama3.1-14b-educator-Q4_K_M.gguf models/ 2>/dev/null || true

echo "Done. Test inference: python3 scripts/inference/pipeline.py 'Write a poem about winter'"
