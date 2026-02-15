# Architecture overview

GPM is a two-model poetry system: **Educator** (mentor/critic, builds briefs and gives feedback) and **Poet** (generates poems from briefs). Cloud training on Modal, local inference via llama.cpp.

**Pipeline:** Raw poetry (good + bad) → Claude API for hard tasks (critiques, comparisons, revision briefs, poet pairs) → interim educator trained on seed → local educator generates briefs/autopsies/lessons → full training data prepared → train educator + poet on Modal → export GGUF → run locally with `pipeline.py` or `serve_gpm.py`.

**Optional:** Rhyme-focused poet: `config/rhyme_training.yaml`, `train_rhyme_poet.py`, data in `data/rhyme_training/`. For 32B poet when both models don’t fit in memory, use `scripts/inference/swapping_pipeline.py` (load one model at a time).

Full design spec and requirements: [DESIGN.md](DESIGN.md).
