# Ablate — CMU two-pass

See [CARD.yaml](CARD.yaml).

**Pass 1:** Identical single-shot poet generation to baseline.

**Pass 2:** `analyze_rhyme` with benchmark `form` / `variant`, formatted for the model; the poet produces a full revised poem from `models/prompts/inference/poet_cmu_revision.json`.
