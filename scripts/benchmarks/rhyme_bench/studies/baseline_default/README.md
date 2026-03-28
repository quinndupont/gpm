# Baseline (default inference)

See [CARD.yaml](CARD.yaml) for machine-readable metadata.

This is the reference condition for rhyme bench: single poet pass, default `poet_generation` template (`default`), no backward instructions, no CMU second pass.

**Generated baseline on disk:** artifacts live under [`data/rhyme_bench/studies/baseline_default/`](../../../../../data/rhyme_bench/studies/baseline_default/) (`summary.json` plus [`CARD.yaml`](../../../../../data/rhyme_bench/studies/baseline_default/CARD.yaml)). Reference run: `run_timestamp` 20260318_111958, model `trained-llama3.1-8b-q6`, 47 prompts, `mean_strict_rhyme_density` 0.65, `matches_form_rate` 0.19, `max_revisions: 0`.
