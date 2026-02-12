# Good Poetry Model (GPM) — Poetry Generator

Locally-trained poetry **generation** model. Takes writing prompts (theme, form, style) and produces original poems. Ollama + Llama 3.2 for synthetic data, MLX for fine-tuning. Optimized for Mac Mini M4 (24GB RAM).

## Setup

```bash
./setup_gpm.sh
source venv/bin/activate
```

Or manually:

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
```

## Pipeline

```bash
# Full pipeline
python orchestrator.py --phase full

# Or run phases individually
python orchestrator.py --phase data      # Build corpus from HuggingFace (~30 min)
python orchestrator.py --phase prepare  # Generate prompt-poem pairs via Ollama (~8–12 hrs)
python orchestrator.py --phase validate # Filter low-quality pairs (~15 min)
python orchestrator.py --phase train    # MLX LoRA fine-tune (~3–4 hrs)

# Resume after interruption
python orchestrator.py --resume

# Quick test (10 random poems, 5 synthetic, 2 style, 50 iters; trains to models/adapters/gpm_lora_test)
python orchestrator.py --test
```

## Test

```bash
python test_gpm.py              # Generate villanelle (requires trained model)
python test_trained_poetry.py   # Generate sonnet (requires trained model)
python serve_gpm.py             # Ollama-compatible API on port 11435
```

## Config

[config/gpm_config.yaml](config/gpm_config.yaml):

- **ollama**: model, temperature, rate_limit_delay
- **data_preparation**: reverse_prompt_limit, synthetic_count, style_topics
- **validation**: min_poem_length, min_lines, check_repetition
- **training**: base_model, iterations, learning_rate, lora_rank, lora_alpha, max_seq_length (1024), eval_prompts

## Structure

- `agents/` — Data, DataPreparation, Validation, Training
- `config/gpm_config.yaml` — Central config
- `data/` — processed corpus → gpm_generator_train.jsonl → validated
- `models/adapters/` — LoRA output

Training uses ChatML format (system + user + assistant) with `--mask-prompt` so loss is computed only on the poem.
