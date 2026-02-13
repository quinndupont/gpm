# Poetry Chatbot — Educator + Poet (v3)

Two-model poetry system: **Educator** (mentor/critic) + **Poet** (generator). Cloud training on Modal, local inference via llama.cpp on Mac Mini M4. Training data via Claude API.

See [poetry_chatbot_plan_v3.md](poetry_chatbot_plan_v3.md) for full spec.

## Model choice

**Base: Llama 3.1 14B Instruct** (`meta-llama/Llama-3.1-14B-Instruct`). Llama 3.2 has no 14B text model (only 1B/3B or 11B/90B vision). The plan requires capacity for voice; 3B struggles with subtle personality. 14B fits two Q4_K_M models (~8GB each) on 24GB Mac Mini.

## Setup

```bash
pip install -r requirements.txt
# .env: ANTHROPIC_API_KEY, HF_TOKEN (accept Llama 3.1 license at huggingface.co)
modal token set
modal secret create huggingface-secret HF_TOKEN=your_token
```

## Data prep

Add raw poetry to `data/raw/good/` and `data/raw/bad/` (`.txt` or `.jsonl` with `text`/`poem` per line).

## Pipeline

### 1. Data generation (Claude API)

| Script | Usage | Params |
|--------|-------|--------|
| `generate_critiques` | T1 workshop critiques | `--limit N`, `--output PATH`, `--model MODEL` |
| `generate_briefs` | T2 generation briefs | `--input PATH`, `--limit N`, `--output PATH`, `--model MODEL` |
| `generate_lessons` | T6 craft lessons | `--questions PATH`, `--limit N`, `--output PATH`, `--model MODEL` |
| `generate_autopsies` | T4 cliché autopsies | `--limit N`, `--output PATH`, `--model MODEL` |
| `generate_comparisons` | T3 comparative workshop | `--limit N`, `--output PATH`, `--model MODEL` |
| `generate_dialogues` | T5 revision follow-ups | `--critiques PATH`, `--limit N`, `--output PATH`, `--model MODEL` |
| `generate_poet_pairs` | Brief → poem pairs | `--briefs PATH`, `--limit N`, `--output PATH`, `--model MODEL` |

Claude default: `claude-3-5-sonnet-20241022`.

```bash
python scripts/data_generation/generate_briefs.py --limit 100
python scripts/data_generation/generate_lessons.py --limit 50
python scripts/data_generation/generate_critiques.py --limit 200
# ... then autopsies, comparisons, dialogues, poet_pairs
```

### 2. Prepare training data

```bash
python scripts/data_generation/prepare_training_data.py [--educator-only] [--poet-only] [--min-samples N] [--seed N]
```

Combines outputs into Llama 3 chat format. `--min-samples` caps dataset for quick test.

### 3. Quality gate (optional)

```bash
python scripts/data_generation/quality_gate.py INPUT [--output PATH] [--reject-log PATH]
```

### 4. Upload to Modal

```bash
python scripts/modal/upload_data.py
```

Uploads `data/educator_training/{train,valid}.jsonl` and `data/poet_training/{train,valid}.jsonl` to `poetry-data` volume.

### 5. Train + export (Modal)

```bash
# Educator
modal run scripts/modal/train_educator.py [--num-epochs-override N]
modal run scripts/modal/export_gguf.py

# Poet
modal run scripts/modal/train_poet.py [--num-epochs-override N]
modal run scripts/modal/export_gguf.py poet

# Or orchestrate both
modal run scripts/modal/modal_app.py [--educator-only] [--poet-only] [--train-only] [--num-epochs N]
```

### 6. Download + inference

```bash
modal volume get poetry-gguf llama3.1-14b-educator-Q4_K_M.gguf models/
modal volume get poetry-gguf llama3.1-14b-poet-Q4_K_M.gguf models/

python scripts/inference/pipeline.py "Write a poem about winter light" [--config PATH]
```

## First test

```bash
./scripts/run_first_test.sh
```

Runs: 3 briefs + 3 lessons → prepare (5 samples) → upload → train (1 epoch) → export.

## Config

| File | Purpose |
|------|---------|
| `config/educator_training.yaml` | QLoRA (r=64, α=128, 4 epochs, max_seq 1024) |
| `config/poet_training.yaml` | QLoRA (r=64, α=128, 6 epochs, max_seq 512) |
| `config/export_pipeline.yaml` | Merge + GGUF Q4_K_M |
| `config/inference_config.yaml` | llama.cpp n_ctx, temperature, etc. |

## Structure

```
data/raw/{good,bad}/     # Poetry input
data/annotated/          # Claude outputs (critiques, autopsies, comparisons)
data/educator_training/  # train.jsonl, valid.jsonl
data/poet_training/      # train.jsonl, valid.jsonl
persona/                 # pedagogy_design_doc.md, persona_condensed.txt, anti_llm_isms.txt
scripts/{data_generation,modal,eval,inference}/
```
