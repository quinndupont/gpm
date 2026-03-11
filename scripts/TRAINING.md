# Fine-tuning guide

QLoRA fine-tuning for **educator** and **poet**, plus optional **REINFORCE** stage on poet. You can run on **Modal** or **Amazon SageMaker** and choose the **base model** from the registry.

## Quick start

```bash
# From repo root. Default: Modal backend, models from config YAML.
python scripts/modal/modal_app.py --educator-only

# SageMaker, 1 epoch (e.g. for a quick test)
python scripts/modal/modal_app.py --backend sagemaker --educator-only --num-epochs 1

# Override base model (HuggingFace ID)
python scripts/modal/modal_app.py --backend modal --base-model "meta-llama/Llama-3.1-8B-Instruct" --poet-only
```

## Backend: Modal vs SageMaker

| | Modal | SageMaker |
|---|--------|-----------|
| **Select** | `--backend modal` (default) | `--backend sagemaker` |
| **Config** | None (uses Modal secrets/volumes) | `config/sagemaker.yaml` (bucket, role, region) |
| **Data** | `python scripts/modal/upload_data.py` | `python scripts/sagemaker/upload_to_s3.py` |
| **Validate** | — | `python scripts/sagemaker/validate_setup.py` |
| **Train** | `modal run scripts/modal/train_educator.py` etc. | `python scripts/sagemaker/train_sagemaker.py --task educator` |
| **Export** | `modal run scripts/modal/export_gguf.py::export_educator` | `python scripts/sagemaker/export_gguf_sagemaker.py --task educator` |
| **Download** | `modal volume get poetry-gguf <file> models/` | `python scripts/sagemaker/download_models.py` |

- **Modal**: Sign up at modal.com, `modal token new`, create secret `huggingface-secret` with your HF token. No AWS needed.
- **SageMaker**: AWS account, S3 bucket, IAM role, Secrets Manager secret for HF token. See `config/sagemaker.yaml` and run `scripts/sagemaker/validate_setup.py` before training.

## Tasks

- **educator** – Critique, briefs, lessons (config: `config/educator_training.yaml`).
- **poet** – Brief → poem (config: `config/poet_training.yaml`).
- **reinforce** – Optional REINFORCE stage on poet (config: `config/reinforce_training.yaml`).

Use `--educator-only`, `--poet-only`, or neither to run educator + poet. Use `--reinforce` to run only the REINFORCE stage.

## Choosing the base model

Base model is set in the task YAML (`config/educator_training.yaml`, etc.) or overridden at run time.

**Override via CLI (recommended):**

```bash
# HuggingFace model ID
python scripts/modal/modal_app.py --base-model "meta-llama/Llama-3.1-8B-Instruct" --educator-only
python scripts/sagemaker/train_sagemaker.py --task poet --base-model "Qwen/Qwen2.5-7B-Instruct"
```

**Override via config:** Edit `base_model` in `config/educator_training.yaml`, `config/poet_training.yaml`, or `config/reinforce_training.yaml`.

**Registered models:** See `config/model_registry.yaml` for supported HuggingFace IDs and short names (e.g. `llama3.1-8b`, `qwen2.5-7b`). Gated models (e.g. Llama) require a HuggingFace token (Modal: `huggingface-secret`; SageMaker: Secrets Manager or `HF_TOKEN` in env).

## Orchestrator (modal_app.py)

```bash
python scripts/modal/modal_app.py [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--backend modal \| sagemaker` | Backend (default: modal) |
| `--base-model HF_ID` | Base model to fine-tune (e.g. `meta-llama/Llama-3.1-8B-Instruct`) |
| `--educator-only` | Train only educator |
| `--poet-only` | Train only poet |
| `--reinforce` | Train only REINFORCE stage on poet |
| `--train-only` | Skip GGUF export after training |
| `--num-epochs N` | Override epoch count |
| `--interactive`, `-i` | Interactive model discovery/training |

## Shell workflow (run_full_workflow.sh)

Full pipeline including data generation and training:

```bash
./scripts/run_full_workflow.sh [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--backend modal \| sagemaker` | Backend (default: modal) |
| `--base-model HF_ID` | Base model for training (passed to orchestrator) |
| `--with-generation` | Run full pipeline including data generation |
| `--skip-generation` | Use existing data, only train (default) |
| `--generation-only` | Generate data and stop before final training |
| `--quality-gate` | Run quality gate on educator data; fail if pass rate < 90% |

## Direct training (no orchestrator)

**Modal:**

```bash
python scripts/modal/upload_data.py
modal run scripts/modal/train_educator.py [--num-epochs-override N] [--base-model "HF_ID"]
modal run scripts/modal/export_gguf.py::export_educator
```

**SageMaker:**

```bash
python scripts/sagemaker/upload_to_s3.py
python scripts/sagemaker/train_sagemaker.py --task educator [--num-epochs-override N] [--base-model "HF_ID"]
python scripts/sagemaker/export_gguf_sagemaker.py --task educator
python scripts/sagemaker/download_models.py
```

## Data layout

- **Educator/poet:** `data/educator_training/train.jsonl`, `valid.jsonl`; `data/poet_training/`; `data/rhyme_training/` (from `prepare_training_data.py`, `prepare_rhyme_training_data.py`). REINFORCE uses rhyme data.
- Upload: Modal → `upload_data.py` (volumes); SageMaker → `upload_to_s3.py` (S3 prefix `data/`).

## Validating training data

- **Quality gate:** `python scripts/data_generation/quality_gate.py data/educator_training/train.jsonl` filters educator examples (voice consistency, anti-rubric, LLM-isms, specificity). Use `--quality-gate` with `prepare_training_data.py` or `run_full_workflow.sh` to fail the pipeline if pass rate < 90%.
- **Data tests:** `pytest tests/ -m data` validates schema, chat format, and rhyme density on fixtures and (when present) generated data.

## Config files

| File | Purpose |
|------|---------|
| `config/educator_training.yaml` | Educator QLoRA (base_model, LoRA, training) |
| `config/poet_training.yaml` | Poet QLoRA |
| `config/reinforce_training.yaml` | REINFORCE stage on poet |
| `config/export_pipeline.yaml` | GGUF merge/quantization (base_model, Q4_K_M) |
| `config/model_registry.yaml` | HuggingFace ID ↔ short name |
| `config/sagemaker.yaml` | SageMaker bucket, role, region, instance (SageMaker only) |

## Credentials

- **HuggingFace:** Required for gated models. Modal: secret `huggingface-secret`. SageMaker: secret in Secrets Manager (`config/sagemaker.yaml` `hf_secret_name`) or env `HF_TOKEN`.
- **AWS (SageMaker):** `~/.aws/credentials` or env `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. IAM user needs SageMaker, S3, Secrets Manager, and `iam:PassRole` on the execution role.
