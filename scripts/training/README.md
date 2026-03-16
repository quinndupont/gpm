# Training Scripts

This directory contains training scripts for the Educator and Poet models.

## Overview

| Script | Purpose | Stage |
|--------|---------|-------|
| `qlora_train.py` | QLoRA supervised fine-tuning | Educator, Poet Stage 1 |
| `srpo_train.py` | SRPO (Self-Refinement Policy Optimization) | Poet Stage 2 |

## Educator Training

Single-stage QLoRA SFT on educator tasks (critique, brief, comparison, etc.).

```bash
# Modal (recommended)
modal run scripts/modal/train_educator.py

# SageMaker
python scripts/sagemaker/train_sagemaker.py --task educator
```

**Config:** `config/educator_training.yaml`
**Data:** `data/educator_training/train.jsonl`

## Poet Training

Two-stage training pipeline:

### Stage 1: Filtered SFT (Warm Start)

QLoRA SFT on high-quality poem examples filtered by rhyme compliance.

```bash
# Modal
modal run scripts/modal/train_poet.py

# SageMaker
python scripts/sagemaker/train_sagemaker.py --task poet
```

**Config:** `config/poet_training.yaml`
**Data:** `data/poet_training/train.jsonl`

### Stage 2: SRPO (Recommended)

SRPO trains the Poet to both **generate** AND **self-revise** in a single training run.

**Algorithm:**
```
L = α · L_gen + (1-α) · L_rev + β · KL(π || π_ref)

Where:
  L_gen = -log P(draft_0 | prompt)                              # Generation
  L_rev = -w(r) · log P(draft_1 | prompt, draft_0, critique)    # Revision
  w(r) = clip((reward_1 - reward_0) / 0.2, 0, 2)                # Improvement weight
  α = 0.4, β = 0.08
```

**Key advantage:** At inference, the Poet can revise its own work based on Educator feedback, eliminating 4 Educator calls per revision round.

#### Generate SRPO Training Data

First, generate trajectories `(prompt, draft_0, critique, draft_1)`:

```bash
# Modal (recommended)
modal run scripts/modal/generate_srpo_data.py --max-trajectories 5000

# Local
python scripts/data/generate_srpo_data.py --limit 100
```

**Config:** `config/srpo_data_generation.yaml`
**Output:** `data/srpo_training/trajectories.jsonl`

##### LLM Backend: Bedrock vs Anthropic API

Data generation uses Claude for critiques and revisions. Two backends are supported:

**Amazon Bedrock (recommended):**
```bash
# 1. Install AWS CLI and configure credentials
brew install awscli
aws configure  # or: aws configure sso && aws sso login

# 2. Verify Bedrock access
aws bedrock list-foundation-models --query "modelSummaries[?contains(modelId, 'claude')]"

# 3. Install Python dependency
pip install anthropic[bedrock]

# 4. Enable Bedrock backend
export USE_BEDROCK=1
export AWS_REGION=us-east-1  # optional, defaults to us-east-1

# 5. Run data generation
python scripts/data/generate_srpo_data.py --limit 100
```

**Direct Anthropic API:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/data/generate_srpo_data.py --limit 100
```

**Modal with Bedrock:**

Create an AWS secret in Modal:
```bash
modal secret create aws-secret \
  AWS_ACCESS_KEY_ID=your-key \
  AWS_SECRET_ACCESS_KEY=your-secret \
  AWS_REGION=us-east-1 \
  USE_BEDROCK=1
```

Then update `scripts/modal/generate_srpo_data.py` to use `aws-secret` instead of `anthropic-secret`.

#### Run SRPO Training

```bash
# Modal
modal run scripts/modal/train_poet_srpo.py \
  --sft-checkpoint /vol/checkpoints/poet/final

# SageMaker
python scripts/sagemaker/train_sagemaker.py --task srpo \
  --sft-s3 s3://BUCKET/checkpoints/poet/JOB_NAME/output/model.tar.gz

# Local (for testing)
python scripts/training/srpo_train.py \
  --sft-checkpoint checkpoints/poet/final \
  --data-dir data/srpo_training \
  --checkpoint-dir checkpoints/poet_srpo
```

**Config:** `config/srpo_training.yaml`
**Data:** `data/srpo_training/trajectories.jsonl`

## Inference Mode

After training, configure the inference pipeline to use SRPO revision mode:

```yaml
# config/inference_config.yaml
revision_mode: "srpo"  # Poet self-revises (recommended for SRPO-trained models)
# revision_mode: "educator"  # Educator generates revision instructions (legacy)
```

**SRPO mode** simplifies the revision loop:
```
Educator critiques → Poet self-revises
```

**Educator mode** (legacy):
```
Educator critiques → Educator summarizes → Educator builds revision brief →
Educator builds poet instructions → Poet revises
```

## Hyperparameters

### SRPO (`config/srpo_training.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alpha` | 0.4 | Balance between generation (40%) and revision (60%) |
| `beta_kl` | 0.08 | KL penalty to prevent drift from SFT |
| `reward_normalization` | 0.2 | Divide improvement by this for w(r) |
| `max_reward_weight` | 2.0 | Clip w(r) at this value |
| `learning_rate` | 5e-6 | Learning rate for stable training |
| `num_epochs` | 3 | Training epochs |
| `max_seq_length` | 1536 | Longer to fit (prompt + draft + critique) |

## File Structure

```
scripts/training/
├── README.md                 # This file
├── qlora_train.py           # QLoRA SFT trainer (Stage 1)
├── srpo_train.py            # SRPO trainer (Stage 2)
├── train_interactive.py     # Interactive training launcher
├── model_registry.py        # Base model registry
└── model_discovery.py       # Model discovery utilities

config/
├── educator_training.yaml   # Educator QLoRA config
├── poet_training.yaml       # Poet Stage 1 config
├── srpo_training.yaml       # SRPO Stage 2 config
├── srpo_data_generation.yaml # SRPO data generation config
└── inference_config.yaml    # Inference config (revision_mode)

data/
├── educator_training/
│   ├── train.jsonl
│   └── valid.jsonl
├── poet_training/
│   ├── train.jsonl
│   └── valid.jsonl
└── srpo_training/
    └── trajectories.jsonl   # (prompt, draft_0, critique, draft_1)
```

## SRPO (Stage 2)

**Self-Refinement Policy Optimization** teaches the poet both generation and self-revision in a single training run.

### Algorithm

Dual-objective loss combining generation and revision:

```python
# For each trajectory (prompt, draft_0, critique, draft_1):
#   1. Generation loss: L_gen = -log P(draft_0 | prompt)
#   2. Revision loss: L_rev = -w(r) · log P(draft_1 | prompt, draft_0, critique)
#   3. Reward weight: w(r) = clip((r1 - r0) / 0.2, 0, 2)
#   4. Total: L = 0.4·L_gen + 0.6·L_rev + 0.08·KL(policy || ref)
```

### Training Data

- **Format:** JSONL trajectories with `(prompt, draft_0, critique, draft_1, reward_0, reward_1)`
- **Generation:** Use `scripts/data/generate_srpo_data.py` with Llama 4 Maverick via Bedrock
- **Location:** `data/srpo_training/trajectories.jsonl`

Each trajectory shows:
1. Initial poem attempt (draft_0)
2. Critique with specific line references and rhyme analysis
3. Improved revision (draft_1)
4. Reward scores (r0, r1) from deterministic rhyme analyzer

### What The Model Learns

- **Generation (40% weight):** Write poems from prompts
- **Self-revision (60% weight):** Improve poems based on critique feedback

The trained model can both generate initial poems AND self-revise based on Educator critiques.

### Benefits

- **Fewer API calls at inference:** 2 calls per revision (Educator critique + Poet self-revise) instead of 6 calls with legacy educator-driven revision
- **Single model for dual skills:** One checkpoint handles generation and revision
- **Sample-efficient:** Learns from both positive (generation) and improvement (revision) examples
- **Production-ready:** Designed for multi-round revision workflows

### Requirements

- Pre-generated trajectories from `generate_srpo_data.py`
- AWS Bedrock access with Llama 4 Maverick enabled
- Longer sequence length (1536) to fit prompt + draft + critique
- Stage 1 SFT checkpoint as base model
