# Training Scripts

This directory contains training scripts for the Educator and Poet models.

## Overview

| Script | Purpose | Stage |
|--------|---------|-------|
| `qlora_train.py` | QLoRA supervised fine-tuning | Educator, Poet Stage 1 |
| `srpo_train.py` | SRPO (Self-Refinement Policy Optimization) | Poet Stage 2 (recommended) |
| `reinforce_train.py` | REINFORCE reward-weighted regression | Poet Stage 2 (legacy) |

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

### Stage 2: REINFORCE (Legacy)

REINFORCE only teaches generation (π), not self-revision. Use SRPO for models that should self-revise.

```bash
# Modal
modal run scripts/modal/train_poet_reinforce.py \
  --sft-checkpoint /vol/checkpoints/poet/final

# SageMaker
python scripts/sagemaker/train_sagemaker.py --task reinforce \
  --sft-s3 s3://BUCKET/checkpoints/poet/JOB_NAME/output/model.tar.gz
```

**Config:** `config/reinforce_training.yaml`

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
| `learning_rate` | 5e-6 | Lower than REINFORCE for stability |
| `num_epochs` | 3 | Training epochs |
| `max_seq_length` | 1536 | Longer to fit (prompt + draft + critique) |

### REINFORCE (`config/reinforce_training.yaml`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_completions` | 4 | Completions per prompt |
| `beta_kl` | 0.1 | KL penalty coefficient |
| `temperature` | 0.8 | Sampling temperature |
| `learning_rate` | 1e-5 | Learning rate |
| `num_epochs` | 2 | Training epochs |

## File Structure

```
scripts/training/
├── README.md                 # This file
├── qlora_train.py           # QLoRA SFT trainer (Stage 1)
├── srpo_train.py            # SRPO trainer (Stage 2, recommended)
├── reinforce_train.py       # REINFORCE trainer (Stage 2, legacy)
├── train_interactive.py     # Interactive training launcher
├── model_registry.py        # Base model registry
└── model_discovery.py       # Model discovery utilities

config/
├── educator_training.yaml   # Educator QLoRA config
├── poet_training.yaml       # Poet Stage 1 config
├── srpo_training.yaml       # SRPO Stage 2 config
├── srpo_data_generation.yaml # SRPO data generation config
├── reinforce_training.yaml  # REINFORCE Stage 2 config
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

## Comparison: SRPO vs REINFORCE

| Aspect | SRPO | REINFORCE |
|--------|------|-----------|
| **Skills learned** | Generation + Self-revision | Generation only |
| **Training data** | Trajectories (draft → critique → revision) | Prompts + completions |
| **Reward signal** | Improvement (r1 - r0) | Absolute score |
| **Inference calls** | 2 per revision (critique + self-revise) | 6 per revision (multiple Educator calls) |
| **Recommended for** | Production use | Experimentation |
