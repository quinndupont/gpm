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

These are **alternative Stage 2 training approaches** - choose one based on your use case. They don't run together.

### REINFORCE ([reinforce_train.py](reinforce_train.py))

**Algorithm:** Reward-weighted regression with KL penalty

```python
# For each prompt:
#   1. Generate N=4 completions from policy
#   2. Score each with compute_reward() (deterministic rhyme analyzer)
#   3. Normalize advantages: Â = (score - mean) / std
#   4. Loss = -Σ(Â_i · log P_policy(completion_i | prompt)) + β_kl · KL(policy || ref)
```

**Training data:**
- Format: JSONL with `messages` field (system + user)
- Just prompts - no pre-generated completions needed
- Example: `data/poet_training/train.jsonl` (same as Stage 1)

**During training:**
- Generates N=4 completions per prompt on-the-fly
- Uses temperature=0.8 sampling for diversity
- Scores with deterministic rhyme analyzer (perfect/slant/none rhyme detection)

**What it learns:**
- Policy to generate high-reward poems directly
- No self-revision capability

**Pros:**
- Simpler setup - just needs prompts
- Direct reward optimization
- Good for pure generation tasks

**Cons:**
- Only learns generation, not revision
- Requires Educator for all revisions at inference (6 calls per revision round)
- Less sample-efficient than SRPO

---

### SRPO ([srpo_train.py](srpo_train.py))

**Algorithm:** Self-Refinement Policy Optimization with dual objectives

```python
# For each trajectory (prompt, draft_0, critique, draft_1):
#   1. Generation loss: L_gen = -log P(draft_0 | prompt)
#   2. Revision loss: L_rev = -w(r) · log P(draft_1 | prompt, draft_0, critique)
#   3. Reward weight: w(r) = clip((r1 - r0) / 0.2, 0, 2)
#   4. Total: L = 0.4·L_gen + 0.6·L_rev + 0.08·KL(policy || ref)
```

**Training data:**
- Format: JSONL trajectories with `(prompt, draft_0, critique, draft_1, reward_0, reward_1)`
- Pre-computed using `scripts/data/generate_srpo_data.py`
- Requires Educator model for generating critiques during data preparation
- Example: `data/srpo_training/trajectories.jsonl`

**During training:**
- No generation - learns from pre-computed trajectories
- Optimizes both generation (40%) and revision (60%) objectives simultaneously
- Uses improvement-based reward weighting

**What it learns:**
- **Generation**: How to write poems from scratch (like REINFORCE)
- **Self-revision**: How to improve poems based on critique feedback
- Two skills in one model

**Pros:**
- Poet can self-revise at inference (eliminates 4 Educator calls per revision)
- More sample-efficient (learns from both generation and revision examples)
- Production-ready for multi-round revision workflows

**Cons:**
- Requires trajectory generation beforehand
- More complex data pipeline
- Slightly longer training time (3 epochs vs 2)

---

### Quick Comparison Table

| Aspect | SRPO | REINFORCE |
|--------|------|-----------|
| **Algorithm** | Self-Refinement Policy Optimization | Reward-weighted regression |
| **Skills learned** | Generation + Self-revision | Generation only |
| **Training data** | Pre-computed trajectories | Just prompts (JSONL) |
| **Data preparation** | Requires trajectory generation | None (reuses Stage 1 data) |
| **During training** | No generation (learns from trajectories) | Generates N=4 completions per prompt |
| **Reward signal** | Improvement (r1 - r0) weighted | Absolute rhyme score |
| **Inference calls** | 2 per revision (critique + self-revise) | 6 per revision (multiple Educator calls) |
| **Learning rate** | 5e-6 (lower for stability) | 1e-5 |
| **Epochs** | 3 | 2 |
| **Max sequence length** | 1536 (fits prompt + draft + critique) | 1024 |
| **Recommended for** | Production use with revision | Experimentation, pure generation |

---

### Which Should I Use?

**Use SRPO if:**
- You need multi-round revision at inference
- You want to minimize Educator API calls
- You're deploying to production
- You want a single model that can both generate and self-revise

**Use REINFORCE if:**
- You only need generation (no revision)
- You want simpler data pipeline
- You're experimenting with reward functions
- You don't have an Educator model for trajectory generation

**Note:** Both training modes benefit from the same optimizations (timing metrics, batched log probs, KV cache, etc.) implemented in this codebase.
