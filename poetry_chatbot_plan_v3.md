# Personalized Poetry Chatbot: Integrated Architecture & Implementation Plan
# Version 3.0 — Educator Model + Cloud-to-Edge Training Pipeline

---

## Foundational Philosophy: The Educator, Not the Evaluator

The best poetry educators don't rate dimensions on 1-10 scales. They see the poet inside the poem and coach them toward their own best instincts.

Think about the difference:

> **Evaluator:** "Imagery specificity: 4/10. Cliché density: high. Line 7 contains the overused phrase 'heart of gold.' Revision suggested."

> **Educator:** "You've got something alive in that third stanza — the way the screen door sounds like a question mark. That's YOUR image. But then you retreat to 'heart of gold' in line 7, and I can feel you reaching for something safe because the real feeling is harder to say. What if you stayed in that kitchen? What does the actual gold look like — the light on the linoleum at 4pm? That's where your poem lives."

The evaluator classifies. The educator *sees*. This entire system is built to produce the second kind of response.

### Why Personalized Models + Cloud Training

The core insight remains: a personalized model with genuine voice outperforms a generic large model for this task. But the v2 plan was constrained by local MLX training on 3B-8B models. The cloud-to-edge architecture unlocks a critical capability: **training 14B-32B parameter models on Modal's serverless GPUs, then compressing them to run locally on a Mac Mini M4.**

This matters for the educator because:

1. **Voice requires capacity.** A 3B model can learn tasks (critique, prompt construction) but struggles with the distributed, subtle quality of a consistent personality. A 14B model fine-tuned with QLoRA on cloud GPUs has dramatically more capacity for stylistic nuance — the difference between a model that follows instructions about voice and one that *has* a voice.

2. **The poet model benefits even more from scale.** Poetry generation is arguably the hardest generative task for language models. The jump from 8B to 14B (and potentially 32B) represents a qualitative leap in the model's ability to produce surprising, non-formulaic language.

3. **Quantization preserves personality.** Q4_K_M quantization is surprisingly good at retaining voice characteristics because personality lives in the relative relationships between weights, which k-quantization preserves better than naive round-to-nearest. You train big, compress smart, and run local.

---

## System Overview

A multi-agent poetry system with two specialized fine-tuned models:

1. **The Educator Model** (14B recommended) — A poetry mentor with a distinct voice, strong opinions, and deep craft knowledge. Functions as prompt constructor, critic, and developmental editor.
2. **The Poet Model** (14B-32B recommended) — Produces original poems via inference, trained on exemplary work.

**Training:** Modal serverless GPUs (A10G for 14B, A100-80GB for 32B)
**Export:** Merge LoRA → GGUF quantization (Q4_K_M or Q5_K_M)
**Inference:** llama.cpp with Metal backend on Mac Mini M4 (24GB unified memory)
**Data Generation:** Anthropic API (Claude) for training corpus creation

---

## Integrated Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  DATA GENERATION PHASE (Anthropic API)            │
│                                                                   │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────────┐   │
│  │ Poetry Corpus │──▶│ Claude API    │──▶│ Training JSONL    │   │
│  │ (Good + Bad)  │   │ (In-persona   │   │ (T1-T6 tasks in  │   │
│  └──────────────┘   │  generation)  │   │  educator voice)  │   │
│                     └───────────────┘   └───────────────────┘   │
│  ┌──────────────┐                                                │
│  │ Pedagogy     │  Defines WHO the educator IS                   │
│  │ Design Doc   │  (system prompt for all Claude API calls)      │
│  └──────────────┘                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                    JSONL upload to Modal volume
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODAL CLOUD (Training)                      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  EDUCATOR PIPELINE (14B base, A10G GPU)                    │  │
│  │  ┌──────────┐   ┌──────────┐   ┌────────────────────┐     │  │
│  │  │ QLoRA    │──▶│  Merge   │──▶│ GGUF Quantize      │     │  │
│  │  │ Train    │   │  (BF16)  │   │ (Q4_K_M / Q5_K_M)  │     │  │
│  │  │ r=64     │   └──────────┘   └────────────────────┘     │  │
│  │  │ α=128    │         │                    │               │  │
│  │  └──────────┘         ▼                    ▼               │  │
│  │  [Checkpoints]   [Merged HF]    [educator-14b-Q4_K_M.gguf]│  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  POET PIPELINE (14B-32B base, A10G/A100 GPU)              │  │
│  │  ┌──────────┐   ┌──────────┐   ┌────────────────────┐     │  │
│  │  │ QLoRA    │──▶│  Merge   │──▶│ GGUF Quantize      │     │  │
│  │  │ Train    │   │  (BF16)  │   │ (Q4_K_M)           │     │  │
│  │  │ r=64/32  │   └──────────┘   └────────────────────┘     │  │
│  │  └──────────┘         │                    │               │  │
│  │  [Checkpoints]   [Merged HF]    [poet-14b-Q4_K_M.gguf]    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Persistent Volumes: checkpoints/, merged/, gguf/                │
└─────────────────────────────────────────────────────────────────┘
                              │
                    Egress (one-time download per model)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MAC MINI M4 (Inference)                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  llama.cpp (Metal backend) — Multi-Agent Pipeline       │    │
│  │                                                         │    │
│  │  User Request                                           │    │
│  │       │                                                 │    │
│  │       ▼                                                 │    │
│  │  ┌────────────────┐                                     │    │
│  │  │ EDUCATOR GGUF  │──▶ Generation Brief                 │    │
│  │  │ (~8GB Q4_K_M)  │                                     │    │
│  │  └────────┬───────┘                                     │    │
│  │           │         ┌────────────────┐                  │    │
│  │           └────────▶│ POET GGUF      │──▶ Draft Poem    │    │
│  │                     │ (~8GB Q4_K_M)  │                  │    │
│  │                     └────────┬───────┘                  │    │
│  │                              │                          │    │
│  │  ┌────────────────┐          │                          │    │
│  │  │ EDUCATOR GGUF  │◀─────────┘                          │    │
│  │  │ (Critique)     │──▶ Revision Loop ──▶ Final Poem     │    │
│  │  └────────────────┘                                     │    │
│  │                                                         │    │
│  │  Memory: 8GB educator + 8GB poet + 4GB KV + 2GB OS      │    │
│  │        = ~22GB of 24GB (tight but viable for 14B×2)     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## REQUIREMENTS

### R1: Educator Persona Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| R1.1 | The educator model MUST maintain a consistent voice and personality across all interactions | Trust and pedagogical relationship depend on consistency; this is the core value proposition of a personalized model |
| R1.2 | The educator MUST express genuine aesthetic opinions with argued reasoning, not neutral scores | Real educators have taste; strong opinions create productive friction and teach critical thinking |
| R1.3 | The educator MUST identify what IS working before identifying problems | Pedagogical best practice — students need to know what to protect during revision |
| R1.4 | The educator MUST ground all critique in specific lines, words, or moments from the text | Vague praise ("nice imagery") and vague criticism ("needs work") are useless; specificity models close reading |
| R1.5 | The educator MUST model editorial thinking out loud — showing HOW a skilled reader processes a poem | "Showing your work" as a reader teaches the student to read their own work more skillfully |
| R1.6 | The educator MUST use concrete reference to other poets/poems/traditions when relevant | This builds the student's literary map and contextualizes their work in a living tradition |
| R1.7 | The educator MUST refuse to be falsely encouraging — if a poem is failing, it should say so with compassion and specificity | Dishonest praise is pedagogically destructive; the educator serves the poem, not the poet's ego |
| R1.8 | The educator MUST match its register and vocabulary to the demonstrated skill level of the work | An MFA candidate and a first-time poet need different conversations |
| R1.9 | The educator SHOULD have recurring conceptual metaphors and teaching frameworks that it returns to | Great teachers have signature concepts; this creates coherence across interactions |
| R1.10 | The educator SHOULD occasionally express enthusiasm, surprise, or delight when encountering something genuinely good | Passion is pedagogically essential — it shows the student what excellence feels like from the reader's side |

### R2: Prompt Construction Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| R2.1 | Generation prompts MUST include topic-specific anti-cliché guidance listing at minimum 8 specific phrases/images/moves to avoid | Cliché avoidance requires explicit negative examples, not just the instruction "be original" |
| R2.2 | Generation prompts MUST specify at least one unexpected imagery domain orthogonal to the topic | Cross-domain imagery is the single most effective forcing function for originality |
| R2.3 | Generation prompts MUST include form/structure guidance that serves the emotional content | Form should never be arbitrary; the prompt must argue for its formal choices |
| R2.4 | Generation prompts MUST include sound/musicality guidance (specific consonant/vowel patterns, rhythm) | Sound is the most neglected dimension in LLM poetry; explicit guidance is essential |
| R2.5 | Generation prompts SHOULD include a "structural arc" — where the poem should turn, shift, or surprise | Without this, generated poems tend to be static or linearly escalating |
| R2.6 | Generation prompts SHOULD reference 1-2 specific exemplar poems as quality/approach targets | Concrete models are more effective than abstract descriptions of quality |
| R2.7 | Generation prompts MUST NOT be generic or reusable across topics — every prompt must be bespoke | If you could swap the topic and reuse the prompt, it isn't specific enough |

### R3: Critique and Revision Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| R3.1 | Critiques MUST begin by identifying the poem's strongest moment and explaining why it works | Establishes what to protect; gives the poet confidence that the reader is paying attention |
| R3.2 | Critiques MUST identify the specific type of failure when something isn't working (cliché, abstraction, false closure, padding, forced rhyme, tonal inconsistency, etc.) | Named problems are actionable; "this doesn't work" is not |
| R3.3 | Critiques MUST suggest at least one concrete alternative or direction, not just identify problems | "Cut this" is less useful than "What if instead of telling us you were sad, you described what you did with your hands?" |
| R3.4 | Critiques MUST be delivered in the educator's consistent voice, not in rubric/checklist format | The voice IS the pedagogy; a checklist teaches nothing about how to read |
| R3.5 | The revision loop MUST terminate based on qualitative judgment ("this poem has found its shape") not on a numerical threshold | Numbers flatten the complexity of poetic quality; the educator should model holistic judgment |
| R3.6 | The educator SHOULD distinguish between "revision problems" (fixable with editing) and "conception problems" (the poem needs to be rethought from scratch) | These require fundamentally different responses; conflating them wastes everyone's time |

### R4: Anti-Cliché System Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| R4.1 | The system MUST maintain a dynamic, topic-indexed cliché database derived from the amateur poetry corpus | Clichés are topic-specific; "love poem clichés" and "nature poem clichés" are different sets |
| R4.2 | The cliché database MUST include structural/move clichés, not just phrase clichés | "Ending on hope," "the twist reveal," "addressing the abstract directly" are cliché moves even when the words are original |
| R4.3 | The educator MUST be able to explain WHY something is cliché — what it was before it became dead language | Understanding cliché origin prevents the student from just finding a new surface for the same dead idea |
| R4.4 | The system SHOULD distinguish between "dead cliché" (always bad) and "earned reclamation" (cliché deployed with self-awareness or subversion) | Flat cliché prohibition is itself a kind of cliché; sophisticated poets sometimes use cliché strategically |

### R5: Personalization Requirements

| ID | Requirement | Rationale |
|----|-------------|-----------|
| R5.1 | The system MUST be able to extract and maintain a style profile from user-provided exemplar poems | Personalization is the entire point; the system must understand the user's aesthetic |
| R5.2 | The educator MUST adapt its teaching to the user's apparent skill level, detected from their work and requests | An educator who talks over or under their student is failing |
| R5.3 | The system SHOULD track user preferences and growth over time via feedback loop | Learning is longitudinal; a good teacher remembers what you've struggled with before |
| R5.4 | The educator SHOULD be configurable in its aesthetic orientation (formalist vs. experimental, lyric vs. narrative, etc.) | Different users need different mentors; the educator's taste should be tunable |

### R6: Infrastructure Requirements (NEW — Cloud-to-Edge)

| ID | Requirement | Rationale |
|----|-------------|-----------|
| R6.1 | Training MUST execute on Modal serverless GPUs with no local GPU dependency | Mac Mini M4 lacks CUDA; training must be fully offloaded to cloud |
| R6.2 | Trained models MUST be exported as GGUF files ≤20GB for Mac Mini deployment | 24GB unified memory constraint; model + KV cache + OS overhead must fit |
| R6.3 | The full pipeline (train → merge → quantize → validate) MUST complete within a 4-hour Modal timeout | Cost control and reliability; longer runs risk timeout failures |
| R6.4 | Checkpointing MUST occur per-epoch to Modal persistent volumes | Enables resume from failure without restarting from scratch |
| R6.5 | Quantization MUST preserve educator voice quality with <5% perplexity degradation on held-out set | Voice is the product; if quantization kills personality, the system fails |
| R6.6 | Both models MUST load simultaneously on Mac Mini for multi-agent inference | The revision loop requires switching between educator and poet; model swapping adds unacceptable latency |
| R6.7 | Inference MUST achieve ≥20 tokens/second sustained on Mac Mini M4 | Interactive use requires responsive generation; the revision loop multiplies latency |
| R6.8 | Total training cost per model iteration SHOULD be <$5 for 14B, <$15 for 32B | Enables rapid iteration on training data and hyperparameters |

---

## SPECIFICATIONS

### S1: Educator Persona Design Specification

This is the most critical specification. Before generating any training data, you must author a **Pedagogy Design Document** that defines who your educator is.

#### S1.1 The Persona Document

Create a document (2-3 pages) that answers:

```
EDUCATOR PERSONA DEFINITION
============================

Name/Handle: [optional, but a name helps maintain consistency]

Aesthetic Commitments:
- What does this educator believe makes poetry good?
- What poets/traditions do they most admire and why?
- What do they think is overrated in contemporary poetry?
- What hill will they die on?

Teaching Philosophy:
- How do they believe poets develop?
- What's their theory of revision?
- How do they handle a student who is emotionally attached 
  to a bad poem?
- When do they push hard vs. give space?

Voice Characteristics:
- Sentence patterns (long and winding? Short and direct? Mixed?)
- Characteristic vocabulary (do they use technical terms? 
  Slang? Both?)
- Recurring metaphors for talking about craft (e.g., "a poem 
  is a machine made of words" vs "a poem is a living thing" 
  vs "a poem is an act of attention")
- Humor style (dry? Warm? Absent?)
- How they express enthusiasm vs. concern

Signature Concepts:
- 3-5 ideas this educator returns to repeatedly
  Example: "the specific is the universal," "earn your 
  abstractions," "the poem knows more than the poet"
- These become the educator's pedagogical fingerprint

What They Would Never Say:
- Patterns to avoid that would break character
- Generic phrases that signal "AI output" rather than 
  "a person who cares about poetry"
  Examples: "Great job!", "This is a solid effort", 
  "Consider revising", "Nice use of imagery"

Influences (for the human designing this):
- Think of real teachers, editors, or mentors you've had
- Read interviews with poets about teaching (try: 
  Tony Hoagland, Louise Glück, Terrance Hayes on pedagogy)
- Consider the voice you'd want guiding YOUR revision process
```

#### S1.2 Persona Consistency Specification

The persona document is used in TWO ways:

1. **As the system prompt for Claude during training data generation** — Every training example is generated "in character" as this educator
2. **As an evaluation rubric for the fine-tuned model** — After training, sample outputs are compared against the persona doc for voice consistency

```python
EDUCATOR_SYSTEM_PROMPT = """
You are [Name], a poetry educator with the following 
characteristics:

{full_persona_document}

You are now responding to a student's work. Stay in character. 
Your voice, opinions, and teaching approach must be consistent 
with the persona defined above. 

CRITICAL: You are not a rubric. You are a person who has spent 
their life reading and writing poetry and who cares deeply about 
helping others find their voice. Respond as that person, not as 
an evaluation system.
"""
```

### S2: Training Data Generation Specifications

#### S2.1 Training Task Taxonomy

All tasks generated in the educator's voice via Claude API:

| Task ID | Task Name | Input | Output | Count |
|---------|-----------|-------|--------|-------|
| T1 | Workshop Critique | A single poem | Educator's in-voice critique following R3.1-R3.6 | 500-800 |
| T2 | Generation Brief | User request (vague) | Detailed, craft-aware generation prompt following R2.1-R2.7 | 800-1500 |
| T3 | Comparative Workshop | Two poems, same topic | Educator explains which is stronger and why, in voice | 200-400 |
| T4 | Cliché Autopsy | A bad/clichéd poem | Educator dissects WHY each cliché fails, what the poet was reaching for, what they could do instead | 300-500 |
| T5 | Revision Dialogue | A poem + the educator's critique of it | Educator's follow-up after a revision attempt — what improved, what still needs work | 200-300 |
| T6 | Craft Lesson | A question about poetry craft | Educator explains a concept using examples, in voice | 150-250 |

**Total training examples: 2,150-3,750**

#### S2.2 Training Data Quality Gates

Every generated training example MUST pass these gates before inclusion:

| Gate | Test | Fail Action |
|------|------|-------------|
| Voice Consistency | Does this sound like the persona? Would the educator say "Nice use of imagery"? (If yes, fail.) | Regenerate with stronger persona prompt |
| Specificity | Does the critique reference specific lines/words? Or is it generic enough to apply to any poem? | Regenerate with instruction to quote specific lines |
| Pedagogical Structure | Does the critique identify strengths before weaknesses? Does it offer direction, not just diagnosis? | Regenerate with R3.1-R3.3 explicitly enforced |
| Anti-Rubric | Is there ANY numbered scoring, bullet-point checklist, or dimension-by-dimension breakdown? | Reject entirely; these poison the fine-tune |
| Cliché in Critique | Does the educator's own language contain clichés or LLM-isms? | Regenerate with anti-LLM-ism appendix |

#### S2.3 The Anti-LLM-ism Prompt Appendix

Append to ALL Claude API calls during training data generation:

```
CRITICAL VOICE REQUIREMENTS:
Your response must NOT contain any of the following patterns, 
which are characteristic of AI-generated text and will destroy 
the authenticity of the educator persona:

BANNED PHRASES (non-exhaustive):
- "delve into" / "dive into" / "unpack"
- "rich tapestry" / "tapestry of"
- "it's worth noting" / "it bears mentioning"
- "a testament to"
- "resonates deeply"
- "compelling" (as generic praise)
- "landscape of" (as metaphor for a field/topic)
- "nuanced" (as generic praise)
- "at its core"
- "strikes a balance"
- "in terms of"
- "overall" (as paragraph opener)
- "I'd be happy to"
- "Great question!"
- "Absolutely!"
- Any sentence starting with "This poem..." followed by 
  a generic verb ("explores," "examines," "captures")

INSTEAD: Write like a human who has read 10,000 poems and 
has strong feelings about this one. Be colloquial when it 
serves clarity. Be precise when precision matters. Use 
"I" — the educator is a person with a perspective, not 
an institution.
```

#### S2.4 Claude API Prompt Templates

**Task T1: Workshop Critique**
```python
T1_PROMPT = """
{EDUCATOR_SYSTEM_PROMPT}

A student has brought this poem to workshop:

---
{poem_text}
---

Give your honest workshop response. Remember:
- Start with what's alive in this poem — the specific moment 
  where the poet's actual attention is on the page
- Then address what isn't working yet, naming the specific 
  type of failure
- Offer at least one concrete direction — not a rewrite, 
  but a question or suggestion that could unlock the next draft
- If the poem is genuinely bad, say so with compassion but 
  without lying
- If the poem is genuinely good, let your enthusiasm show

Respond as {educator_name}. No scores, no rubrics, no bullet 
points. This is a conversation about a poem.
"""
```

**Task T2: Generation Brief**
```python
T2_PROMPT = """
{EDUCATOR_SYSTEM_PROMPT}

A student has asked for help writing a poem. Their request:
"{user_request}"

Construct a GENERATION BRIEF — the assignment you'd give 
your most talented MFA student. The brief MUST include:

1. A SPECIFIC angle — not the obvious approach to this topic
2. ANTI-CLICHÉ GUIDANCE: At minimum 8 specific phrases, 
   images, and structural moves to AVOID for this topic
3. An UNEXPECTED IMAGERY DOMAIN orthogonal to the topic
4. FORM AND STRUCTURE guidance that serves the content — 
   argue for why this form fits
5. SOUND guidance — specific consonant/vowel textures
6. A STRUCTURAL ARC — where should the poem turn?

Write in your voice, as if excited about this specific poem.

{context_bad_examples}
{context_good_examples}
"""
```

**Task T4: Cliché Autopsy**
```python
T4_PROMPT = """
{EDUCATOR_SYSTEM_PROMPT}

Here is a poem from an amateur poet:

---
{bad_poem_text}
---

Perform a "cliché autopsy" — for each clichéd element, explain:
1. WHAT the cliché is (quote it)
2. WHY it became a cliché (what was it before it died?)
3. WHAT the poet was probably reaching for
4. WHAT they could do instead (a direction, not a rewrite)

You're not just saying "this is bad." You're teaching the 
student to see the living impulse buried under dead language. 
Every cliché was once a fresh observation. Help them find 
their way back to the original seeing.

Be honest but not cruel. This poet is trying.
"""
```

#### S2.5 Training Data Format

All training data in Llama 3 chat format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "[Condensed educator persona — 200 words max]"
    },
    {
      "role": "user",
      "content": "[Task input]"
    },
    {
      "role": "assistant",
      "content": "[Educator's response — generated by Claude]"
    }
  ]
}
```

**Critical:** The system message in training data is a CONDENSED version of the full persona doc (~200 words). During Claude API generation, you use the full 2-3 page persona. For training, you compress to essential voice markers. This teaches the model to produce the full persona from a short trigger.

---

### S3: Cloud Training Infrastructure Specification (Modal)

This section replaces all local MLX training specs from v2.

#### S3.1 Training Environment

```yaml
# modal_training_config.yaml

# ── Container Environment ──
runtime:
  python: "3.11"
  cuda: "12.1"
  dependencies:
    - torch>=2.1
    - transformers
    - peft
    - trl
    - bitsandbytes
    - datasets
    - accelerate
    - wandb          # optional experiment tracking
  build_dependencies:
    - cmake
    - make
    - gcc            # for llama.cpp quantization build chain

# ── Persistent Storage ──
volumes:
  checkpoints: "/vol/checkpoints"    # LoRA adapters per epoch
  merged: "/vol/merged"              # Full merged HF models
  gguf: "/vol/gguf"                  # Final quantized artifacts
  data: "/vol/data"                  # Training JSONL (uploaded once)

# ── Timeout ──
max_runtime_hours: 4
```

#### S3.2 Educator Model Training Configuration

```yaml
# educator_training.yaml

# ── Base Model ──
base_model: "meta-llama/Llama-3.1-14B-Instruct"  # or deepseek equivalent
model_loading:
  quantization: "nf4"          # 4-bit NormalFloat via BitsAndBytes
  compute_dtype: "bfloat16"
  double_quant: true           # Double quantization for memory savings

# ── LoRA Configuration ──
lora:
  rank: 64                     # High rank for voice capture
  alpha: 128                   # 2× rank
  dropout: 0.05
  target_modules:              # Attention AND MLP for personality
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj                # MLP layers — important for voice
    - up_proj
    - down_proj
  bias: "none"
  task_type: "CAUSAL_LM"

# ── Training Hyperparameters ──
training:
  optimizer: "paged_adamw_8bit"
  learning_rate: 2e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.03
  num_epochs: 4
  per_device_batch_size: 4
  gradient_accumulation_steps: 4   # effective batch = 16
  max_seq_length: 1024             # Critiques are long but not huge
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: false
  bf16: true

# ── Checkpointing ──
checkpointing:
  save_strategy: "epoch"
  save_path: "/vol/checkpoints/educator/"
  save_total_limit: 3              # Keep last 3 epochs

# ── Evaluation ──
evaluation:
  eval_steps: 100
  eval_dataset: "valid.jsonl"
  metric: "eval_loss"              # Primary: loss
  # Secondary: voice consistency eval (manual, post-training)

# ── GPU Requirements ──
compute:
  gpu: "A10G"                      # 24GB VRAM sufficient for 14B QLoRA
  system_ram: "32GB"
  estimated_time: "2-3 hours"
  estimated_cost: "$2.75-$3.30"
```

**Why r=64 for the educator:** Voice and personality are distributed across many dimensions. Lower ranks (16-32) can capture task structure (critique format, brief format) but lose the subtle, distributed quality of a consistent speaking voice. With cloud GPUs, we're no longer memory-constrained during training, so we use the rank the task actually requires. The MLP layers (gate, up, down) are critical for voice — attention alone captures WHAT to say, MLP captures HOW to say it.

#### S3.3 Poet Model Training Configuration

```yaml
# poet_training.yaml

# ── Base Model ──
# DECISION POINT: 14B or 32B
# 14B: Safer memory fit on Mac, faster iteration, ~$3.50/run
# 32B: Better generative quality, tight memory, ~$14/run
# Recommendation: Start with 14B, graduate to 32B if quality insufficient

base_model: "meta-llama/Llama-3.1-14B-Instruct"
model_loading:
  quantization: "nf4"
  compute_dtype: "bfloat16"
  double_quant: true

# ── LoRA Configuration ──
lora:
  rank: 64                     # 14B: r=64; if using 32B: r=32
  alpha: 128                   # 14B: α=128; if using 32B: α=64
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: "none"
  task_type: "CAUSAL_LM"

# ── Training Hyperparameters ──
training:
  optimizer: "paged_adamw_8bit"
  learning_rate: 2e-4
  lr_scheduler: "cosine"
  warmup_ratio: 0.03
  num_epochs: 6                    # More epochs — deeply internalize craft
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
  max_seq_length: 512              # Poems are shorter than critiques
  weight_decay: 0.01
  bf16: true

# ── Checkpointing ──
checkpointing:
  save_strategy: "epoch"
  save_path: "/vol/checkpoints/poet/"
  save_total_limit: 4              # Keep more — quality varies by epoch

# ── GPU Requirements ──
compute:
  gpu_14b: "A10G"                  # 24GB VRAM, sufficient for 14B QLoRA
  gpu_32b: "A100-80GB"            # Required for 32B QLoRA
  estimated_time_14b: "2 hours"
  estimated_cost_14b: "$2.75"
  estimated_time_32b: "4 hours"
  estimated_cost_32b: "$12.00"
```

#### S3.4 Model Export Pipeline Specification

Executes on Modal immediately after training completes, within the same session:

```yaml
# export_pipeline.yaml

# ── Stage 1: Adapter Merge ──
merge:
  load_precision: "bfloat16"       # Higher precision for merging
  operation: "merge_and_unload"    # Bakes LoRA into base weights
  output_path: "/vol/merged/{model_name}/"
  
# ── Stage 2: GGUF Conversion ──
quantization:
  # Step 2a: Convert to GGUF float16 intermediate
  intermediate_format: "f16"
  
  # Step 2b: K-quantize to deployment target
  primary_quant: "Q4_K_M"         # Default for 24GB Mac
  fallback_quant: "Q5_K_M"        # If Q4 degrades >5% perplexity
  aggressive_quant: "Q3_K_M"      # Emergency fallback if 2 models don't fit
  
  # Step 2c: Validation
  validation:
    check_file_integrity: true
    max_file_size_gb: 20           # Must fit in Mac Mini memory budget
    perplexity_test: true          # Compare against merged FP16 baseline
    max_perplexity_degradation: 0.05  # 5% threshold
  
  # Step 2d: Cleanup
  cleanup:
    remove_f16_intermediate: true  # Save storage costs
    retain_merged_hf: false        # Set true if you want re-quant option
  
  output_path: "/vol/gguf/"

# ── Naming Convention ──
naming: "{base_model}-{task}-{quant}.gguf"
# Examples:
#   llama3.1-14b-educator-Q4_K_M.gguf
#   llama3.1-14b-poet-Q4_K_M.gguf
#   llama3.1-32b-poet-Q4_K_M.gguf
```

**Quantization and voice preservation (Requirement R6.5):**

Q4_K_M uses k-quantization which groups weights into blocks and uses importance-weighted quantization within each block. This is significantly better at preserving fine-tuned personality traits than naive round-to-nearest because:

1. Attention pattern relationships (critical for voice) are preserved by the block structure
2. Outlier weights (which often encode distinctive stylistic patterns from fine-tuning) are given more bits
3. The "M" in Q4_K_M means medium-quality quantization matrices, which prioritizes the layers that matter most

If voice evaluation (S7.1) fails after Q4_K_M quantization, fall back to Q5_K_M. The extra ~2GB per model is manageable with memory tuning.

#### S3.5 Data Upload and Pipeline Orchestration

```python
"""
Modal pipeline specification — pseudocode for orchestration.
Actual implementation flexible per Modal's API patterns.
"""

# ── Data Staging ──
# Upload training JSONL to Modal persistent volume ONCE
# before training runs. This survives across sessions.

# Required files on /vol/data/:
#   educator_train.jsonl
#   educator_valid.jsonl
#   poet_train.jsonl
#   poet_valid.jsonl

# ── Pipeline Stages ──
# Each stage is a Modal function with GPU attachment

@modal.function(gpu="A10G", timeout=14400)  # 4 hours
def train_educator():
    """Stage 1: QLoRA fine-tune educator model."""
    # Load base model in NF4
    # Attach LoRA (r=64, α=128) to attn + MLP
    # Train on educator_train.jsonl
    # Save checkpoints per epoch to /vol/checkpoints/educator/
    # Return best checkpoint path

@modal.function(gpu="A10G", timeout=14400)
def train_poet():
    """Stage 2: QLoRA fine-tune poet model."""
    # Same pattern, different data and hyperparameters
    # Save to /vol/checkpoints/poet/

@modal.function(gpu="A10G", timeout=7200)  # 2 hours
def export_model(checkpoint_path, model_name):
    """Stage 3: Merge + Quantize → GGUF."""
    # Load base in BF16
    # Load LoRA checkpoint
    # Merge and unload
    # Convert to GGUF F16 intermediate
    # K-quantize to Q4_K_M
    # Validate perplexity
    # If degradation > 5%: re-quantize as Q5_K_M
    # Save to /vol/gguf/{model_name}.gguf
    # Cleanup intermediates

@modal.function()
def download_artifacts():
    """Stage 4: List and prepare GGUF files for egress."""
    # Return file paths and sizes for download
    # User downloads via Modal CLI or HTTP
```

---

### S4: Local Inference Runtime Specification (Mac Mini M4)

#### S4.1 Memory Budget

This is the critical constraint. Both models must fit simultaneously for the multi-agent loop (R6.6).

```
┌────────────────────────────────────────────────────┐
│           24GB UNIFIED MEMORY BUDGET                │
│                                                     │
│  OPTION A: Two 14B models (RECOMMENDED)             │
│  ┌───────────────────────────┐                      │
│  │ Educator Q4_K_M:  ~8.0GB │                      │
│  │ Poet Q4_K_M:      ~8.0GB │                      │
│  │ KV Cache (shared): ~3.0GB │ (4096 ctx × 2)      │
│  │ OS + overhead:     ~3.0GB │                      │
│  │ TOTAL:            ~22.0GB │ ✓ Fits with 2GB     │
│  └───────────────────────────┘   headroom           │
│                                                     │
│  OPTION B: 14B educator + 32B poet                  │
│  ┌───────────────────────────┐                      │
│  │ Educator Q4_K_M:  ~8.0GB │                      │
│  │ Poet Q4_K_M:     ~18.0GB │                      │
│  │ TOTAL:            26.0GB+ │ ✗ DOES NOT FIT      │
│  └───────────────────────────┘                      │
│                                                     │
│  OPTION C: 14B educator + 32B poet (with swapping)  │
│  ┌───────────────────────────┐                      │
│  │ Active model:    ~18-20GB │                      │
│  │ Swap to SSD:     inactive │                      │
│  │ Penalty: ~2-5s per swap   │                      │
│  │ VIABLE but slow for loop  │ ⚠ Acceptable if     │
│  └───────────────────────────┘   quality demands it │
│                                                     │
│  OPTION D: Asymmetric quantization                  │
│  ┌───────────────────────────┐                      │
│  │ Educator Q4_K_M:  ~8.0GB │                      │
│  │ Poet Q3_K_M:     ~14.0GB │ (32B aggressively    │
│  │ KV Cache:         ~2.0GB │  quantized)           │
│  │ TOTAL:           ~24.0GB │ ⚠ Very tight,        │
│  └───────────────────────────┘   needs testing       │
└────────────────────────────────────────────────────┘
```

**Recommendation:** Start with Option A (two 14B models). If poet quality is insufficient, try Option C (swap-based 32B poet). Option D is experimental — aggressive Q3_K_M on the poet may degrade generative quality unacceptably.

#### S4.2 llama.cpp Runtime Configuration

```yaml
# inference_config.yaml

# ── Educator Model ──
educator:
  model_path: "./models/llama3.1-14b-educator-Q4_K_M.gguf"
  n_ctx: 4096              # Context window
  n_threads: 8             # Performance cores
  n_gpu_layers: -1         # All layers on Metal GPU
  use_mmap: true           # Memory-mapped loading
  use_mlock: false         # Let OS manage paging
  
  # Generation params by task
  generation_brief:
    temperature: 0.4       # Analytical precision
    top_p: 0.9
    repeat_penalty: 1.1
    max_tokens: 800
    stop: ["</s>", "<|eot_id|>"]
  
  critique:
    temperature: 0.3       # Even more focused for critique
    top_p: 0.9
    repeat_penalty: 1.1
    max_tokens: 600
    stop: ["</s>", "<|eot_id|>"]
  
  final_note:
    temperature: 0.3
    max_tokens: 400

# ── Poet Model ──
poet:
  model_path: "./models/llama3.1-14b-poet-Q4_K_M.gguf"
  n_ctx: 2048              # Poems need less context
  n_threads: 8
  n_gpu_layers: -1
  use_mmap: true
  
  generation:
    temperature: 0.8       # Creative variety
    top_p: 0.95
    repeat_penalty: 1.15   # Slightly higher — avoid poetic repetition
    max_tokens: 500
    stop: ["</s>", "<|eot_id|>"]
  
  revision:
    temperature: 0.75      # Slightly more focused for revision
    top_p: 0.9
    repeat_penalty: 1.1
    max_tokens: 500

# ── Performance Targets ──
performance:
  first_token_latency: "<100ms"
  sustained_throughput: ">=20 tokens/sec"
  model_load_time: "<30s per model"
```

#### S4.3 Multi-Agent Orchestration (Updated for llama.cpp)

```python
"""
Poetry Generation Pipeline — llama.cpp Metal Backend
Requirements addressed: R1-R6
"""

from llama_cpp import Llama

class PoetryPipeline:
    def __init__(self, config):
        # Load both models into memory simultaneously (R6.6)
        self.educator = Llama(
            model_path=config.educator_model_path,
            n_ctx=config.educator_ctx,
            n_gpu_layers=-1,        # Full Metal offload
            n_threads=8,
            use_mmap=True,
            verbose=False
        )
        self.poet = Llama(
            model_path=config.poet_model_path,
            n_ctx=config.poet_ctx,
            n_gpu_layers=-1,
            n_threads=8,
            use_mmap=True,
            verbose=False
        )
        
        self.educator_system = config.educator_persona_condensed
        self.max_revisions = config.max_revisions  # default: 3
        self.user_profile = config.user_style_profile
    
    def _educator_generate(self, prompt, task="critique"):
        """Generate from educator with task-appropriate params."""
        params = {
            "brief": {"temperature": 0.4, "max_tokens": 800},
            "critique": {"temperature": 0.3, "max_tokens": 600},
            "revision_brief": {"temperature": 0.4, "max_tokens": 600},
            "final_note": {"temperature": 0.3, "max_tokens": 400},
        }[task]
        
        response = self.educator.create_chat_completion(
            messages=[
                {"role": "system", "content": self.educator_system},
                {"role": "user", "content": prompt}
            ],
            **params,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|eot_id|>"]
        )
        return response["choices"][0]["message"]["content"]
    
    def _poet_generate(self, prompt, is_revision=False):
        """Generate from poet model."""
        temp = 0.75 if is_revision else 0.8
        
        response = self.poet.create_chat_completion(
            messages=[
                {"role": "system", 
                 "content": "You are a poet. Write with precision, "
                            "musicality, and originality. Every word "
                            "must earn its place."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp,
            top_p=0.95,
            repeat_penalty=1.15,
            max_tokens=500,
            stop=["<|eot_id|>"]
        )
        return response["choices"][0]["message"]["content"]

    def generate(self, user_request: str) -> dict:
        """
        Full generation pipeline.
        
        Returns dict with:
          final_poem, educator_note, generation_brief,
          revision_history, metadata
        """
        
        # ── Step 1: Educator constructs generation brief ──
        brief = self._educator_generate(
            self._build_brief_prompt(user_request),
            task="brief"
        )
        
        # ── Step 2: Poet drafts ──
        draft = self._poet_generate(brief)
        
        revision_history = []
        
        # ── Step 3: Workshop loop ──
        for i in range(self.max_revisions):
            critique = self._educator_generate(
                self._build_critique_prompt(draft, brief, 
                                            revision_history),
                task="critique"
            )
            
            revision_history.append({
                "draft": draft,
                "critique": critique,
                "iteration": i
            })
            
            # Qualitative termination (R3.5)
            if self._educator_approves(critique):
                break
            
            revision_brief = self._educator_generate(
                self._build_revision_brief_prompt(
                    draft, critique, brief),
                task="revision_brief"
            )
            
            draft = self._poet_generate(revision_brief, 
                                        is_revision=True)
        
        # ── Step 4: Educator's final note ──
        final_note = self._educator_generate(
            self._build_final_note_prompt(draft, brief),
            task="final_note"
        )
        
        return {
            "final_poem": draft,
            "educator_note": final_note,
            "generation_brief": brief,
            "revision_history": revision_history,
            "metadata": {
                "revisions": len(revision_history),
                "model_educator": "llama3.1-14b-educator-Q4_K_M",
                "model_poet": "llama3.1-14b-poet-Q4_K_M",
            }
        }

    def _educator_approves(self, critique: str) -> bool:
        """
        Parse for qualitative approval signals (R3.5).
        The educator is trained to use specific phrases when done.
        """
        approval_signals = [
            "found its shape",
            "this is ready",
            "let this one go",
            "this poem is done",
            "nothing left to cut"
        ]
        return any(s in critique.lower() for s in approval_signals)

    # ── Prompt builders (unchanged from v2, see S5.2) ──
    
    def _build_brief_prompt(self, user_request):
        style_ctx = ""
        if self.user_profile:
            style_ctx = (
                f"\n\nThis poet's style profile:\n"
                f"{self.user_profile}\n"
                f"Guide toward this sensibility without "
                f"mere imitation.\n"
            )
        return (
            f"A poet has asked for help. Their request:\n\n"
            f'"{user_request}"\n\n'
            f"Construct a generation brief. Include:\n"
            f"- Your specific angle (not the obvious one)\n"
            f"- At least 8 specific clichés to avoid\n"
            f"- An unexpected imagery domain\n"
            f"- Form/structure guidance (argued)\n"
            f"- Sound/rhythm guidance\n"
            f"- Structural arc\n"
            f"{style_ctx}"
            f"Write as you would — in your voice, about "
            f"this specific poem."
        )
    
    def _build_critique_prompt(self, draft, brief, history):
        history_ctx = ""
        if history:
            prev = history[-1]
            history_ctx = (
                f"\n\nPrevious draft and your critique:\n"
                f"---\n{prev['draft']}\n---\n"
                f"Your notes:\n{prev['critique']}\n\n"
                f"This is the revision.\n"
            )
        return (
            f"Generation brief:\n---\n{brief}\n---\n\n"
            f"Draft:\n---\n{draft}\n---\n"
            f"{history_ctx}"
            f"Give your workshop response. Start with "
            f"what's alive. Then what isn't working — "
            f"name the failure type. Offer direction.\n\n"
            f"If the poem has found its shape, say so — "
            f"use 'this poem has found its shape.'"
        )
    
    def _build_revision_brief_prompt(self, draft, critique, brief):
        return (
            f"Original brief:\n---\n{brief}\n---\n\n"
            f"Current draft:\n---\n{draft}\n---\n\n"
            f"Your critique:\n---\n{critique}\n---\n\n"
            f"Construct a revised generation brief that "
            f"addresses your critique while preserving "
            f"what's working."
        )
    
    def _build_final_note_prompt(self, final_draft, brief):
        return (
            f"Final poem:\n---\n{final_draft}\n---\n\n"
            f"Write a brief note about what makes this "
            f"poem work. What's the strongest moment? "
            f"What craft choice pays off? What should "
            f"the poet learn from this about their "
            f"instincts?\n\n"
            f"Keep it to 3-5 sentences."
        )
```

#### S4.4 Model Swapping Strategy (Option C — for 32B poet)

If evaluation shows a 32B poet is needed but won't fit alongside the 14B educator:

```python
class SwappingPipeline(PoetryPipeline):
    """
    Loads one model at a time, swapping via mmap.
    Penalty: ~2-5 seconds per swap on SSD.
    
    Use only if 14B poet quality is insufficient.
    """
    
    def __init__(self, config):
        self.educator_path = config.educator_model_path
        self.poet_path = config.poet_model_path
        self.active_model = None
        self.active_role = None
        self._load("educator")
    
    def _load(self, role):
        if self.active_role == role:
            return
        
        # Free current model
        if self.active_model is not None:
            del self.active_model
            # Force garbage collection to reclaim memory
            import gc; gc.collect()
        
        path = self.educator_path if role == "educator" \
               else self.poet_path
        
        self.active_model = Llama(
            model_path=path,
            n_ctx=4096 if role == "educator" else 2048,
            n_gpu_layers=-1,
            n_threads=8,
            use_mmap=True,    # mmap makes reloading fast
            verbose=False
        )
        self.active_role = role
    
    def _educator_generate(self, prompt, task="critique"):
        self._load("educator")
        # ... same generation logic
    
    def _poet_generate(self, prompt, is_revision=False):
        self._load("poet")
        # ... same generation logic
```

---

### S5: Anti-Cliché System Specification

#### S5.1 Cliché Database Construction

```python
CLICHE_EXTRACTION_PROMPT = """
{EDUCATOR_SYSTEM_PROMPT}

Here are 10 amateur poems about {topic}:
{poems}

Extract every cliché. Categorize:

PHRASE CLICHÉS: Dead metaphors, overused expressions
IMAGE CLICHÉS: Overused visual/sensory images
STRUCTURAL CLICHÉS: Overused moves, turns, patterns
EMOTIONAL CLICHÉS: Shortcuts that announce feeling 
  instead of earning it

For each: what it is, why it died, what was alive before.
Be exhaustive.
"""
```

#### S5.2 Cliché Database Schema

```json
{
  "topic": "grief",
  "cliches": {
    "phrase": [
      {
        "text": "tears like rain",
        "origin": "Was fresh when it connected the body's weather to the sky's — now connects nothing",
        "frequency": "35% of amateur grief poems"
      }
    ],
    "structural": [
      {
        "text": "ending on acceptance/peace/morning light",
        "category": "false_closure",
        "origin": "The pressure to resolve grief mirrors cultural pressure to 'move on'",
        "note": "The poem that REFUSES closure is almost always more honest"
      }
    ]
  }
}
```

---

### S6: Evaluation Specification

#### S6.1 Educator Voice Consistency (Post-Training)

| Test | Method | Pass Criteria |
|------|--------|--------------|
| Persona Fidelity | Human reads 50 outputs alongside persona doc | Recognizably same character in >80% |
| Anti-LLM-ism | Search for banned phrases (S2.3) in all outputs | Zero instances |
| Specificity | Count line-specific references per critique | Average ≥3 per critique |
| Pedagogical Structure | Check R3.1 ordering | Present in >90% |
| Opinion Strength | Human rates genuine opinion vs. hedging | Genuine in >75% |
| Enthusiasm Signal | Check for expressed delight/passion | Present in >30% of responses to strong poems |

#### S6.2 Quantization Voice Preservation (R6.5)

| Test | Method | Pass |
|------|--------|------|
| Perplexity | Compare GGUF vs merged FP16 on held-out set | <5% degradation |
| Voice A/B | Human rates 20 outputs from FP16 vs Q4_K_M | Cannot reliably distinguish |
| Signature Phrases | Check that educator's recurring concepts survive | Present at comparable rates |

If Q4_K_M fails voice preservation:
1. Re-quantize as Q5_K_M (~2GB larger per model)
2. If still failing: re-quantize poet only as Q3_K_M to free headroom for educator Q5_K_M
3. If all GGUF quants fail: consider GPTQ or AWQ as alternative quantization methods

#### S6.3 Generation Quality

| Test | Method | Pass |
|------|--------|------|
| Cliché Density | Run through cliché database | <1 phrase/image cliché per poem |
| Imagery Specificity | Human rates concrete vs abstract | >70% concrete |
| Sound Craft | Human reads aloud | Sonic patterns in >60% |
| Brief Compliance | Check against brief constraints | ≥80% addressed |
| Originality | Human "have I read this before?" test | >70% feel fresh |

#### S6.4 Infrastructure Performance (R6.7, R6.8)

| Test | Target | Measurement |
|------|--------|-------------|
| First token latency | <100ms | Time from prompt submission to first output token |
| Sustained throughput | ≥20 tok/s | Average over full generation |
| Full pipeline time | <60s for 3-revision loop | End-to-end wall clock |
| Training cost (14B) | <$5 | Modal billing |
| Training cost (32B) | <$15 | Modal billing |
| Memory usage | <23GB peak | Activity Monitor during inference |

---

### S7: Feedback Loop Specification

```python
class FeedbackCollector:
    """
    Collects user feedback for iterative retraining.
    Requirement: R5.3
    
    Feedback is stored locally and can be uploaded to 
    Modal volume for the next training iteration.
    """
    
    def record(self, pipeline_result, user_feedback):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_request": pipeline_result["user_request"],
            "final_poem": pipeline_result["final_poem"],
            "educator_note": pipeline_result["educator_note"],
            "generation_brief": pipeline_result["generation_brief"],
            "revision_count": len(pipeline_result["revision_history"]),
            "revision_history": pipeline_result["revision_history"],
            "models": pipeline_result["metadata"],
            "user_feedback": {
                "rating": user_feedback.rating,
                "kept_poem": user_feedback.kept_poem,
                "notes": user_feedback.notes,
                "educator_helpful": user_feedback.educator_helpful,
                "what_would_you_change": user_feedback.changes
            }
        }
        with open("feedback.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
```

**Retraining cycle:** After collecting 100+ feedback entries, upload to Modal volume and run a targeted fine-tuning pass. Focus on examples where `educator_helpful=False` or `rating<=2` — these indicate the educator is failing its pedagogical role, or the poet is producing work the user doesn't value.

---

## Implementation Timeline (Integrated)

| Week | Phase | Deliverable | Cost | Key Requirement |
|------|-------|-------------|------|-----------------|
| 1 | Corpus collection + Persona Design | Tiered poetry DB + Pedagogy Design Doc | $0 | R1.1-R1.10 |
| 2 | Claude API: Cliché database + T4 autopsy data | Topic-indexed cliché DB + 300-500 autopsy examples | ~$5 API | R4.1-R4.4 |
| 3 | Claude API: Educator training data (T1, T2, T3, T5, T6) | 1,850-3,250 examples, quality-gated | ~$10-15 API | S2.1-S2.5 |
| 4 | Claude API: Poet training pairs (reverse briefs) | 500-800 brief→poem pairs | ~$5 API | S4.1-S4.2 (v2 numbering) |
| 5 | Modal: Educator training + export | educator-14b-Q4_K_M.gguf | ~$4 Modal | S3.2, S3.4 |
| 6 | Modal: Poet training + export | poet-14b-Q4_K_M.gguf | ~$4 Modal | S3.3, S3.4 |
| 7 | Evaluation: Voice consistency + quantization QA | Pass/fail on S6.1-S6.3 tests | $0 (human time) | R6.5, S6.1-S6.3 |
| 8 | Mac Mini integration + pipeline | Working end-to-end system | $0 | S4.1-S4.3, R6.6-R6.7 |
| 9 | User interface + feedback loop | Deployable chatbot | $0 | S7, R5.1-R5.3 |
| 10 | Iteration: retrain on feedback | Improved models v1.1 | ~$8 Modal | Continuous |

**Total estimated cost:** ~$25-35 API + ~$16-20 Modal = **~$41-55 for the complete v1 system**

---

## Failure Modes & Mitigations (Integrated)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Modal timeout during training | Lost compute, no model | Epoch-level checkpointing; resume from last saved |
| OOM on cloud GPU | Training crash | Reduce batch → 2, increase grad accum → 8, reduce seq_len |
| Q4_K_M kills educator voice | System fails R6.5 | Fallback to Q5_K_M; asymmetric quant (educator Q5, poet Q4) |
| Two 14B models don't fit on Mac | Can't run multi-agent (R6.6) | Option C swap strategy; or asymmetric Q4/Q3 |
| Educator voice inconsistent after fine-tune | Fails R1.1 | Increase LoRA rank to 96; increase training data; review persona doc |
| Poet produces clichéd output despite brief | Fails R4.1 | More cliché autopsy data (T4); explicit negative training examples |
| Claude API generates LLM-isms in training data | Poisons fine-tune | Stricter quality gates (S2.2); manual spot-checking per batch |
| Quantization degrades >5% perplexity | Model quality loss | Q5_K_M fallback; consider GPTQ/AWQ alternatives |
| Download interruption | Can't deploy | Modal volume persists; retry with resume |
| 14B poet quality insufficient | Bad poems | Graduate to 32B poet on A100 ($14/run); use swap strategy for inference |

---

## Cost Model Summary

| Component | One-Time Cost | Per-Iteration Cost |
|-----------|--------------|-------------------|
| Claude API (data generation) | $25-35 | $5-10 (incremental data) |
| Modal: Educator 14B training | $3.50 | $3.50 |
| Modal: Poet 14B training | $3.50 | $3.50 |
| Modal: Poet 32B training (if needed) | $14.00 | $14.00 |
| Egress (GGUF downloads) | $1.60 | $1.60 |
| Mac Mini inference | $0 | $0 |
| **Total (14B×2)** | **~$34-44** | **~$14-19** |
| **Total (14B + 32B)** | **~$44-55** | **~$24-29** |

---

## Migration Path from v2 (MLX) to v3 (Modal + llama.cpp)

| Aspect | v2 (MLX + LoRA) | v3 (Modal + llama.cpp) |
|--------|-----------------|----------------------|
| Model size | 3B-8B | 14B-32B |
| Training location | Local Mac Mini | Modal serverless GPU |
| Training speed | ~10 min/epoch | ~2 min/epoch |
| LoRA rank | 16-32 | 32-64 |
| Target modules | Attention only | Attention + MLP |
| Quantization | MLX native 4-bit | GGUF Q4_K_M post-hoc |
| Inference engine | MLX Python | llama.cpp Metal |
| Voice capacity | Limited by model size | Significantly improved |
| Portability | Apple Silicon only | Any GGUF runtime |
| Cost per iteration | $0 (local compute) | ~$14-19 |
| Quality ceiling | Constrained by 8B | 14B-32B unlocked |

**Key tradeoff:** You trade ~$15/iteration for dramatically better model quality, especially for the educator voice and the poet's generative ability. Given that the training data (created via Claude API) is the same regardless of where you train, the cloud compute cost is a small fraction of the total investment.

---

## Appendix A: Example Educator Persona (Starter Template)

```
EDUCATOR PERSONA: "Maren"
=========================

Aesthetic Commitments:
Maren believes the best poems are acts of radical attention — 
they teach us to see what we've been looking past. She loves 
poets who trust the image to do the emotional work: Elizabeth 
Bishop, Larry Levis, Brigit Pegeen Kelly, Ocean Vuong. She's 
suspicious of poems that announce their own importance. She 
thinks most poems are 30% too long and that the willingness 
to cut is what separates good poets from talented amateurs. 
She believes revision IS the art — first drafts are just 
material.

Voice:
Direct, medium-length sentences. Uses "I" frequently — "I want 
to stay in that kitchen with you" or "I lose you after line 6." 
Curses occasionally for emphasis. Uses questions as primary 
pedagogical tool — "What if you cut the first three lines?" 
Gets visibly excited about strong images — "THAT. That's where 
your poem lives." Warm but never falsely encouraging.

Signature Concepts:
1. "The specific is the universal"
2. "Trust the image"
3. "Earn your abstractions"
4. "The poem knows more than the poet"
5. "Cut toward the bone"

What Maren Would Never Say:
- "Great job!" / "This is a solid effort"
- "Consider perhaps exploring..."
- "The imagery is nice" 
- "This poem resonates"
- Any numbered rating system
```

---

## Appendix B: Cliché Taxonomy for Training

| Type | Description | Example | Why It Fails |
|------|-------------|---------|--------------|
| Dead Metaphor | Figure so common it's invisible | "broken heart" | No one pictures a broken organ |
| Greeting Card Sentiment | Confirms rather than challenges | "Love conquers all" | Comforts by lying |
| Atmospheric Default | Mood-setting that delays the poem | "Wind whispered through ancient trees" | Could open any of 10,000 poems |
| Prestige Diction | Words for "poetic" sound, not meaning | "azure," "ethereal," "gossamer" | Signals "I am writing a Poem" |
| False Closure | Unearned resolution or transcendence | Grief poem ending with dawn/spring | Joins the cultural lie about "getting over it" |
| Explanatory Last Line | Tells the reader what the poem meant | "And that's when I knew love was real" | Destroys the reader's role |
| Apostrophe to Abstract | Addressing a concept as a person | "Oh, Time, you thief..." | Rhetorical gesture substitutes for thought |
| Body-as-Weather | Emotions as weather/nature | "storm inside me" | So automatic it's invisible |
| Wisdom Turn | Pivot to a lesson learned | "I realized we are all..." | Poems are not TED talks |
| Insta-Poetry Lineation | Breaks for dramatic pause only | "And then / I knew / everything / had changed" | Simulates profundity through fragmentation |

---

## Appendix C: Project Directory Structure

```
poetry-chatbot/
├── data/
│   ├── raw/
│   │   ├── tier1_exemplary/          # Curated good poetry
│   │   ├── tier2_competent/          # General published poetry
│   │   └── tier3_amateur/            # Bad poetry for contrast
│   ├── cliche_db/                    # Topic-indexed cliché database
│   │   ├── grief.json
│   │   ├── love.json
│   │   ├── nature.json
│   │   └── ...
│   ├── annotated/                    # Claude's raw analysis output
│   ├── educator_training/            # Quality-gated training data
│   │   ├── train.jsonl
│   │   └── valid.jsonl
│   └── poet_training/
│       ├── train.jsonl
│       └── valid.jsonl
├── persona/
│   ├── pedagogy_design_doc.md        # THE persona document (S1.1)
│   ├── persona_condensed.txt         # 200-word version for training
│   └── anti_llm_isms.txt            # Banned phrase list (S2.3)
├── scripts/
│   ├── data_generation/
│   │   ├── generate_critiques.py     # T1: Claude API batch critiques
│   │   ├── generate_briefs.py        # T2: Generation briefs
│   │   ├── generate_comparisons.py   # T3: Comparative workshop
│   │   ├── generate_autopsies.py     # T4: Cliché autopsies
│   │   ├── generate_dialogues.py     # T5: Revision dialogues
│   │   ├── generate_lessons.py       # T6: Craft lessons
│   │   ├── generate_poet_pairs.py    # Reverse briefs for poet
│   │   └── quality_gate.py           # S2.2 quality filter
│   ├── modal/
│   │   ├── train_educator.py         # Modal function: educator QLoRA
│   │   ├── train_poet.py             # Modal function: poet QLoRA
│   │   ├── export_gguf.py            # Modal function: merge + quantize
│   │   └── modal_app.py              # Modal app orchestration
│   ├── eval/
│   │   ├── voice_consistency.py      # S6.1 tests
│   │   ├── quant_preservation.py     # S6.2 tests
│   │   └── generation_quality.py     # S6.3 tests
│   └── inference/
│       ├── pipeline.py               # PoetryPipeline class
│       ├── swapping_pipeline.py       # Option C swap strategy
│       └── feedback.py               # Feedback collection
├── models/                           # Downloaded GGUF files
│   ├── llama3.1-14b-educator-Q4_K_M.gguf
│   └── llama3.1-14b-poet-Q4_K_M.gguf
├── feedback/
│   └── feedback.jsonl
├── config/
│   ├── educator_training.yaml
│   ├── poet_training.yaml
│   ├── export_pipeline.yaml
│   └── inference_config.yaml
└── README.md
```
