# Models

Weights, adapters, and prompts for the GPM poetry pipeline. This directory is the output target for trained models and the source for all prompt templates.

## Directory Layout

```
models/
├── README.md
├── prompts/           # Prompt templates (JSON) and loader
│   ├── loader.py      # get_persona(), get_prompt(), render_prompt()
│   ├── personas/      # System prompts (educator, poet)
│   ├── tuning/       # Prompts for data generation → data/
│   └── inference/    # Prompts for runtime pipeline
├── adapters/         # LoRA checkpoints (QLoRA)
├── *.gguf            # Quantized GGUF models for inference
└── Llama-*/          # Downloaded base models
```

## Prompts

Prompts are stored as JSON in `prompts/` and loaded via `models.prompts.loader`.

### Personas (`prompts/personas/`)

System prompts that define model behavior:  

| ID | Description |
|----|-------------|
| `educator_neutral` | SFT-style workshop voice: critique and briefs for a revision/rhyme-trained poet |
| `educator_condensed` | Short persona for training |
| `poet` | SRPO/rhyme-trained: generate and revise under form constraints; poem-only output |

### Tuning (`prompts/tuning/`)

Used by data-generation scripts to produce training data in `data/`:

| Prompt | Script | Output |
|--------|--------|--------|
| `critique` | generate_critiques_seed | data/annotated/critiques_seed.jsonl |
| `brief` | generate_briefs | data/educator_training/briefs.jsonl |
| `comparison` | generate_comparisons | data/annotated/comparisons.jsonl |
| `revision_brief` | generate_revision_briefs | data/educator_training/revision_briefs_seed.jsonl |
| `lesson` | generate_lessons | data/educator_training/lessons.jsonl |
| `autopsy` | generate_autopsies | data/annotated/autopsies.jsonl |
| `dialogue` | generate_dialogues | data/educator_training/dialogues.jsonl |
| `approval` | generate_approval_examples | data/educator_training/approval_examples.jsonl |
| `poet_generation` | generate_poet_pairs | data/poet_training/pairs.jsonl |
| `rhyme_pairs` | generate_rhyme_pairs | data/poet_training/rhyme_pairs.jsonl, data/educator_training/rhyme_critiques.jsonl |

### Inference (`prompts/inference/`)

Used by the runtime pipeline (`scripts/inference/pipeline.py`):

| Prompt | Purpose |
|--------|---------|
| `brief` | User request → generation brief |
| `critique` | Brief + selective revision memory + implementation diagnostic → workshop critique |
| `critique_diagnostic` | Prior draft vs current vs last critique — implementation audit (internal step) |
| `revision_brief` | Draft + critique → revised brief |
| `poet_generation` | Brief → poem |
| `poet_revision_instructions` | Educator → compact revision instructions |
| `poet_revision` | Revision instructions → poet revises |
| `final_note` | Final poem → wrap-up note |

### Relation to `data/`

```
prompts/tuning/*  →  scripts/data_generation/*.py  →  data/educator_training/, data/poet_training/, data/annotated/
                                                              ↓
                                              prepare_training_data.py, prepare_rhyme_training_data.py
                                                              ↓
                                              train.jsonl, valid.jsonl
                                                              ↓
                                              scripts/modal/train_*.py  →  models/adapters/, models/*.gguf
```

### Adding Prompt Variants

Each prompt JSON has a `templates` object. Use multiple keys for A/B experiments:

```json
{
  "id": "critique",
  "templates": {
    "default": "...",
    "terse": "...",
    "detailed": "..."
  }
}
```

Load with `get_prompt("tuning", "critique", template="terse")` or `render_prompt("tuning", "critique", template="terse", poem_text=...)`.

## Models Produced

| Model | Source | Output |
|-------|--------|--------|
| Educator | Qwen2.5-7B-Instruct + educator training data | qwen2.5-7b-educator-Q4_K_M.gguf |
| Poet | Qwen2.5-7B-Instruct + poet pairs | qwen2.5-7b-poet-Q4_K_M.gguf |
| Rhyme poet | Qwen2.5-7B-Instruct + rhyme pairs | (rhyme adapter or merged GGUF) |

Training: QLoRA on Modal (A10G). Export: merge LoRA → GGUF quantization via `scripts/modal/export_gguf.py`. See `config/model_registry.yaml` for base model options.
