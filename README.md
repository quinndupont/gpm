# Good Poetry Model (GPM)

Locally-trained poetry analysis model: Ollama + Llama 3.2 for synthetic data, MLX for fine-tuning. Optimized for Mac Mini M4 (24GB RAM).

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
# Full pipeline (~16-20 hrs)
python orchestrator.py --phase full

# Or run phases individually
python orchestrator.py --phase data      # ~30 min
python orchestrator.py --phase annotate # ~8-12 hrs (requires ollama serve)
python orchestrator.py --phase validate  # ~15 min
python orchestrator.py --phase train     # ~3-4 hrs

# Resume after interruption
python orchestrator.py --resume
```

## Test

```bash
python test_gpm.py
```

## Structure

- `agents/` — Data, Annotation, Validation, Training agents
- `config/gpm_config.yaml` — Central config
- `data/` — raw → processed → annotated → training
- `models/adapters/` — LoRA output
