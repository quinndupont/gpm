#!/usr/bin/env python3
"""Modal: List checkpoints on volume and return base model metadata. For model discovery."""
import json
from pathlib import Path

import modal

CHECKPOINT_VOLUME = "poetry-checkpoints"
checkpoint_vol = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)

app = modal.App("poetry-list-checkpoints")


@app.function(volumes={"/vol/checkpoints": checkpoint_vol})
def list_checkpoints() -> list[dict]:
    """List all checkpoints and read base_model from adapter_config.json."""
    checkpoint_vol.reload()
    base = Path("/vol/checkpoints")
    results = []
    if not base.exists():
        return results
    for subdir in base.iterdir():
        if not subdir.is_dir():
            continue
        config_path = subdir / "final" / "adapter_config.json"
        if not config_path.exists():
            continue
        try:
            cfg = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        hf_id = cfg.get("base_model_name_or_path") or cfg.get("model")
        if not hf_id:
            continue
        name = subdir.name
        if name.startswith("educator"):
            task = "educator"
        elif name.startswith("poet_rhyme"):
            task = "poet_rhyme"
        elif name.startswith("poet"):
            task = "poet"
        else:
            task = name
        results.append({
            "path": f"modal:{name}/final",
            "task": task,
            "base_model_hf_id": hf_id,
            "quant": None,
            "source": "modal_checkpoint",
        })
    return results


if __name__ == "__main__":
    import sys
    with app.run():
        data = list_checkpoints.remote()
        print(json.dumps(data))
