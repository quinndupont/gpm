#!/usr/bin/env python3
"""Interactive CLI for training: discover existing models, replace or train new."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
REGISTRY_PATH = CONFIG_DIR / "model_registry.yaml"
EDUCATOR_CONFIG = CONFIG_DIR / "educator_training.yaml"
POET_CONFIG = CONFIG_DIR / "poet_training.yaml"
RHYME_CONFIG = CONFIG_DIR / "rhyme_training.yaml"
EXPORT_CONFIG = CONFIG_DIR / "export_pipeline.yaml"

sys.path.insert(0, str(ROOT))
from scripts.training.model_discovery import (
    discover_all,
    discover_by_task,
    hf_to_short_name,
)
from scripts.training.data_generation_status import (
    format_status,
    get_data_generation_status,
)


def load_registry() -> list[dict]:
    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("models", [])


def update_training_config(path: Path, base_model: str, save_path: str | None = None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["base_model"] = base_model
    if save_path:
        if "checkpointing" not in cfg:
            cfg["checkpointing"] = {}
        cfg["checkpointing"]["save_path"] = save_path
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def update_export_config(
    base_model: str,
    primary_quant: str,
    checkpoint_path: str | None = None,
    out_name: str | None = None,
):
    with open(EXPORT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["base_model"] = base_model
    if "quantization" not in cfg:
        cfg["quantization"] = {}
    cfg["quantization"]["primary_quant"] = primary_quant
    if checkpoint_path:
        cfg["checkpoint_path"] = checkpoint_path
    if out_name:
        cfg["out_name"] = out_name
    with open(EXPORT_CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def prompt_choice(prompt: str, options: str) -> str:
    while True:
        val = input(f"{prompt} [{options}]: ").strip().lower()
        if val in options.lower() or (len(options) == 1 and val == ""):
            return val or options[0].lower()
        print(f"  Choose one of: {options}")


def select_model_from_registry(registry: list[dict], task: str) -> dict:
    fit_key = "educator_fit" if task == "educator" else "poet_fit"
    note_key = "educator_note" if task == "educator" else "poet_note"
    print(f"\nSelect base model for {task}:")
    print("  Educator: instruct models best (critique, briefs). Poet: creative models often less formulaic.")
    for i, m in enumerate(registry, 1):
        short = m["short_name"]
        param = m["param_b"]
        quant = m["recommended_inference_quant"]
        fit = m.get(fit_key, "")
        note = m.get(note_key, "")
        badge = " ★" if fit == "recommended" else (" ✓" if fit == "good" else "")
        advice = f" — {note}" if note else ""
        print(f"  {i}. {short} ({param}B) -> {quant}{badge}{advice}")
    while True:
        raw = input("Choice [1]: ").strip() or "1"
        try:
            idx = int(raw)
            if 1 <= idx <= len(registry):
                return registry[idx - 1]
        except ValueError:
            pass
        print("  Invalid choice.")


def run_training(
    task: str,
    base_model: str,
    short_name: str,
    quant: str,
    save_path: str,
    replace: bool,
    num_epochs: int | None,
    train_only: bool,
):
    if task == "poet_rhyme":
        update_training_config(RHYME_CONFIG, base_model, save_path)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "data_generation" / "prepare_rhyme_training_data.py")],
            cwd=str(ROOT),
            check=True,
        )
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "modal" / "upload_data.py")],
            cwd=str(ROOT),
            check=True,
        )
        extra = ["--num-epochs-override", str(num_epochs)] if num_epochs else []
        subprocess.run(
            [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / "train_rhyme_poet.py")] + extra,
            cwd=str(ROOT),
            check=True,
        )
        print("Rhyme training done. Download: modal volume get poetry-checkpoints poet_rhyme/final ./models/")
        return

    config = EDUCATOR_CONFIG if task == "educator" else POET_CONFIG
    update_training_config(config, base_model, save_path)

    script = "train_educator.py" if task == "educator" else "train_poet.py"
    extra = ["--num-epochs-override", str(num_epochs)] if num_epochs else []
    subprocess.run(
        [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / script)] + extra,
        cwd=str(ROOT),
        check=True,
    )

    if train_only:
        return

    out_name = f"{short_name}-{task}" if replace else f"{short_name}-{task}-v2"
    update_export_config(base_model, quant, save_path.rstrip("/") + "/final", out_name)

    export_fn = "export_educator" if task == "educator" else "export_poet"
    subprocess.run(
        [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / "export_gguf.py") + f"::{export_fn}"],
        cwd=str(ROOT),
        check=True,
    )
    gguf_name = f"{short_name}-{task}-{quant}.gguf" if replace else f"{short_name}-{task}-v2-{quant}.gguf"
    print(f"Done. Download: modal volume get --force poetry-gguf {gguf_name} ./models/")


def run_data_generation():
    """Run full prompt generation (Opus + interim educator + local + poet pairs)."""
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_full_workflow.sh"), "--generation-only"],
        cwd=str(ROOT),
        check=True,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive training with model discovery")
    parser.add_argument("--educator-only", action="store_true")
    parser.add_argument("--poet-only", action="store_true")
    parser.add_argument("--train-rhyme", action="store_true", help="Train rhyme-focused poet")
    parser.add_argument("--train-only", action="store_true", help="Skip export")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--include-modal", action="store_true", help="Discover Modal checkpoints")
    parser.add_argument("--skip-generation-prompt", action="store_true", help="Skip the run generation prompt (assume skip)")
    args = parser.parse_args()

    # Data generation: enumerate and prompt (default: skip)
    if not args.skip_generation_prompt:
        phases = get_data_generation_status()
        print("\n" + format_status(phases))
        run_gen = prompt_choice("\nRun prompt generation (Opus + interim educator + local)?", "ny")
        if run_gen == "y":
            run_data_generation()
        else:
            print("Skipping generation. Using existing data.")

    # Verify required data exists for selected tasks
    educator_data = ROOT / "data" / "educator_training" / "train.jsonl"
    poet_data = ROOT / "data" / "poet_training" / "train.jsonl"
    rhyme_data = ROOT / "data" / "rhyme_training" / "train.jsonl"
    needs_edu = not args.poet_only and not args.train_rhyme
    needs_poet = not args.educator_only and not args.train_rhyme
    needs_rhyme = args.train_rhyme
    if needs_edu and not educator_data.exists():
        print(f"ERROR: Educator data not found. Run generation or create {educator_data}")
        return
    if needs_poet and not poet_data.exists():
        print(f"ERROR: Poet data not found. Run generation or create {poet_data}")
        return
    if needs_rhyme and not rhyme_data.exists():
        print(f"ERROR: Rhyme data not found. Run prepare_rhyme_training_data or create {rhyme_data}")
        return

    registry = load_registry()
    discovered = discover_all(include_modal=args.include_modal)

    tasks_to_run: list[str] = []
    if args.train_rhyme:
        tasks_to_run.append("poet_rhyme")
    else:
        if not args.poet_only:
            tasks_to_run.append("educator")
        if not args.educator_only:
            tasks_to_run.append("poet")

    if not tasks_to_run:
        print("No tasks selected. Use --educator-only, --poet-only, or --train-rhyme.")
        return

    for task in tasks_to_run:
        existing = discover_by_task(task, include_modal=args.include_modal)
        if existing:
            print(f"\n--- {task.upper()} ---")
            for m in existing:
                print(f"  Found: {m.path}")
                print(f"    Base: {m.base_model_hf_id}  Quant: {m.quant or 'N/A'}")
            choice = prompt_choice("Replace existing or train new (keep both)?", "rn")
            replace = choice == "r"
            if replace:
                base_model = existing[0].base_model_hf_id
                short = hf_to_short_name(base_model)
                entry = next((e for e in registry if e["hf_id"] == base_model), None)
                quant = entry["recommended_inference_quant"] if entry else "Q4_K_M"
                save_path = "/vol/checkpoints/educator/" if task == "educator" else "/vol/checkpoints/poet/"
                if task == "poet_rhyme":
                    save_path = "/vol/checkpoints/poet_rhyme/"
            else:
                entry = select_model_from_registry(registry, task)
                base_model = entry["hf_id"]
                short = entry["short_name"]
                quant = entry["recommended_inference_quant"]
                save_path = f"/vol/checkpoints/{task}_{short}/"
        else:
            entry = select_model_from_registry(registry, task)
            base_model = entry["hf_id"]
            short = entry["short_name"]
            quant = entry["recommended_inference_quant"]
            save_path = f"/vol/checkpoints/{task}/" if task in ("educator", "poet") else f"/vol/checkpoints/{task}_{short}/"
            replace = True

        run_training(
            task, base_model, short, quant, save_path, replace, args.num_epochs, train_only=args.train_only
        )


if __name__ == "__main__":
    main()
