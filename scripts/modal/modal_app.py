#!/usr/bin/env python3
"""Modal app orchestration — train, export. Run with: modal run scripts/modal/modal_app.py"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode: discover models, replace or train new")
    parser.add_argument("--educator-only", action="store_true")
    parser.add_argument("--poet-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Skip export")
    parser.add_argument("--train-rhyme", action="store_true", help="Modal rhyme fine-tune (80%% strong_rhyme_poems + 20%% general, anti-collapse)")
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--include-modal", action="store_true", help="Discover checkpoints on Modal (with --interactive)")
    parser.add_argument("--skip-generation-prompt", action="store_true", help="Skip run generation prompt (with --interactive)")
    args = parser.parse_args()

    if args.interactive:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "training" / "train_interactive.py")]
            + (["--educator-only"] if args.educator_only else [])
            + (["--poet-only"] if args.poet_only else [])
            + (["--train-rhyme"] if args.train_rhyme else [])
            + (["--train-only"] if args.train_only else [])
            + (["--include-modal"] if args.include_modal else [])
            + (["--skip-generation-prompt"] if args.skip_generation_prompt else [])
            + (["--num-epochs", str(args.num_epochs)] if args.num_epochs else []),
            cwd=str(ROOT),
            check=True,
        )
        return

    if args.train_rhyme:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "data_generation" / "prepare_rhyme_training_data.py")], cwd=str(ROOT), check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "modal" / "upload_data.py")], cwd=str(ROOT), check=True)
        extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
        subprocess.run(
            [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / "train_rhyme_poet.py")] + extra,
            cwd=str(ROOT),
            check=True,
        )
        print("Rhyme training done. Download: modal volume get poetry-checkpoints poet_rhyme/final ./models/")
        return

    run_edu = not args.poet_only
    run_poet = not args.educator_only

    def _prepare_export_for(task: str):
        """Set export config to use default paths for this task."""
        import yaml
        export_cfg = ROOT / "config" / "export_pipeline.yaml"
        train_cfg = ROOT / "config" / ("educator_training.yaml" if task == "educator" else "poet_training.yaml")
        with open(export_cfg) as f:
            cfg = yaml.safe_load(f)
        if train_cfg.exists():
            with open(train_cfg) as f:
                train = yaml.safe_load(f)
            cfg["base_model"] = train.get("base_model", cfg.get("base_model"))
        cfg.pop("checkpoint_path", None)
        cfg.pop("out_name", None)
        with open(export_cfg, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def run_script(script: str, extra: list[str] = None):
        cmd = [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / script)]
        if extra:
            cmd.extend(extra)
        subprocess.run(cmd, check=True)

    if run_edu:
        extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
        run_script("train_educator.py", extra)
        if not args.train_only:
            _prepare_export_for("educator")
            subprocess.run([sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / "export_gguf.py") + "::export_educator"], check=True)

    if run_poet:
        extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
        run_script("train_poet.py", extra)
        if not args.train_only:
            _prepare_export_for("poet")
            subprocess.run(
                [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / "export_gguf.py") + "::export_poet"],
                check=True,
            )

    print("Done. Download GGUF: modal volume get --force poetry-gguf <filename> ./models/")


if __name__ == "__main__":
    main()
