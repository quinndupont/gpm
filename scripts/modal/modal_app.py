#!/usr/bin/env python3
"""Modal app orchestration — train, export. Run with: modal run scripts/modal/modal_app.py"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["modal", "sagemaker"], default="modal",
        help="Fine-tuning backend",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="HuggingFace base model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interactive: discover models, replace or train new",
    )
    parser.add_argument("--educator-only", action="store_true")
    parser.add_argument("--poet-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Skip export")
    parser.add_argument(
        "--reinforce", action="store_true",
        help="Run REINFORCE (Stage 2) after SFT for poet model",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--include-modal", action="store_true",
        help="Discover checkpoints on Modal (with --interactive)",
    )
    parser.add_argument(
        "--skip-generation-prompt", action="store_true",
        help="Skip run generation prompt (with --interactive)",
    )
    args = parser.parse_args()

    if args.interactive:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "training" / "train_interactive.py")]
            + (["--educator-only"] if args.educator_only else [])
            + (["--poet-only"] if args.poet_only else [])
            + (["--reinforce"] if args.reinforce else [])
            + (["--train-only"] if args.train_only else [])
            + (["--include-modal"] if args.include_modal else [])
            + (["--skip-generation-prompt"] if args.skip_generation_prompt else [])
            + (["--num-epochs", str(args.num_epochs)] if args.num_epochs else []),
            cwd=str(ROOT),
            check=True,
        )
        return

    run_edu = not args.poet_only
    run_poet = not args.educator_only

    def _prepare_export_for(task: str):
        """Set export config to use default paths for this task."""
        import yaml
        export_cfg = ROOT / "config" / "export_pipeline.yaml"
        train_cfg = ROOT / "config" / (
            "educator_training.yaml" if task == "educator" else "poet_training.yaml"
        )
        with open(export_cfg) as f:
            cfg = yaml.safe_load(f)
        if train_cfg.exists():
            with open(train_cfg) as f:
                train = yaml.safe_load(f)
            cfg["base_model"] = train.get("base_model", cfg.get("base_model"))
        cfg.pop("checkpoint_path", None)
        cfg.pop("out_name", None)
        with open(export_cfg, "w") as f:
            yaml.dump(
                cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True,
            )

    if args.backend == "sagemaker":
        upload_s3 = str(ROOT / "scripts" / "sagemaker" / "upload_to_s3.py")
        subprocess.run([sys.executable, upload_s3], cwd=str(ROOT), check=True)
        extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
        if args.base_model:
            extra.extend(["--base-model", args.base_model])
        if run_edu:
            train_sm = str(ROOT / "scripts" / "sagemaker" / "train_sagemaker.py")
            subprocess.run(
                [sys.executable, train_sm, "--task", "educator"] + extra,
                cwd=str(ROOT), check=True,
            )
            if not args.train_only:
                _prepare_export_for("educator")
                exp = str(ROOT / "scripts" / "sagemaker" / "export_gguf_sagemaker.py")
                subprocess.run(
                    [sys.executable, exp, "--task", "educator"],
                    cwd=str(ROOT), check=True,
                )
        if run_poet:
            train_sm = str(ROOT / "scripts" / "sagemaker" / "train_sagemaker.py")
            subprocess.run(
                [sys.executable, train_sm, "--task", "poet"] + extra,
                cwd=str(ROOT), check=True,
            )
            if args.reinforce:
                subprocess.run(
                    [sys.executable, train_sm, "--task", "reinforce"] + extra,
                    cwd=str(ROOT), check=True,
                )
            if not args.train_only:
                _prepare_export_for("poet")
                exp = str(ROOT / "scripts" / "sagemaker" / "export_gguf_sagemaker.py")
                task = "poet_reinforce" if args.reinforce else "poet"
                subprocess.run(
                    [sys.executable, exp, "--task", task],
                    cwd=str(ROOT), check=True,
                )
        print("Done. Download GGUF: python scripts/sagemaker/download_models.py")
    else:
        def run_script(script: str, extra: list[str] = None):
            script_path = str(ROOT / "scripts" / "modal" / script)
            cmd = [sys.executable, "-m", "modal", "run", script_path]
            if extra:
                cmd.extend(extra)
            subprocess.run(cmd, check=True)

        if run_edu:
            extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
            if args.base_model:
                extra.extend(["--base-model", args.base_model])
            run_script("train_educator.py", extra)
            if not args.train_only:
                _prepare_export_for("educator")
                exp_gguf = str(ROOT / "scripts" / "modal" / "export_gguf.py") + "::export_educator"
                subprocess.run([sys.executable, "-m", "modal", "run", exp_gguf], check=True)

        if run_poet:
            extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
            if args.base_model:
                extra.extend(["--base-model", args.base_model])
            run_script("train_poet.py", extra)
            if args.reinforce:
                run_script("train_poet_reinforce.py", extra)
            if not args.train_only:
                _prepare_export_for("poet")
                if args.reinforce:
                    exp = str(ROOT / "scripts" / "modal" / "export_gguf.py") + "::export_poet_reinforce"
                else:
                    exp = str(ROOT / "scripts" / "modal" / "export_gguf.py") + "::export_poet"
                subprocess.run([sys.executable, "-m", "modal", "run", exp], check=True)

        print(
            "Done. Download GGUF: modal volume get --force poetry-gguf <filename> ./models/"
        )


if __name__ == "__main__":
    main()
