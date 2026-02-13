#!/usr/bin/env python3
"""Modal app orchestration — train, export. Run with: modal run scripts/modal/modal_app.py"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--educator-only", action="store_true")
    parser.add_argument("--poet-only", action="store_true")
    parser.add_argument("--train-only", action="store_true", help="Skip export")
    parser.add_argument("--num-epochs", type=int, default=None)
    args = parser.parse_args()

    def run_script(script: str, extra: list[str] = None):
        cmd = [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / script)]
        if extra:
            cmd.extend(extra)
        subprocess.run(cmd, check=True)

    run_edu = not args.poet_only
    run_poet = not args.educator_only

    if run_edu:
        extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
        run_script("train_educator.py", extra)
        if not args.train_only:
            run_script("export_gguf.py")  # defaults to educator

    if run_poet:
        extra = ["--num-epochs-override", str(args.num_epochs)] if args.num_epochs else []
        run_script("train_poet.py", extra)
        if not args.train_only:
            subprocess.run(
                [sys.executable, "-m", "modal", "run", str(ROOT / "scripts" / "modal" / "export_gguf.py"), "poet"],
                check=True,
            )

    print("Done. Download GGUF: modal volume get poetry-gguf <filename> ./models/")


if __name__ == "__main__":
    main()
