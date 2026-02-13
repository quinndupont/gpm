#!/usr/bin/env python3
"""S6.3 Generation quality — cliché density, specificity, etc."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLICHE_DB = ROOT / "data" / "cliche_db"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+", help="JSONL with generated poems")
    parser.add_argument("--cliche-db", type=Path, default=CLICHE_DB)
    args = parser.parse_args()
    # TODO: Load cliché DB, run poems through, count violations
    # Pass: <1 phrase/image cliché per poem
    raise NotImplementedError("Requires populated cliché database")


if __name__ == "__main__":
    main()
