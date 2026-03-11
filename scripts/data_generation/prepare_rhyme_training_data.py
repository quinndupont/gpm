#!/usr/bin/env python3
"""DEPRECATED: Rhyme data is now part of the unified poet pipeline.

Run prepare_training_data.py instead — it includes strong-rhyme poems
and rhyme pairs in the poet training data automatically.
"""
import sys


def main():
    print(
        "DEPRECATED: prepare_rhyme_training_data.py is no longer needed.\n"
        "Rhyme data (strong-rhyme poems + rhyme pairs) is now included in\n"
        "prepare_training_data.py. Run that script instead.",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
