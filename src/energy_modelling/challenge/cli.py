"""CLI for building public and private challenge datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from energy_modelling.challenge.data import write_challenge_data

_DEFAULT_DATASET = Path("kaggle_upload/dataset_de_lu.csv")
_DEFAULT_OUTPUT_DIR = Path("data/challenge")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build daily hackathon challenge datasets.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help="Path to the hourly DE-LU dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where the daily challenge CSV files will be written.",
    )
    parser.add_argument(
        "--include-hidden-test",
        action="store_true",
        help="Also write the private full hidden-test file for organizers.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    written = write_challenge_data(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        include_hidden_test=args.include_hidden_test,
    )

    for name, path in written.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
