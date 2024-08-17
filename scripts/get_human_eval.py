import argparse
from pathlib import Path

import pandas as pd


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=Path, help="Path to the reference containing the 'gold' preferences.")
    parser.add_argument("--annotation_path", type=Path, help="Path to the annotations file.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    reference = pd.read_csv(args.reference_path)
    annotations = pd.read_csv(args.annotation_path)

    breakpoint()


if __name__ == "__main__":
    main()
