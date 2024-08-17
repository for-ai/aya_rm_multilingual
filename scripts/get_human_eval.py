import argparse
from pathlib import Path
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO)


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
    # All gold 'preference' in answer key is in completion_a
    reference["gold_preference"] = "A"
    annotations = annotations[["id", "human_preference", "notes"]]

    # Combine in single dataframe and bring back random swaps
    df = pd.merge(reference, annotations, on="id")
    df["human_preference"] = df.apply(
        lambda row: (
            "B"
            if row["human_preference"] == "A" and row["swapped"] == 1
            else "A" if row["human_preference"] == "B" and row["swapped"] == 1 else row["human_preference"]
        ),
        axis=1,
    )


if __name__ == "__main__":
    main()
