import argparse
from pathlib import Path
import logging

import pandas as pd
from pycm import ConfusionMatrix

logging.basicConfig(level=logging.INFO)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=Path, help="Path to the reference containing the 'gold' preferences.")
    parser.add_argument("--annotation_path", type=Path, help="Path to the annotations file.")
    parser.add_argument("--dropna", default=False, action="store_true", help="Drop instances with no annotations")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    reference = pd.read_csv(args.reference_path)
    annotations = pd.read_csv(args.annotation_path)
    # All gold 'preference' in answer key is in completion_a
    reference["gold_preference"] = "A"
    annotations = annotations[["id", "human_preference", "notes"]]

    # Combine in single dataframe and apply random swaps
    # We swap the gold preferences because if they're all the same value,
    # it affects the random-corrected chance in the IAA measures
    df = pd.merge(reference, annotations, on="id")
    df["gold_preference"] = df.apply(lambda row: "B" if row["swapped"] == 1 else "A", axis=1)
    if args.dropna:
        df = df.dropna(subset=["human_preference"])
        logging.info(f"Dropped instances with no annotations. No. of instances: {len(df)}")

    cm = ConfusionMatrix(
        actual_vector=df["gold_preference"].to_list(),
        predict_vector=df["human_preference"].to_list(),
    )
    print(
        f"*** Overall metrics ***\n",
        f"Accuracy: {cm.Overall_ACC}\n",
        f"F1-score: {cm.F1_Macro}\n",
        f"Per-class accuracy: {cm.ACC}\n",
        f"Cohen's Kappa: {cm.Kappa}\n",
        f"Krippendorff Alpha: {cm.Alpha}\n",
        f"Gwet's AC1: {cm.AC1}\n",
    )


if __name__ == "__main__":
    main()
