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
    df = pd.merge(reference, annotations, on="id")
    df["llm_preference"] = df.apply(lambda row: "A" if row["scores"] == 1 else "B", axis=1)
    df["human_preference"] = df.apply(
        lambda row: (
            # fmt: off
            "B" if row["human_preference"] == "A" and row["swapped"] == 1
            else "A" if row["human_preference"] == "B" and row["swapped"] == 1 else row["human_preference"]
            # fmt: on
        ),
        axis=1,
    )

    if args.dropna:
        df = df.dropna(subset=["human_preference"])
        logging.info(f"Dropped instances with no human annotations. No. of instances left: {len(df)}")

    # Compute accuracy
    acc_human_vs_gold_all = (df["human_preference"] == df["gold_preference"]).sum() / len(df)
    logging.info(f"Accuracy (human vs. gold): {acc_human_vs_gold_all * 100:.2f}")
    subset_df = df[df["scores"] != 0.5]  # get instances with LLM scores
    acc_human_vs_gold_sub = (subset_df["human_preference"] == subset_df["gold_preference"]).sum() / len(subset_df)
    acc_human_vs_llm_sub = (subset_df["human_preference"] == subset_df["llm_preference"]).sum() / len(subset_df)
    logging.info(f"Accuracy (human vs. gold) subset: {acc_human_vs_gold_sub*100:.2f}")
    logging.info(f"Accuracy (human vs. llm) subset: {acc_human_vs_llm_sub*100:.2f}")

    # Save disagreements
    lang_code = df["language"].to_list()[0]
    disagree_human_vs_gold = df[df["human_preference"] != df["gold_preference"]]
    disagree_human_vs_llm = subset_df[subset_df["human_preference"] != subset_df["llm_preference"]]
    disagree_human_vs_gold.to_csv(f"{lang_code}-disagreement-human-vs-gold.csv", index=False)
    disagree_human_vs_llm.to_csv(f"{lang_code}-disagreement-human-vs-llm.csv", index=False)


if __name__ == "__main__":
    main()
