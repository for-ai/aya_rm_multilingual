import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Create annotation CSV for a given language.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the annotation CSV files.")
    parser.add_argument("--langs", nargs="*", required=True, type=str, help="Languages to create annotation files on.")
    parser.add_argument("--pred_dataset", type=str, default="aya-rm-multilingual/eval-results", help="HuggingFace dataset containing the results.")
    parser.add_argument("--gold_dataset", type=str, default="aya-rm-multilingual/multilingual-reward-bench", help="HuggingFace dataset containing the gold labels.")
    parser.add_argument("--use_model", type=str, default=None, help="If set, will use model outputs as basis for sampling. Will sample equal number of wins/losses/ties. Only works for Generative RMs for now.")
    parser.add_argument("--sample_size", type=int, default=None, help="Total number of instances to sample.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    pred_dir = Path(snapshot_download(args.pred_dataset, repo_type="dataset"))
    lang_folders = [d for d in pred_dir.iterdir() if d.is_dir()]

    # for lang in args.langs:
    for lang_dir in lang_folders:
        if lang_dir.name in args.langs:
            lang = lang_dir.name
            gold_dataset = load_dataset(args.gold_dataset, lang, split="filtered")
            annotation_df = gold_dataset.to_pandas()
            if args.use_model:
                logging.info(f"Will sample based on {args.use_model} results")
                scores = get_per_instance_scores(model_name=args.use_model, lang_dir=lang_dir)
                annotation_df["scores"] = scores

                if args.sample_size:
                    logging.info(f"Sampling {args.sample_size} examples")
                    annotation_df = stratified_sampling(annotation_df, n=args.sample_size, column="scores")

            logging.info(f"Number of annotation tasks: {len(annotation_df)}")
            logging.info("Randomly swapping the completions")
            swap_mask = np.random.rand(len(annotation_df)) < 0.5
            annotation_df["swapped"] = swap_mask.astype(int)
            annotation_df = annotation_df.rename(columns={"chosen": "completion_a", "rejected": "completion_b"})

            # Save the answer key before swapping and removing some other columns
            answer_key_df = annotation_df.copy()
            # Start swapping
            annotation_df.loc[swap_mask, ["completion_a", "completion_b"]] = annotation_df.loc[
                swap_mask, ["completion_b", "completion_a"]
            ].values
            annotation_df = annotation_df.drop(
                columns=["chosen_model", "rejected_model", "subset", "scores", "swapped"]
            )

            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            answer_key_output = output_dir / f"{lang}-answer_key.csv"
            answer_key_df.to_csv(answer_key_output, index=False)
            annotation_file_output = output_dir / f"{lang}-annotation.csv"
            annotation_df.to_csv(annotation_file_output, index=False)
            logging.info(f"Saved answer key and annotation file to {output_dir}")


def get_per_instance_scores(model_name: str, lang_dir: Path) -> List[float]:
    model_file = [
        file for file in lang_dir.iterdir() if file.suffix == ".json" and model_name.replace("/", "___") in str(file)
    ]
    if len(model_file) == 0:
        logging.error(f"Can't find model '{model_name}' in {lang_dir.name} results")
        sys.exit(1)

    with open(model_file[0], "r") as f:
        results = json.load(f)

    scores = results["scores"]["results"]
    return scores


def stratified_sampling(df: "pd.DataFrame", n: int, column: str = "scores") -> "pd.DataFrame":
    counts = df[column].value_counts()
    min_count = counts.min()
    num_categories = len(counts)
    samples_per_category = min(n // num_categories, min_count)

    # Sample the rows
    samples = []
    for score in counts.index:
        score_df = df[df[column] == score]
        sampled_df = score_df.sample(n=samples_per_category, random_state=42)
        samples.append(sampled_df)

    # Concatenate the samples
    sampled_df = pd.concat(samples).reset_index(drop=True)
    return sampled_df


if __name__ == "__main__":
    main()
