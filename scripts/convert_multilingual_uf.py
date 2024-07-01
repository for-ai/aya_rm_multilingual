"""Convert multilingual ultrafeedback into a format acceptable for RewardBench

We need to follow the load_preference_dataset setup in RewardBench as
shown here: https://github.com/allenai/reward-bench/blob/main/rewardbench/utils.py#L136
So we need three columns:
    - prompt (str)
    - chosen (list[dict[str, str]]), and
    - rejected (list[dict[str, str]])
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert a HuggingFace dataset into the RewardBench format."
    )

    # fmt: off
    parser.add_argument("--dataset", type=str, default="nthakur/multilingual-ultrafeedback-dpo-v0.1", help="Dataset to convert.")
    parser.add_argument("--output_path", type=Path, default="data/multilingual-ultrafeedback-dpo-v0.1.json", help="Path to save converted dataset as JSON file.")
    # fmt: on

    return parser.parse_args()


def main():
    args = get_args()
    if args.output_path:
        args.output_path.parents[0].mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split="test")

    def _convert_to_turn_based(example):
        example["chosen"] = [
            {"content": example["prompt"], "role": "user"},
            {"content": example["chosen_raw"], "role": "assistant"},
        ]
        example["rejected"] = [
            {"content": example["prompt"], "role": "user"},
            {"content": example["rejected_raw"], "role": "assistant"},
        ]
        return example

    cols = ["id", "source", "language", "input", "chosen", "rejected"]
    rename_map = {"input": "prompt", "chosen": "chosen_raw", "rejected": "rejected_raw"}
    dataset = (
        dataset.select_columns(cols)
        .rename_columns(rename_map)
        .map(_convert_to_turn_based)
        .remove_columns(["chosen_raw", "rejected_raw"])
    )
    dataset.to_json(args.output_path)
    logging.info(f"Saved file to {args.output_path}.")


if __name__ == "__main__":
    main()
