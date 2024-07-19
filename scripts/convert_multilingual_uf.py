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
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert a HuggingFace dataset into the RewardBench format.")
    parser.add_argument("--dataset", type=str, default="nthakur/multilingual-ultrafeedback-dpo-v0.1", help="Dataset to convert.")
    parser.add_argument("--output_path", type=Path, default="data/multilingual-ultrafeedback-dpo-v0.1.json", help="Path to save converted dataset as JSON file.")
    parser.add_argument("--en", action="store_true", help="Use the english columns.")
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

    prefix = "en_" if args.en else ""
    cols = [
        "id",
        "source",
        "language",
        f"{prefix}input",
        f"{prefix}chosen",
        f"{prefix}rejected",
    ]
    rename_map = {
        f"{prefix}input": "prompt",
        f"{prefix}chosen": "chosen_raw",
        f"{prefix}rejected": "rejected_raw",
    }
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
