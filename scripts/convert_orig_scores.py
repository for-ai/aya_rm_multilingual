from pathlib import Path
import json
import requests
import argparse
import logging

from rewardbench.utils import calculate_scores_per_section
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING

logging.basicConfig(level=logging.INFO)

REWARDBENCH_RESULTS_BASEURL = "https://huggingface.co/datasets/allenai/reward-bench-results/resolve/main/eval-set"


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert RewardBench scores to our formatting")
    parser.add_argument("--model", required=True, help="JSON URL to convert.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    url = f"{REWARDBENCH_RESULTS_BASEURL}/{args.model}.json"
    response = requests.get(url)
    if response.ok:
        data = json.loads(response.text)
    else:
        raise ValueError("Cannot parse response")

    results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, data)

    dataset_name = "allenai/reward-bench"
    if data["model_type"] == "Generative RM":
        results_dict = {
            "dataset": dataset_name,
            "model": data["model"],
            "chat_template": data["chat_template"],
            "scores": {},  # can't compute now
            "leaderboard": results_leaderboard,
            "subset": data,
        }
    else:
        results_dict = {
            "accuracy": None,  # can't compute now
            "num_prompts": 2985,
            "model": data["model"],
            "ref_model": None if "ref_model" not in data else data["ref_model"],
            "tokenizer": None,  # we don't know for now
            "chat_template": data["chat_template"],
            "extra_results": data,
        }

    output_file = f"{data['model'].split('/')[1]}-eng_Latn.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f)

    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
