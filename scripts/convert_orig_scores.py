import argparse
import logging

logging.basicConfig(level=logging.INFO)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Convert RewardBench scores to our formatting")
    parser.add_argument("--filepath", required=True, help="Filepath of JSON to download.")
    parser.add_argument("--output_file", required=True, help="Filepath to save the updated JSON file.")
    parser.add_argument("--dataset_name", default="allenai/reward-bench", help="HuggingFace dataset to obtain datasets from.")
    # fmt: on
    return parser.parse_args()


def main():
    pass
