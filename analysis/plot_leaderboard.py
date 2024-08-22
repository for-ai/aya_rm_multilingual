import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="aya-rm-multilingual/eval-results", help="HuggingFace dataset that stores the eval results.")
    parser.add_argument("--force_download", action="store_true", help="If set, will redownload the dataset.")
    parser.add_argument("--output_path", type=Path, help="Path to save the output plot."),
    # fmt: on
    pass


def main():
    args = get_args()


if __name__ == "__main__":
    main()
