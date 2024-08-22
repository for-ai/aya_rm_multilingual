import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

from analysis.plot_utils import get_scores


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
    dataset_dir = Path(snapshot_download(args.dataset, repo_type="dataset", force_download=args.force_download))
    lang_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]

    lang_scores = {}
    for lang_dir in lang_folders:
        model_scores = get_scores(lang_dir)
        lang_scores[lang_dir.name] = {score["model"]: score["score"] for score in model_scores}

    lang_scores_df = pd.DataFrame(lang_scores)
    breakpoint()


if __name__ == "__main__":
    main()
