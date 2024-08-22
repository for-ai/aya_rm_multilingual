import argparse
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

from analysis.plot_utils import get_scores

logging.basicConfig(level=logging.INFO)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, help="Directory to save the output plots."),
    parser.add_argument("--dataset", type=str, default="aya-rm-multilingual/eval-results", help="HuggingFace dataset that stores the eval results.")
    parser.add_argument("--force_download", action="store_true", help="If set, will redownload the dataset.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    leaderboard_df = get_leaderboard(dataset=args.dataset, force_download=args.force_download)

    # Get average of non eng_Latn
    leaderboard_df["Avg"] = leaderboard_df.drop(["eng_Latn", "Type"], axis=1).mean(axis=1, skipna=False)
    leaderboard_df["Std"] = leaderboard_df.drop(["eng_Latn", "Type"], axis=1).std(axis=1, skipna=False)
    leaderboard_df = leaderboard_df.sort_values(by=["Type", "Avg"], ascending=False)
    # overall_non_eng = pd.concat(
    #     [
    #         avg.rename("Avg_Multilingual"),
    #         std.rename("Std_Multilingual"),
    #         leaderboard_df["Type"],
    #     ],
    #     axis=1,
    # )
    # .sort_values(by=["Type", "Avg_Multilingual"], ascending=False)
    breakpoint()


def get_leaderboard(dataset: str, force_download: bool) -> "pd.DataFrame":
    dataset_dir = Path(snapshot_download(dataset, repo_type="dataset", force_download=force_download))
    lang_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]

    lang_scores = {}
    # Track model type
    model_type = {}
    for lang_dir in lang_folders:
        model_scores = get_scores(lang_dir)
        lang_scores[lang_dir.name] = {score["model"]: score["score"] for score in model_scores}
        for model in model_scores:
            model_name = model.get("model")
            if model_name not in model_type.keys():
                model_type[model_name] = model.get("model_type")

    lang_scores_df = pd.DataFrame(lang_scores).merge(
        pd.Series(model_type).rename("Type"),
        left_index=True,
        right_index=True,
    )
    return lang_scores_df


if __name__ == "__main__":
    main()
