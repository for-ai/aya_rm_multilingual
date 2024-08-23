import argparse
import logging
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download

from analysis.plot_utils import get_scores, PLOT_PARAMS

logging.basicConfig(level=logging.INFO)

plt.rcParams.update(PLOT_PARAMS)


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
    output_dir = Path(args.output_dir)
    leaderboard_df = get_leaderboard(dataset=args.dataset, force_download=args.force_download)

    # Get average of non eng_Latn
    leaderboard_df["Avg"] = leaderboard_df.drop(["eng_Latn", "Type"], axis=1).mean(axis=1, skipna=False)
    leaderboard_df["Std"] = leaderboard_df.drop(["eng_Latn", "Type"], axis=1).std(axis=1, skipna=False)
    leaderboard_df = leaderboard_df.sort_values(by=["Type", "Avg"], ascending=False)

    # Save per model type
    model_types = leaderboard_df["Type"].unique().tolist()
    for model_type in model_types:
        model_type_df = leaderboard_df[leaderboard_df["Type"] == model_type]
        data = model_type_df.drop(["eng_Latn", "Type", "Std"], axis=1)
        avg_col = "Avg"
        data = data[[avg_col] + [c for c in data.columns if c != avg_col]]
        data = data.dropna()

        if "Generative" in model_type:
            figsize = (24, 8)
        else:
            figsize = (24, 3)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data, annot=True, cmap="BuPu", ax=ax, annot_kws={"size": 14})
        ax.xaxis.tick_top()
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="left", fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        fig.tight_layout()
        fig.savefig(output_dir / f"leaderboard-{model_type.replace(' ', '_')}.png", dpi=120)


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
