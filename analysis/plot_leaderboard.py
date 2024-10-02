import argparse
import logging
from pathlib import Path
from typing import Optional

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
    parser.add_argument("--dataset", type=str, default="aya-rm-multilingual/eval-results-gtranslate-v2", help="HuggingFace dataset that stores the eval results.")
    parser.add_argument("--force_download", action="store_true", help="If set, will redownload the dataset.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    # *** Leaderboard scores ***
    logging.info("Plotting leaderboard scores for all models and languages")
    leaderboard_df = get_leaderboard(dataset=args.dataset, force_download=args.force_download)
    chat_leaderboard_df = get_leaderboard(
        dataset=args.dataset,
        force_download=args.force_download,
        category="Chat",
    )
    chat_hard_leaderboard_df = get_leaderboard(
        dataset=args.dataset,
        force_download=args.force_download,
        category="Chat Hard",
    )
    safety_leaderboard_df = get_leaderboard(
        dataset=args.dataset,
        force_download=args.force_download,
        category="Safety",
    )
    reasoning_leaderboard_df = get_leaderboard(
        dataset=args.dataset,
        force_download=args.force_download,
        category="Reasoning",
    )

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
        output_file = output_dir / f"leaderboard-{model_type.replace(' ', '_')}.png"
        fig.savefig(output_file, dpi=120)
        logging.info(f"Saved to {output_file}")

    # *** English drop ***
    eng_drop_df = pd.DataFrame(
        {
            "Overall": get_eng_drop(leaderboard_df)["Percentage_Change"],
            "Chat": get_eng_drop(chat_leaderboard_df)["Percentage_Change"],
            "Chat Hard": get_eng_drop(chat_hard_leaderboard_df)["Percentage_Change"],
            "Safety": get_eng_drop(safety_leaderboard_df)["Percentage_Change"],
            "Reasoning": get_eng_drop(reasoning_leaderboard_df)["Percentage_Change"],
        }
    )
    # Only get top-3 and bottom-3. Put bottom 3 at the top rows
    top_bottom_n = pd.concat([eng_drop_df.nsmallest(3, "Overall"), eng_drop_df.nlargest(3, "Overall")])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(top_bottom_n, annot=True, cmap="Reds_r", fmt=".1f", annot_kws={"size": 18}, cbar=False)
    ax.xaxis.tick_top()
    fig.tight_layout()
    output_file = output_dir / "eng-drop-overall.png"
    fig.savefig(output_file, dpi=120)
    logging.info(f"Saved to {output_file}")


def get_eng_drop(df: pd.DataFrame) -> pd.DataFrame:
    eng_drop_df = df[["eng_Latn", "Avg"]].rename(columns={"eng_Latn": "English", "Avg": "Multilingual_Avg"})
    eng_drop_df["Percentage_Change"] = (
        (eng_drop_df["Multilingual_Avg"] - eng_drop_df["English"]) / eng_drop_df["English"]
    ) * 100
    eng_drop_df = eng_drop_df.dropna().sort_values(by="Percentage_Change", ascending=True)
    return eng_drop_df


def get_leaderboard(dataset: str, force_download: bool, category: Optional[str] = None) -> "pd.DataFrame":
    dataset_dir = Path(snapshot_download(dataset, repo_type="dataset", force_download=force_download))
    lang_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]

    lang_scores = {}
    # Track model type
    model_type = {}
    for lang_dir in lang_folders:
        model_scores = get_scores(lang_dir)
        if category:
            lang_scores[lang_dir.name] = {score["model"]: score["category_scores"][category] for score in model_scores}
        else:
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

    # Get average but dont include eng_Latn
    lang_scores_df["Avg"] = lang_scores_df.drop(["eng_Latn", "Type"], axis=1).mean(axis=1, skipna=False)
    lang_scores_df["Std"] = lang_scores_df.drop(["eng_Latn", "Type"], axis=1).std(axis=1, skipna=False)
    lang_scores_df = lang_scores_df.sort_values(by=["Type", "Avg"], ascending=False)
    return lang_scores_df


if __name__ == "__main__":
    main()
