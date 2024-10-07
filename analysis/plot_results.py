import argparse
import logging
from pathlib import Path
from inspect import signature
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

FONT_SIZES = {"small": 12, "medium": 16, "large": 18}

PLOT_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX"],
    "font.size": FONT_SIZES.get("medium"),
    "axes.titlesize": FONT_SIZES.get("large"),
    "axes.labelsize": FONT_SIZES.get("large"),
    "xtick.labelsize": FONT_SIZES.get("large"),
    "ytick.labelsize": FONT_SIZES.get("large"),
    "legend.fontsize": FONT_SIZES.get("medium"),
    "figure.titlesize": FONT_SIZES.get("medium"),
    "text.usetex": False,
}

plt.rcParams.update(PLOT_PARAMS)

logging.basicConfig(level=logging.INFO)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Plotting utilities", formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    shared_args = argparse.ArgumentParser(add_help=False)
    shared_args.add_argument("--output_path", type=Path, required=True, help="Path to save the PDF plot.")
    shared_args.add_argument("--figsize", type=int, nargs=2, default=[10, 10], help="Matplotlib figure size.")

    parser_main_results = subparsers.add_parser("main_heatmap", help="Plot results as a heatmap.", parents=[shared_args])
    parser_main_results.add_argument("--input_path", type=Path, required=True, help="Path to the results file.")

    parser_eng_drop = subparsers.add_parser("eng_drop_line", help="Plot english drop as a line chart.", parents=[shared_args])
    parser_eng_drop.add_argument("--input_path", type=Path, required=True, help="Path to the results file.")
    parser_eng_drop.add_argument("--top_n", default=None, type=int, help="If set, will only show the .")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    cmd_map = {
        "main_heatmap": plot_main_heatmap,
        "eng_drop_line": plot_eng_drop_line,
    }

    def _filter_args(func, kwargs):
        func_params = signature(func).parameters
        return {k: v for k, v in kwargs.items() if k in func_params}

    if args.command in cmd_map:
        plot_fn = cmd_map[args.command]
        kwargs = _filter_args(plot_fn, vars(args))
        plot_fn(**kwargs)
    else:
        logging.error(f"Unknown plotting command: {args.command}")


def plot_main_heatmap(
    input_path: Path,
    output_path: Path,
    figsize: Optional[tuple[int, int]] = (18, 5),
):

    df = pd.read_csv(input_path)
    # Remove unnecessary column
    df.pop("eng_Latn")

    df = df.sort_values(by="Avg_Multilingual", ascending=False).head(10).reset_index(drop=True)
    data = df[[col for col in df.columns if col not in ["Model_Type"]]].rename(columns={"Avg_Multilingual": "Avg"})
    data = data.set_index("Model")
    data = data * 100
    data = data[sorted(data.columns)]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(data, ax=ax, cmap="YlGn", annot=True, annot_kws={"size": 14}, fmt=".2f", cbar=False)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("")
    ax.set_yticklabels([f"{model}     " for model in data.index])

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


def plot_eng_drop_line(
    input_path: Path,
    output_path: Path,
    figsize: Optional[tuple[int, int]] = (18, 5),
    top_n: Optional[int] = None,

    from scipy.stats import pearsonr, spearmanr

    df = pd.read_csv(input_path)
    df = df[["Model", "Model_Type", "eng_Latn", "Avg_Multilingual"]]
    df = df.sort_values(by="Avg_Multilingual", ascending=False).reset_index(drop=True)
    data = df.set_index("Model").dropna()
    data = data[["eng_Latn", "Avg_Multilingual"]] * 100
    model_types = df.dropna().pop("Model_Type")
    if top_n:
        logging.info(f"Showing top {top_n}")
        data = data.head(top_n)
        model_types = model_types[:top_n]
        
    fig, ax = plt.subplots(figsize=figsize)
    mrewardbench_scores = data["Avg_Multilingual"]
    rewardbench_scores = data["eng_Latn"]
    r, _ = pearsonr(mrewardbench_scores, rewardbench_scores)
    res = spearmanr(mrewardbench_scores, rewardbench_scores)

    colormap = {"Generative RM": "green", "Sequence Classifier": "blue", "DPO": "red"}
    colors = [colormap[model_type] for model_type in model_types]

    ax.scatter(rewardbench_scores, mrewardbench_scores, marker="o", s=30, color=colors)

    min_val = min(rewardbench_scores.min(), mrewardbench_scores.min())
    max_val = max(rewardbench_scores.max(), mrewardbench_scores.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", alpha=0.25)
    ax.set_xlabel("RewardBench (Lambert et al., 2024)")
    ax.set_ylabel("M-RewardBench")
    ax.set_aspect("equal")

    model_names = [model.split("/")[1] for model in data.index]
    texts = [
        ax.text(
            rewardbench_scores[idx],
            mrewardbench_scores[idx],
            model_names[idx],
            fontsize=12,
        )
        for idx in range(len(data))
    ]
    adjust_text(
        texts,
        ax=ax,
        # force_static=0.15,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    ax.text(
        0.1,
        0.9,
        s=f"Pearson-r: {r:.2f}\nSpearman-r: {res.statistic:.2f}",
        fontsize=14,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")

    logging.info("Showing top-10 models with biggest deltas")
    delta_df = data.copy(deep=True)
    delta_df["delta"] = delta_df["eng_Latn"] - delta_df["Avg_Multilingual"]
    delta_df = delta_df.sort_values(by="delta", ascending=False)
    print(delta_df.to_latex())


if __name__ == "__main__":
    main()
