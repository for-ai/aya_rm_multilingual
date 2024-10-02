import argparse
import logging
from pathlib import Path
from inspect import signature
from typing import Optional

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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
    parser_main_results.add_argument("--input_path", action="append", help="Path to the results file and model category (e.g., DPO::path/to/dpo_results.csv).")
    parser_main_results.add_argument("--top_ten_only", action="store_true", help="If set, will only show the top-10 of all models.")
    parser_main_results.add_argument("--print_latex", action="store_true", help="If set, print LaTeX table.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    cmd_map = {
        "main_heatmap": plot_main_heatmap,
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
    input_path: list[str],
    output_path: Optional[Path] = None,
    figsize: Optional[tuple[int, int]] = None,
    top_ten_only: bool = False,
    print_latex: bool = False,
):
    category_results = {path.split("::")[0]: pd.read_csv(path.split("::")[1]) for path in input_path}

    if top_ten_only:
        logging.info("Passed --top_ten_only tag, will print LaTeX table of top ten models")
        df_with_tags = []
        for category, df in category_results.items():
            df = df.set_index(df.columns[0]) * 100
            df["model_type"] = category
            df.index.name = "model"
            df_with_tags.append(df)
        top_ten_df = pd.concat(df_with_tags).sort_values(by="Avg", ascending=False).head(10)
        model_type_col = top_ten_df.pop("model_type")
        avg_col = top_ten_df.pop("Avg")
        top_ten_df = top_ten_df.reindex(sorted(top_ten_df.columns), axis=1)
        top_ten_df.insert(0, "Model", model_type_col)
        top_ten_df.insert(1, "Avg", avg_col)

        if print_latex:
            top_ten_df.columns = top_ten_df.columns.str.replace("_", r"\_", regex=False)
            print(top_ten_df.to_latex(float_format="%.2f"))

        # Plot
        top_ten_df.pop("Model")
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sns.heatmap(
            top_ten_df,
            ax=ax,
            cmap="BuPu",
            cbar=False,
            annot=True,
            annot_kws={"size": 14},
            fmt=".2f",
        )

        # cbar = ax.collections[0].colorbar
        # cbar.set_label("Score")
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("")

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")

    else:
        fig, axs = plt.subplots(3, 1, figsize=figsize, gridspec_kw={"height_ratios": [4, 2, 2]}, sharex=True)
        cbar_ax = fig.add_axes([1.05, 0.3, 0.03, 0.4])
        for idx, (ax, (category, df)) in enumerate(zip(axs, category_results.items())):
            df = df.set_index(df.columns[0]) * 100
            df.index.name = "model"
            sns.heatmap(
                df,
                ax=ax,
                cmap="BuPu",
                annot=True,
                annot_kws={"size": 12},
                fmt=".2f",
                # Ticklabels and colorbar on first heatmap only
                xticklabels=(idx == 0),
                cbar=(idx == 0),
                cbar_ax=None if idx else cbar_ax,
            )

            if idx == 0:
                cbar = ax.collections[0].colorbar
                cbar.set_label("Score")
                ax.xaxis.set_ticks_position("top")
                ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
