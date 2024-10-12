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
    "font.serif": ["Times", "Times New Roman", "STIX"],
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

MODEL_STANDARDIZATION = {
    "openai/gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "openai/gpt-4o-2024-05-13": "GPT-4o",
    "google/gemma-2-9b-it": "Gemma 2 9B",
    "LxzGordon/URM-LLaMa-3.1-8B": "URM LlaMa 3.1 8B",
    "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama 3.1 70B",
    "meta-llama/Meta-Llama-3-70B-Instruct": "Llama 3 70B",
    "CIR-AMS/BTRM_Qwen2_7b_0613": "BTRM Qwen 2 7B",
    "cohere/command-r-plus-08-2024": "Command R+",
    "allenai/tulu-2-dpo-13b": "Tulu 2 13B DPO",
    "cohere/c4ai-aya-23-35b": "Aya 23 35B",
}

LANG_STANDARDIZATION = {
    "arb": "ar",
    "ces": "cs",
    "deu": "de",
    "ell": "el",
    "fra": "fr",
    "heb": "he",
    "hin": "hi",
    "ind": "id",
    "ita": "it",
    "jpn": "jp",
    "kor": "kr",
    "nld": "nl",
    "pes": "fa",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "spa": "es",
    "tur": "tr",
    "ukr": "uk",
    "vie": "vi",
    "zho": "zh",
}


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

    parser_ling_dims = subparsers.add_parser("ling_dims", help="Plot performance with respect to linguistic dimensions.", parents=[shared_args])
    parser_ling_dims.add_argument("--input_path", type=Path, required=True, help="Path to the results file.")
    parser_ling_dims.add_argument("--langdata", type=Path, required=True, help="Path to the language data file.")
    parser_ling_dims.add_argument("--top_n", type=int, required=False, default=None, help="Aggregate only the scores for top-n.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()

    cmd_map = {
        "main_heatmap": plot_main_heatmap,
        "eng_drop_line": plot_eng_drop_line,
        "ling_dims": plot_ling_dims,
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
    df.pop("Family")

    df = df.sort_values(by="Avg_Multilingual", ascending=False).head(10).reset_index(drop=True)
    data = df[[col for col in df.columns if col not in ["Model_Type"]]].rename(columns={"Avg_Multilingual": "Avg"})
    data["Model"] = data["Model"].replace(MODEL_STANDARDIZATION)
    data = data.set_index("Model")
    data = data * 100
    data["zho"] = data[["zho_Hans", "zho_Hant"]].mean(axis=1)
    data.pop("zho_Hans")
    data.pop("zho_Hant")
    data = data[sorted(data.columns)]
    data.columns = [col.split("_")[0] for col in data.columns]
    data["Var"] = data[list(LANG_STANDARDIZATION.keys())].var(axis=1)
    data = data.rename(columns=LANG_STANDARDIZATION)

    lang_results = data[list(LANG_STANDARDIZATION.values())]
    avg = data[["Avg"]]
    var = data[["Var"]]

    fig, axs = plt.subplots(ncols=3, figsize=figsize, gridspec_kw={"width_ratios": [0.5, 0.5, 9]}, sharey=True)
    cmap = "Greys"
    fmt = ".1f"

    sns.heatmap(avg, ax=axs[0], cmap=cmap, annot=True, annot_kws={"size": 16}, fmt=fmt, cbar=False)
    axs[0].xaxis.set_ticks_position("top")
    axs[0].set_xticklabels(avg.columns, fontsize=20)
    axs[0].tick_params(axis="x")
    axs[0].set_ylabel("")
    axs[0].set_yticklabels([f"{model}     " for model in avg.index], fontsize=20)

    sns.heatmap(var, ax=axs[1], cmap=cmap, annot=True, annot_kws={"size": 16}, fmt=fmt, cbar=False)
    axs[1].xaxis.set_ticks_position("top")
    axs[1].set_xticklabels(var.columns, fontsize=20)
    axs[1].tick_params(axis="x")
    axs[1].set_ylabel("")
    axs[1].tick_params(axis="y", length=0)
    axs[1].set_yticklabels([f"{model}     " for model in var.index], fontsize=20)

    sns.heatmap(lang_results, ax=axs[2], cmap=cmap, annot=True, annot_kws={"size": 16}, fmt=fmt, cbar=False)
    axs[2].xaxis.set_ticks_position("top")
    axs[2].set_xticklabels(lang_results.columns, fontsize=20)
    axs[2].tick_params(axis="x")
    axs[2].tick_params(axis="y", length=0)
    axs[2].set_ylabel("")
    axs[2].set_yticklabels([f"{model}     " for model in lang_results.index], fontsize=20)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


def plot_eng_drop_line(
    input_path: Path,
    output_path: Path,
    figsize: Optional[tuple[int, int]] = (18, 5),
    top_n: Optional[int] = None,
):
    from scipy.stats import pearsonr, spearmanr

    df = pd.read_csv(input_path)
    df = df[["Model", "Model_Type", "eng_Latn", "Avg_Multilingual"]]
    df = df.sort_values(by="Avg_Multilingual", ascending=False).reset_index(drop=True)
    data = df.set_index("Model").dropna()
    data[data.select_dtypes(include="number").columns] = data.select_dtypes(include="number") * 100
    data["Model_Type"] = data["Model_Type"].replace({"DPO": "Implicit RM", "Sequence Classifier": "Classifier RM"})
    if top_n:
        logging.info(f"Showing top {top_n}")
        data = data.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["red", "green", "blue"]
    markers = ["o", "*", "D"]
    for (label, group), marker in zip(data.groupby("Model_Type"), markers):
        mrewardbench_scores = group["Avg_Multilingual"]
        rewardbench_scores = group["eng_Latn"]
        ax.scatter(rewardbench_scores, mrewardbench_scores, marker=marker, s=60, label=label, color="k")

    mrewardbench_scores = data["Avg_Multilingual"]
    rewardbench_scores = data["eng_Latn"]
    r, _ = pearsonr(mrewardbench_scores, rewardbench_scores)
    res = spearmanr(mrewardbench_scores, rewardbench_scores)

    # ax.scatter(rewardbench_scores, mrewardbench_scores, marker="o", s=30, color=colors, label=model_types)

    min_val = min(rewardbench_scores.min(), mrewardbench_scores.min())
    max_val = max(rewardbench_scores.max(), mrewardbench_scores.max())
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", alpha=0.25)
    ax.set_xlabel("RewardBench (Lambert et al., 2024)")
    ax.set_ylabel("M-RewardBench")
    ax.grid(color="gray", alpha=0.2, which="both")
    ax.set_aspect("equal")
    ax.legend(frameon=False, handletextpad=0.2, fontsize=12)

    if top_n:
        model_names = [MODEL_STANDARDIZATION[model] for model in data.index]
        texts = [
            ax.text(
                rewardbench_scores[idx],
                mrewardbench_scores[idx],
                model_names[idx],
                fontsize=14,
            )
            for idx in range(len(data))
        ]
        adjust_text(
            texts,
            ax=ax,
            # force_static=0.15,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # ax.text(
    #     0.6,
    #     0.8,
    #     s=f"Pearson-r: {r:.2f}  Spearman-r: {res.statistic:.2f}",
    #     fontsize=14,
    #     transform=ax.transAxes,
    #     verticalalignment="top",
    #     rotation=45,
    #     color="gray",
    #     # bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    # )

    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")

    logging.info("Showing top-10 models with biggest deltas")
    delta_df = data.copy(deep=True)
    delta_df["delta"] = delta_df["eng_Latn"] - delta_df["Avg_Multilingual"]
    delta_df = delta_df.sort_values(by="delta", ascending=False)
    print(delta_df.to_latex())


def plot_ling_dims(
    input_path: Path,
    langdata: Path,
    output_path: Path,
    top_n: Optional[int] = None,
    figsize: Optional[tuple[int, int]] = (18, 5),
):
    raw = pd.read_csv(input_path).set_index("Model")
    if top_n:
        raw = raw.head(top_n)
    raw = raw[[col for col in raw.columns if col not in ("Model_Type", "eng_Latn", "Avg_Multilingual", "Family")]]
    raw = raw.T
    langdata = pd.read_csv(langdata).set_index("Language")
    combined = raw.merge(langdata, left_index=True, right_index=True)
    combined["Avg"] = raw.mean(axis=1) * 100
    combined["Std"] = raw.std(axis=1) * 100

    combined = combined.rename(columns={"Resource_Type": "Resource Availability"})
    # Remove Class 0 because it's misleading
    combined = combined[combined["Resource Availability"] != "Class-0"].reset_index()

    linguistic_dims = [
        "Resource Availability",
        "Family",
        "Script",
    ]
    fig, axs = plt.subplots(1, len(linguistic_dims), figsize=figsize, sharex=True)
    for ax, dim in zip(axs, linguistic_dims):
        lingdf = combined.groupby(dim).agg({"Avg": "mean", "Std": "mean"}).reset_index()
        if dim != "Resource Availability":
            lingdf = lingdf.sort_values(by="Avg", ascending=False)
        else:
            lingdf = lingdf[::-1]

        ax.grid(color="gray", alpha=0.2, which="both", axis="x")
        ax.set_axisbelow(True)
        sns.barplot(
            x="Avg",
            y=dim,
            data=lingdf,
            ax=ax,
            color="green",
            width=0.4 if dim == "Resource Availability" else 0.7,
        )
        ax.set_title(dim)
        ax.set_xlim([60, 70])
        ax.set_ylabel("")
        ax.set_xlabel("M-RewardBench Score")

        # ax.spines["right"].set_visible(False)
        # ax.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    main()
