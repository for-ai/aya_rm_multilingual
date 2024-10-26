import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from datasets import load_dataset

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


FONT_SIZES = {"small": 12, "medium": 16, "large": 18}
COLORS = {
    "green": "#355145",
    "purple": "#d8a6e5",
    "orange": "#fe7759",
    "blue": "#4c6ee6",
}

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

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}


def find_key(d: dict[str, list[str]], value: str) -> str | None:
    for key, values in d.items():
        if value in values:
            return key
    return None


plt.rcParams.update(PLOT_PARAMS)

# annotations = Path("data/hin_Deva_histogram.csv")
lang = "hin_Deva"
lang = "ind_Latn"
annotations = Path(f"plots/{lang}_histogram.csv")
reference = Path("plots/eng_Latn_histogram.csv")

annot_df = pd.read_csv(annotations).set_index("model").T
ref_df = pd.read_csv(reference).set_index("model").T

cohen_scores: dict[str, float] = {}
for (idx, annot), (_, ref) in zip(annot_df.iterrows(), ref_df.iterrows()):
    cohen_scores[idx] = cohen_kappa_score(annot.to_list(), ref.to_list(), labels=[0, 1, 2])


df = pd.DataFrame([cohen_scores]).T.reset_index().rename(columns={0: "cohen", "index": "instance_id"}).dropna()
sdf = load_dataset(
    "aya-rm-multilingual/multilingual-reward-bench-gtranslate", "ind_Latn", split="filtered"
).to_pandas()
sdf = sdf[["prompt", "chosen", "rejected", "subset", "id"]].rename(columns={"id": "instance_id"})
sdf["instance_id"] = sdf["instance_id"].apply(lambda x: str(x))
combi = df.merge(sdf, on="instance_id").sort_values(by="cohen", ascending=False).reset_index(drop=True)
combi["category"] = combi["subset"].apply(lambda x: find_key(SUBSET_MAPPING, x))

# df_grouped = combi.groupby(["category", "cohen"]).count().groupby(level=0).apply(lambda x: x / x.sum()).reset_index()
combi["count"] = 1

# Bin the data and compute percentages


fig, axs = plt.subplots(
    2,
    1,
    figsize=(8, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [5, 2]},
)
sns.histplot(
    df["cohen"],
    ax=axs[0],
    stat="count",
    fill=True,
    color=COLORS.get("orange"),
)

axs[0].axvline(x=0, color=COLORS.get("green"), linestyle="--", linewidth=1)
axs[0].axvline(x=0.60, color=COLORS.get("green"), linestyle="--", linewidth=1)

sns.histplot(
    data=combi,
    x="cohen",
    # weights="count",
    hue="category",
    multiple="fill",
    ax=axs[1],
    palette=[
        COLORS.get("green"),
        COLORS.get("purple"),
        COLORS.get("orange"),
        COLORS.get("blue"),
    ],
)

lang_code = LANG_STANDARDIZATION[lang.split("_")[0]]
axs[1].set_xlabel("Cohen's Kappa (Language: Indonesian)")
axs[1].set_ylabel("Percentage")

# annot_df["model_annotations"] = [i for i in annot_df.values]
# annot_df["eng_reference"] = [i for i in ref_df.values]
# annotations = annot_df[["model_annotations", "eng_reference"]].reset_index().rename(columns={"index": "instance_id"})
# df = df.merge(annotations, how="left", on="instance_id")

axs[0].set_axisbelow(True)
axs[0].grid(True, color="gray", axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(f"plots/cohen_k_histogram_{lang}.svg", bbox_inches="tight")
