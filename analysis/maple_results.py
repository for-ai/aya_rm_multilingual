import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from huggingface_hub import snapshot_download

FONT_SIZES = {"small": 12, "medium": 16, "large": 18}

PLOT_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX"],
    "font.size": FONT_SIZES.get("medium"),
    "axes.titlesize": FONT_SIZES.get("large"),
    "axes.labelsize": FONT_SIZES.get("large"),
    "xtick.labelsize": FONT_SIZES.get("large"),
    "ytick.labelsize": FONT_SIZES.get("small"),
    "legend.fontsize": FONT_SIZES.get("medium"),
    "figure.titlesize": FONT_SIZES.get("medium"),
    "text.usetex": False,
}

logging.basicConfig(level=logging.INFO)

plt.rcParams.update(PLOT_PARAMS)


def load_json(json_file_path):
    with open(json_file_path, "r") as file:
        json_data = json.load(file)
    return json_data


results_dir = "data/eval-results-maple"
results_path = Path(results_dir)

results_all = []
for result_file in results_path.glob("*.json"):
    raw_results = load_json(result_file)
    if "leaderboard" in raw_results.keys():
        model_id = raw_results["model"]
        subset_results = raw_results["subset"]
        overall = raw_results["scores"]["accuracy"]
        remove_key = ["model", "model_type", "chat_template"]
        for key in remove_key:
            del subset_results[key]
    elif "subset_results" in raw_results.keys():
        model_id = raw_results["model"]
        subset_results = raw_results["subset_results"]
        overall = raw_results["accuracy"]
    else:
        model_id = raw_results["model"]
        subset_results = raw_results["extra_results"]
        overall = raw_results["accuracy"]
    # print(model_id, overall)
    # print("\t", subset_results)
    # results_all.append([model_id, overall, subset_results])
    results_all.append({"Model": model_id, "Avg": overall, **subset_results})

    # import ipdb; ipdb.set_trace()

TOP = 10
# results_all.sort(key=lambda x: x[1], reverse=True)
# results_all = results_all[:TOP]
# print(results_all)

df_results = pd.DataFrame(results_all)
df_results = df_results.sort_values(by="Avg", ascending=False).reset_index(drop=True)
df_results = df_results.head(10).reset_index(drop=True)

df_results.columns = df_results.columns.str.replace("^maple-", "", regex=True)
df_results = df_results.set_index("Model")
df_results = df_results * 100
fig, ax = plt.subplots(1, 1, figsize=(18, 5))

sns.heatmap(df_results, ax=ax, cmap="YlGn", annot=True, annot_kws={"size": 16}, fmt=".1f", cbar=False)

ax.xaxis.set_ticks_position("top")
ax.tick_params(axis="x", labelrotation=45)
ax.set_ylabel("")
ax.set_yticklabels([f"{model}     " for model in df_results.index])

plt.tight_layout()

plt.savefig("plots/maple.pdf", bbox_inches="tight")
# import ipdb; ipdb.set_trace()
