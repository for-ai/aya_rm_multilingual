import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FONT_SIZES = {"small": 12, "medium": 16, "large": 18}
COLORS = {"green": "#355145", "purple": "#d8a6e5", "orange": "#fe7759"}

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


data = {
    "LlaMa 3.1 8B": [0.3533086666014079, 0.052422082615756406],
    "Aya 23 35B": [0.43767196047824003, 0.026040919354464294],
    # "Aya 23 8B": [0.013483014909052663, 0.03363706833599835],
    "Command R": [0.374457668650282, 0.02926089754079793],
    "Command R+": [0.3830841816733316, 0.020185255968455686],
    "Gemma 1.1 7B": [0.5190375637539242, 0.027757722654111305],
    "Gemma 2 9B": [0.5181663123111222, 0.031090119385244894],
    "LlaMa 3 70B": [0.5685224105896568, 0.04853344616275034],
    "LlaMa 3 8B": [0.37936948540837095, 0.032172769265151994],
    "LlaMa 3.1 70B": [0.603536768244583, 0.027191895488989915],
    "Mistal 7B v0.2": [0.4071166722276529, 0.04577594028555328],
    "Mistral 7B v0.3": [0.41195018984687265, 0.056184679972755454],
    "GPT-4 Turbo": [0.6106943361444249, 0.02932446842558468],
    "GPT-4o": [0.5833874065757011, 0.023695391445384514],
}

sorted_data = dict(sorted(data.items(), key=lambda item: item[1][0]))
labels_sorted = list(sorted_data.keys())
means_sorted = [v[0] for v in sorted_data.values()]
std_devs_sorted = [v[1] for v in sorted_data.values()]

# sns.set(style="whitegrid")
# palette = sns.color_palette("coolwarm", len(labels_sorted))

plt.figure(figsize=(7, 7))
x_pos_sorted = np.arange(len(labels_sorted))

ax1 = sns.barplot(
    x=x_pos_sorted,
    y=means_sorted,
    errorbar=None,
    color=COLORS.get("orange"),
    edgecolor=COLORS.get("green"),
)
plt.errorbar(x_pos_sorted, means_sorted, yerr=std_devs_sorted, fmt="none", c="black", capsize=5)

# ax1.spines["top"].set_color("black")
# ax1.spines["right"].set_color("black")
# ax1.spines["left"].set_color("black")
# ax1.spines["bottom"].set_color("black")
# for spine in ax1.spines.values():
#     spine.set_linewidth(2)  # Make the border thicker
plt.grid(color="gray", axis="y", alpha=0.2)

plt.ylim(0, 0.8)
plt.gca().set_axisbelow(True)

plt.xticks(x_pos_sorted, labels_sorted, rotation=45, ha="right")
plt.ylabel("Cohen's Kappa")
plt.title("Average Inner-Model Agreement Across Languages")

plt.tight_layout()
plt.savefig("plots/innermodel_agreement_green_oracle.pdf", bbox_inches="tight")
