import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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


# data = {
#     "LlaMa 3.1 8B": [0.3533086666014079, 0.052422082615756406],
#     "Aya 23 35B": [0.43767196047824003, 0.026040919354464294],
#     # "Aya 23 8B": [0.013483014909052663, 0.03363706833599835],
#     "Command R": [0.374457668650282, 0.02926089754079793],
#     "Command R+": [0.3830841816733316, 0.020185255968455686],
#     "Gemma 1.1 7B": [0.5190375637539242, 0.027757722654111305],
#     "Gemma 2 9B": [0.5181663123111222, 0.031090119385244894],
#     "LlaMa 3 70B": [0.5685224105896568, 0.04853344616275034],
#     "LlaMa 3 8B": [0.37936948540837095, 0.032172769265151994],
#     "LlaMa 3.1 70B": [0.603536768244583, 0.027191895488989915],
#     "Mistal 7B v0.2": [0.4071166722276529, 0.04577594028555328],
#     "Mistral 7B v0.3": [0.41195018984687265, 0.056184679972755454],
#     "GPT-4 Turbo": [0.6106943361444249, 0.02932446842558468],
#     "GPT-4o": [0.5833874065757011, 0.023695391445384514],
# }

data = {
    "Mistral 7B v0.2": [0.41964902527302483, 0.041728704319417186, "Generative RM"],
    "Aya 23 35B": [0.4366594509037704, 0.02590083631166214, "Generative RM"],
    "Command R": [0.370172816882575, 0.02977439059146716, "Generative RM"],
    "Command R+": [0.38117473236836474, 0.020413901190603385, "Generative RM"],
    "Gemma 1.1 7B": [0.5121848983276365, 0.02775593676763153, "Generative RM"],
    "Gemma 2 9B": [0.5239388151608217, 0.029070955636084302, "Generative RM"],
    "Llama 3 70B": [0.5738032949863474, 0.04813697578838559, "Generative RM"],
    "Llama 3 8B": [0.3426278270154337, 0.028673093628218196, "Generative RM"],
    "Llama 3.1 70B": [0.6074197074501972, 0.028414614724563008, "Generative RM"],
    "Llama 3.1 8B": [0.34965468089191665, 0.056407978898463204, "Generative RM"],
    "Mistral 7B v0.3": [0.4166882337797498, 0.05085550655767351, "Generative RM"],
    "GPT-4 Turbo": [0.6096953791655624, 0.028784709595173846, "Generative RM"],
    "GPT-4o": [0.5833907047087866, 0.023692522150173454, "Generative RM"],
    "Tulu 2 DPO 13B": [0.3416546214690787, 0.1304713944811808, "Implicit RM"],
    "BTRM Qwen 2 7B": [0.4893276344968342, 0.07031889836622843, "Classifier RM"],
    "Eurus RM 7B": [0.3586485854871021, 0.09638527344174744, "Classifier RM"],
    "Zephyr 7B Beta": [0.35011426942621166, 0.176041224588175, "Implicit RM"],
    "Hermes 2 Mistral 7B DPO": [0.1902062108486662, 0.08462799373351747, "Implicit RM"],
    "Qwen1.5 4B": [0.38751934608609767, 0.055096683780610285, "Implicit RM"],
    "StableLM Zephyr 3B": [0.1708047069636795, 0.06315971482897487, "Implicit RM"],
    "Tulu 2.5 13B RM": [0.3038059897554214, 0.1147333149007323, "Classifier RM"],
    "URM LLaMa 3.1 8B": [0.3969881479982245, 0.07787037973169045, "Classifier RM"],
}


sorted_data = dict(sorted(data.items(), key=lambda item: item[1][0]))
labels_sorted = list(sorted_data.keys())
means_sorted = [v[0] for v in sorted_data.values()]
std_devs_sorted = [v[1] for v in sorted_data.values()]
model_type = [v[2] for v in sorted_data.values()]

df = pd.DataFrame({"means": means_sorted, "std": std_devs_sorted, "model_type": model_type})


plt.figure(figsize=(12, 7))
x_pos_sorted = np.arange(len(labels_sorted))

ax1 = sns.barplot(
    x=df.index,
    y="means",
    data=df,
    errorbar=None,
    hue="model_type",
    hue_order=["Classifier RM", "Generative RM", "Implicit RM"],
    palette=[COLORS.get("green"), COLORS.get("purple"), COLORS.get("orange")],
    # color=COLORS.get("orange"),
    # edgecolor=COLORS.get("green"),
)
plt.errorbar(x_pos_sorted, means_sorted, yerr=std_devs_sorted, fmt="none", c="black", capsize=5)

plt.grid(color="gray", axis="y", alpha=0.2)

plt.ylim(0, 0.8)
plt.gca().set_axisbelow(True)
plt.legend(frameon=False)

plt.xticks(x_pos_sorted, labels_sorted, rotation=45, ha="right")
plt.ylabel("Cohen's Kappa")
plt.title("Average Inner-Model Agreement Across Languages", fontsize=18)

plt.tight_layout()
plt.savefig("plots/innermodel_agreement_green_oracle.pdf", bbox_inches="tight")
