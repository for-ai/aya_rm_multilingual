import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = {
  "meta-llama/Meta-Llama-3.1-8B-Instruct": [
    0.3533086666014079,
    0.052422082615756406
  ],
  "cohere/c4ai-aya-23-35b": [
    0.43767196047824003,
    0.026040919354464294
  ],
  "cohere/c4ai-aya-23-8b": [
    0.013483014909052663,
    0.03363706833599835
  ],
  "cohere/command-r-08-2024": [
    0.374457668650282,
    0.02926089754079793
  ],
  "cohere/command-r-plus-08-2024": [
    0.3830841816733316,
    0.020185255968455686
  ],
  "google/gemma-1.1-7b-it": [
    0.5190375637539242,
    0.027757722654111305
  ],
  "google/gemma-2-9b-it": [
    0.5181663123111222,
    0.031090119385244894
  ],
  "meta-llama/Meta-Llama-3-70B-Instruct": [
    0.5685224105896568,
    0.04853344616275034
  ],
  "meta-llama/Meta-Llama-3-8B-Instruct": [
    0.37936948540837095,
    0.032172769265151994
  ],
  "meta-llama/Meta-Llama-3.1-70B-Instruct": [
    0.603536768244583,
    0.027191895488989915
  ],
  "mistralai/Mistral-7B-Instruct-v0.2": [
    0.4071166722276529,
    0.04577594028555328
  ],
  "mistralai/Mistral-7B-Instruct-v0.3": [
    0.41195018984687265,
    0.056184679972755454
  ],
  "openai/gpt-4-turbo-2024-04-09": [
    0.6106943361444249,
    0.02932446842558468
  ],
  "openai/gpt-4o-2024-05-13": [
    0.5833874065757011,
    0.023695391445384514
  ]
}

sorted_data = dict(sorted(data.items(), key=lambda item: item[1][0]))
labels_sorted = list(sorted_data.keys())
means_sorted = [v[0] for v in sorted_data.values()]
std_devs_sorted = [v[1] for v in sorted_data.values()]

sns.set(style="whitegrid")
palette = sns.color_palette("coolwarm", len(labels_sorted))

plt.figure(figsize=(10, 6))
x_pos_sorted = np.arange(len(labels_sorted))

ax1 = sns.barplot(x=x_pos_sorted, y=means_sorted, palette=palette, errorbar=None)
plt.errorbar(x_pos_sorted, means_sorted, yerr=std_devs_sorted, fmt='none', c='black', capsize=5)

ax1.spines['top'].set_color('black')
ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_color('black')
for spine in ax1.spines.values():
    spine.set_linewidth(2)  # Make the border thicker

plt.ylim(0, 0.8)

plt.xticks(x_pos_sorted, labels_sorted, rotation=90)
plt.ylabel("Cohen's Kappa")
plt.title('Average Inner-Model Agreement Across Languages')

plt.tight_layout()
plt.savefig(f"./innermodel_agreement.pdf", bbox_inches='tight')