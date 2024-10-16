import pandas as pd
from pathlib import Path
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from itertools import combinations

file = Path("data/hin_Deva_histogram.csv")
file = Path("data/ind_Latn_histogram.csv")
df = pd.read_csv(file)
df = df.set_index("model")
df = df.T
data = df.values


def calculate_fleiss_kappa(row: pd.Series) -> float:
    categories_count = np.bincount(row, minlength=3)
    breakpoint()
    return fleiss_kappa(categories_count[np.newaxis, :], method="fleiss")


def fleiss_kappa_per_instance(ratings, n_categories=3):
    """
    Calculate Fleiss' Kappa for a single instance
    ratings: array of ratings for one instance
    n_categories: number of possible categories (0, 1, 2 in this case)
    """
    n_raters = len(ratings)

    n_i = np.zeros(n_categories)
    for rating in ratings:
        n_i[rating] += 1

    P_j = n_i / (n_raters)
    P_i = (np.sum(n_i * (n_i - 1))) / (n_raters * (n_raters - 1))
    Pe = np.sum(P_j**2)
    if Pe == 1:  # Handle edge case
        return 1.0
    kappa = (P_i - Pe) / (1 - Pe)
    return kappa


def compute_percentage_agreement(annotations):
    """
    Compute percentage agreement for a single instance across all annotators
    annotations: array of annotations (0, 1, or 2) for one instance
    """
    n_annotators = len(annotations)
    n_agreements = 0
    n_pairs = 0

    # Compare each pair of annotators
    for a1, a2 in combinations(annotations, 2):
        if a1 == a2:
            n_agreements += 1
        n_pairs += 1

    return n_agreements / n_pairs if n_pairs > 0 else 0


def analyze_annotations(data):
    """
    Compute agreement scores for each instance
    data: numpy array of shape (n_instances, n_annotators)
    """
    n_instances = len(data)
    agreement_scores = []

    for i in range(n_instances):
        instance_annotations = data[i]
        agreement = compute_percentage_agreement(instance_annotations)
        agreement_scores.append(agreement)

    return agreement_scores


scores = analyze_annotations(data)
df["agreement"] = scores

print(df["agreement"].value_counts().to_markdown())


import matplotlib.pyplot as plt

# Create a histogram
plt.hist(scores, bins=10, alpha=0.7, color="b")
plt.xlabel("Agreement Score")
plt.ylabel("Frequency")
plt.title("Agreement Score Histogram")
plt.show()
