from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

# annotations = Path("data/hin_Deva_histogram.csv")
annotations = Path("plots/ind_Latn_histogram.csv")
reference = Path("plots/eng_Latn_histogram.csv")

annot_df = pd.read_csv(annotations).set_index("model").T
ref_df = pd.read_csv(reference).set_index("model").T

cohen_scores: dict[str, float] = {}
for (idx, annot), (_, ref) in zip(annot_df.iterrows(), ref_df.iterrows()):
    cohen_scores[idx] = cohen_kappa_score(annot.to_list(), ref.to_list())

iaa_df = pd.DataFrame([cohen_scores]).T.rename(columns={0: "cohen"})
iaa_df.hist(bins=8)
plt.show()
