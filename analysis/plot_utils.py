import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING

logging.basicConfig(level=logging.INFO)


PLOT_PARAMS = {
    "text.usetex": True,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
}


def get_scores(lang_dir: Path) -> List[Dict[str, Any]]:
    """Get scores for a single language, returns the category scores and the per-subset scores per model"""
    files = [file for file in lang_dir.iterdir() if file.suffix == ".json"]
    logging.debug(f"Found {len(files)} model results for {lang_dir.name}")

    def _compute_category_scores(results: Dict[str, float]) -> Dict[str, float]:
        """Weighted average of each dataset"""
        category_scores = {}
        for category, subsets in SUBSET_MAPPING.items():
            subset_results = [results[subset] for subset in subsets]
            subset_lengths = [EXAMPLE_COUNTS[subset] for subset in subsets]
            wt_avg = sum(v * w for v, w in zip(subset_results, subset_lengths)) / sum(subset_lengths)
            category_scores[category] = wt_avg
        return category_scores

    model_scores = []
    for file in files:
        with open(file, "r") as f:
            result = json.load(f)
        # The Generative and Clasifier RMs have different JSON schemas
        # so we need to handle them separately
        if "subset" in result:
            # Most likely generative
            model_scores.append(
                {
                    "model": result["subset"].pop("model"),
                    "model_type": result["subset"].pop("model_type"),
                    "chat_template": (
                        result["subset"].pop("chat_template") if "chat_template" in result["subset"] else None
                    ),
                    # The rewardbench score is the average of the weighted average of the four category scores
                    "score": sum(result["leaderboard"].values()) / len(result["leaderboard"]),
                    "category_scores": result["leaderboard"],
                    "subset_scores": result["subset"],
                }
            )
        else:
            category_scores = _compute_category_scores(result["extra_results"])
            model_scores.append(
                {
                    "model": result["model"],
                    "model_type": "Sequence Classifier",
                    "chat_template": result["chat_template"],
                    "score": sum(category_scores.values()) / len(category_scores),
                    "category_scores": category_scores,
                    "subset_scores": result["extra_results"],
                }
            )
    return model_scores
