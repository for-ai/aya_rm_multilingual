import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from huggingface_hub import snapshot_download
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING

logging.basicConfig(level=logging.INFO)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Get evaluation results")
    parser.add_argument("--dataset", type=str, default="aya-rm-multilingual/eval-results", help="HuggingFace dataset that stores the eval results.")
    parser.add_argument("--langs", nargs="*", required=False, type=str, help="If set, will only show the results for the particular language codes provided.")
    parser.add_argument("--show_subsets", action="store_true", help="If set, will show subset results instead of per-category results.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    dataset_dir = Path(snapshot_download(args.dataset, repo_type="dataset"))
    lang_folders = [d for d in dataset_dir.iterdir() if d.is_dir()]

    if args.langs:
        logging.info(f"Only showing detailed results for the ff languages: {','.join(args.langs)}")
        for lang_dir in lang_folders:
            if lang_dir.name in args.langs:
                model_scores = get_scores(lang_dir)
                df = pd.DataFrame(model_scores)
                metadata_df = df[["model", "model_type", "score"]]
                key = "subset_scores" if args.show_subsets else "category_scores"
                scores_df = pd.DataFrame(df[key].tolist())
                lang_scores_df = pd.concat([metadata_df, scores_df], axis=1).sort_values(by="score", ascending=False)
                print(f"\n*** Results for {lang_dir.name} ***\n")
                print(lang_scores_df.to_markdown(tablefmt="github", index=False))

    else:
        logging.info("Showing m-rewardbench scores for all languages")
        lang_scores = {}
        for lang_dir in lang_folders:
            model_scores = get_scores(lang_dir)
            lang_scores[lang_dir.name] = {score["model"]: score["score"] for score in model_scores}

        lang_scores_df = pd.DataFrame(lang_scores)
        print(lang_scores_df.to_markdown(tablefmt="github"))


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
                    "chat_template": result["subset"].pop("chat_template"),
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


if __name__ == "__main__":
    main()
