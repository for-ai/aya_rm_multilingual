import argparse
import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from analysis.plot_utils import get_scores


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Get evaluation results")
    parser.add_argument("--dataset", type=str, default="aya-rm-multilingual/eval-results", help="HuggingFace dataset that stores the eval results.")
    parser.add_argument("--langs", nargs="*", required=False, type=str, help="If set, will only show the results for the particular language codes provided.")
    parser.add_argument("--show_subsets", action="store_true", help="If set, will show subset results instead of per-category results.")
    parser.add_argument("--force_download", action="store_true", help="If set, will redownload the dataset.")
    # fmt: on
    return parser.parse_args()


def main():
    args = get_args()
    dataset_dir = Path(snapshot_download(args.dataset, repo_type="dataset", force_download=args.force_download))
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


if __name__ == "__main__":
    main()
