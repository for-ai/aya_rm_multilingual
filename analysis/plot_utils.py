import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)


PLOT_PARAMS = {
    "text.usetex": False,
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
        elif result.get("ref_model"):
            # Most likely DPO:
            category_scores = _compute_category_scores(result["extra_results"])
            model_scores.append(
                {
                    "model": result["model"],
                    "model_type": "DPO",
                    "chat_template": result["chat_template"],
                    "score": sum(category_scores.values()) / len(category_scores),
                    "category_scores": category_scores,
                    "subset_scores": result["extra_results"],
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


EXAMPLE_COUNTS = {
    "alpacaeval-easy": 79,
    "alpacaeval-length": 79,
    "alpacaeval-hard": 76,
    "mt-bench-easy": 24,
    "mt-bench-med": 38,
    "mt-bench-hard": 35,
    "math-prm": 983,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 76,
    "llmbar-adver-neighbor": 124,
    "llmbar-adver-GPTInst": 87,
    "llmbar-adver-GPTOut": 42,
    "llmbar-adver-manual": 43,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 247,
    "donotanswer": 135,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 163,
    "hep-rust": 164,
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

SUBSET_NAME_TO_PAPER_READY = {
    "alpacaeval-easy": "AlpacaEval Easy",
    "alpacaeval-length": "AlpacaEval Length",
    "alpacaeval-hard": "AlpacaEval Hard",
    "mt-bench-easy": "MT Bench Easy",
    "mt-bench-med": "MT Bench Medium",
    "mt-bench-hard": "MT Bench Hard",
    "llmbar-natural": "LLMBar Natural",
    "llmbar-adver-neighbor": "LLMBar Adver. Neighbor",
    "llmbar-adver-GPTInst": "LLMBar Adver. GPTInst",
    "llmbar-adver-GPTOut": "LLMBar Adver. GPTOut",
    "llmbar-adver-manual": "LLMBar Adver. Manual",
    "refusals-dangerous": "Refusals Dangerous",
    "refusals-offensive": "Refusals Offensive",
    "xstest-should-refuse": "XSTest Should Refuse",
    "xstest-should-respond": "XSTest Should Respond",
    "donotanswer": "Do Not Answer",
    "math-prm": "PRM Math",
    "hep-cpp": "HumanEvalPack CPP",
    "hep-go": "HumanEvalPack Go",
    "hep-java": "HumanEvalPack Java",
    "hep-js": "HumanEvalPack Javascript",
    "hep-python": "HumanEvalPack Python",
    "hep-rust": "HumanEvalPack Rust",
    "anthropic_harmless": "Anthropic Harmless",
    "anthropic_helpful": "Anthropic Helpful",
    "anthropic_hhh": "Anthropic HHH",
    "mtbench_gpt4": "MT Bench GPT-4",
    "mtbench_human": "MT Bench Human",
    "shp": "SHP",
    "summarize": "Summarize",
}
