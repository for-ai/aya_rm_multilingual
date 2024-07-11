"""
Modified version of https://github.com/allenai/reward-bench/blob/045c7f8291f804d193bb102f590fd9db8d52cec3/scripts/run_generative.py
Updated to accommodate custom preference datasets
"""

# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run a generative RM. For now, this requires openai and anthropic to be installed
# Examples:
# python scripts/run_generative.py --dataset_name <DATASET_NAME> --model gpt-3.5-turbo
# python scripts/run_generative.py --dataset_name <DATASET_NAME> --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset
from fastchat.conversation import get_conv_template
from rewardbench.generative import ANTHROPIC_MODEL_LIST, API_MODEL_LIST
from rewardbench.generative import GEMINI_MODEL_LIST, OPENAI_MODEL_LIST
from rewardbench.generative import format_judge_answers, process_judgement
from rewardbench.generative import run_judge_pair
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
if not HF_TOKEN:
    raise ValueError("Missing value for HF_TOKEN environment variable")


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset to test on")
    parser.add_argument("--split", default="test", type=str, required=True, help="dataset split to evaluate")
    parser.add_argument("--model", type=str, nargs="+", required=True, help="name of model to use")
    parser.add_argument("--chat_template", type=str, default=None, help="fastchat chat template (optional)")
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--debug", action="store_true", help="run on debug mode (show additional info, etc.)")
    parser.add_argument("--sample", type=int, default=None, help="sample a few instances for testing")
    parser.add_argument("--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples")
    parser.add_argument("--force_local", action="store_true", default=False, help="force local run, even if model is on Together API")
    # fmt: on
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(
        f"Running reward model on {args.model} with chat template {args.chat_template}"
    )

    model_type = "Generative RM"

    # if model is list, make type + PoLL and check multiple is odd
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        model_type += " PoLL"
        # assert that is odd and > 1
        assert len(args.model) % 2 == 1

    # define variable if is API or local
    is_api_models = (
        isinstance(args.model, list)
        or args.model in API_MODEL_LIST
        or not args.force_local
    )

    # if model isn't API, load via vllm
    if not is_api_models:
        # load model
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "Llama-3" in args.model or "llama3-8b" in args.model:
            stop_token_ids = [128009]
        else:
            stop_token_ids = []

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1,
            max_tokens=1024,
            stop_token_ids=stop_token_ids,
        )

    # handle off-case models
    is_prometheus = False  # handles output tokens differently (less flexible)
    # use different prompt for prometheus/gemini models
    if "prometheus" in args.model:
        model_modifier = "prometheus"
        is_prometheus = True
    elif "gemini" in args.model:
        model_modifier = "gemini"
    else:
        model_modifier = None

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    dataset = load_dataset(args.dataset_name, split=args.split)
    # Rename columns for compatibility with existing API
    dataset = dataset.rename_columns(
        {"chosen": "text_chosen", "rejected": "text_rejected"}
    )

    if args.sample:
        logger.debug(f"Running on first {args.sample} examples")
        dataset = dataset.select(range(args.sample))

    if is_api_models:
        ############################
        # Run inference via API
        ############################
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write(
                "\r[{}{}] {}/{}".format(
                    "#" * progress, "." * (50 - progress), done, total
                )
            )
            sys.stdout.flush()

        def get_judgement(batch, debug=args.debug):
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["prompt"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
                winner, request, judgement = run_judge_pair(
                    prompt,
                    answer_a,
                    answer_b,
                    args.model,
                    multi_turn=mult_turn,
                    model_modifier=model_modifier,
                )
                if debug:
                    print(f"Prompt: {request}")
                    print(f"Judgement: {judgement}")

                # handle voting
                if isinstance(winner, list):
                    # print votes if debug
                    if debug:
                        print(winner)
                    winner = max(set(winner), key=winner.count)

                if winner == winner_text:
                    return 1
                elif winner == loser_text:
                    return 0
                else:  # if "error"
                    return 0.5  # effectively a tie

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads
            # results = list(executor.map(get_judgement, dataset))

            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {
                    executor.submit(get_judgement, x): i for i, x in enumerate(dataset)
                }

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()
    else:
        ############################
        # Run model weights with vllm
        ############################

        def format_judgements(batch, optional_chat_template=None):
            # TODO expand this to include fastchat chat templates if needed
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["prompt"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a

            system_prompt, user_prompt = format_judge_answers(
                prompt,
                answer_a,
                answer_b,
                multi_turn=mult_turn,
                model_modifier=model_modifier,
            )

            if optional_chat_template is not None:
                optional_chat_template.set_system_message(system_prompt)
                optional_chat_template.messages = []
                optional_chat_template.append_message(
                    optional_chat_template.roles[0], user_prompt
                )
                optional_chat_template.append_message(
                    optional_chat_template.roles[1], None
                )
                prompt = optional_chat_template.get_prompt()
            elif model_modifier:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            batch["text"] = prompt
            batch["is_shuffled"] = is_shuffled
            return batch

        # format the dataset for the model, with optional fastchat templating
        chat_template = (
            get_conv_template(args.chat_template)
            if args.chat_template is not None
            else None
        )
        dataset_prompts = dataset.map(
            format_judgements, fn_kwargs={"optional_chat_template": chat_template}
        )

        # collect texts of dataset in list
        prompts = dataset_prompts["text"]
        is_shuffled = dataset_prompts["is_shuffled"]

        # generate
        logger.info("*** Run inference ***")
        outputs = model.generate(prompts, sampling_params)
        logger.info("*** Inference done ***")

        answers = [o.outputs[0].text for o in outputs]
        winners = [process_judgement(a, is_prometheus=is_prometheus) for a in answers]

        def process_shuffled(win, shuffle):
            winner_text, loser_text = "B", "A" if shuffle else "A", "B"
            if win == winner_text:
                return 1
            elif win == loser_text:
                return 0
            else:  # if "error"
                return 0.5  # effectively a tie

        results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # model name concat if list
    if isinstance(args.model, list):
        model_name = "_".join(args.model)
        model_name = "PoLL/" + model_name
    else:
        model_name = args.model
    # if model in openai or Anthropic list, append org to model name
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name

    # compute scores
    num_correct = sum(out_dataset["results"])
    num_total = len(out_dataset["results"])
    print(f"{args.dataset_name}: {num_correct}/{num_total} ({num_correct/num_total})")

    # save results
    results_dict = {
        "dataset": args.dataset_name,
        "model": model_name,
        "chat_template": args.chat_template,
        "scores": {
            "accuracy": num_correct / num_total,
            "num_correct": num_correct,
            "num_total": num_total,
            "results": results,
        },
    }

    file_path = f"{model_name.replace('/', '___')}.json"
    with open(file_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    logger.info(f"Saved results to {file_path}")


if __name__ == "__main__":
    main()
