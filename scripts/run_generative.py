


import torch
import argparse
import logging
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock

import numpy as np
from datasets import load_dataset
from fastchat.conversation import get_conv_template
from rewardbench.generative import ANTHROPIC_MODEL_LIST, API_MODEL_LIST
from rewardbench.generative import GEMINI_MODEL_LIST, OPENAI_MODEL_LIST
from rewardbench.generative import format_judge_answers, process_judgement
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Define your Cohere model list
COHERE_MODEL_LIST = ["command-r-plus", "command-r", "command"]

# Get tokens from environment variables
HF_TOKEN = os.getenv("HF_TOKEN", None)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", None)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", None)

# Check if at least one token is set
if not HF_TOKEN:
    raise ValueError("Missing value for HF_TOKEN environment variable")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set, will use Hugging Face models if applicable.")

if not COHERE_API_KEY:
    print("Warning: COHERE_API_KEY not set, will use Hugging Face models if applicable.")

def get_args():
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

def mock_cohere_api_call(*args, **kwargs):
    # This function will mock the response from the Cohere API
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'id': 'mock-id',
        'generations': [
            {'text': 'Mock response from Cohere API'}
        ]
    }
    return mock_response

def generate_huggingface(model_name, tokenizer, prompts):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        output = model.generate(**inputs, max_length=1024)
        outputs.append(tokenizer.decode(output[0], skip_special_tokens=True))
    return outputs

def generate_vllm(model_name, prompts, num_gpus):
    model = LLM(
        model_name,
        trust_remote_code=True,
        tensor_parallel_size=num_gpus,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "Llama-3" in model_name or "llama3-8b" in model_name:
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

    outputs = []
    for prompt in prompts:
        tokenized_prompt = tokenizer(prompt, return_tensors="pt").input_ids
        result = model.generate(tokenized_prompt, sampling_params=sampling_params)
        outputs.append(result[0].outputs[0].text)
    return outputs

def run_judge_pair(prompt, answer_a, answer_b, model_name, multi_turn, model_modifier):
    # Mock implementation of run_judge_pair
    request = f"{prompt}\nAnswer A: {answer_a}\nAnswer B: {answer_b}"
    judgement = {"winner": "A" if len(answer_a) > len(answer_b) else "B"}
    return judgement["winner"], request, judgement

def main():
    args = get_args()
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

    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        model_type += " PoLL"
        assert len(args.model) % 2 == 1

    is_api_models = (
        isinstance(args.model, list)
        or args.model in API_MODEL_LIST
        or args.model in COHERE_MODEL_LIST
        or not args.force_local
    )

    logger.info("*** Load dataset ***")
    dataset = load_dataset(args.dataset_name, split=args.split)
    dataset = dataset.rename_columns(
        {"chosen": "text_chosen", "rejected": "text_rejected"}
    )

    if args.sample:
        logger.debug(f"Running on first {args.sample} examples")
        dataset = dataset.select(range(args.sample))

    # Extract prompts from the dataset
    prompts = dataset['prompt']

    if not is_api_models:
        model_name = args.model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        outputs = generate_huggingface(model_name, tokenizer, prompts)
        for i, output in enumerate(outputs):
            print(f"Output {i + 1}: {output}")
        return  # Exiting after testing Hugging Face models with actual dataset prompts

    is_prometheus = False
    if "prometheus" in args.model:
        model_modifier = "prometheus"
        is_prometheus = True
    elif "gemini" in args.model:
        model_modifier = "gemini"
    else:
        model_modifier = None

    if is_api_models:
        def update_progress_bar(done, total):
            progress = int(50 * done / total)
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

            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if len(batch["text_chosen"]) <= 4:
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

                if isinstance(winner, list):
                    if debug:
                        print(winner)
                    winner = max(set(winner), key=winner.count)

                if winner == winner_text:
                    return 1
                elif winner == loser_text:
                    return 0
                else:
                    return 0.5

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            results = [None] * len(dataset)
            done_tasks = 0

            future_to_index = {
                executor.submit(get_judgement, x): i for i, x in enumerate(dataset)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                done_tasks += 1
                update_progress_bar(done_tasks, len(dataset))

            print()
    else:
        def format_judgements(batch, optional_chat_template=None):
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["prompt"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

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

        chat_template = (
            get_conv_template(args.chat_template)
            if args.chat_template is not None
            else None
        )
        dataset_prompts = dataset.map(
            format_judgements, fn_kwargs={"optional_chat_template": chat_template}
        )

        prompts = dataset_prompts["text"]
        is_shuffled = dataset_prompts["is_shuffled"]

        logger.info("*** Run inference ***")
        if args.model in COHERE_MODEL_LIST:
            outputs = [mock_cohere_api_call(prompt).json()['generations'][0]['text'] for prompt in prompts]
        else:
            outputs = generate_vllm(args.model, prompts, args.num_gpus)
        logger.info("*** Inference done ***")

        winners = [process_judgement(a, is_prometheus=is_prometheus) for a in outputs]

        def process_shuffled(win, shuffle):
            winner_text, loser_text = "B", "A" if shuffle else "A", "B"
            if win == winner_text:
                return 1
            elif win == loser_text:
                return 0
            else:
                return 0.5

        results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]

    out_dataset = dataset.add_column("results", results)

    if isinstance(args.model, list):
        model_name = "_".join(args.model)
        model_name = "PoLL/" + model_name
    else:
        model_name = args.model

    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name
    elif args.model in COHERE_MODEL_LIST:
        model_name = "cohere/" + model_name

    num_correct = sum(out_dataset["results"])
    num_total = len(out_dataset["results"])
    print(f"{args.dataset_name}: {num_correct}/{num_total} ({num_correct/num_total})")

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







