"""
Modified version of https://github.com/allenai/reward-bench/blob/main/rewardbench/rewardbench.py
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

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from rewardbench import DPO_MODEL_CONFIG, REWARD_MODEL_CONFIG
from rewardbench import check_tokenizer_chat_template, load_eval_dataset
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.utils import load_multilingual_eval_dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def torch_dtype_mapping(dtype_str):
    """
    Helper function for argparse to map string to torch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if dtype_str not in dtype_map:
        raise argparse.ArgumentTypeError(f"Invalid torch dtype: {dtype_str}")
    return dtype_map[dtype_str]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a reward model.")

    # fmt: off
    parser.add_argument("--dataset_name", type=str, default="aya-rm-multilingual/multilingual-reward-bench", help="the dataset to evaluate on")
    parser.add_argument("--lang_code", type=str, default=None, help="the language code to use")
    parser.add_argument("--split", type=str, default="filtered", help="the split to evaluate on")
    parser.add_argument("--model", type=str, required=True, help="the model to evaluate")
    parser.add_argument("--ref_model", type=str, default=None, help="the reference model to compare against")
    parser.add_argument("--tokenizer", type=str, default=None, help="the tokenizer to use (defaults to model)")
    parser.add_argument("--chat_template", type=str, default=None, help="the chat template to use (defaults to from tokenizer, from chattemplate)")
    parser.add_argument("--not_quantized", action="store_true", help="disable quantization for models that are quantized by default")
    # inference args
    parser.add_argument("--batch_size", type=int, default=8, help="the batch size to use")
    parser.add_argument("--max_length", type=int, default=512, help="the max length to use")
    # system args
    parser.add_argument("--load_json", action="store_true", default=False, help="load dataset as json")
    parser.add_argument("--trust_remote_code", action="store_true", default=False, help="set trust remote code in HuggingFace to true")
    parser.add_argument("--debug", action="store_true", default=False, help="enable debug mode")
    parser.add_argument("--output_dir", type=str, default="results/", help="the output directory to save results")
    parser.add_argument("--save_all", action="store_true", default=False, help="save all results (include scores per instance)")
    parser.add_argument("--force_truncation", action="store_true", default=False, help="force truncation (if model errors)")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32", "float64"], help="set PyTorch dtype (default: float16)")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Attention implementation to use (default: None)")
    # fmt: on
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)

    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    # basic checks from config
    if args.ref_model:
        is_dpo = True
        MODEL_CONFIGS = DPO_MODEL_CONFIG
        assert args.model != args.ref_model, "policy and reference model should be different"
        from rewardbench import DPOInference
        from trl.trainer.utils import DPODataCollatorWithPadding
    else:
        is_dpo = False
        MODEL_CONFIGS = REWARD_MODEL_CONFIG

    if args.chat_template:
        from fastchat.conversation import get_conv_template

        conv = get_conv_template(args.chat_template)
    else:
        conv = None

    if args.model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model]
    else:
        config = MODEL_CONFIGS["default"]
    logger.info(f"Using reward model config: {config}")

    torch_dtype = config.get("torch_dtype", None)
    if torch_dtype is None:
        # if datatype is bfloat16, then manually turn off quantizaiton (done with bitsandbytes)
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    # Default entries
    # "model_builder": AutoModelForSequenceClassification.from_pretrained,
    # "pipeline_builder": pipeline,
    # "quantized": True,
    # "custom_dialogue": False,
    # "model_type": "Seq. Classifier"

    if not is_dpo:
        quantized = config["quantized"]  # only Starling isn't quantized for now
        # if llama-3 in name, switch quantized to False (severely degrades performance)
        if (
            ("llama-3" in args.model)
            or ("Llama3" in args.model)
            or ("Llama-3" in args.model)
            or ("LLaMA3" in args.model)
            or ("llama3" in args.model)
            or args.not_quantized
        ):
            quantized = False
            logger.info(f"Disabling quantization for llama-3 or override flag (--not_quantized: {args.not_quantized})")
        custom_dialogue = config["custom_dialogue"]
        pipeline_builder = config["pipeline_builder"]
        _ = config["model_type"]
        if custom_dialogue:
            raise NotImplementedError("Custom dialogue not implemented yet for simpler data formatting.")

    model_builder = config["model_builder"]

    #########################
    # load dataset
    #########################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if args.dataset_name == "allenai/reward-bench":
        logger.info("Running core eval dataset.")
        # primary set compiles slightly more information
        dataset, subsets = load_eval_dataset(
            core_set=True,
            conv=conv,
            custom_dialogue_formatting=False,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "prompt"],
        )
    else:
        dataset, subsets = load_multilingual_eval_dataset(
            dataset_name=args.dataset_name,
            conv=conv,
            lang_code=args.lang_code,
            custom_dialogue_formatting=False,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "prompt"],
        )

    if args.debug:
        dataset = dataset.select(range(10))

    logger.info("*** Load reward model ***")

    ############################
    # Load DPO model pipeline
    ############################
    if is_dpo:
        tokenizer.pad_token = tokenizer.eos_token
        # if no BOS token, set as pad token, e.g. QWEN models
        if tokenizer.bos_token is None:
            tokenizer.bos_token_id = tokenizer.eos_token_id
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
        }
        model = model_builder(
            args.model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs,
        )

        # use internal inference functions in DPO trainer
        dpo = DPOInference(
            model,
            ref_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            # norm is norm, avg is average, sum is sum
        )

        # tokenize dataset
        column_names = list(dataset.features)

        tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=args.batch_size,
            collate_fn=DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=dpo.label_pad_token_id,
                is_encoder_decoder=dpo.is_encoder_decoder,
            ),
            # collate_fn = lambda x: x, # fix weird batching error
            shuffle=False,
            drop_last=False,
        )

    ############################
    # Load classifier model pipeline
    ############################
    else:

        # padding experiments for determinism
        tokenizer.padding_side = "left"
        truncation = False
        if args.force_truncation:
            truncation = True
            tokenizer.truncation_side = "left"

        reward_pipeline_kwargs = {
            "batch_size": args.batch_size,  # eval_args.inference_batch_size,
            "truncation": truncation,
            "padding": True,
            "max_length": args.max_length,
            "function_to_apply": "none",  # Compute raw logits
            "return_token_type_ids": False,
        }
        if quantized:
            model_kwargs = {
                "load_in_8bit": True,
                "device_map": {"": current_device},
                "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
            }
        else:
            # note, device map auto does not work for quantized models
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch_dtype,
            }

        # if attn_implementation is not specified, this falls back to Hugging Face's default
        # strategy (which chooses between sdpa and eager depending on pytorch version)
        if args.attn_implementation:
            model_kwargs["attn_implementation"] = args.attn_implementation

        model = model_builder(args.model, **model_kwargs, trust_remote_code=args.trust_remote_code)
        reward_pipe = pipeline_builder(
            "text-classification",  # often not used
            model=model,
            tokenizer=tokenizer,
        )

        # set pad token to eos token if not set
        if reward_pipe.tokenizer.pad_token_id is None:
            reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
            reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
        # For models whose config did not contains `pad_token_id`
        if reward_pipe.model.config.pad_token_id is None:
            reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

        # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
        if not check_tokenizer_chat_template(tokenizer):
            reward_pipe.tokenizer.add_eos_token = True

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = accelerator.prepare(reward_pipe.model)
        reward_pipe.model = model

    ############################
    # Run inference
    ############################

    results = []
    scores_chosen = []
    scores_rejected = []
    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")

        if is_dpo:
            rewards_chosen, rewards_rejected = dpo.inference_step(batch)
        else:
            rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
            rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            score_chosen_batch = [result["score"] for result in rewards_chosen]
            score_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            # Cast to float in case of bfloat16
            score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
            score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

        # log results
        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(score_chosen_batch, score_rejected_batch)
        ]
        scores_chosen.extend(score_chosen_batch)
        scores_rejected.extend(score_rejected_batch)

    ############################
    # compile scores
    ############################
    # calculate accuracy
    accuracy = sum(results) / len(results)
    logger.info(f"Results: {accuracy}, on {len(results)} prompts")

    # compute mean and std of scores, chosen and rejected, then margin between them
    logger.info(f"Mean chosen: {np.mean(scores_chosen)}, std: {np.std(scores_chosen)}")
    logger.info(f"Mean rejected: {np.mean(scores_rejected)}, std: {np.std(scores_rejected)}")
    logger.info(f"Mean margin: {np.mean(np.array(scores_chosen) - np.array(scores_rejected))}")

    if "reward-bench" in args.dataset_name:
        logger.info("Computing grouped results")
        out_dataset = dataset.add_column("results", results)
        if args.debug:
            subsets = subsets[:10]
        out_dataset = out_dataset.add_column("subsets", subsets)
        out_dataset = out_dataset.to_pandas()  # I know this is meh

        results_grouped = {}
        present_subsets = np.unique(out_dataset["subsets"])
        for subset in present_subsets:
            subset_dataset = out_dataset[out_dataset["subsets"] == subset]
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            logger.info(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

        results_section = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        logger.info(f"Results: {results_section}")

    ############################
    # compile scores
    ############################
    # save score in json to args.output_dir + args.model + ".json"
    output_dir = Path(args.output_dir)
    output_path = output_dir / f"{args.model}-{args.lang_code}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # get core dataset
    results_grouped = {}
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset[out_dataset["subsets"] == subset]
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    logger.info(f"Saving to {output_path}")
    with output_path.open("w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "num_prompts": len(results),
                "model": args.model,
                "ref_model": args.ref_model,
                "tokenizer": tokenizer_path,
                "chat_template": args.chat_template,
                "extra_results": results_grouped if "reward-bench" in args.dataset_name else None,
                "subset_results": results_grouped,
            },
            f,
        )

    # if save_all is passed, save a large jsonl with all scores_chosen, scores_rejected
    if args.save_all:
        output_path = output_dir / f"{args.model}-{args.lang_code}-all.jsonl"
        logger.info(f"Saving 'all' results to {output_path}")

        with output_path.open("w") as f:
            for chosen, rejected in zip(scores_chosen, scores_rejected):
                f.write(json.dumps({"chosen": chosen, "rejected": rejected}) + "\n")


if __name__ == "__main__":
    main()
