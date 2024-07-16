"""Convert multilingual ultrafeedback into a format acceptable for RewardBench

We need to follow the load_preference_dataset setup in RewardBench as
shown here: https://github.com/allenai/reward-bench/blob/main/rewardbench/utils.py#L136
So we need three columns:
    - prompt (str)
    - chosen (list[dict[str, str]]), and
    - rejected (list[dict[str, str]])
    

** Translation: 2000/2000 [2:36:00<00:00,  4.68s/ examples]
"""


import argparse
import logging
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import ctranslate2
import transformers

import unicodedata

logging.basicConfig(level=logging.INFO)


def get_args():
  parser = argparse.ArgumentParser(
      description="Translation a HuggingFace dataset into the RewardBench format."
  )

  parser.add_argument("--dataset", type=str, default="nthakur/multilingual-ultrafeedback-dpo-v0.1", help="Dataset to convert.")
  # parser.add_argument("--output_path", type=Path, default="data/multilingual-ultrafeedback-dpo-v0.1-test-ben_Beng.json", help="Path to save converted dataset as JSON file.")
  # fmt: on
  parser.add_argument("--target", type=str, help="Target-lang")

  return parser.parse_args()


def main():
  args = get_args()
  
  # model_id = "facebook/nllb-moe-54b"
  model_id = "facebook/nllb-200-3.3B"
  src_lang = "eng_Latn"

  # tgt_lang = "fra_Latn"
  # tgt_lang = "spa_Latn"
  # tgt_lang = "ben_Beng"
  tgt_lang = args.target

  output_path = Path(f"data/multilingual-ultrafeedback-dpo-v0.1-test-{tgt_lang}.json")
  
  # ct2-transformers-converter \
	# --model facebook/nllb-200-3.3B --output_dir facebook-nllb-200-3.3B

  model_dir = f"./nllb/{model_id.replace('/', '-')}/"

  translator = ctranslate2.Translator(model_dir, device='cuda')
  tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, src_lang=src_lang)

  target_prefix = [tgt_lang]

  def translate(source, unicode_norm='NFKC'):
    # batched_input = [source]
    batched_input = source.split("\n")
    tokenized_input = tokenizer(batched_input, return_attention_mask=False).input_ids
    source = [tokenizer.convert_ids_to_tokens(x) for x in tokenized_input]
    results = translator.translate_batch(source, target_prefix=[target_prefix]*len(batched_input))
    target = [result.hypotheses[0][1:] for result in results]
    target = [tokenizer.convert_tokens_to_ids(x) for x in target]
    translated = tokenizer.batch_decode(target)
    
    translated = [x.replace("\n", "") for x in translated]
    translated = "\n".join(translated)
    translated = unicodedata.normalize(unicode_norm, translated)
    # translated = " ".join(translated.splitlines())
    # import ipdb; ipdb.set_trace()
    return translated

  
  if output_path:
    output_path.parents[0].mkdir(parents=True, exist_ok=True)

  dataset = load_dataset(args.dataset, split="test")

  # dataset = dataset.train_test_split(test_size=2.0/2000)['test']
  
  def _convert_to_turn_based(example):
    input = translate(example['en_input'])
    print(f"{src_lang}: {example['en_input']}\n{tgt_lang}: {input}")
    chosen = translate(example['en_chosen'])
    rejected = translate(example['en_rejected'])
    # import ipdb; ipdb.set_trace()
    
    example['language'] = tgt_lang
    example['prompt'] = input
    
    example["chosen"] = [
      {"content": example["prompt"], "role": "user"},
      {"content": chosen, "role": "assistant"},
    ]
    example["rejected"] = [
      {"content": example["prompt"], "role": "user"},
      {"content": rejected, "role": "assistant"},
    ]
    return example

  # cols = ["id", "source", "language", "input", "chosen", "rejected"]
  rename_map = {"input": "prompt", "chosen": "chosen_raw", "rejected": "rejected_raw"}
  cols = ["id", "source", "language", "input", "chosen", "rejected",
    "en_input", "en_chosen", "en_rejected",
  ]
  remove_cols = ["chosen_raw", "rejected_raw", "en_input", "en_chosen", "en_rejected"]
  
  dataset = (
    dataset.select_columns(cols)
    .rename_columns(rename_map)
    .map(_convert_to_turn_based)
    .remove_columns(remove_cols)
  )
  dataset.to_json(output_path)
  logging.info(f"Saved file to {output_path}.")


if __name__ == "__main__":
    main()
    
