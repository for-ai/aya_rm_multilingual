import argparse
import json
import os
from google.cloud import translate_v2 as translate
from datasets import load_dataset
from tqdm import tqdm

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path_to_credentials.json>"

def translate_text_batch(texts, client, target_lang_code):
    """
    Translates a batch of texts using Google Translate API.
    """
    result = client.translate(texts, target_language=target_lang_code, format_="text")
    # result = [ {"translatedText": "test"} for txt in texts]
    return [res['translatedText'] for res in result]

def validate_columns(dataset, columns):
    for subset in dataset.keys():
        for column in columns:
            if column not in dataset[subset].column_names:
                raise ValueError(f"Column '{column}' not found in subset '{subset}' of the dataset")

def create_batches(dataset, batch_size):
    """
    Create batches of examples from the dataset.
    """
    batches = []
    current_batch = []

    for example in dataset:
        current_batch.append(example)
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = []

    if current_batch:
        batches.append(current_batch)

    return batches

def load_processed_ids(output_file):
    """
    Load the set of processed example ids from the output file.
    """
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {example["id"] for example in data}
    return set()

def save_translated_batch(translated_batch, output_file):
    """
    Save the translated batch to the output file.
    """
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(translated_batch)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

def translate_subset(dataset, columns_to_translate, target_language, client, translate_prompt_only=False, batch_size=32, output_file=None):
    translated_data = []

    # Load previously processed ids
    processed_ids = load_processed_ids(output_file)

    # Create batches from the dataset
    batches = create_batches(dataset, batch_size)

    for batch in tqdm(batches, desc=f"Translating subset"):
        translated_batch = []

        # Skip batches with examples already processed
        batch_to_process = [example for example in batch if example["id"] not in processed_ids]
        if not batch_to_process:
            print("Skipping current batch as it is already translated")
            continue

        if translate_prompt_only:
            prompts = [example['prompt'] for example in batch_to_process]
            translated_prompts = translate_text_batch(prompts, client, target_language)

            for i, example in enumerate(batch_to_process):
                translated_example = {'prompt': translated_prompts[i]}
                for key in example.keys():
                    if key != 'prompt':
                        translated_example[key] = example[key]
                translated_example["target_language"] = target_language
                translated_batch.append(translated_example)
        else:
            for col in columns_to_translate:
                texts_to_translate = [example[col] for example in batch_to_process]
                translated_texts = translate_text_batch(texts_to_translate, client, target_language)

                for i, example in enumerate(batch_to_process):
                    if i >= len(translated_batch):
                        translated_batch.append({})
                    translated_batch[i][col] = translated_texts[i]

            for i, example in enumerate(batch_to_process):
                for key in example.keys():
                    if key not in translated_batch[i]:
                        translated_batch[i][key] = example[key]
                translated_batch[i]["target_language"] = target_language

        translated_data.extend(translated_batch)

        # Save the translated batch
        save_translated_batch(translated_batch, output_file)

    return translated_data

def translate_dataset(dataset, columns_to_translate, target_language, subset_size=None, output_dir="translations", batch_size=32):
    # Initialize the Google Cloud Translate client
    client = translate.Client()

    # Validate columns
    validate_columns(dataset, columns_to_translate)
    
    dataset = dataset["filtered"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Filter for "hep-" subset and non-"hep-" subset
    code_dataset = dataset.filter(lambda x: x['subset'].startswith('hep-'))
    non_code_dataset = dataset.filter(lambda x: not x['subset'].startswith('hep-'))

    dataset_name = args.dataset_name.replace("/", "_")
    output_file = os.path.join(output_dir, f"{dataset_name}_{args.target_language}_translated.json")

    # Translate non-"hep-" subset: Translate all columns
    non_code_translated = translate_subset(non_code_dataset, columns_to_translate, target_language, client, batch_size=batch_size, output_file=output_file)

    # Translate "hep-" subset: Translate only the 'prompt' column
    code_translated = translate_subset(code_dataset, columns_to_translate, target_language, client, translate_prompt_only=True, batch_size=batch_size, output_file=output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate dataset columns using Google Cloud Translate API.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name.")
    parser.add_argument("--target_language", type=str, required=True, help="Target language code (e.g., fr).")
    parser.add_argument("--columns_to_translate", type=str, nargs="+", required=True, help="Columns to translate.")
    parser.add_argument("--subset_size", type=int, help="Size of the random subset to translate.")
    parser.add_argument("--output_dir", type=str, default="translations", help="Output directory to save translations.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for Google Translate API.")

    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name)

    translate_dataset(
        dataset,
        args.columns_to_translate,
        args.target_language,
        args.subset_size,
        args.output_dir,
        args.batch_size,
    )
