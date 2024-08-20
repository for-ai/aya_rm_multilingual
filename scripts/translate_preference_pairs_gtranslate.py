import argparse
import json
import os
import random
from google.cloud import translate_v2 as translate
from datasets import load_dataset
from tqdm import tqdm

# Steps to setup:
# 1. https://cloud.google.com/python/docs/setup#linux
# 2. https://cloud.google.com/sdk/docs/install

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path_to_credentials_json>"

def translate_texts(texts, client, target_lang_code):
    results = client.translate(texts, target_language=target_lang_code)
    return [result['translatedText'] for result in results]


def validate_columns(dataset, columns):
    for subset in dataset.keys():
        for column in columns:
            if column not in dataset[subset].column_names:
                raise ValueError(f"Column '{column}' not found in subset '{subset}' of the dataset")


def translate_dataset(
    dataset, columns_to_translate, target_language, subset_size=None, output_dir="translations", batch_size=10
):
    # Initialize the Google Cloud Translate client
    client = translate.Client()

    # Validate columns
    validate_columns(dataset, columns_to_translate)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subset in dataset.keys():
        if subset == "raw":
            continue
        translated_data = []
        data_length = len(dataset[subset])

        # Randomly select a subset of the data if subset_size is specified
        if subset_size:
            indices = random.sample(range(data_length), min(subset_size, data_length))
            dataset[subset] = dataset[subset].select(indices)

        for start_idx in tqdm(range(0, data_length, batch_size), desc=f"Translating {subset} subset"):
            end_idx = min(start_idx + batch_size, data_length)
            batch = dataset[subset].select(range(start_idx, end_idx))

            # Initialize a dictionary to hold the translated batch
            translated_batch = {col: [] for col in columns_to_translate}

            for col in columns_to_translate:
                # Translate each column in the batch
                texts_to_translate = batch[col]
                translated_texts = translate_texts(texts_to_translate, client, target_language)
                translated_batch[col] = translated_texts

            # Add other columns as-is
            other_columns = {key: batch[key] for key in batch.column_names if key not in translated_batch}
            
            # Combine translated and other columns into a list of examples
            for i in range(len(translated_batch[columns_to_translate[0]])):
                translated_example = {col: translated_batch[col][i] for col in columns_to_translate}
                translated_example["target_language"] = target_language
                for key in other_columns:
                    translated_example[key] = other_columns[key][i]
                translated_data.append(translated_example)

        # Save translated data to JSON file
        dataset_name = args.dataset_name.replace("/", "_")
        output_file = os.path.join(output_dir, f"{dataset_name}_{subset}_{args.target_language}_translated.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)

        print(f"Translated data for subset '{subset}' saved to {output_file}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Translate dataset columns using Google Cloud Translate API.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name.")
    parser.add_argument("--target_language", type=str, required=True, help="Target language code (e.g., fr).")
    parser.add_argument("--columns_to_translate", type=str, nargs="+", required=True, help="Columns to translate.")
    parser.add_argument("--subset_size", type=int, help="Size of the random subset to translate.")
    parser.add_argument("--output_dir", type=str, default="translations", help="Output directory to save translations.")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of texts to translate in each batch.")
    # fmt: on

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_name)

    # Translate dataset
    translate_dataset(
        dataset,
        args.columns_to_translate,
        args.target_language,
        args.subset_size,
        args.output_dir,
        args.batch_size
    )
