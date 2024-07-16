import argparse
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import random

def translate_batch(texts, model, tokenizer, target_language, device, max_length):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_language], max_length=max_length
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

def validate_columns(dataset, columns):
    for subset in dataset.keys():
        for column in columns:
            if column not in dataset[subset].column_names:
                raise ValueError(f"Column '{column}' not found in subset '{subset}' of the dataset")

def validate_language_code(tokenizer, target_language):
    if target_language not in tokenizer.lang_code_to_id:
        raise ValueError(f"Target language code '{target_language}' is not valid for the given tokenizer")

def translate_dataset(dataset, columns_to_translate, target_language, max_length, batch_size, subset_size=None, output_dir="translations"):
    # Check if GPU is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and tokenizer
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Validate language code
    validate_language_code(tokenizer, target_language)
    
    # Validate columns
    validate_columns(dataset, columns_to_translate)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subset in dataset.keys():
        translated_data = []
        data_length = len(dataset[subset])
        
        # Randomly select a subset of the data if subset_size is specified
        if subset_size:
            indices = random.sample(range(data_length), min(subset_size, data_length))
            dataset[subset] = dataset[subset].select(indices)
        
        num_batches = len(dataset[subset]) // batch_size + (1 if len(dataset[subset]) % batch_size != 0 else 0)
        for i in tqdm(range(num_batches), desc=f'Translating {subset} subset'):
            batch = dataset[subset].select(range(i * batch_size, min((i + 1) * batch_size, len(dataset[subset]))))
            translations = {col: translate_batch(batch[col], model, tokenizer, target_language, device, max_length) for col in columns_to_translate}
            
            for j in range(len(batch[columns_to_translate[0]])):
                translated_example = {col: translations[col][j] for col in columns_to_translate}
                translated_example["target_language"] = target_language
                # Add other columns as-is
                for key in batch.column_names:
                    if key not in translated_example:
                        translated_example[key] = batch[key][j]
                translated_data.append(translated_example)
        
        # Save translated data to JSON file
        args.dataset_name = args.dataset_name.replace("/", "_")
        output_file = os.path.join(output_dir, f"{args.dataset_name}_{subset}_{args.target_language}_translated.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)
        
        print(f"Translated data for subset '{subset}' saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate dataset columns using a specified translation model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name.")
    parser.add_argument("--target_language", type=str, required=True, help="Target language code (e.g., fra_Latn).")
    parser.add_argument("--columns_to_translate", type=str, nargs='+', required=True, help="Columns to translate.")
    parser.add_argument("--max_length", type=int, default=30, help="Maximum length for translation.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for translation.")
    parser.add_argument("--subset_size", type=int, help="Size of the random subset to translate.")
    parser.add_argument("--output_dir", type=str, default="translations", help="Output directory to save translations.")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset_name)
    
    # Translate dataset
    translate_dataset(dataset, args.columns_to_translate, args.target_language, args.max_length, args.batch_size, args.subset_size, args.output_dir)

# Reference: Language and FLORES-200 codes
# ```markdown
# | Language                     | FLORES-200 code |
# |------------------------------|-----------------|
# | Arabic                       | arb_Arab        |
# | Chinese (Simplified)         | zho_Hans        |
# | Chinese (Traditional)        | zho_Hant        |
# | Czech                        | ces_Latn        |
# | Dutch                        | nld_Latn        |
# | English                      | eng_Latn        |
# | French                       | fra_Latn        |
# | German                       | deu_Latn        |
# | Greek                        | ell_Grek        |
# | Hebrew                       | heb_Hebr        |
# | Hindi                        | hin_Deva        |
# | Indonesian                   | ind_Latn        |
# | Italian                      | ita_Latn        |
# | Japanese                     | jpn_Jpan        |
# | Korean                       | kor_Hang        |
# | Persian                      | pes_Arab        |
# | Polish                       | pol_Latn        |
# | Portuguese                   | por_Latn        |
# | Romanian                     | ron_Latn        |
# | Russian                      | rus_Cyrl        |
# | Spanish                      | spa_Latn        |
# | Turkish                      | tur_Latn        |
# | Ukrainian                    | ukr_Cyrl        |
# | Vietnamese                   | vie_Latn        |
# ```

# Example command to run the script:
# python translate_dataset.py --dataset_name my_dataset --target_language fra_Latn --columns_to_translate prompt chosen rejected --max_length 30 --batch_size 8 --subset_size 100 --output_dir translations
