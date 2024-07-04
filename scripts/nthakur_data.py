import logging
from datasets import load_dataset, DatasetDict
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_dataset(dataset):
    def transform(example):
        # Check if 'chosen' or 'en_chosen' exists, and use the appropriate one
        if 'chosen' in example and isinstance(example['chosen'], str):
            chosen_content = example['chosen']
        elif 'en_chosen' in example:
            chosen_content = example['en_chosen']
        else:
            chosen_content = "No chosen content available"
        
        # Do the same for 'rejected' or 'en_rejected'
        if 'rejected' in example and isinstance(example['rejected'], str):
            rejected_content = example['rejected']
        elif 'en_rejected' in example:
            rejected_content = example['en_rejected']
        else:
            rejected_content = "No rejected content available"
        
        # Set 'chosen' and 'rejected' as lists of dictionaries with 'content' keys
        example['chosen'] = [{'content': chosen_content}]
        example['rejected'] = [{'content': rejected_content}]
        
        # Rename 'input' to 'prompt'
        prompt_content = example.get('input', "No prompt available")
        
        return {
            'prompt': prompt_content,
            'chosen': example['chosen'],
            'rejected': example['rejected']
        }

    for split in dataset:
        # Print column names and first two rows before transformation
        columns = dataset[split].column_names
        first_two_rows = [dataset[split][i] for i in range(min(2, len(dataset[split])))]
        logger.info(f"Columns: {columns}")
        logger.info(f"First two rows before transformation ({split}): {first_two_rows}")

        dataset[split] = dataset[split].map(transform, num_proc=8)

        # Print first two rows after transformation
        first_two_rows_transformed = [dataset[split][i] for i in range(min(2, len(dataset[split])))]
        logger.info(f"First two rows after transformation ({split}): {first_two_rows_transformed}")

    return dataset

# Load the dataset
dataset_name = 'nthakur/multilingual-ultrafeedback-dpo-v0.1'
dataset = load_dataset(dataset_name)

# Preprocess the dataset
dataset = preprocess_dataset(dataset)

# Save the processed dataset
processed_data_path = os.path.join(os.path.dirname(__file__), "preprocessed_dataset_extracted")
dataset.save_to_disk(processed_data_path)

# Tokenize and save shards
def save_tokenized_shards(dataset, shard_size=10000):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    def tokenize_function(example):
        return tokenizer(example['prompt'], truncation=True, padding='max_length', max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["id", "en_chosen", "en_rejected", "en_input", "source", "language"])

    for split in tokenized_datasets:
        split_dir = os.path.join(os.path.dirname(__file__), f"tokenized_shards/{split}")
        os.makedirs(split_dir, exist_ok=True)
        num_shards = len(tokenized_datasets[split]) // shard_size + 1
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = (shard_idx + 1) * shard_size
            shard = tokenized_datasets[split].select(range(start_idx, min(end_idx, len(tokenized_datasets[split]))))
            shard.save_to_disk(os.path.join(split_dir, f"shard_{shard_idx}.arrow"))
            logger.info(f"Saved shard {shard_idx} for split {split}")

save_tokenized_shards(dataset)

