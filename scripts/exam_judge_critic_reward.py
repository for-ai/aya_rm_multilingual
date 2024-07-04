import logging
import os
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, logging as transformers_logging
import torch
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the logging level for the transformers library
transformers_logging.set_verbosity_error()

class ExamTaker:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        logger.info(f"Loaded ExamTaker model: {model_name}")

    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info("Generated response")
        return response

class RewardModel:
    def __init__(self, model_name):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        logger.info(f"Loaded RewardModel: {model_name}")

    def evaluate_response(self, prompt, response):
        inputs = self.tokenizer(prompt + response, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        scores = torch.softmax(outputs.logits, dim=-1)[0]
        score = scores.max().item()
        logger.info("Evaluated response")
        return score
'''
class JudgeModel:
    def __init__(self, model_name):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token 
            logger.info(f"Loaded JudgeModel: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load JudgeModel: {model_name}", exc_info=True)
            raise e

    def evaluate_response(self, prompt, response):
        try:
            inputs = self.tokenizer(prompt + response, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=-1)[0]
            score = scores.max().item()  # Use max score as the reward
            logger.info("Evaluated response")
            return score
        except Exception as e:
            logger.error("Error evaluating response", exc_info=True)
            raise e
'''
class JudgeModel:
    def __init__(self, model_name):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token 
            logger.info(f"Loaded JudgeModel: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load JudgeModel: {model_name}", exc_info=True)
            raise e

    def evaluate_pairwise(self, prompt, response_a, response_b):
        try:
            # Prepare inputs for the pairwise comparison
            prompt_combined = f"Prompt: {prompt}\nResponse A: {response_a}\nResponse B: {response_b}\nWhich response is better?"
            inputs = self.tokenizer(prompt_combined, return_tensors="pt", truncation=True, padding=True, max_length=1024)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=-1)[0]
            logger.info("Evaluated pairwise responses")
            return scores  # Return the scores to determine the better response
        except Exception as e:
            logger.error("Error evaluating pairwise responses", exc_info=True)
            raise e

    def evaluate_single(self, prompt, response):
        try:
            inputs = self.tokenizer(f"Prompt: {prompt}\nResponse: {response}\nScore the response.", return_tensors="pt", truncation=True, padding=True, max_length=1024)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            scores = torch.softmax(outputs.logits, dim=-1)[0]
            score = scores.max().item()
            logger.info("Evaluated single response")
            return score
        except Exception as e:
            logger.error("Error evaluating single response", exc_info=True)
            raise e


class CriticModel:
    def __init__(self, model_name, reward_model_name, length_modifier=0.1):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
            self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
            self.reward_model.to(self.device)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token  
            if self.reward_tokenizer.pad_token is None:
                self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token  
            self.length_modifier = length_modifier
            logger.info(f"Loaded CriticModel: {model_name} and RewardModel: {reward_model_name}")
        except Exception as e:
            logger.error(f"Failed to load CriticModel or RewardModel: {model_name}, {reward_model_name}", exc_info=True)
            raise e

    def force_sampling(self, input_text, num_samples=4):
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            samples = []
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=True, num_return_sequences=1)
                samples.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
            logger.info("Performed force sampling")
            return samples
        except Exception as e:
            logger.error("Error during force sampling", exc_info=True)
            raise e

    def score_samples(self, samples):
        try:
            scores = []
            for sample in samples:
                inputs = self.reward_tokenizer(sample, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with torch.no_grad():
                    outputs = self.reward_model(**inputs)
                rm_scores = torch.softmax(outputs.logits, dim=-1)[0]
                rm_score = rm_scores.max().item()
                num_highlights = sample.count("```")
                score = rm_score + self.length_modifier * num_highlights
                scores.append((sample, score))
            logger.info("Scored samples")
            return scores
        except Exception as e:
            logger.error("Error scoring samples", exc_info=True)
            raise e

    def generate_critique(self, input_text, total_samples=28, length_modifiers=[0.1, 0.25, 0.5, 0.75], top_k=2):
        try:
            best_critiques = []
            for modifier in length_modifiers:
                self.length_modifier = modifier
                samples = self.force_sampling(input_text + "```", num_samples=total_samples // len(length_modifiers))
                scored_samples = self.score_samples(samples)
                scored_samples.sort(key=lambda x: x[1], reverse=True)
                best_critiques.extend(scored_samples[:top_k])
            final_critique = max(best_critiques, key=lambda x: x[1])[0]
            logger.info("Generated critique using FSBS")
            return final_critique
        except Exception as e:
            logger.error("Error generating critique", exc_info=True)
            raise e

def load_tokenized_shards(data_path):
    dataset = DatasetDict()
    for split in ['train', 'test']:
        split_path = os.path.join(data_path, split)
        if not os.path.isdir(split_path):
            logger.warning(f"Split directory not found: {split_path}")
            continue

        shards = [f for f in os.listdir(split_path) if f.startswith('shard_') and os.path.isdir(os.path.join(split_path, f))]
        logger.info(f"Found {len(shards)} potential shards in {split_path}")

        split_dataset = None
        for shard in shards:
            shard_path = os.path.join(split_path, shard)
            logger.info(f"Loading shard: {shard_path}")
            
            try:
                shard_dataset = load_from_disk(shard_path)
                if split_dataset is None:
                    split_dataset = shard_dataset
                else:
                    split_dataset = concatenate_datasets([split_dataset, shard_dataset])
                logger.info(f"Successfully loaded shard: {shard_path}")
            except Exception as e:
                logger.warning(f"Error loading shard {shard_path}: {str(e)}")

        if split_dataset is not None:
            dataset[split] = split_dataset
            logger.info(f"Created dataset for split: {split}")
        else:
            logger.warning(f"No valid shards found for split: {split}")

    if dataset:
        logger.info(f"Loaded Dataset from {data_path}")
        logger.info(f"Example from the loaded dataset: {dataset['train'][0] if 'train' in dataset else dataset[list(dataset.keys())[0]][0]}")
    else:
        logger.warning("No data loaded")

    return dataset
'''
def run_pipeline(model_name, reward_model_name, judge_model_name, critic_model_name, data_path, batch_size=8):
    try:
        logger.info("Loading dataset...")
        dataset = load_tokenized_shards(data_path)
        logger.info("Dataset loaded successfully.")

        logger.info("Loading models...")
        exam_taker = ExamTaker(model_name)
        reward_model = RewardModel(reward_model_name)
        judge_model = JudgeModel(judge_model_name) if judge_model_name else None
        critic_model = CriticModel(critic_model_name, reward_model_name) if critic_model_name else None
        logger.info("Models loaded successfully.")

        for example in dataset['test']:
            prompt = example['prompt'][:512]  # Limit prompt length for exam_taker 
            chosen = example['chosen'][0]['content'][:512]  # Limit chosen response for evaluation

            logger.info(f"Processing example with prompt: {prompt}")
            generated_response = exam_taker.generate_response(prompt)
            reward_score = reward_model.evaluate_response(prompt, generated_response)

            logger.info(f"Prompt: {prompt}")
            logger.info(f"Chosen: {chosen}")
            logger.info(f"Generated Response: {generated_response}")
            logger.info(f"Reward Score: {reward_score}")

            if judge_model:
                judge_score = judge_model.evaluate_response(prompt, generated_response)
                logger.info(f"Judge Score: {judge_score}")

            if critic_model:
                critique = critic_model.generate_critique(generated_response)
                logger.info(f"Critique: {critique}")
                # Compare reward scores before and after critique (if desired)

    except Exception as e:
        logger.error("Error running pipeline", exc_info=True)
        raise e
'''
def run_pipeline(model_name, reward_model_name, judge_model_name, critic_model_name, data_path, batch_size=8):
    try:
        logger.info("Loading dataset...")
        dataset = load_tokenized_shards(data_path)
        logger.info("Dataset loaded successfully.")

        logger.info("Loading models...")
        exam_taker = ExamTaker(model_name)
        reward_model = RewardModel(reward_model_name)
        judge_model = JudgeModel(judge_model_name) if judge_model_name else None
        critic_model = CriticModel(critic_model_name, reward_model_name) if critic_model_name else None
        logger.info("Models loaded successfully.")

        for example in dataset['test']:
            prompt = example['prompt'][:512]  # Limit prompt length for exam_taker 
            chosen = example['chosen'][0]['content'][:512]  # Limit chosen response for evaluation

            logger.info(f"Processing example with prompt: {prompt}")
            generated_response = exam_taker.generate_response(prompt)
            reward_score = reward_model.evaluate_response(prompt, generated_response)

            logger.info(f"Prompt: {prompt}")
            logger.info(f"Chosen: {chosen}")
            logger.info(f"Generated Response: {generated_response}")
            logger.info(f"Reward Score: {reward_score}")

            if judge_model:
                # Example usage of pairwise evaluation
                judge_scores = judge_model.evaluate_pairwise(prompt, chosen, generated_response)
                logger.info(f"Judge Scores: {judge_scores}")

                # Example usage of single evaluation
                judge_score = judge_model.evaluate_single(prompt, generated_response)
                logger.info(f"Judge Single Score: {judge_score}")

            if critic_model:
                critique = critic_model.generate_critique(generated_response)
                logger.info(f"Critique: {critique}")
                # Compare reward scores before and after critique (if desired)

    except Exception as e:
        logger.error("Error running pipeline", exc_info=True)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a smaller pipeline for model and reward model testing.")
    parser.add_argument("--data_path", type=str, default="tokenized_shards", help="Path to the tokenized shards")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--judge_model", type=str, help="Name of the judge model to use (optional)")
    parser.add_argument("--critic_model", type=str, help="Name of the critic model to use (optional)")  # Add critic model argument

    args = parser.parse_args()

    model_name = "distilgpt2"
    reward_model_name = "lvwerra/distilbert-imdb"
    judge_model_name = args.judge_model if args.judge_model else None
    critic_model_name = args.critic_model if args.critic_model else None

    run_pipeline(model_name, reward_model_name, judge_model_name, critic_model_name, args.data_path, args.batch_size)
