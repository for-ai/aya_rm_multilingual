import logging
import os
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, logging as transformers_logging
import torch
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the logging level for the transformers library
transformers_logging.set_verbosity_error()

# System prompts
prompt_v2 = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below in multiple languages. "
    "You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider "
    "factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by "
    "comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were "
    "presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names "
    "of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "
    '"[[A]]" if assistant A is better, "[[B]]" if assistant B is better.'
)

exam_taker_system_prompt = (
    "You are an AI assistant responsible for generating high-quality responses to user queries in multiple languages. Your responses should be:\n"
    "1. Highly accurate and relevant to the user's question.\n"
    "2. Detailed, providing comprehensive information and covering all aspects of the query.\n"
    "3. Clear and concise, avoiding unnecessary information and focusing on key points.\n"
    "4. Well-structured, with logical flow and proper formatting to enhance readability.\n"
    "5. Helpful, offering valuable insights and actionable advice wherever applicable.\n"
    "6. Free of errors, including grammatical, factual, and logical mistakes.\n\n"
    "Always ensure your answers are objective and unbiased. In case of ambiguities in the question, make reasonable assumptions and state them clearly. "
    "Aim to provide responses that exceed user expectations in quality and usefulness. Remember to handle queries in multiple languages efficiently."
)

critic_system_prompt = (
    "Please review the following response in a multilingual context and identify any errors or areas for improvement. "
    "Highlight specific issues and provide detailed feedback on each problem. "
    "Your critique should be comprehensive, accurate, and avoid nitpicks. Focus on critical issues that could "
    "affect the correctness and quality of the response. Use a tree of thought approach to break down the critique into logical steps and reasoning."
)

class ExamTaker:
    def __init__(self, model_name, system_prompt):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.system_prompt = system_prompt
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  
        logger.info(f"Loaded ExamTaker model: {model_name}")

    def generate_response(self, prompt):
        combined_prompt = f"{self.system_prompt}\n{prompt}"
        inputs = self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
        return score

class JudgeModel:
    def __init__(self, model_name, system_prompt):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.system_prompt = system_prompt
        logger.info(f"Loaded JudgeModel: {model_name}")

    def evaluate_pairwise(self, prompt, response_a, response_b):
        prompt_combined = self.system_prompt.format(prompt=prompt, response_a=response_a, response_b=response_b)
        inputs = self.tokenizer(prompt_combined, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        judgment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.parse_judgment(judgment)

    def parse_judgment(self, judgment):
        if "[[A]]" in judgment:
            return "A"
        elif "[[B]]" in judgment:
            return "B"
        else:
            return "error"

class CriticModel:
    def __init__(self, model_name, reward_model_name, system_prompt, length_modifier=0.1):
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
        self.system_prompt = system_prompt
        self.length_modifier = length_modifier
        logger.info(f"Loaded CriticModel: {model_name} and RewardModel: {reward_model_name}")

    def force_sampling(self, input_text, num_samples=4):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        samples = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100, do_sample=True, num_return_sequences=1)
            samples.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return samples

    def score_samples(self, samples):
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
        return scores

    def generate_critique(self, input_text, total_samples=28, length_modifiers=[0.1, 0.25, 0.5, 0.75], top_k=2):
        best_critiques = []
        for modifier in length_modifiers:
            self.length_modifier = modifier
            samples = self.force_sampling(input_text + "```", num_samples=total_samples // len(length_modifiers))
            scored_samples = self.score_samples(samples)
            scored_samples.sort(key=lambda x: x[1], reverse=True)
            best_critiques.extend(scored_samples[:top_k])
        final_critique = max(best_critiques, key=lambda x: x[1])[0]
        return final_critique

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

def run_experiment(model_name, reward_model_name, judge_model_name, critic_model_name, data_path, batch_size=8):
    dataset = load_tokenized_shards(data_path)

    exam_taker = ExamTaker(model_name, exam_taker_system_prompt)
    reward_model = RewardModel(reward_model_name)
    judge_model = JudgeModel(judge_model_name, prompt_v2) if judge_model_name else None
    critic_model = CriticModel(critic_model_name, reward_model_name, critic_system_prompt) if critic_model_name else None

    for i, example in enumerate(dataset['test']):
        prompt = example['prompt'][:512]  # Limit prompt length for exam_taker 
        chosen = example['chosen'][0]['content'][:512]  # Limit chosen response for evaluation
        rejected = example['rejected'][0]['content'][:512]  # Limit rejected response for evaluation

        generated_response = exam_taker.generate_response(prompt)
        reward_score_before = reward_model.evaluate_response(prompt, generated_response)

        judge_scores_before = judge_model.evaluate_pairwise(prompt, chosen, generated_response) if judge_model else "N/A"

        critique = critic_model.generate_critique(generated_response) if critic_model else None
        reward_score_after = reward_model.evaluate_response(prompt, critique) if critic_model else "N/A"

        judge_scores_after = judge_model.evaluate_pairwise(prompt, chosen, critique) if judge_model and critic_model else "N/A"

        print("==================")
        print(f"Iteration {i + 1}:")
        print(f"- Generated response reward: {reward_score_before}")
        print(f"- Judge Evaluated Output before Critique: {judge_scores_before}")
        print(f"- Reward After Critic Score: {reward_score_after}")
        print(f"- Judge Evaluated Output after Critique: {judge_scores_after}")
        print("==================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment for model and reward model testing.")
    parser.add_argument("--data_path", type=str, default="tokenized_shards", help="Path to the tokenized shards")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--judge_model", type=str, help="Name of the judge model to use (optional)")
    parser.add_argument("--critic_model", type=str, help="Name of the critic model to use (optional)")  # Add critic model argument

    args = parser.parse_args()

    model_name = "distilgpt2"
    reward_model_name = "lvwerra/distilbert-imdb"
    judge_model_name = args.judge_model if args.judge_model else None
    critic_model_name = args.critic_model if args.critic_model else None

    run_experiment(model_name, reward_model_name, judge_model_name, critic_model_name, args.data_path, args.batch_size)
