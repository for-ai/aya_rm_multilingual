
import logging
import os
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, logging as transformers_logging
import torch
import argparse

# Set up logging
logging.basicConfig(level=logging.WARN)
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
    "5. Helpful, offering valuable grammatical, factual, and logical mistakes.\n\n"
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

    def extract_text(self, response):
        if isinstance(response, list):
            response_texts = []
            for item in response:
                if isinstance(item, dict):
                    response_texts.append(' '.join(str(value) for value in item.values()))
                else:
                    response_texts.append(str(item))
            response = ' '.join(response_texts)
        elif isinstance(response, dict):
            response = ' '.join(str(value) for value in response.values())
        return response

    def evaluate_responses(self, prompt, chosen_response, rejected_response):
        # Convert responses to strings if necessary
        chosen_response = self.extract_text(chosen_response)
        rejected_response = self.extract_text(rejected_response)

        # Tokenize inputs
        chosen_inputs = self.tokenizer(prompt + chosen_response, return_tensors="pt", truncation=True, padding=True, max_length=512)
        rejected_inputs = self.tokenizer(prompt + rejected_response, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Move inputs to the device
        chosen_inputs = {key: value.to(self.device) for key, value in chosen_inputs.items()}
        rejected_inputs = {key: value.to(self.device) for key, value in rejected_inputs.items()}

        # Get model outputs
        with torch.no_grad():
            chosen_outputs = self.model(**chosen_inputs)
            rejected_outputs = self.model(**rejected_inputs)

        # Calculate preference probabilities using Bradley-Terry model
        chosen_score = torch.softmax(chosen_outputs.logits, dim=-1)[0]
        rejected_score = torch.softmax(rejected_outputs.logits, dim=-1)[0]

        # Calculate the probability that chosen_response is preferred over rejected_response
        p_chosen = torch.exp(chosen_score[1]) / (torch.exp(chosen_score[1]) + torch.exp(rejected_score[1]))
        p_rejected = torch.exp(rejected_score[1]) / (torch.exp(chosen_score[1]) + torch.exp(rejected_score[1]))

        logger.info(f"Chosen score: {chosen_score}, Rejected score: {rejected_score}")
        logger.info(f"Probability chosen is preferred: {p_chosen}, Probability rejected is preferred: {p_rejected}")

        return p_chosen.item(), p_rejected.item()


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
        prompt_combined_1 = (
            f"{self.system_prompt}\nUser question: {prompt}\nAssistant A response: {response_a}\nAssistant B response: {response_b}\n"
        )
        prompt_combined_2 = (
            f"{self.system_prompt}\nUser question: {prompt}\nAssistant A response: {response_b}\nAssistant B response: {response_a}\n"
        )
        
        inputs_1 = self.tokenizer(prompt_combined_1, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs_2 = self.tokenizer(prompt_combined_2, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        inputs_1 = {key: value.to(self.device) for key, value in inputs_1.items()}
        inputs_2 = {key: value.to(self.device) for key, value in inputs_2.items()}
        
        with torch.no_grad():
            outputs_1 = self.model.generate(**inputs_1, max_new_tokens=100)
            outputs_2 = self.model.generate(**inputs_2, max_new_tokens=100)
        
        judgment_1 = self.tokenizer.decode(outputs_1[0], skip_special_tokens=True)
        judgment_2 = self.tokenizer.decode(outputs_2[0], skip_special_tokens=True)
        
        logger.info(f"Prompt combined 1: {prompt_combined_1}")
        logger.info(f"Prompt combined 2: {prompt_combined_2}")
        logger.info(f"Judgment 1: {judgment_1}")
        logger.info(f"Judgment 2: {judgment_2}")
        
        result_1 = self.parse_judgment(judgment_1)
        result_2 = self.parse_judgment(judgment_2)
        
        final_judgment = judgment_1 if result_1 == result_2 else "tie"
        return result_1, final_judgment

    def parse_judgment(self, judgment):
        if "[[A]]" in judgment:
            return "A"
        elif "[[B]]" in judgment:
            return "B"
        else:
            return "error"

    def evaluate_single(self, prompt, response):
        combined_prompt = f"{self.system_prompt}\n{prompt}\n{response}"
        inputs = self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return evaluation

    def evaluate_with_reference(self, prompt, response, reference):
        combined_prompt = f"{self.system_prompt}\n{prompt}\nReference answer:\n{reference}\nResponse:\n{response}"
        inputs = self.tokenizer(combined_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        evaluation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return evaluation

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
        logger.info(f"Generated critique: {final_critique}")
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

def load_rewardbench(subset):
    try:
        rewardbench_dataset = load_dataset("allenai/reward-bench")
        if subset:
            rewardbench_dataset = rewardbench_dataset.filter(lambda ex: ex["subset"] == subset)
        logger.info(f"Loaded RewardBench dataset with subset: {subset}.")
        logger.info(f"RewardBench dataset structure: {rewardbench_dataset}")

        # Add a sample inspection of the dataset
        for split in rewardbench_dataset.keys():
            logger.info(f"Sample from {split} split: {rewardbench_dataset[split][0]}")

        return rewardbench_dataset
    except Exception as e:
        logger.error(f"Error loading RewardBench dataset with subset {subset}: {str(e)}")
        return None

def run_experiment(model_name, reward_model_name, judge_model_name, critic_model_name, data_path, batch_size=8, rewardbench=False, rewardbench_subset=None):
    if rewardbench:
        dataset = load_rewardbench(rewardbench_subset)
    else:
        dataset = load_tokenized_shards(data_path)

    if not dataset:
        logger.error("Dataset could not be loaded. Exiting the experiment.")
        return

    exam_taker = ExamTaker(model_name, exam_taker_system_prompt)
    reward_model = RewardModel(reward_model_name)
    judge_model = JudgeModel(judge_model_name, prompt_v2) if judge_model_name else None
    critic_model = CriticModel(critic_model_name, reward_model_name, critic_system_prompt) if critic_model_name else None

    for split_name, split_data in dataset.items():
        logger.info(f"Processing split: {split_name}")
        for i, example in enumerate(split_data):
            # Print only the desired output
            prompt = example['prompt']
            chosen = example['chosen']
            rejected = example['rejected']

            generated_response = exam_taker.generate_response(prompt)
            reward_score_before = reward_model.evaluate_responses(prompt, chosen, generated_response)

            judge_scores_before, judge_text = judge_model.evaluate_pairwise(prompt, chosen, generated_response) if judge_model else ("N/A", "N/A")

            # Update the system prompt with the judge's text
            new_system_prompt = f"{critic_system_prompt}\n{judge_text}"
            new_generated_response = exam_taker.generate_response(new_system_prompt)
            reward_score_after_judgement = reward_model.evaluate_responses(prompt, new_generated_response, rejected)

            if critic_model:
                critic_model.system_prompt = new_system_prompt
                critique = critic_model.generate_critique(new_generated_response)
                logger.info(f"Critique generated: {critique}")
                final_response = new_generated_response + " " + critique
                reward_score_after = reward_model.evaluate_responses(prompt, final_response, rejected)
            else:
                final_response = new_generated_response
                reward_score_after = reward_score_after_judgement

            print("==================")
            print(f"RewardBench Subset: {rewardbench_subset}:")
            print(f"Iteration {i + 1}:")
            print(f"- Generated Response Reward: {reward_score_before}")
            print(f"- Generated Response Reward after judgement: {reward_score_after_judgement}")
            print(f"- Generated Response Reward after judgement and critic: {reward_score_after}")
            print("==================")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment for model and reward model testing.")
    parser.add_argument("--data_path", type=str, default="tokenized_shards", help="Path to the tokenized shards")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--model", type=str, help="Name of the exam taker model to use (optional)")
    parser.add_argument("--reward_model", type=str, help="Name of the reward model to use (optional)")
    parser.add_argument("--judge_model", type=str, help="Name of the judge model to use (optional)")
    parser.add_argument("--critic_model", type=str, help="Name of the critic model to use (optional)")  # Add critic model argument
    parser.add_argument("--rewardbench", action="store_true", help="Use RewardBench dataset for evaluation")
    parser.add_argument("--rewardbench_subset", type=str, help="Subset of RewardBench dataset to use")

    args = parser.parse_args()

    exam_taker_model_name = args.model if args.model else None
    reward_model_name = args.reward_model if args.reward_model else None
    judge_model_name = args.judge_model if args.judge_model else None
    critic_model_name = args.critic_model if args.critic_model else None

    run_experiment(exam_taker_model_name, reward_model_name, judge_model_name, critic_model_name, args.data_path, args.batch_size, args.rewardbench, args.rewardbench_subset)
