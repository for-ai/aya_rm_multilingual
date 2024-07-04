# Aya Expedition: Reward Model Multilingual

Repository for Aya Expedition Project : Reward Model Multilingual

Project Docs: [docs](https://docs.google.com/document/d/11l7Mb60JMRpdJpp9-B7VjWOF4FshBdjzY0FDOTq9sMk/edit?usp=sharing)

# Multilingual Model Pipeline

This repository contains scripts to preprocess data, tokenize it into shards, and run various models including ExamTaker, Judge, Critic, and Reward models.

## Getting Started

### Prerequisites

Ensure you have the necessary Python packages installed. You can use the provided `requirements.txt` if available, or manually install the necessary packages:
```bash
pip install datasets transformers torch

1. Navigate to the directory containing your scripts:

cd ./aya_rm_multilingual/scripts

2. Run the data preprocessing and tokenization script:

python nthakur_data.py

This script will preprocess the dataset, tokenize it, and save the shards in the tokenized_shards directory

3. How to run the models:

a) Run exam taker and reward model:

python exam_judge_critic_reward.py --data_path ../tokenized_shards --batch_size 8 --model_name distilgpt2 --reward_model_name lvwerra/distilbert-imdb

b) Run exam taker, reward model and judge model:

python exam_judge_critic_reward.py --data_path ../tokenized_shards --batch_size 8 --model_name distilgpt2 --reward_model_name lvwerra/distilbert-imdb --judge_model lvwerra/distilbert-imdb

c) Run exam taker, reward model, judge model, and critic model:

python exam_judge_critic_reward.py --data_path ../tokenized_shards --batch_size 8 --model_name distilgpt2 --reward_model_name lvwerra/distilbert-imdb --judge_model lvwerra/distilbert-imdb --critic_model distilgpt2

4. 3. Model Selection
You can choose different models by specifying their names in the command-line arguments. Ensure that the models you choose are supported and available.

Model Name: <MODEL_NAME>
Reward Model Name: <REWARD_MODEL_NAME>
Judge Model Name: <JUDGE_MODEL_NAME>
Critic Model Name: <CRITIC_MODEL_NAME>






