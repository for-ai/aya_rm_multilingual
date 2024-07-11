# Aya Expedition: Reward Model Multilingual

Repository for Aya Expedition Project : Reward Model Multilingual

Project Docs: [docs](https://docs.google.com/document/d/11l7Mb60JMRpdJpp9-B7VjWOF4FshBdjzY0FDOTq9sMk/edit?usp=sharing)

## Setup and installation

We recommend installing the dependencies inside a [virtual environment](https://docs.python.org/3/library/venv.html):

```sh
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate
# Install the dependencies (within venv context)
pip install -r requirements.txt
```

Note that the [`rewardbench`](https://pypi.org/project/rewardbench/) package requires Python 3.10 and above.

## Running experiments

First, you need to set a [HuggingFace token](https://huggingface.co/settings/tokens) as an environment variable (`HF_TOKEN`):

```sh
export HF_TOKEN=<your huggingface token>
```

If you're planning to use some closed-source APIs, you also need to set the tokens for each:


```sh
export OPENAI_API_KEY=<your openai token>
export ANTHROPIC_API_KEY=<your anthropic token>
export GEMINI_API_KEY=<your gemini token>
```

You can find all runnable experiments in the `scripts` directory.
Their filename should explicitly tell you their purpose. 
For example, `scripts/run_rm_evals.sh` runs the RewardBench inference pipeline on a select number of models given a dataset:

```sh
./scripts/run_rm_evals.sh
```
