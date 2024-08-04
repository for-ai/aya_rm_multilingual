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

You can find all runnable experiments in the `scripts` directory.
Their filename should explicitly tell you their purpose.

### Running translation

We currently use [`facebook/nllb-200-3.3B`](https://huggingface.co/facebook/nllb-200-3.3B) for translation. First install sentence splitter using:

```
pip install git+https://github.com/mediacloud/sentence-splitter.git
```

To translate reward bench into [22 Aya languages](https://arxiv.org/abs/2405.15032) run the following:

```
cd scripts
bash run_nllb.sh
```

You can also translate a specifc preference dataset from huggingface to a specifc target language using `scripts/translate_preference_pairs_nllb.py`.

### Getting rewards from a Reward Model (RM) on a HuggingFace dataset

Here, we use the `scripts/run_rewardbench.py` command-line interface and pass a HuggingFace dataset.
This is useful if the reward model is trained as a Custom classifier (🛠️), Sequence classifier (🔢), or via DPO (🎯).
For example, if we want to get the reward score of the UltraRM-13b reward model on a preference dataset, we run:

```sh
python -m scripts.run_rewardbench \
    --model openbmb/UltraRM-13b \
    --chat_template openbmb \
    --dataset_name $DATASET \
    --lang_code $LANG_CODE \
    --split "filtered" \
    --output_dir $OUTDIR \
    --batch_size 8 \
    --trust_remote_code \
    --force_truncation \
    --save_all
```

The evaluation parameters can be found in the [allenai/reward-bench](https://github.com/allenai/reward-bench/blob/main/scripts/configs/eval_configs.yaml) repository.
This runs the reward model on the (prompt, chosen, rejected) triples and give us the reward score for each instance.
The results are saved into a JSON file inside the `$OUTDIR` directory.
Finally, you can find some experiments in the `experiments/run_rm_evals.sh` script.

### Getting rewards from a Generative RM on a HuggingFace dataset

Here we use `scripts/run_generative.py`, a modified version of the [same script in RewardBench](https://github.com/allenai/reward-bench/blob/main/scripts/run_generative.py) to obtain rewards from a Generative RM (🗨️).
The only difference is that this script accepts any arbitrary HuggingFace preference dataset (which we plan to conribute upstream later on) instead of just the RewardBench dataset.

For Generative RMs, we prompt a model in a style akin to LLM-as-a-judge, and then parse the output to obtain the preference.
This can be done for closed-source APIs (e.g., GPT-4, Claude) or open-source LMs (done via vLLM).
If you're planning to use some closed-source APIs, you also need to set the tokens for each:

```sh
export OPENAI_API_KEY=<your openai token>
export CO_API_KEY=<your cohere api token>
export ANTHROPIC_API_KEY=<your anthropic token>
```

**You can also store all your API keys in a .env file.**
It will be loaded using the [python-dotenv library](https://github.com/theskumar/python-dotenv).
Say we want to obtain the preferences of `gpt-4-2024-04-09`:

```sh
export OPENAI_API_KEY=<your openai token>
python -m scripts.run_generative \
    --dataset_name $DATASET \
    --model gpt-4-turbo-2024-04-09 \
    --split "filtered" \
    --lang_code $LANG_CODE \
    --output_dir $OUTDIR
```

You can also run open-source LMs in a generative fashion.
The inference is then routed through [vLLM](https://github.com/vllm-project/vllm).
Here's an example using `meta-llama/Meta-Llama-3-70B-Instruct`:

```sh
python -m scripts/run_generative.py \
    --dataset_name $DATASET \
    --lang_code $LANG_CODE \
    --split "filtered" \
    --model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --num_gpus 4 \
    --output_dir $OUTDIR
```

To improve prompt output especially on multilingual cases, we recommend passing a tuple to the `--include_languages` parameter.
The first value should be the language a prompt was written in, and the second value should be the language the assistant should use in its answer.

```diff
python -m scripts/run_generative.py \
    --dataset_name $DATASET \
    --lang_code deu_Latn \
    --split $SPLIT \
    --model "meta-llama/Meta-Llama-3-70B-Instruct" \
    --num_gpus 4 \
+   --include_languages German English
    --output_dir $OUTDIR
```

## Testing and Development

This codebase contains minimal tests, mostly we test functions that were added or patched from RewardBench.
First, you need to install all the development dependencies:

```sh
pip install -r requirements-dev.txt
```

Then, you can run the tests by:

```sh
pytest tests/ -v --capture=no
pytest tests/ -m "not api" -v --capture=no  # to ignore tests that make use of third-party APIs
```

When developing, we format the code using [black](https://black.readthedocs.io/en/stable/index.html) and [isort](https://pycqa.github.io/isort/), to be consistent with the RewardBench codebase.
You can automatically format your code by running:

```
make style
```
