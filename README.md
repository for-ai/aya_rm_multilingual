# Expedition Aya: Reward Model Multilingual

In this project, we evaluate reward models on their multilingual capabilities.
We translated [RewardBench](https://huggingface.co/datasets/allenai/reward-bench) into 23 diverse languages and evaluated several open-source and multilingual LLMs on their chat, safety, and reasoning capabilities.
This project is part of [Cohere for AI's Expedition Aya 2024](https://sites.google.com/cohere.com/expedition-aya/home), a 6-week open build challenge.

<p align="center">
<b><a href="https://huggingface.co/datasets/aya-rm-multilingual/multilingual-reward-bench">🤗 Dataset</a></b>
|
<b><a href="https://docs.google.com/presentation/d/1nEWUGw8qaHUa-FroNyFYLInRJ2yAKgQBIK5n5cGX9sA/edit?usp=sharing">💬 Presentation</a></b>
|
<b><a href="https://github.com/for-ai/aya_rm_multilingual/blob/main/docs.md">📚 Documentation</a></b>
|
<b><a href="https://arxiv.org/abs/2410.15522">📄Paper</a></b>
</p>

## News

- [2024-08-28] We won **Silver Prize** in Expedition Aya 2024! We're also releasing the v1 of the multilingual RewardBench on [HuggingFace](https://huggingface.co/datasets/aya-rm-multilingual/multilingual-reward-bench).

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

## Team Members

The team is composed of Srishti Gureja ([@srishti-git1110](https://github.com/srishti-git1110)), Shayekh Bin Islam, ([@ShayekhBinIslam](https://github.com/ShayekhBinIslam)), Rishabh Maheshwary ([@RishabhMaheshwary](https://github.com/RishabhMaheshwary)), Drishti Sharma ([@DrishtiShrrrma](https://github.com/DrishtiShrrrma)), Gusti Winata ([@sanggusti](https://github.com/sanggusti)), and Lj Miranda ([@ljvmiranda921](https://github.com/ljvmiranda921)).
