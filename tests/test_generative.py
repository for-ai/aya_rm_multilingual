"""Testing new functions in generative.py"""

import pytest
from dotenv import load_dotenv

from scripts.generative import chat_completion_cohere, format_judge_answers
from scripts.generative import process_judgement, run_judge_pair

load_dotenv(verbose=True)


def test_format_judge_answers_multilingual_includes_language():
    question = "Ano ang sagot sa (2+3) * 4? Ipaliwanag ang iyong sagot"
    ans_a = "20. Unahing i-add ang nasa loob ng parenthesis. Tapos i-multiply sa 4."
    ans_b = "Ang sagot ay 20."
    answer_a = [{"role": "user", "content": question}, {"role": "assistant", "content": ans_a}]
    answer_b = [{"role": "user", "content": question}, {"role": "assistant", "content": ans_b}]
    src_lang = "Filipino"  # language the prompt is written on
    tgt_lang = "English"  # language the assistant should reply on
    include_languages = [src_lang, tgt_lang]
    sys_prompt, _ = format_judge_answers(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        include_langs=include_languages,
    )

    assert src_lang in sys_prompt
    assert tgt_lang in sys_prompt


@pytest.mark.parametrize("judgment,expected", [("[[A]]", "A"), ("[[B]]", "B"), ("I don't know", "error")])
def test_process_judgment_answers(judgment, expected):
    answer = process_judgement(judgment, is_prometheus=False)
    assert answer == expected


@pytest.mark.api
@pytest.mark.parametrize("multilingual", [True, False])
def test_cohere_api(multilingual):
    from fastchat.conversation import get_conv_template

    if multilingual:
        question = "Quelle est la capitale du Japon?"
        ans_a = "Tokyo"
        ans_b = "La capitale du Japon est Tokyo"
        include_langs = ["Japanese", "English"]
    else:
        question = "What is the capital of Japan?"
        ans_a = "Tokyo"
        ans_b = "The capital of Japan is Tokyo"
        include_langs = None

    sys_prompt, user_prompt = format_judge_answers(
        question=question,
        answer_a=[{"role": "user", "content": question}, {"role": "assistant", "content": ans_a}],
        answer_b=[{"role": "user", "content": question}, {"role": "assistant", "content": ans_b}],
        include_langs=include_langs,
    )

    conv = get_conv_template("raw")
    conv.append_message(conv.roles[0], user_prompt)
    conv.set_system_message(sys_prompt)
    judgement = chat_completion_cohere(
        conv=conv,
        model="command-r",
        temperature=0,
        max_tokens=2048,
    )

    assert judgement
    assert isinstance(judgement, str)


@pytest.mark.api
@pytest.mark.parametrize("multilingual", [True, False])
def test_run_judge_pair(multilingual):
    if multilingual:
        question = "Quelle est la capitale du Japon?"
        ans_a = "Tokyo"
        ans_b = "La capitale du Japon est Tokyo"
        include_langs = ["Japanese", "English"]
    else:
        question = "What is the capital of Japan?"
        ans_a = "Tokyo"
        ans_b = "The capital of Japan is Tokyo"
        include_langs = None

    answer_a = ([{"role": "user", "content": question}, {"role": "assistant", "content": ans_a}],)
    answer_b = ([{"role": "user", "content": question}, {"role": "assistant", "content": ans_b}],)

    winner, user_prompt, judgement = run_judge_pair(
        question,
        answer_a=answer_a,
        answer_b=answer_b,
        model="command-r",
        multi_turn=False,
        model_modifier=None,
        include_langs=include_langs,
    )

    assert winner in ["A", "B", "none"]
    assert judgement
    assert user_prompt
