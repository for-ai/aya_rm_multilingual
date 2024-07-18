import pytest

from scripts.generative import chat_completion_anthropic
from scripts.generative import chat_completion_cohere, chat_completion_gemini
from scripts.generative import chat_completion_together, format_judge_answers
from scripts.generative import process_judgement, run_judge_pair


def test_format_judge_answers_multilingual_includes_language():
    question = "Ano ang sagot sa (2+3) * 4? Ipaliwanag ang iyong sagot"
    answer_a = [
        {
            "role": "user",
            "content": "Ano ang sagot sa (2+3) * 4? Ipaliwanag ang iyong sagot",
        },
        {
            "role": "assistant",
            "content": "20. Unahing i-add ang nasa loob ng parenthesis. Tapos i-multiply sa 4.",
        },
    ]
    answer_b = [
        {
            "role": "user",
            "content": "Ano ang sagot sa (2+3) * 4? Ipaliwanag ang iyong sagot",
        },
        {
            "role": "assistant",
            "content": "Ang sagot ay 20.",
        },
    ]
    src_lang = "Filipino"  # language the prompt is written on
    tgt_lang = "English"  # language the assistant should reply on
    include_languages = [src_lang, tgt_lang]
    sys_prompt, user_prompt = format_judge_answers(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        include_langs=include_languages,
    )

    assert src_lang in sys_prompt
    assert tgt_lang in sys_prompt
