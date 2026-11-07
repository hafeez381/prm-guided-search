"""Edge cases beyond the public smoke tests."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import submission_adapter as adapter

SPEC_TABLE = [
    ("Final Answer: 18", 18),
    ("Final Answer: $1,234", 1234),
    ("Final Answer: 18.00", 18),
    ("Final Answer: 18.5", None),
    ("Final Answer: -5", -5),
    ("Final Answer: 18 dollars.", 18),
    ("**Final Answer:** 18", 18),
    ("final answer 18", 18),
    ("Final Answer: 10\nscratch\nFinal Answer: 42", 42),
    ("No answer here.", None),
    ("", None),
]

EXTRA = [
    (None, None),
    (12345, None),
    ("Final Answer: 1,234,567", 1234567),
    ("Final Answer: −5", -5),
    ("Final Answer: - 5", -5),
    ("Final Answer:\n18", 18),
    ("Final Answer: 7 * 2 = 14", 14),
    ("Final Answer: 18 (9 eggs x 2)", 18),
    ("Final Answer: $3.50", None),
    ("### Final Answer: `24`", 24),
    ("Step 1: 2 + 2 = 4.\nFinal Answer: 4\n", 4),
    ("x" * 10_000, None),
    ("Final Answer: " + "9" * 400, int("9" * 400)),
]


@pytest.mark.parametrize("text,expected", SPEC_TABLE + EXTRA)
def test_extract_answer(text, expected):
    assert adapter.extract_answer(text) == expected


def test_prompt_bans_drift_formats():
    prompt = adapter.format_step_prompt("Q?").lower()
    for banned in ("markdown", "latex", "boxed", "bold"):
        assert banned in prompt
    assert "final answer:" in prompt


@pytest.mark.parametrize("answers,expected", [
    ([None, None], None),
    ([], None),
    ([3, 5, 5, 3], 3),
    ([5, 3, 3, 4, 4], 3),
    ([None, 7], 7),
    ([None, 4, None, 9, 9, 4], 4),
    ([0, None, 0], 0),
])
def test_majority_vote(answers, expected):
    assert adapter.compute_majority_vote(answers) == expected


def _recording_generate_fn(texts):
    calls = []

    def generate_fn(prompt, num_samples, gen_kwargs):
        calls.append((prompt, num_samples, gen_kwargs))
        return [{"text": texts[i % len(texts)], "num_tokens": 7 + i} for i in range(num_samples)]

    generate_fn.calls = calls
    return generate_fn


def test_greedy_does_not_share_kwargs_dict():
    from src.utils import GREEDY_GEN_KWARGS
    before = dict(GREEDY_GEN_KWARGS)
    fn = _recording_generate_fn(["Step 1: 1 + 1 = 2.\nFinal Answer: 2"])
    res = adapter.run_greedy(question="Q", generate_fn=fn)
    fn.calls[0][2]["do_sample"] = True
    assert GREEDY_GEN_KWARGS == before
    assert res["generated_tokens"] == 7 and res["prompt"] == fn.calls[0][0]


def test_self_consistency_counts_and_kwargs():
    texts = ["Step 1: a\nFinal Answer: 4", "no delimiter here", "Step 1: b\nFinal Answer: 6"]
    fn = _recording_generate_fn(texts)
    res = adapter.run_self_consistency(question="Q", n=5, temperature=0.7, generate_fn=fn)
    _, n, kwargs = fn.calls[0]
    assert n == 5 and len(fn.calls) == 1
    assert kwargs == {"do_sample": True, "temperature": 0.7, "top_p": 1.0, "top_k": 0,
                      "repetition_penalty": 1.0, "max_new_tokens": 512}
    assert [c["answer"] for c in res["candidates"]] == [4, None, 6, 4, None]
    assert res["answer"] == 4
    assert res["generation_calls"] == 5
    assert res["generated_tokens"] == sum(7 + i for i in range(5))
