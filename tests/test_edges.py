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
