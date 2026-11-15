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


def test_compliance_check_never_reads_ground_truth():
    src = open(os.path.join(os.path.dirname(__file__), "..", "compliance_check.py")).read()
    assert "ground_truth" not in src and "accuracy" not in src.lower()


@pytest.mark.parametrize("text,score,rationale", [
    ("Verdict: **INVALID**\nRationale: Wrong sum.", -1.0, "Wrong sum."),
    ("VALID? No. Verdict: INVALID", -1.0, ""),
    ("The step is not valid.", -1.0, ""),
    ("INVALIDATED", -1.0, ""),
    ("VALIDATED", -1.0, ""),
    ("Verdict: VALIDATED\nThe step is valid.", 1.0, ""),
    ("Rationale: 2.5 + 2.5 = 5. Next.\nVerdict: VALID", 1.0, "2.5 + 2.5 = 5."),
    ("Verdict: VALID\nRationale: ok", 1.0, "ok"),
    ("Verdict: VALID\nRationale: line one\nline two.", 1.0, "line one"),
    ("Verdict: INVALID\nRationale:", -1.0, ""),
    (None, -1.0, ""),
])
def test_parse_judge_decision_edges(text, score, rationale):
    assert adapter.parse_judge_decision(text) == (score, rationale)


def test_judge_prompt_numbering_and_current_step():
    p = adapter.build_prm_judge_prompt("Q", ["a = 1", "b = 2"], "a + b = 3")
    assert "Step 1: a = 1\nStep 2: b = 2" in p
    assert "Current Step: a + b = 3" in p
    assert "(no preceding steps)" not in p
    assert "Step 3" not in p


def test_score_path_judges_each_prefix_once():
    prompts = []

    def judge_fn(prompt):
        prompts.append(prompt)
        return {"text": "Verdict: VALID\nRationale: Fine.", "num_tokens": 4}

    res = adapter.score_path(question="Q", candidate={"steps": ["s1", "s2", "s3"]}, judge_fn=judge_fn)
    assert res["judge_calls"] == 3 and res["judge_tokens"] == 12
    assert "(no preceding steps)" in prompts[0] and "Current Step: s1" in prompts[0]
    assert "Step 1: s1\nStep 2: s2" in prompts[2] and "Current Step: s3" in prompts[2]
    empty = adapter.score_path(question="Q", candidate={"steps": []}, judge_fn=judge_fn)
    assert empty["step_scores"] == [] and empty["judge_calls"] == 0 and len(prompts) == 3
    assert adapter.score_path(question="Q", candidate=None, judge_fn=judge_fn)["judge_calls"] == 0


@pytest.mark.parametrize("scores,strategy,lam,expected", [
    ([1, 1, 1, 1], "min", 2.0, 1.0),
    ([1, 1, 1, 1], "product", 2.0, 0.99 ** 4),
    ([1, 1, 1, 1], "step_penalty", 2.0, 1.0),
    ([1, -1], "min", 2.0, -1.0),
    ([1, -1], "product", 2.0, 0.99 * 0.01),
    ([1, -1], "step_penalty", 2.0, -2.0),
    ([1, -1], "step_penalty", 0.0, 0.0),
    ([-1, -1, -1], "product", 2.0, 1e-6),
    ([-1, -1, -1], "step_penalty", 0.5, -2.5),
    ([1, -1], "step_penalty", 0.5, -0.5),
    ([1] * 9 + [-1, -1], "step_penalty", 0.5, 7 / 11 - 1.0),
    ([], "min", 2.0, -1.0),
    ([], "product", 2.0, -1.0),
    ([], "step_penalty", 0.0, -1.0),
])
def test_aggregate_hand_values(scores, strategy, lam, expected):
    assert abs(adapter.aggregate_path_score(scores, strategy, lam) - expected) < 1e-12


def test_aggregate_unknown_strategy_raises_even_when_empty():
    with pytest.raises(ValueError):
        adapter.aggregate_path_score([], strategy="max")


def test_product_never_ties_sound_paths_of_different_length():
    s4 = adapter.aggregate_path_score([1.0] * 4, "product")
    s5 = adapter.aggregate_path_score([1.0] * 5, "product")
    assert s4 - s5 > 1e-9


def test_select_best_path_vote_among_tied_only():
    cands = [{"step_scores": [1.0], "answer": a} for a in [5, 3, 3, 4, 4]]
    win = adapter.select_best_path(cands)
    assert win["answer"] == 3 and win["num_tied"] == 5
    flawed = [{"step_scores": [-1.0], "answer": 7}, {"step_scores": [-1.0], "answer": 7},
              {"step_scores": [1.0], "answer": 9}]
    assert adapter.select_best_path(flawed)["answer"] == 9


def test_select_best_path_product_breaks_tie_by_length():
    cands = [{"step_scores": [1.0] * 5, "answer": 1}, {"step_scores": [1.0] * 4, "answer": 2},
             {"step_scores": [1.0] * 5, "answer": 1}]
    win = adapter.select_best_path(cands, "product")
    assert win["answer"] == 2 and win["num_tied"] == 1
    assert adapter.select_best_path(cands, "min")["answer"] == 1


def test_select_best_path_empty_steps_score_minus_one():
    cands = [{"step_scores": [], "answer": 1}, {"step_scores": [-1.0, 1.0], "answer": 2}]
    win = adapter.select_best_path(cands, "step_penalty", 0.0)
    assert win["answer"] == 2 and win["path_score"] == 0.0


def test_best_of_n_does_not_mutate_shared_pool():
    import copy
    pool = [{"completion": "c", "steps": ["a", "b"], "answer": 3, "num_tokens": 5}]
    snapshot = copy.deepcopy(pool)
    res = adapter.run_prm_best_of_n(question="Q", candidates=pool,
                                    judge_fn=lambda p: {"text": "Verdict: VALID\nRationale: ok.", "num_tokens": 2})
    assert pool == snapshot
    assert res["judge_calls"] == 2 and res["judge_tokens"] == 4 and res["generation_calls"] == 0
    assert res["candidates"][0]["step_scores"] == [1.0, 1.0]


def test_compute_matched_samples_only_the_deficit():
    calls = []

    def gen(prompt, n, kw):
        calls.append(n)
        return [{"text": "Step 1: x\nFinal Answer: 1", "num_tokens": 3}] * n

    full = [{"completion": "", "steps": [], "answer": 2, "num_tokens": 9}] * 15
    res = adapter.run_compute_matched_self_consistency(question="Q", existing_candidates=full, generate_fn=gen)
    assert calls == [] and res["generation_calls"] == 0 and res["total_samples"] == 15
    res = adapter.run_compute_matched_self_consistency(question="Q", target_calls=3, generate_fn=gen)
    assert calls == [3] and res["generated_tokens"] == 9 and res["answer"] == 1


def test_product_logprob_multiplies_probabilities():
    assert abs(adapter.aggregate_path_score([0.9, 0.5], "product_logprob") - 0.45) < 1e-12
    assert adapter.aggregate_path_score([], "product_logprob") == -1.0


def test_pipeline_row_from_mocks():
    import run_all
    gen, judge = run_all.make_mocks()
    row = run_all.run_problem(0, "Q", gen, judge)
    assert len(row["candidates"]) == 15
    assert all("step_scores" in c for c in row["candidates"][:5])
    assert not any("step_scores" in c for c in row["candidates"][5:])
    assert row["judge_calls"] == sum(len(c["steps"]) for c in row["candidates"][:5])
    assert set(row["predictions"]) == {"greedy", "sc5", "sc15", "prm_min", "prm_product", "prm_step_penalty_0.0",
                                       "prm_step_penalty_0.5", "prm_step_penalty_2.0"}
    assert "ground_truth_answer" not in row


def test_paired_bootstrap_identical_vectors_give_zero_ci():
    import numpy as np
    from analysis import stats
    v = np.array([1, 0, 1, 1, 0] * 10, dtype=float)
    acc = {k: v.copy() for _, k in stats.METHODS}
    _, diffs = stats.bootstrap(acc)
    assert all(d["ci"] == [0.0, 0.0] and d["diff"] == 0.0 for d in diffs.values())
