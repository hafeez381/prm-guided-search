"""
Builds the graded artifacts from per_problem.jsonl alone:
results_summary.json, aggregation_ablation.json, sc_scaling_curve.json (+ logprob_bonus.json).
Correctness is computed here and nowhere else.
"""
import json
import os
import sys

from src.generation import compute_majority_vote
from src.search import select_best_path
from src.utils import save_aggregation_ablation, save_results_summary, save_sc_scaling_curve

PRM_METHODS = [
    ("PRM Best-of-5 (min)", "min"),
    ("PRM Best-of-5 (product)", "product"),
    ("PRM Best-of-5 (step_penalty, lambda=2.0)", "step_penalty_2.0"),
]
ABLATION = [("min", None, "min"), ("product", None, "product"), ("step_penalty", 0.0, "step_penalty_0.0"),
            ("step_penalty", 0.5, "step_penalty_0.5"), ("step_penalty", 2.0, "step_penalty_2.0")]


def pct(hits):
    return round(100.0 * sum(hits) / len(hits), 2)


def load_rows(out_dir):
    return [json.loads(line) for line in open(os.path.join(out_dir, "per_problem.jsonl"))]


def build(out_dir="outputs"):
    rows = load_rows(out_dir)
    n = len(rows)
    gt = [r["ground_truth_answer"] for r in rows]
    right = lambda preds: [p is not None and p == g for p, g in zip(preds, gt)]
    pred = lambda key: [r["predictions"][key] for r in rows]

    gen5 = [sum(c["num_tokens"] for c in r["candidates"][:5]) for r in rows]
    gen15 = [sum(c["num_tokens"] for c in r["candidates"][:15]) for r in rows]
    judge_calls = sum(r["judge_calls"] for r in rows)
    judge_tokens = [r["judge_tokens"] for r in rows]
    assert judge_calls == sum(len(c["steps"]) for r in rows for c in r["candidates"][:5])
    assert sum(judge_tokens) == sum(c["judge_tokens"] for r in rows for c in r["candidates"][:5])
    assert sum(judge_tokens) <= 96 * judge_calls, "judge_tokens must count output tokens only"
    assert all(len(r["candidates"]) == 15 for r in rows)
    sc5_time = sum(r["sampling_time"]["sc5"] for r in rows)
    judge_time = sum(r["judge_time"] for r in rows)

    summary = [
        {"Method": "Greedy (N=1)", "Accuracy (%)": pct(right(pred("greedy"))), "Generation Calls": n,
         "Judge Calls": 0, "Total LLM Calls": n,
         "Avg Gen Tokens / Problem": round(sum(r["greedy"]["num_tokens"] for r in rows) / n, 2),
         "Avg Judge Tokens / Problem": 0.0,
         "Total Time (s)": round(sum(r["greedy"]["time_seconds"] for r in rows), 2)},
        {"Method": "Self-Consistency (N=5)", "Accuracy (%)": pct(right(pred("sc5"))), "Generation Calls": 5 * n,
         "Judge Calls": 0, "Total LLM Calls": 5 * n, "Avg Gen Tokens / Problem": round(sum(gen5) / n, 2),
         "Avg Judge Tokens / Problem": 0.0, "Total Time (s)": round(sc5_time, 2)},
        {"Method": "Compute-Matched SC (N=15)", "Accuracy (%)": pct(right(pred("sc15"))),
         "Generation Calls": 15 * n, "Judge Calls": 0, "Total LLM Calls": 15 * n,
         "Avg Gen Tokens / Problem": round(sum(gen15) / n, 2), "Avg Judge Tokens / Problem": 0.0,
         "Total Time (s)": round(sc5_time + sum(r["sampling_time"]["sc_extra10"] for r in rows), 2)},
    ]
    for method, key in PRM_METHODS:
        summary.append({
            "Method": method, "Accuracy (%)": pct(right(pred(f"prm_{key}"))), "Generation Calls": 5 * n,
            "Judge Calls": judge_calls, "Total LLM Calls": 5 * n + judge_calls,
            "Avg Gen Tokens / Problem": round(sum(gen5) / n, 2),
            "Avg Judge Tokens / Problem": round(sum(judge_tokens) / n, 2),
            "Total Time (s)": round(sc5_time + judge_time + sum(r["selection_time"][key] for r in rows), 2),
        })

    ablation = []
    for name, lam, key in ABLATION:
        ablation.append({"Strategy": name, "Lambda": lam, "Accuracy (%)": pct(right(pred(f"prm_{key}"))),
                         "Tie Rate (%)": pct([r["selected"][key]["num_tied"] > 1 for r in rows])})

    curve = []
    for k in range(1, 16):
        votes = [compute_majority_vote([c["answer"] for c in r["candidates"][:k]]) for r in rows]
        curve.append({"N": k, "Accuracy (%)": pct(right(votes)),
                      "Avg Gen Tokens / Problem": round(sum(sum(c["num_tokens"] for c in r["candidates"][:k])
                                                            for r in rows) / n, 2)})

    by_method = {s["Method"]: s["Accuracy (%)"] for s in summary}
    assert curve[4]["Accuracy (%)"] == by_method["Self-Consistency (N=5)"], "curve N=5 must equal SC@5"
    assert curve[14]["Accuracy (%)"] == by_method["Compute-Matched SC (N=15)"], "curve N=15 must equal SC@15"
    for (method, key), ab in zip(PRM_METHODS, [ablation[0], ablation[1], ablation[4]]):
        assert ab["Accuracy (%)"] == by_method[method], f"ablation row {key} must match the summary"

    save_results_summary(summary, os.path.join(out_dir, "results_summary.json"))
    save_aggregation_ablation(ablation, os.path.join(out_dir, "aggregation_ablation.json"))
    save_sc_scaling_curve(curve, os.path.join(out_dir, "sc_scaling_curve.json"))

    def step_probs(c):
        # A step whose output had no verdict token falls back to its binary score, clamped like "product".
        return [p if p is not None else (0.99 if s > 0 else 0.01)
                for p, s in zip(c.get("valid_probs") or [None] * len(c["step_scores"]), c["step_scores"])]

    if any(p is not None for r in rows for c in r["candidates"][:5] for p in c.get("valid_probs") or []):
        picks = [select_best_path([{"step_scores": step_probs(c), "answer": c["answer"]}
                                   for c in r["candidates"][:5]], "product_logprob") for r in rows]
        agree = [(p >= 0.5) == (s > 0) for r in rows for c in r["candidates"][:5]
                 for p, s in zip(c["valid_probs"], c["step_scores"]) if p is not None]
        bonus = {"Strategy": "product_logprob", "Logit comparison": "first verdict token, VALID vs INVALID",
                 "Accuracy (%)": pct(right([p["answer"] for p in picks])),
                 "Tie Rate (%)": pct([p["num_tied"] > 1 for p in picks]),
                 "Verdict agreement (%)": pct(agree)}
        with open(os.path.join(out_dir, "logprob_bonus.json"), "w") as f:
            json.dump(bonus, f, indent=2)

    for s in summary:
        print(json.dumps(s))
    for a in ablation:
        print(json.dumps(a))
    print("curve", [c["Accuracy (%)"] for c in curve])


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "outputs")
