"""Bootstrap CIs, paired differences, win/loss counts and the 10x-greedy token budget, from per_problem.jsonl."""
import json
import sys

import numpy as np

METHODS = [("Greedy (N=1)", "greedy"), ("Self-Consistency (N=5)", "sc5"), ("Compute-Matched SC (N=15)", "sc15"),
           ("PRM Best-of-5 (min)", "prm_min"), ("PRM Best-of-5 (product)", "prm_product"),
           ("PRM Best-of-5 (step_penalty, lambda=2.0)", "prm_step_penalty_2.0")]
PRM = ["prm_min", "prm_product", "prm_step_penalty_2.0"]


def correctness(rows):
    return {key: np.array([r["predictions"][key] is not None and r["predictions"][key] == r["ground_truth_answer"]
                           for r in rows], dtype=float) for _, key in METHODS}


def percentile_ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [round(100 * lo, 2), round(100 * hi, 2)]


def bootstrap(acc, resamples=1000, seed=5326):
    n = len(next(iter(acc.values())))
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    ci = {k: percentile_ci(v[idx].mean(axis=1)) for k, v in acc.items()}
    diffs = {f"{p} - {b}": {"diff": round(100 * (acc[p] - acc[b]).mean(), 2),
                            "ci": percentile_ci((acc[p][idx] - acc[b][idx]).mean(axis=1))}
             for p in PRM for b in ("sc5", "sc15")}
    return ci, diffs


def win_loss(acc):
    out = {}
    for p in PRM:
        for b in ("sc5", "sc15"):
            a, c = acc[p].astype(bool), acc[b].astype(bool)
            out[f"{p} vs {b}"] = {"prm_wins": int((a & ~c).sum()), "baseline_wins": int((~a & c).sum()),
                                  "both_right": int((a & c).sum()), "both_wrong": int((~a & ~c).sum())}
    return out


def budget(rows):
    greedy = sum(r["greedy"]["num_tokens"] for r in rows)
    gen5 = sum(c["num_tokens"] for r in rows for c in r["candidates"][:5])
    judge_out = sum(r["judge_tokens"] for r in rows)
    judge_in = sum(c.get("judge_prompt_tokens", 0) for r in rows for c in r["candidates"][:5])
    calls = sum(r["judge_calls"] for r in rows)
    sc_tokens = [sum(c["num_tokens"] for r in rows for c in r["candidates"][:k]) for k in range(1, 16)]
    limit = 10 * greedy
    largest_n = max([k for k, t in zip(range(1, 16), sc_tokens) if t <= limit], default=0)
    return {"greedy_tokens": greedy, "budget_10x": limit,
            "prm_spent": gen5 + judge_out, "prm_gen_tokens": gen5, "prm_judge_output_tokens": judge_out,
            "prm_judge_prompt_tokens": judge_in, "judge_calls": calls,
            "judge_prompt_tokens_per_call": round(judge_in / max(calls, 1), 1),
            "judge_output_tokens_per_call": round(judge_out / max(calls, 1), 1),
            "gen_tokens_per_sample": round(gen5 / (5 * len(rows)), 1),
            "sc_tokens_by_n": dict(zip(range(1, 16), sc_tokens)),
            "largest_sc_n_within_budget": largest_n, "prm_within_budget": gen5 + judge_out <= limit}


def main(path="outputs/per_problem.jsonl", out="analysis/stats.json"):
    rows = [json.loads(line) for line in open(path)]
    acc = correctness(rows)
    ci, diffs = bootstrap(acc)
    result = {"accuracy": {k: round(100 * v.mean(), 2) for k, v in acc.items()}, "ci95": ci,
              "paired_differences": diffs, "win_loss": win_loss(acc), "budget": budget(rows)}
    json.dump(result, open(out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
