"""Shortlists for the three report failure cases, from outputs/per_problem.jsonl."""
import json
import sys
from collections import Counter


def main(path="outputs/per_problem.jsonl", out="analysis/failures.json"):
    rows = [json.loads(line) for line in open(path)]
    fp, fn, shared = [], [], []
    for r in rows:
        gt = r["ground_truth_answer"]
        for i, c in enumerate(r["candidates"][:5]):
            scores = c["step_scores"]
            if scores and c["answer"] is not None and c["answer"] != gt and min(scores) > 0:
                fp.append({"id": r["id"], "candidate": i, "answer": c["answer"], "ground_truth": gt, "steps": len(scores)})
            if c["answer"] == gt and scores and min(scores) < 0:
                bad = [t for t, s in enumerate(scores) if s < 0]
                fn.append({"id": r["id"], "candidate": i, "invalid_steps": bad,
                           "rationales": [c["rationales"][t] for t in bad]})
        top5 = Counter(c["answer"] for c in r["candidates"][:5] if c["answer"] is not None).most_common(1)
        top15 = Counter(c["answer"] for c in r["candidates"] if c["answer"] is not None).most_common(1)
        if top5 and top5[0][0] != gt and top5[0][1] >= 4 and top15 and top15[0][0] == top5[0][0] and top15[0][1] >= 12:
            shared.append({"id": r["id"], "wrong_answer": top5[0][0], "votes_of_5": top5[0][1],
                           "votes_of_15": top15[0][1], "ground_truth": gt})
    result = {"judge_false_positive_candidates": fp, "judge_false_negative_candidates": fn,
              "shared_flawed_premise_candidates": shared}
    json.dump(result, open(out, "w"), indent=2)
    print({k: len(v) for k, v in result.items()})


if __name__ == "__main__":
    main(*sys.argv[1:])
