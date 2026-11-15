"""
Judge audit. `draw`: seeded stratified sample of judged steps into a blinded sheet (no verdicts) plus a key.
`score`: join human labels with the key; confusion matrix with INVALID as the positive class.
"""
import csv
import json
import random
import sys

SHEET, KEY, OUT = "analysis/audit_sheet.csv", "analysis/audit_key.csv", "analysis/audit.json"


def draw(path="outputs/per_problem.jsonl", n_invalid=15, n_valid=20, seed=5326):
    rows = [json.loads(line) for line in open(path)]
    steps = [(r, i, t) for r in rows for i, c in enumerate(r["candidates"][:5]) for t in range(len(c["step_scores"]))]
    invalid = [s for s in steps if s[0]["candidates"][s[1]]["step_scores"][s[2]] < 0]
    valid = [s for s in steps if s[0]["candidates"][s[1]]["step_scores"][s[2]] > 0]
    rng = random.Random(seed)
    sample = rng.sample(invalid, min(n_invalid, len(invalid))) + rng.sample(valid, min(n_valid, len(valid)))
    rng.shuffle(sample)
    with open(SHEET, "w", newline="") as fs, open(KEY, "w", newline="") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["row", "problem_id", "candidate", "step", "question", "preceding_steps", "current_step", "human_label", "note"])
        wk.writerow(["row", "verdict", "rationale"])
        for k, (r, i, t) in enumerate(sample):
            c = r["candidates"][i]
            prev = "\n".join(f"Step {j + 1}: {s}" for j, s in enumerate(c["steps"][:t])) or "(no preceding steps)"
            ws.writerow([k, r["id"], i, t + 1, r["question"], prev, c["steps"][t], "", ""])
            wk.writerow([k, "VALID" if c["step_scores"][t] > 0 else "INVALID", c["rationales"][t]])
    print(f"drew {len(sample)} steps ({min(n_invalid, len(invalid))} judge-INVALID of {len(invalid)}, "
          f"{min(n_valid, len(valid))} judge-VALID of {len(valid)})")


def score():
    labels = {r["row"]: r["human_label"].strip().upper() for r in csv.DictReader(open(SHEET))}
    key = {r["row"]: r["verdict"] for r in csv.DictReader(open(KEY))}
    assert all(v in ("VALID", "INVALID") for v in labels.values()), "every row needs a VALID/INVALID label"
    cm = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    for row, human in labels.items():
        judge = key[row]
        cm[{("INVALID", "INVALID"): "TP", ("INVALID", "VALID"): "FP",
            ("VALID", "VALID"): "TN", ("VALID", "INVALID"): "FN"}[(judge, human)]] += 1
    precision = cm["TP"] / (cm["TP"] + cm["FP"]) if cm["TP"] + cm["FP"] else None
    recall = cm["TP"] / (cm["TP"] + cm["FN"]) if cm["TP"] + cm["FN"] else None
    result = {**cm, "n": len(labels), "judge_invalid": sum(v == "INVALID" for v in key.values()),
              "precision_invalid": precision, "recall_invalid": recall}
    json.dump(result, open(OUT, "w"), indent=2)
    print(result)


if __name__ == "__main__":
    {"draw": draw, "score": score}[sys.argv[1]]()
