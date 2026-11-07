"""Greedy format-compliance check over the benchmark. Reports format only, never correctness."""
import argparse
import json
import os
import re

from src.generation import run_greedy
from src.utils import set_seed_for_problem

GATE_MIN_ANSWERED = 48
DRIFT = {"heading": r"(?m)^\s*#", "bold": r"\*\*", "boxed": r"\\boxed", "latex": r"\$\$|\\\(|\\\[|\\frac"}
DELIMITER_LINE = re.compile(r"(?im)^.*final\s+answer.*$")


def mock_generate_fn(prompt, num_samples, gen_kwargs):
    texts = ["Step 1: 2 + 2 = 4.\nFinal Answer: 4", "**Step 1:** 3 * 3 = 9\n\\boxed{9}", "### Step 1: 1 + 1 = 2\nFinal Answer: 2"]
    mock_generate_fn.i = getattr(mock_generate_fn, "i", -1) + 1
    return [{"text": texts[mock_generate_fn.i % 3], "num_tokens": 10}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--out", default="outputs/_smoke/compliance.jsonl")
    args = ap.parse_args()

    questions = [json.loads(line)["question"] for line in open("data/gsm8k_test_50.jsonl")]
    if args.mode == "smoke":
        questions, generate_fn = questions[:3], mock_generate_fn
    else:
        from src.utils import load_models, make_hf_generate_fn
        gen_model, gen_tok, _, _ = load_models("cuda:0", "cuda:1")
        generate_fn = make_hf_generate_fn(gen_model, gen_tok)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    with open(args.out, "w") as f:
        for i, question in enumerate(questions):
            set_seed_for_problem(i)
            res = run_greedy(question=question, generate_fn=generate_fn)
            lines = DELIMITER_LINE.findall(res["completion"])
            row = {
                "problem_index": i,
                "num_steps": len(res["steps"]),
                "answer": res["answer"],
                "final_answer_line": lines[-1].strip() if lines else None,
                "drift": [k for k, pat in DRIFT.items() if re.search(pat, res["completion"])],
                "generated_tokens": res["generated_tokens"],
                "time_seconds": round(res["time_seconds"], 2),
                "completion": res["completion"],
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(f"{i:2d} steps={row['num_steps']:2d} answer={str(row['answer']):>8} tokens={row['generated_tokens']:3d} "
                  f"drift={','.join(row['drift']) or '-':12s} line={row['final_answer_line']!r}", flush=True)

    answered = sum(r["answer"] is not None for r in rows)
    no_steps = [r["problem_index"] for r in rows if r["num_steps"] == 0]
    truncated = sum(r["generated_tokens"] >= 512 for r in rows)
    drift = {k: sum(k in r["drift"] for r in rows) for k in DRIFT}
    print(f"\nanswered {answered}/{len(rows)}  no_steps {no_steps}  hit_512_tokens {truncated}  drift {drift}")
    gate = answered >= min(GATE_MIN_ANSWERED, len(rows)) and not no_steps
    print("D4 compliance gate:", "PASS" if gate else "BREACH")


if __name__ == "__main__":
    main()
