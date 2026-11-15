"""
End-to-end benchmark pipeline: greedy, SC@5, PRM Best-of-5 (min / product / step_penalty),
compute-matched SC@15, one row per problem, then the graded artifacts.

  python run_all.py --mode full                      # all 50 problems, then build outputs/
  python run_all.py --mode full --start 0 --end 25   # one chunk, written to --out
  python run_all.py --merge runs/*/per_problem_*.jsonl   # merge chunks, then build outputs/
  python run_all.py --mode smoke                     # 3 problems with mock models, CPU only
"""
import argparse
import hashlib
import json
import os
import time

from src.generation import format_step_prompt, run_greedy, run_self_consistency
from src.prm_judge import build_prm_judge_prompt
from src.search import run_compute_matched_self_consistency, run_prm_best_of_n, select_best_path
from src.utils import set_seed_for_problem

DATA = "data/gsm8k_test_50.jsonl"
LOCK = "prompts.lock"
STRATEGIES = [("min", None), ("product", None), ("step_penalty", 0.0), ("step_penalty", 0.5), ("step_penalty", 2.0)]


def strategy_key(name, lam):
    return name if lam is None else f"{name}_{lam}"


def prompt_hashes():
    gen = format_step_prompt("X")
    judge = build_prm_judge_prompt("X", ["a"], "b")
    return {"generator": hashlib.sha256(gen.encode()).hexdigest(), "judge": hashlib.sha256(judge.encode()).hexdigest()}


def completions_hash(cands):
    return hashlib.sha256("\x00".join(c["completion"] for c in cands).encode()).hexdigest()


def assert_no_ground_truth(*objs):
    for obj in objs:
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            if isinstance(item, dict):
                bad = [k for k in item if "ground_truth" in k or "correct" in k]
                assert not bad, f"answer key leaked into the search inputs: {bad}"


def make_mocks():
    """Deterministic stand-ins for the two models, for --mode smoke."""
    texts = [
        "Step 1: 10 + 8 = 18.\nStep 2: 18 * 1 = 18.\nFinal Answer: 18",
        "Step 1: 10 + 7 = 17.\nFinal Answer: 17",
        "Step 1: 9 * 2 = 18.\nFinal Answer: 18",
        "Step 1: 3 + 3 = 6.\nStep 2: 6 + 1 = 7.\nStep 3: 7 * 2 = 14.\nFinal Answer: 14",
        "Step 1: no answer line here.",
    ]
    state = {"g": 0, "j": 0}

    def generate_fn(prompt, n, kw):
        out = []
        for _ in range(n):
            out.append({"text": texts[state["g"] % len(texts)], "num_tokens": 20 + state["g"] % 7})
            state["g"] += 1
        return out

    def judge_fn(prompt):
        state["j"] += 1
        verdict = "INVALID" if state["j"] % 4 == 0 else "VALID"
        return {"text": f"Verdict: {verdict}\nRationale: Checked the arithmetic.", "num_tokens": 9,
                "prompt_tokens": 300, "valid_prob": 0.2 if verdict == "INVALID" else 0.9}

    return generate_fn, judge_fn


def run_problem(index, question, generate_fn, judge_fn):
    set_seed_for_problem(index)
    greedy = run_greedy(question=question, generate_fn=generate_fn)
    sc5 = run_self_consistency(question=question, n=5, temperature=0.7, generate_fn=generate_fn)
    pool = sc5["candidates"]
    assert_no_ground_truth(pool)

    bon = run_prm_best_of_n(question=question, n=5, aggregation_strategy="min", candidates=pool, judge_fn=judge_fn)
    judged = [dict(c, cand_index=i) for i, c in enumerate(bon["candidates"])]
    assert completions_hash(judged) == completions_hash(pool), "PRM rows must judge the SC@5 pool"
    selected, selection_time = {}, {}
    for name, lam in STRATEGIES:
        start = time.perf_counter()
        win = select_best_path(judged, name, 2.0 if lam is None else lam)
        selection_time[strategy_key(name, lam)] = time.perf_counter() - start
        selected[strategy_key(name, lam)] = {"index": win["cand_index"], "answer": win["answer"],
                                             "path_score": win["path_score"], "num_tied": win["num_tied"]}

    cm = run_compute_matched_self_consistency(question=question, target_calls=15, temperature=0.7,
                                              existing_candidates=pool, generate_fn=generate_fn)
    assert completions_hash(cm["candidates"][:5]) == completions_hash(pool), "SC@15 must extend the SC@5 pool"

    candidates = []
    for i, c in enumerate(cm["candidates"]):
        row = {k: c[k] for k in ("completion", "steps", "answer", "num_tokens")}
        if i < 5:
            j = judged[i]
            row.update({"step_scores": j["step_scores"], "rationales": j["rationales"],
                        "judge_outputs": j["judge_outputs"], "judge_tokens": j["judge_tokens"],
                        "judge_prompt_tokens": j["judge_prompt_tokens"], "valid_probs": j["valid_probs"]})
        candidates.append(row)

    return {
        "problem_index": index,
        "seed": 5326 + index,
        "greedy": {"completion": greedy["completion"], "steps": greedy["steps"], "answer": greedy["answer"],
                   "num_tokens": greedy["generated_tokens"], "time_seconds": greedy["time_seconds"]},
        "candidates": candidates,
        "sampling_time": {"sc5": sc5["time_seconds"], "sc_extra10": cm["time_seconds"]},
        "judge_time": bon["time_seconds"],
        "judge_calls": bon["judge_calls"],
        "judge_tokens": bon["judge_tokens"],
        "selection_time": selection_time,
        "selected": selected,
        "predictions": {
            "greedy": greedy["answer"],
            "sc5": sc5["answer"],
            "sc15": cm["answer"],
            **{f"prm_{k}": v["answer"] for k, v in selected.items()},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", default=None, help="per-problem jsonl to append to (resumable)")
    ap.add_argument("--pilot", action="store_true", help="allow a full-mode run before prompts.lock exists")
    ap.add_argument("--merge", nargs="+", help="chunk files to merge into outputs/per_problem.jsonl")
    args = ap.parse_args()

    from build_artifacts import build

    if args.merge:
        rows = {}
        for path in args.merge:
            for line in open(path):
                row = json.loads(line)
                rows[row["problem_index"]] = row
        problems = [json.loads(line) for line in open(DATA)]
        assert sorted(rows) == list(range(len(problems))), f"missing problems: {set(range(len(problems))) - set(rows)}"
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/per_problem.jsonl", "w") as f:
            for i, problem in enumerate(problems):
                f.write(json.dumps({"id": problem["id"], "question": problem["question"],
                                    "ground_truth_answer": problem["ground_truth_answer"], **rows[i]}) + "\n")
        build("outputs")
        return

    problems = [json.loads(line) for line in open(DATA)]
    if args.mode == "smoke":
        generate_fn, judge_fn = make_mocks()
        end, out_dir = min(args.end or 3, 3), "outputs/_smoke"
    else:
        if not args.pilot:
            assert os.path.exists(LOCK), "prompts.lock missing: freeze the prompts before the full run"
            assert json.load(open(LOCK)) == prompt_hashes(), "prompts changed after prompts.lock was written"
        from src.utils import load_models, make_hf_generate_fn, make_hf_judge_fn_with_scores
        gen_model, gen_tok, judge_model, judge_tok = load_models("cuda:0", "cuda:1")
        generate_fn = make_hf_generate_fn(gen_model, gen_tok)
        judge_fn = make_hf_judge_fn_with_scores(judge_model, judge_tok)
        end, out_dir = args.end or len(problems), "outputs"

    out = args.out or os.path.join(out_dir, f"per_problem_{args.start}_{end}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {json.loads(line)["problem_index"] for line in open(out)}
    with open(out, "a") as f:
        for i in range(args.start, end):
            if i in done:
                continue
            t0 = time.perf_counter()
            row = run_problem(i, problems[i]["question"], generate_fn, judge_fn)
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"problem {i:2d} done in {time.perf_counter() - t0:6.1f}s  judge_calls={row['judge_calls']:3d}  "
                  f"preds={row['predictions']}", flush=True)
    print("chunk", out, "sha256", hashlib.sha256(open(out, "rb").read()).hexdigest())

    if args.mode == "smoke" or (args.start == 0 and end == len(problems)):
        with open(os.path.join(out_dir, "per_problem.jsonl"), "w") as f:
            for line in open(out):
                row = json.loads(line)
                problem = problems[row["problem_index"]]
                f.write(json.dumps({"id": problem["id"], "question": problem["question"],
                                    "ground_truth_answer": problem["ground_truth_answer"], **row}) + "\n")
        build(out_dir)


if __name__ == "__main__":
    main()
