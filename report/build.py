"""Writes report/tables/*.tex and report/tables/numbers.tex from outputs/*.json and analysis/*.json only."""
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = os.path.join(ROOT, "report", "tables")
load = lambda p: json.load(open(os.path.join(ROOT, p)))

SHORT = {"Greedy (N=1)": "Greedy ($N=1$)", "Self-Consistency (N=5)": "Self-Consistency ($N=5$)",
         "Compute-Matched SC (N=15)": "Compute-Matched SC ($N=15$)", "PRM Best-of-5 (min)": "PRM Best-of-5 (min)",
         "PRM Best-of-5 (product)": "PRM Best-of-5 (product)",
         "PRM Best-of-5 (step_penalty, lambda=2.0)": "PRM Best-of-5 (step penalty, $\\lambda=2$)"}
KEY = {"Greedy (N=1)": "greedy", "Self-Consistency (N=5)": "sc5", "Compute-Matched SC (N=15)": "sc15",
       "PRM Best-of-5 (min)": "prm_min", "PRM Best-of-5 (product)": "prm_product",
       "PRM Best-of-5 (step_penalty, lambda=2.0)": "prm_step_penalty_2.0"}
PRM_NAME = {"prm_min": "min", "prm_product": "product", "prm_step_penalty_2.0": "step penalty"}


def fmt(x, d=1):
    return f"{x:,.{d}f}".replace(",", "{,}")


def sg(x):
    return f"${x:+.0f}$" if round(x) else "$0$"


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text + ("\\bottomrule\n" if name != "numbers.tex" else ""))


def main():
    summary = load("outputs/results_summary.json")
    ablation = load("outputs/aggregation_ablation.json")
    stats = load("analysis/stats.json")
    n = summary[0]["Generation Calls"]

    lines = []
    for r in summary:
        lo, hi = stats["ci95"][KEY[r["Method"]]]
        lines.append(f"{SHORT[r['Method']]} & {r['Accuracy (%)']:.0f} & [{lo:.0f}, {hi:.0f}] & {r['Generation Calls']} & "
                     f"{fmt(r['Judge Calls'], 0)} & {fmt(r['Total LLM Calls'], 0)} & {fmt(r['Avg Gen Tokens / Problem'])} & "
                     f"{fmt(r['Avg Judge Tokens / Problem'])} & {fmt(r['Total Time (s)'], 0)} \\\\")
    write("results.tex", "\n".join(lines) + "\n")

    lines = []
    for p in ("prm_min", "prm_product", "prm_step_penalty_2.0"):
        for b, bname in (("sc5", "SC@5"), ("sc15", "SC@15")):
            wl = stats["win_loss"][f"{p} vs {b}"]
            d = stats["paired_differences"][f"{p} - {b}"]
            lines.append(f"{PRM_NAME[p]} vs {bname} & {wl['prm_wins']} & {wl['baseline_wins']} & {wl['both_right']} & "
                         f"{wl['both_wrong']} & {sg(d['diff'])} & [{sg(d['ci'][0])}, {sg(d['ci'][1])}] \\\\")
    write("paired.tex", "\n".join(lines) + "\n")

    lines = []
    for a in ablation:
        lam = "--" if a["Lambda"] is None else f"{a['Lambda']:.1f}"
        name = a["Strategy"].replace("_", " ")
        lines.append(f"{name} & {lam} & {a['Accuracy (%)']:.0f} & {a['Tie Rate (%)']:.0f} \\\\")
    if os.path.exists(os.path.join(ROOT, "outputs/logprob_bonus.json")):
        b = load("outputs/logprob_bonus.json")
        lines.append(f"product of $p_t$ (verdict logits) & -- & {b['Accuracy (%)']:.0f} & {b['Tie Rate (%)']:.0f} \\\\")
    write("ablation.tex", "\n".join(lines) + "\n")

    acc = {r["Method"]: r["Accuracy (%)"] for r in summary}
    bud = stats["budget"]
    macros = {
        "accGreedy": acc["Greedy (N=1)"], "accSCfive": acc["Self-Consistency (N=5)"],
        "accSCfifteen": acc["Compute-Matched SC (N=15)"], "accPRMmin": acc["PRM Best-of-5 (min)"],
        "accPRMprod": acc["PRM Best-of-5 (product)"], "accPRMpen": acc["PRM Best-of-5 (step_penalty, lambda=2.0)"],
        "judgeCalls": summary[3]["Judge Calls"], "Kbar": round(summary[3]["Judge Calls"] / (5 * n), 2),
        "callsPRMperProblem": round(summary[3]["Total LLM Calls"] / n, 1),
        "greedyTokens": bud["greedy_tokens"], "budgetTenX": bud["budget_10x"], "prmSpent": bud["prm_spent"],
        "prmGenTokens": bud["prm_gen_tokens"], "prmJudgeOut": bud["prm_judge_output_tokens"],
        "prmJudgeIn": bud["prm_judge_prompt_tokens"], "judgeInPerCall": bud["judge_prompt_tokens_per_call"],
        "judgeOutPerCall": bud["judge_output_tokens_per_call"], "genPerSample": bud["gen_tokens_per_sample"],
        "largestSCN": bud["largest_sc_n_within_budget"],
        "scTokensFifteen": bud["sc_tokens_by_n"]["15"], "scTokensFive": bud["sc_tokens_by_n"]["5"],
    }
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "outputs/per_problem.jsonl"))]
    s_min = lambda c: min(c["step_scores"]) if c["step_scores"] else -1.0
    macros.update({
        "coverFive": sum(any(c["answer"] == r["ground_truth_answer"] for c in r["candidates"][:5]) for r in rows),
        "coverFifteen": sum(any(c["answer"] == r["ground_truth_answer"] for c in r["candidates"]) for r in rows),
        "mixedPools": sum(len({s_min(c) for c in r["candidates"][:5]}) > 1 for r in rows),
        "allInvalidPools": sum(all(s_min(c) < 0 for c in r["candidates"][:5]) for r in rows),
        "allValidPools": sum(all(s_min(c) > 0 for c in r["candidates"][:5]) for r in rows),
        "minChangedAnswer": sum(r["predictions"]["prm_min"] != r["predictions"]["sc5"] for r in rows),
        "stepsInvalidPct": round(100 * sum(s < 0 for r in rows for c in r["candidates"][:5] for s in c["step_scores"])
                                 / summary[3]["Judge Calls"]),
        "judgeTimeOverSC": round((summary[3]["Total Time (s)"] - summary[1]["Total Time (s)"]) / summary[1]["Total Time (s)"], 1),
        "judgeInOverOut": round(bud["judge_prompt_tokens_per_call"] / bud["judge_output_tokens_per_call"], 1),
        "runSeconds": round(summary[0]["Total Time (s)"] + summary[2]["Total Time (s)"]
                            + summary[3]["Total Time (s)"] - summary[1]["Total Time (s)"]),
        "accSCeleven": next(c["Accuracy (%)"] for c in load("outputs/sc_scaling_curve.json") if c["N"] == 11),
    })
    for k in ("min", "product"):
        row = next(a for a in ablation if a["Strategy"] == k)
        macros[f"tie{k.capitalize()}"] = row["Tie Rate (%)"]
    for lam, name in ((0.0, "Zero"), (0.5, "Half"), (2.0, "Two")):
        row = next(a for a in ablation if a["Strategy"] == "step_penalty" and a["Lambda"] == lam)
        macros[f"accPen{name}"] = row["Accuracy (%)"]
        macros[f"tiePen{name}"] = row["Tie Rate (%)"]
    if os.path.exists(os.path.join(ROOT, "outputs/logprob_bonus.json")):
        b = load("outputs/logprob_bonus.json")
        macros.update({"accLogprob": b["Accuracy (%)"], "tieLogprob": b["Tie Rate (%)"],
                       "agreeLogprob": b["Verdict agreement (%)"]})
    if os.path.exists(os.path.join(ROOT, "analysis/audit.json")):
        au = load("analysis/audit.json")
        macros.update({"auditN": au["n"], "auditTP": au["TP"], "auditFP": au["FP"], "auditTN": au["TN"],
                       "auditFN": au["FN"], "auditJudgeInvalid": au["judge_invalid"],
                       "auditPrecision": round(100 * au["precision_invalid"]) if au["precision_invalid"] is not None else "--",
                       "auditRecall": round(100 * au["recall_invalid"]) if au["recall_invalid"] is not None else "--"})

    def show(v):
        if isinstance(v, float):
            return f"{v:.0f}" if v == round(v) and abs(v) >= 10 else f"{v:g}"
        return f"{v:,}".replace(",", "{,}") if isinstance(v, int) else str(v)

    write("numbers.tex", "".join(f"\\newcommand{{\\{k}}}{{{show(v)}}}\n" for k, v in macros.items()))
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
