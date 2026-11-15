"""The two scaling figures for report Section 1, from outputs/*.json and analysis/stats.json."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SC, PRM_COLORS = "#3a6ea5", {"min": "#c0392b", "product": "#e08e0b", "step_penalty": "#2e8b57"}


def main(out_dir="outputs", fig_dir="report/figures"):
    import os
    os.makedirs(fig_dir, exist_ok=True)
    summary = {r["Method"]: r for r in json.load(open(f"{out_dir}/results_summary.json"))}
    curve = json.load(open(f"{out_dir}/sc_scaling_curve.json"))
    n = summary["Greedy (N=1)"]["Generation Calls"]
    prm = [(m, m.split("(")[1].split(",")[0].rstrip(")")) for m in summary if m.startswith("PRM")]

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False})
    for name, x_sc, x_prm, label in [
        ("fig_calls", [c["N"] for c in curve], lambda r: r["Total LLM Calls"] / n, "LLM calls per problem"),
        ("fig_tokens", [c["Avg Gen Tokens / Problem"] for c in curve],
         lambda r: r["Avg Gen Tokens / Problem"] + r["Avg Judge Tokens / Problem"], "Output tokens per problem"),
    ]:
        fig, ax = plt.subplots(figsize=(2.7, 2.1))
        ax.plot(x_sc, [c["Accuracy (%)"] for c in curve], "-o", color=SC, ms=2.5, lw=1.2, label="SC, $N$ = 1 to 15")
        for method, key in prm:
            r = summary[method]
            ax.scatter([x_prm(r)], [r["Accuracy (%)"]], color=PRM_COLORS[key], marker="D", s=22, zorder=3,
                       label=f"PRM, {key.replace('_', ' ')}")
        ax.set_xlabel(label)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.25, lw=0.5)
        if name == "fig_calls":
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/{name}.pdf", bbox_inches="tight")
        print("wrote", f"{fig_dir}/{name}.pdf")


if __name__ == "__main__":
    main(*sys.argv[1:])
