# Process-reward search over reasoning paths on GSM8K

Inference-time search for multi-step math word problems with no weight updates. A policy model (`Qwen/Qwen2.5-1.5B-Instruct`, fp16) writes numbered reasoning steps, and a larger verifier (`Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4) labels every step VALID or INVALID under local validity. Step labels are aggregated into a path score (min, product or step penalty), and PRM-guided Best-of-5 is compared with greedy decoding, self-consistency over the same 5 samples, and compute-matched self-consistency over 15 samples, on a fixed 50-problem subset of the GSM8K test split.

Results on the 50 problems (one NVIDIA L4, seed 5326 + i). The 95 % CI is a percentile bootstrap over problems.

| Method | Accuracy (%) | 95 % CI | Generation calls | Judge calls | Total LLM calls | Generated tokens / problem | Judge tokens / problem |
|---|---:|---|---:|---:|---:|---:|---:|
| Greedy (N=1) | 62 | [48, 74] | 50 | 0 | 50 | 207.8 | 0.0 |
| Self-Consistency (N=5) | 76 | [64, 86] | 250 | 0 | 250 | 902.6 | 0.0 |
| Compute-Matched SC (N=15) | 82 | [72, 92] | 750 | 0 | 750 | 2774.9 | 0.0 |
| PRM Best-of-5 (min) | 76 | [64, 86] | 250 | 1203 | 1453 | 902.6 | 617.8 |
| PRM Best-of-5 (product) | 74 | [62, 84] | 250 | 1203 | 1453 | 902.6 | 617.8 |
| PRM Best-of-5 (step_penalty, lambda=2.0) | 74 | [62, 84] | 250 | 1203 | 1453 | 902.6 | 617.8 |

Accuracy is exact match on the integer after the last `Final Answer:` line; a missing or non-integer answer counts as wrong. Calls and tokens follow the bookkeeping in `run_all.py`: one generated sequence is one generation call, one judged step is one judge call, and tokens are output tokens counted from token ids.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install numpy pytest matplotlib
.venv/bin/python -m pytest -q tests/                      # unit tests, no model or GPU needed
.venv/bin/python tests/test_public_adapter_smoke.py      # adapter contract
.venv/bin/python run_all.py --mode smoke                 # whole pipeline on 3 problems with mock models
```

The GPU run used torch 2.14.1, transformers 5.22.1 and bitsandbytes 0.50.2 (`pip install -U bitsandbytes` on top of the Colab image). Every script has `--mode smoke` (mock models, CPU, writes `outputs/_smoke/`) and `--mode full`. The seed is `5326 + i` for the problem at 0-based position `i`, set once before each problem.

## Compute

The benchmark ran on one NVIDIA L4 (Google Colab) with both models on `cuda:0`. On a two-GPU machine `load_models` places the generator on `cuda:0` and the verifier on `cuda:1`. The launcher is `colab/colab_run.sh`; `kaggle/push_job.sh` runs the same command as a Kaggle T4 ×2 kernel.

```bash
bash colab/colab_run.sh new L4
bash colab/colab_run.sh setup                 # ships git archive HEAD, upgrades bitsandbytes
bash colab/colab_run.sh launch full python run_all.py --mode full --start 0 --end 50 --out /content/out/per_problem_0_50.jsonl
bash colab/colab_run.sh fetch /content/out/per_problem_0_50.jsonl runs/colab/per_problem_0_50.jsonl
bash colab/colab_run.sh stop
```

## Reproducing the results

**1. Format check.** Greedy decoding on all 50 problems, reporting step counts and the `Final Answer` line only.

```bash
python compliance_check.py --mode full --out runs/compliance.jsonl
```

**2. Benchmark run.** Greedy, the shared 5-sample pool, step-level judging, the three aggregation rules with λ ∈ {0.0, 0.5, 2.0}, and 10 extra samples, one row per problem. The run resumes from its output file. The prompts are checked against `prompts.lock` first.

```bash
python run_all.py --mode full --start 0 --end 50 --out runs/per_problem_0_50.jsonl
```

**3. Artifacts.** Merges per-problem rows and writes `outputs/per_problem.jsonl`, `results_summary.json`, `aggregation_ablation.json`, `sc_scaling_curve.json` and `logprob_bonus.json`.

```bash
python run_all.py --merge runs/per_problem_0_50.jsonl
```

**4. Analysis.** Bootstrap confidence intervals (1,000 resamples over problems, seed 5326), paired win/loss counts, the 10× greedy token budget, the scaling figures, failure-case shortlists and the blinded judge-audit sheet.

```bash
python analysis/stats.py
python analysis/plots.py
python analysis/failures.py
python analysis/audit.py draw      # then fill human_label in analysis/audit_sheet.csv
python analysis/audit.py score
```

## Attribution

- Starter files: `submission_adapter.py`, `src/utils.py` (model loading and the HF generate/judge factories), `parse_steps` in `src/extraction.py`, `tests/test_public_adapter_smoke.py`, `tests/test_token_counting.py`, `make_submission.sh` and `data/gsm8k_test_50.jsonl` come from the starter repository.
- Libraries: PyTorch, Hugging Face Transformers and Accelerate, bitsandbytes, NumPy, matplotlib.
- Models and data: Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (Qwen Team), GSM8K (Cobbe et al., 2021).

No code was copied from public implementations. Code was written with LLM assistance.

## References

- Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.
- Lightman et al. (2023). Let's Verify Step by Step. arXiv:2305.20050.
- Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. arXiv:2203.11171.
- Snell et al. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. arXiv:2408.03314.
- Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2412.15115.
