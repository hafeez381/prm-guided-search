"""
src/search.py - Path aggregation, Best-of-N search, and compute-matched search.
"""
import time
from typing import Any, Callable

from .generation import compute_majority_vote, run_self_consistency, sample_candidates
from .prm_judge import score_path

TIE_TOLERANCE = 1e-9

GenerateFn = Callable[[str, int, dict], list[dict]]
JudgeFn = Callable[[str], dict]


def aggregate_path_score(
    step_scores: list[float], 
    strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> float:
    """
    Aggregates a list of step scores (+1.0 or -1.0) into a scalar trajectory score.
    
    Rules:
      - An empty list returns -1.0 for every strategy.
      - 'min': min(step_scores).
      - 'product': p_t = clamp((r_t + 1)/2, 0.01, 0.99); returns the raw product.
      - 'step_penalty': mean(step_scores) - penalty_lambda * count(r_t == -1.0).
      - An unknown strategy raises ValueError.
    """
    if strategy not in ("min", "product", "step_penalty", "product_logprob"):
        raise ValueError(f"unknown aggregation strategy: {strategy!r}")
    if not step_scores:
        return -1.0
    if strategy == "product_logprob":
        # Bonus: step_scores are the judge's P(VALID) per step, taken from its verdict-token logits.
        product = 1.0
        for p in step_scores:
            product *= min(max(float(p), 1e-6), 1.0)
        return product
    if strategy == "min":
        return float(min(step_scores))
    if strategy == "product":
        product = 1.0
        for r in step_scores:
            product *= min(max((r + 1.0) / 2.0, 0.01), 0.99)
        return product
    num_invalid = sum(1 for r in step_scores if r == -1.0)
    return sum(step_scores) / len(step_scores) - penalty_lambda * num_invalid


def select_best_path(
    candidate_paths: list[dict[str, Any]], 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0
) -> dict[str, Any]:
    """
    Selects the winning candidate path from candidate_paths.
    
    Input Dictionary Contract:
      Each candidate dictionary in candidate_paths must contain 'step_scores' (list[float])
      and 'answer' (int | None). Any other keys present in the candidate dictionary
      are passed through unchanged in the returned copy.
    
    Selection & Tie-Breaking Rules:
      1. Compute path_score for each candidate with aggregate_path_score.
         A candidate with an empty step_scores list gets -1.0.
      2. Eligibility: if at least one candidate has a non-None answer, candidates
         with answer=None are excluded. If all answers are None, all remain eligible.
      3. Find maximum path_score among eligible candidates (tolerance 1e-9).
         The tied set is all eligible candidates within tolerance of the maximum.
      4. If the tied set has more than one candidate, compute compute_majority_vote
         over their answers. Return the lowest-index candidate in the tied set whose
         answer equals the vote. If the vote is None, return the lowest-index tied candidate.
      5. Returns a shallow COPY of the chosen dict with two added keys:
         'path_score' (float) and 'num_tied' (int, size of the tied set).
         Does not mutate the inputs.
    """
    if not candidate_paths:
        raise ValueError("select_best_path needs at least one candidate")
    scores = [aggregate_path_score(c.get("step_scores") or [], aggregation_strategy, penalty_lambda)
              for c in candidate_paths]
    eligible = [i for i, c in enumerate(candidate_paths) if c.get("answer") is not None]
    if not eligible:
        eligible = list(range(len(candidate_paths)))
    best = max(scores[i] for i in eligible)
    tied = [i for i in eligible if abs(scores[i] - best) <= TIE_TOLERANCE]
    pick = tied[0]
    if len(tied) > 1:
        vote = compute_majority_vote([candidate_paths[i].get("answer") for i in tied])
        if vote is not None:
            pick = next(i for i in tied if candidate_paths[i].get("answer") == vote)
    chosen = dict(candidate_paths[pick])
    chosen["path_score"] = float(scores[pick])
    chosen["num_tied"] = len(tied)
    return chosen


def run_prm_best_of_n(
    model: Any = None, 
    judge_model: Any = None, 
    gen_tokenizer: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    aggregation_strategy: str = "min", 
    penalty_lambda: float = 2.0, 
    candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Executes PRM-guided Best-of-N search over reasoning trajectories.
    
    If candidates is provided (shared candidate pool), does NOT sample new candidates
    (generation_calls = 0, generated_tokens = 0). Provided candidates use the
    run_self_consistency candidate format (at minimum 'steps' and 'answer').
    Otherwise, samples n candidates using run_self_consistency.
    
    Scores each candidate's steps using score_path.
    Selects the winning candidate via select_best_path.
    
    Returns:
      selected, answer, candidates [each augmented with step_scores, rationales],
      generation_calls, judge_calls, generated_tokens, judge_tokens, time_seconds
    """
    start = time.perf_counter()
    if candidates is None:
        sampled = run_self_consistency(model, gen_tokenizer, question, n=n, generate_fn=generate_fn)
        pool, generation_calls, generated_tokens = sampled["candidates"], sampled["generation_calls"], sampled["generated_tokens"]
    else:
        pool, generation_calls, generated_tokens = candidates, 0, 0
    augmented, judge_calls, judge_tokens = [], 0, 0
    for candidate in pool:
        scored = score_path(judge_model, judge_tokenizer, question, candidate, judge_fn=judge_fn)
        augmented.append(dict(candidate, **scored))
        judge_calls += scored["judge_calls"]
        judge_tokens += scored["judge_tokens"]
    selected = select_best_path(augmented, aggregation_strategy, penalty_lambda)
    return {
        "selected": selected,
        "answer": selected.get("answer"),
        "candidates": augmented,
        "generation_calls": generation_calls,
        "judge_calls": judge_calls,
        "generated_tokens": generated_tokens,
        "judge_tokens": judge_tokens,
        "time_seconds": time.perf_counter() - start,
    }


def run_compute_matched_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    target_calls: int = 15, 
    temperature: float = 0.7, 
    existing_candidates: list[dict[str, Any]] | None = None, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes compute-matched Self-Consistency aggregating target_calls paths.
    If existing_candidates are supplied, samples only the deficit:
    target_calls - len(existing_candidates). Returned candidates = existing_candidates followed by new samples, in sampling order.
    
    Returns:
      candidates, answer, generation_calls (new samples only), total_samples,
      generated_tokens (new samples only), time_seconds
    """
    start = time.perf_counter()
    existing = list(existing_candidates or [])
    deficit = max(target_calls - len(existing), 0)
    new = sample_candidates(model, tokenizer, question, deficit, temperature, generate_fn)
    pool = existing + new
    return {
        "candidates": pool,
        "answer": compute_majority_vote([c.get("answer") for c in pool]),
        "generation_calls": len(new),
        "total_samples": len(pool),
        "generated_tokens": sum(c["num_tokens"] for c in new),
        "time_seconds": time.perf_counter() - start,
    }
