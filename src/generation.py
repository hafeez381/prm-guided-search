"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
import time
from typing import Any, Callable

from .extraction import extract_answer, parse_steps
from .utils import GREEDY_GEN_KWARGS, SAMPLING_GEN_KWARGS, make_hf_generate_fn

GenerateFn = Callable[[str, int, dict], list[dict]]
# generate_fn(raw_prompt, num_samples, gen_kwargs) -> list of {"text": str, "num_tokens": int}
# gen_kwargs carries do_sample/temperature/top_p/top_k/repetition_penalty/max_new_tokens


def format_step_prompt(question: str) -> str:
    """
    Formats the raw problem question into a prompt that instructs the 
    model to emit numbered steps ('Step 1: ...') and 'Final Answer: <int>'.
    
    Returns a raw instruction string (no chat template applied).
    """
    return (
        "Solve the following math word problem step by step.\n\n"
        f"Problem: {question}\n\n"
        "Write your solution in plain text using exactly this format:\n"
        "Step 1: <first deduction or calculation>\n"
        "Step 2: <next deduction or calculation>\n"
        "...\n"
        "Step K: <final calculation>\n"
        "Final Answer: <integer>\n\n"
        "Rules:\n"
        "- Start every step on a new line with \"Step <number>:\" and put one deduction or calculation in each step.\n"
        "- Write each calculation with its result, for example 16 - 7 = 9.\n"
        "- Use plain text only: no markdown, no headings, no bold, no LaTeX, no \\boxed{}.\n"
        "- Write math the way you would type it in a text message, for example (1/2) * 2 = 1 or 3 x 4 = 12. "
        "Never use \\( \\), \\[ \\], \\frac, \\times or \\text.\n"
        "- End with one last line \"Final Answer: \" followed by the answer as a plain integer, "
        "with no units, words, currency signs or commas.\n"
        "- Always finish with the Final Answer line, and write nothing after it."
    )


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    counts: dict[int, int] = {}
    for answer in answers:
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return None
    # dicts keep insertion order, so max() returns the earliest-seen answer among equal counts
    return max(counts, key=counts.get)


def to_candidate(output: dict) -> dict[str, Any]:
    """Turns one generate_fn output into a candidate record."""
    text = output["text"]
    return {
        "completion": text,
        "steps": parse_steps(text),
        "answer": extract_answer(text),
        "num_tokens": int(output["num_tokens"]),
    }


def sample_candidates(model: Any, tokenizer: Any, question: str, n: int, temperature: float,
                      generate_fn: GenerateFn | None) -> list[dict[str, Any]]:
    """Draws n temperature-sampled step-structured completions in one generate_fn call."""
    if n <= 0:
        return []
    generate_fn = generate_fn or make_hf_generate_fn(model, tokenizer)
    gen_kwargs = dict(SAMPLING_GEN_KWARGS, temperature=temperature)
    outputs = generate_fn(format_step_prompt(question), n, gen_kwargs)
    assert len(outputs) == n, f"generate_fn returned {len(outputs)} samples, expected {n}"
    return [to_candidate(o) for o in outputs]


def run_greedy(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Executes greedy decoding for a single question.
    Calls generate_fn with num_samples=1 and gen_kwargs containing do_sample=False,
    temperature=None, top_p=None, top_k=None.
    
    Returns:
      prompt, completion, steps, answer, generation_calls, generated_tokens, time_seconds
    """
    generate_fn = generate_fn or make_hf_generate_fn(model, tokenizer)
    prompt = format_step_prompt(question)
    start = time.perf_counter()
    output = generate_fn(prompt, 1, dict(GREEDY_GEN_KWARGS))[0]
    elapsed = time.perf_counter() - start
    candidate = to_candidate(output)
    return {
        "prompt": prompt,
        "completion": candidate["completion"],
        "steps": candidate["steps"],
        "answer": candidate["answer"],
        "generation_calls": 1,
        "generated_tokens": candidate["num_tokens"],
        "time_seconds": elapsed,
    }


def run_self_consistency(
    model: Any = None, 
    tokenizer: Any = None, 
    question: str = "", 
    n: int = 5, 
    temperature: float = 0.7, 
    generate_fn: GenerateFn | None = None
) -> dict[str, Any]:
    """
    Samples n diverse reasoning trajectories and aggregates candidate answers via majority vote.
    
    Returns:
      candidates [each: completion, steps, answer, num_tokens], answer, 
      generation_calls, generated_tokens, time_seconds
    """
    start = time.perf_counter()
    candidates = sample_candidates(model, tokenizer, question, n, temperature, generate_fn)
    elapsed = time.perf_counter() - start
    return {
        "candidates": candidates,
        "answer": compute_majority_vote([c["answer"] for c in candidates]),
        "generation_calls": len(candidates),
        "generated_tokens": sum(c["num_tokens"] for c in candidates),
        "time_seconds": elapsed,
    }
