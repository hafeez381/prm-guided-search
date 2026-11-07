"""
src/generation.py - Prompt formatting and baseline generation routines.
"""
from typing import Any, Callable

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
        "- End with one last line \"Final Answer: \" followed by the answer as a plain integer, "
        "with no units, words, currency signs or commas.\n"
        "- Write nothing after the Final Answer line."
    )


def compute_majority_vote(answers: list[int | None]) -> int | None:
    """
    Given a list of extracted answers, returns the majority answer.
    Ignores None values. Breaks ties deterministically in favor of the answer
    that appeared earliest in the list.
    """
    # TODO: Implement deterministic majority voting
    raise NotImplementedError("Implement compute_majority_vote in src/generation.py")


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
    # TODO: Implement greedy baseline decoding
    raise NotImplementedError("Implement run_greedy in src/generation.py")


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
    # TODO: Implement Self-Consistency sampling and majority voting
    raise NotImplementedError("Implement run_self_consistency in src/generation.py")
