"""
src/prm_judge.py - PRM judge prompt construction, parsing, and step scoring.
"""
import re
from typing import Any, Callable

from .utils import make_hf_judge_fn

JudgeFn = Callable[[str], dict]
# judge_fn(judge_prompt) -> {"text": str, "num_tokens": int}

_VERDICT_LINE = re.compile(r"verdict\s*:\s*\**\s*(VALID|INVALID)\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"\bINVALID\b|\bNOT\s+VALID\b", re.IGNORECASE)
_POSITIVE = re.compile(r"\bVALID\b", re.IGNORECASE)
# Shortest text ending in . ! or ? that is followed by whitespace or the end, else the rest of the line.
_RATIONALE = re.compile(r"rationale\s*:[ \t]*([^\n]*?[.!?](?=\s|$)|[^\n]*)", re.IGNORECASE)


def build_prm_judge_prompt(question: str, preceding_steps: list[str], current_step: str) -> str:
    """
    Builds the verification prompt for the PRM judge to classify current_step
    as VALID or INVALID given the problem question and preceding_steps.
    
    Judge Semantics (Local Validity):
    The judge evaluates current_step ASSUMING ALL PRECEDING STEPS ARE CORRECT.
    A step is VALID if its arithmetic is sound and it follows logically from the
    problem statement plus preceding steps as written. An error in an earlier step
    does NOT make subsequent steps invalid by itself.
    
    Autograder Literal Checks:
    - Preceding steps are rendered as 'Step 1: <text>', 'Step 2: <text>', ... in order.
    - The current step is rendered with the label 'Current Step:'.
    - The prompt contains the phrase 'assume all preceding steps are correct' (case-insensitive).
    - The prompt contains the literal strings 'Verdict: VALID', 'Verdict: INVALID', and 'Rationale:'.
    - With an empty preceding_steps list, the prompt states '(no preceding steps)'.
    """
    if preceding_steps:
        preceding = "\n".join(f"Step {i}: {step}" for i, step in enumerate(preceding_steps, start=1))
    else:
        preceding = "(no preceding steps)"
    return (
        "You are checking one step of a solution to a math word problem.\n\n"
        f"Problem: {question}\n\n"
        f"Preceding steps:\n{preceding}\n\n"
        f"Current Step: {current_step}\n\n"
        "Judge only the Current Step. Assume all preceding steps are correct, even if they contain "
        "mistakes: an error in an earlier step does not by itself make the Current Step invalid.\n"
        "The Current Step is VALID if its arithmetic is correct and it follows logically from the "
        "problem and the preceding steps as written. A step that only restates a known quantity or "
        "says what will be computed next is VALID.\n"
        "The Current Step is INVALID if it contains an arithmetic error, uses a number that does not "
        "follow from the problem or the preceding steps, or misreads what the problem says.\n\n"
        "Reply with exactly two lines and nothing else.\n"
        "Line 1 is either \"Verdict: VALID\" or \"Verdict: INVALID\".\n"
        "Line 2 is \"Rationale:\" followed by one short sentence explaining the verdict."
    )


def parse_judge_decision(judge_output: str) -> tuple[float, str]:
    r"""
    Parses the judge model's raw text completion into (score, rationale).
    
    Rules:
      1. If a line matches /verdict\s*:\s*\**\s*(VALID|INVALID)\b/i, use it.
         If several exist, use the first.
      2. Otherwise, if standalone word INVALID or phrase 'NOT VALID' appears
         (case-insensitive, word boundaries) -> -1.0.
      3. Otherwise, if standalone word VALID appears -> +1.0.
      4. Otherwise -> -1.0 (unparseable defaults to INVALID).
      
    Warning: A naive '"VALID" in text' check is wrong because "INVALID" contains "VALID".
    
    Rationale: Text after 'Rationale:' up to the first sentence end (".", "!", or "?"
    followed by whitespace or end of string, so decimals like "2.5" are not split),
    or a newline, whichever comes first, stripped. If absent, returns empty string "".
    """
    text = judge_output if isinstance(judge_output, str) else ""
    score = -1.0
    for line in text.splitlines():
        match = _VERDICT_LINE.search(line)
        if match:
            score = 1.0 if match.group(1).upper() == "VALID" else -1.0
            break
    else:
        if _NEGATIVE.search(text):
            score = -1.0
        elif _POSITIVE.search(text):
            score = 1.0
    rationale = _RATIONALE.search(text)
    return score, rationale.group(1).strip() if rationale else ""


def score_path(
    judge_model: Any = None, 
    judge_tokenizer: Any = None, 
    question: str = "", 
    candidate: dict[str, Any] | None = None, 
    judge_fn: JudgeFn | None = None
) -> dict[str, Any]:
    """
    Sequentially scores each step in candidate["steps"] using judge_fn.
    Step t is judged with preceding_steps = steps[:t].
    
    The student's code MUST call build_prm_judge_prompt to build the string passed to judge_fn,
    and MUST call parse_judge_decision on the returned text.
    
    Returns:
      step_scores, rationales, judge_calls, judge_tokens
    """
    judge_fn = judge_fn or make_hf_judge_fn(judge_model, judge_tokenizer)
    steps = list((candidate or {}).get("steps") or [])
    step_scores, rationales, outputs = [], [], []
    for t, step in enumerate(steps):
        output = judge_fn(build_prm_judge_prompt(question, steps[:t], step))
        score, rationale = parse_judge_decision(output["text"])
        step_scores.append(score)
        rationales.append(rationale)
        outputs.append(output)
    return {
        "step_scores": step_scores,
        "rationales": rationales,
        "judge_calls": len(steps),
        "judge_tokens": sum(int(o["num_tokens"]) for o in outputs),
        # Optional extras from our HF judge factory; mocks omit them.
        "judge_outputs": [o["text"] for o in outputs],
        "judge_prompt_tokens": sum(int(o.get("prompt_tokens", 0)) for o in outputs),
        "valid_probs": [o.get("valid_prob") for o in outputs],
    }
