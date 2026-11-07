"""
src/extraction.py - Step and answer extraction utilities.
"""
import re

_FINAL_ANSWER = re.compile(r"[*#`_]*\s*final\s+answer\s*[*#`_]*\s*:?\s*[*#`_]*", re.IGNORECASE)
_NUMBER = re.compile(r"-?\s*\d+(?:\.\d+)?")


def parse_steps(completion_text: str) -> list[str]:
    """
    Parses completion text and returns a list of individual step strings.
    Must tolerate markdown decoration around labels (e.g. '**Step 1:**', '### Step 2:')
    and strip leftover '*' or '#' from captured text.
    """
    if not completion_text or not isinstance(completion_text, str):
        return []

    # Pattern matching Step markers with optional markdown decoration (#, *, `, :, .)
    pattern = r"(?:^|\n)\s*(?:[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)(.*?)(?=(?:\n\s*[#*`\s]*Step\s*\d+\s*[:.]?[#*`\s]*)|(?:\n\s*[#*`\s]*Final Answer)|$)"
    matches = re.findall(pattern, completion_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = []
    for m in matches:
        text = m.strip().strip("*#`").strip()
        if text:
            cleaned.append(text)
    return cleaned


def extract_answer(completion_text: str) -> int | None:
    """
    Extracts the final numeric integer from the completion.

    Required Behaviors:
      - 'Final Answer: 18' -> 18
      - 'Final Answer: $1,234' -> 1234
      - 'Final Answer: 18.00' -> 18
      - 'Final Answer: 18.5' -> None (non-integer)
      - 'Final Answer: -5' -> -5
      - 'Final Answer: 18 dollars.' -> 18
      - '**Final Answer:** 18' -> 18
      - 'final answer 18' (case-insensitive, colon optional) -> 18
      - Multiple 'Final Answer' lines -> use the LAST one
      - No 'Final Answer' present -> None
      - Empty string or None-like garbage -> None, and never raise
    """
    try:
        if not isinstance(completion_text, str):
            return None
        delimiters = list(_FINAL_ANSWER.finditer(completion_text))
        if not delimiters:
            return None
        line = completion_text[delimiters[-1].end():].split("\n", 1)[0]
        # Read the result of an equation ("7 * 2 = 14" -> 14), not its first operand.
        line = line.rsplit("=", 1)[-1]
        line = line.replace("\u2212", "-").replace("$", "").replace(",", "")
        number = _NUMBER.search(line)
        if number is None:
            return None
        whole, _, fraction = number.group().replace(" ", "").partition(".")
        return int(whole) if fraction.strip("0") == "" else None
    except Exception:
        return None
