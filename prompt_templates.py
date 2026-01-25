# prompt_templates.py
"""
Prompt templates for two modes:
- accuracy: maximize correctness (still concise to avoid truncation)
- slo: meet latency/throughput targets (shorter prompts, tighter budgets)

Design goals (based on your debugging):
1) GSM8K: avoid truncation + "drop trailing zeros" by:
   - enforcing concise solutions (Option B)
   - early-stopping ONLY after FINAL_ANSWER number is complete
2) MMLU: improve accuracy with a tiny amount of reasoning + few-shot,
   while keeping an easily-extractable final letter.

NOTE:
- We return chat "messages" for Llama-3.x Instruct models.
- We also return a plain formatted_prompt fallback for non-chat models.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, List, Optional

# -----------------------------------------------------------------------------
# Token budgets
# -----------------------------------------------------------------------------

# Budgets are "max_new_tokens" (output tokens), not total context length.
# If you want SLO-mode to be stricter, reduce the "slo" budgets.
TOKEN_BUDGETS: Dict[str, Dict[str, Dict[str, int]]] = {
    "gsm8k": {
        # Enough for 3–6 lines of math + final answer
        "accuracy": {"easy": 192, "medium": 256, "hard": 320},
        # Still workable, but shorter and intended for speed
        "slo": {"easy": 96, "medium": 128, "hard": 192},
    },
    "mmlu": {
        # Short reasoning + ANSWER line
        "accuracy": {"easy": 72, "medium": 96, "hard": 128},
        # Single-letter (or near-single-token) answer
        "slo": {"easy": 2, "medium": 2, "hard": 2},
    },
}

DEFAULT_MODE = "slo"
DEFAULT_DIFFICULTY = "medium"


def _norm_mode(prompt_mode: Optional[str]) -> str:
    mode = (prompt_mode or DEFAULT_MODE).strip().lower()
    return "accuracy" if mode == "accuracy" else "slo"


def _norm_diff(difficulty: Optional[str]) -> str:
    d = (difficulty or DEFAULT_DIFFICULTY).strip().lower()
    return d if d in ("easy", "medium", "hard") else DEFAULT_DIFFICULTY


def get_max_tokens(difficulty: str, dataset_type: str = "gsm8k", prompt_mode: str = DEFAULT_MODE) -> int:
    """Max output tokens by difficulty/dataset/mode."""
    d = _norm_diff(difficulty)
    mode = _norm_mode(prompt_mode)
    ds = dataset_type.lower()
    if ds not in TOKEN_BUDGETS:
        ds = "gsm8k"
    return TOKEN_BUDGETS[ds][mode][d]


# -----------------------------------------------------------------------------
# Few-shot blocks (kept small; avoid bloating context)
# -----------------------------------------------------------------------------

GSM8K_FEWSHOT_ACCURACY = """Example (format):
Problem: A book has 10 pages and you read 3 pages. How many pages are left?
Solution:
10 - 3 = 7
FINAL_ANSWER: 7

Example (format):
Problem: A pack has 6 bottles. You buy 4 packs. How many bottles?
Solution:
6 * 4 = 24
FINAL_ANSWER: 24
"""

# SLO-mode uses just ONE tiny example to reduce prompt length.
GSM8K_FEWSHOT_SLO = """Example (format):
Problem: A book has 10 pages and you read 3 pages. How many pages are left?
Solution:
10 - 3 = 7
FINAL_ANSWER: 7
"""

# MMLU few-shot: keep it short & generic.
MMLU_FEWSHOT_ACCURACY = """Example (format):
Question: What is 2 + 2?
A) 1
B) 4
C) 5
D) 6
Reason: 2+2 equals 4.
ANSWER: B

Example (format):
Question: Which is a mammal?
A) Shark
B) Salmon
C) Dolphin
D) Trout
Reason: Dolphins are mammals; the others are fish.
ANSWER: C
"""


# -----------------------------------------------------------------------------
# Template builders
# -----------------------------------------------------------------------------

def _parse_mmlu_prompt(raw: str) -> str:
    """
    MMLU examples in your processed jsonl appear as a single string that already
    contains the question + options. We keep it as-is, but normalize whitespace.
    """
    raw = (raw or "").strip()
    return raw


def build_messages(example: Dict[str, Any], dataset_type: str, prompt_mode: str = DEFAULT_MODE) -> Tuple[List[Dict[str, str]], int]:
    """
    Build chat messages (system + user) plus max_new_tokens.

    Returns:
        messages, max_new_tokens
    """
    ds = dataset_type.lower()
    mode = _norm_mode(prompt_mode)
    diff = _norm_diff(example.get("difficulty", DEFAULT_DIFFICULTY))
    max_tokens = get_max_tokens(diff, ds, mode)

    if ds == "gsm8k":
        system = "You are a careful math problem solver."
        problem = (example.get("prompt") or "").strip()

        # Option B: concise solutions to avoid truncation + keep latency down.
        if mode == "accuracy":
            fewshot = GSM8K_FEWSHOT_ACCURACY
            user = f"""Solve the following math problem.

{fewshot}
Now solve:
Problem: {problem}

Write a SHORT solution (max 4 lines). Do NOT restate the problem.
End with EXACTLY one final line:
FINAL_ANSWER: <number>

Rules:
- The final line MUST start with 'FINAL_ANSWER:'.
- After the colon, output ONLY the number (no commas, no units, no extra words).
- Do not write anything after the FINAL_ANSWER line.

Solution:"""
        else:
            fewshot = GSM8K_FEWSHOT_SLO
            user = f"""Solve the following math problem.

{fewshot}
Now solve:
Problem: {problem}

Be concise (max 3–4 lines).
End with:
FINAL_ANSWER: <number>

Solution:"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return messages, max_tokens

    if ds == "mmlu":
        # Accuracy mode: allow 1 short sentence of reasoning + final ANSWER line.
        # SLO mode: answer letter only.
        question_block = _parse_mmlu_prompt(example.get("prompt") or "")

        if mode == "accuracy":
            system = "You are a knowledgeable assistant. Answer multiple-choice questions accurately."
            user = f"""Answer the following multiple-choice question.

{MMLU_FEWSHOT_ACCURACY}
Now answer:
{question_block}

Write:
- Reason: <one short sentence>
- ANSWER: <letter>

Constraints:
- The final line MUST be: ANSWER: A|B|C|D
- Do not add anything after the ANSWER line.
"""
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return messages, max_tokens
        else:
            system = "You are a knowledgeable assistant. Answer multiple-choice questions accurately."
            user = f"""Answer the following multiple-choice question.

{question_block}

Select the correct option.
Answer with ONLY the letter (A, B, C, or D)."""
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return messages, max_tokens

    # Fallback (unknown dataset)
    system = "You are a helpful assistant."
    user = (example.get("prompt") or "").strip()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return messages, get_max_tokens(diff, "gsm8k", mode)


def build_llama_formatted_prompt(
    example: Dict[str, Any],
    dataset_type: str,
    prompt_mode: str = DEFAULT_MODE,
) -> Tuple[List[Dict[str, str]], str, int]:
    """
    Backwards-compatible wrapper used by the rest of the pipeline.

    Returns:
        (messages, formatted_prompt_fallback, max_new_tokens)
    """
    messages, max_tokens = build_messages(example, dataset_type, prompt_mode=prompt_mode)

    # Plain-text fallback prompt (non-chat models)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    user = messages[1]["content"] if len(messages) > 1 else ""
    formatted = (system.strip() + "\n\n" + user.strip()).strip()

    return messages, formatted, max_tokens
