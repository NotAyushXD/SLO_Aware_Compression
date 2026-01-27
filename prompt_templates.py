# prompt_templates.py
"""
Prompt templates for (MMLU + GSM8K) with two explicit modes:

- prompt_mode="accuracy": maximize correctness / instruction adherence.
  * GSM8K: includes 1-2 few-shot exemplars, higher max_new_tokens, ends with "Solution:".
  * MMLU: concise, answer-only.

- prompt_mode="slo": smaller budgets / shorter outputs (kept for later SLO work).

Design goals (accuracy mode):
1) Avoid "template continuation" failures (do NOT end prompts with bullet lists).
2) Avoid placeholder traps like "FINAL_ANSWER: <number>" or "[number only]" which models often copy.
3) Make the model reliably reach a FINAL_ANSWER line.

This file is *model-agnostic*: it returns plain-text prompts. If you use an Instruct model,
server.py should wrap (system,user) into the model's chat template.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

PROMPT_MODES = ("slo", "accuracy")

# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------

# Keep MMLU tiny (we want a single letter). We'll also hard-restrict in server.py.
MMLU_MAX_NEW_TOKENS = {
    "slo":      {"easy": 2, "medium": 2, "hard": 2},
    "accuracy": {"easy": 2, "medium": 2, "hard": 2},
}

# GSM8K needs room to finish. Accuracy mode budgets are intentionally larger.
GSM8K_MAX_NEW_TOKENS = {
    "slo": {
        # SLO mode should be short, but still long enough to reliably reach
        # FINAL_ANSWER on multi-step problems.
        # Keeping the same cap across difficulties improves batching efficiency.
        "easy": 96,
        "medium": 96,
        "hard": 96,
    },
    "accuracy": {
        "easy": 256,
        "medium": 384,
        "hard": 512,
    },
}

DEFAULT_DIFFICULTY = "medium"


def _norm_difficulty(difficulty: str) -> str:
    d = (difficulty or DEFAULT_DIFFICULTY).lower().strip()
    if d not in ("easy", "medium", "hard"):
        logger.warning(f"Unknown difficulty='{difficulty}', using '{DEFAULT_DIFFICULTY}'.")
        return DEFAULT_DIFFICULTY
    return d


def _norm_prompt_mode(prompt_mode: str) -> str:
    m = (prompt_mode or "slo").lower().strip()
    if m not in PROMPT_MODES:
        logger.warning(f"Unknown prompt_mode='{prompt_mode}', using 'slo'.")
        return "slo"
    return m


def get_max_tokens(difficulty: str, dataset_type: str, prompt_mode: str = "slo") -> int:
    """Return max_new_tokens for a dataset+mode+difficulty."""
    difficulty = _norm_difficulty(difficulty)
    prompt_mode = _norm_prompt_mode(prompt_mode)
    dataset_type = (dataset_type or "").lower().strip()

    if dataset_type == "mmlu":
        return MMLU_MAX_NEW_TOKENS[prompt_mode][difficulty]
    if dataset_type == "gsm8k":
        return GSM8K_MAX_NEW_TOKENS[prompt_mode][difficulty]

    # Fallback
    return 128 if prompt_mode == "accuracy" else 64


# ---------------------------------------------------------------------------
# MMLU prompt building
# ---------------------------------------------------------------------------

def _parse_mmlu_prompt(raw_prompt: str) -> Tuple[str, Dict[str, str]]:
    """
    raw_prompt format from preprocessing:
      "{question}\nA) ...\nB) ...\nC) ...\nD) ..."
    """
    raw_prompt = raw_prompt or ""
    lines = [ln.strip() for ln in raw_prompt.splitlines() if ln.strip()]
    question = lines[0] if lines else ""

    choices = {"A": "", "B": "", "C": "", "D": ""}
    for ln in lines[1:]:
        if len(ln) >= 3 and ln[0] in "ABCD" and ln[1] == ")":
            choices[ln[0]] = ln[2:].strip()
    return question, choices


def build_mmlu_prompt(example: Dict[str, Any], prompt_mode: str) -> Tuple[str, str, int, List[str]]:
    difficulty = _norm_difficulty(example.get("difficulty", DEFAULT_DIFFICULTY))
    prompt_mode = _norm_prompt_mode(prompt_mode)

    system = "You are a knowledgeable assistant. Answer accurately and concisely."

    raw = example.get("prompt", "")
    question, choices = _parse_mmlu_prompt(raw)

    # End with an explicit "Answer:" cue; do NOT end with bullet rules.
    user = (
        "Answer the following multiple-choice question.\n\n"
        f"Question: {question}\n\n"
        f"A) {choices['A']}\n"
        f"B) {choices['B']}\n"
        f"C) {choices['C']}\n"
        f"D) {choices['D']}\n\n"
        "Select the correct option.\n"
        "Respond with ONLY the letter (A, B, C, or D).\n\n"
        "Answer:"
    )

    max_new_tokens = get_max_tokens(difficulty, "mmlu", prompt_mode)
    stop_sequences = ["\n"]  # optional (server may enforce token restriction anyway)
    return system, user, max_new_tokens, stop_sequences


# ---------------------------------------------------------------------------
# GSM8K prompt building
# ---------------------------------------------------------------------------

_GSM8K_FEWSHOT = (
    "Example 1:\n"
    "Problem: A book has 10 pages and you read 3 pages. How many pages are left?\n"
    "Solution: 10 - 3 = 7\n"
    "FINAL_ANSWER: 7\n\n"
    "Example 2:\n"
    "Problem: Sara has 4 notebooks. She buys 6 more and then loses 2. How many notebooks does she have now?\n"
    "Solution: 4 + 6 - 2 = 8\n"
    "FINAL_ANSWER: 8\n\n"
)

def build_gsm8k_prompt(example: Dict[str, Any], prompt_mode: str) -> Tuple[str, str, int, List[str]]:
    difficulty = _norm_difficulty(example.get("difficulty", DEFAULT_DIFFICULTY))
    prompt_mode = _norm_prompt_mode(prompt_mode)

    system = "You are a careful math problem solver."

    rules = (
        "You will solve a grade-school math word problem.\n"
        "Write a clear solution.\n"
        "Your LAST line must be exactly:\n"
        "FINAL_ANSWER: <number>\n"
        "Where <number> is the final numeric answer (no units, no extra words).\n"
    )

    question = example.get("prompt", "")

    if prompt_mode == "accuracy":
        user = (
            f"{rules}\n\n"
            f"{_GSM8K_FEWSHOT}"
            "Now solve:\n"
            f"Problem: {question}\n\n"
            "Solution:"
        )
    else:
        # SLO mode: keep the prompt short, but strongly steer the model to
        # (1) be concise and (2) actually emit FINAL_ANSWER within the token cap.
        # A tiny format-only example improves formatting reliability without
        # adding a lot of tokens.
        user = (
            "Solve the math problem.\n"
            "Keep the solution concise (<= 6 short lines).\n"
            "IMPORTANT: your last line MUST use the exact token 'FINAL_ANSWER:' (with underscore).\n"
            "Do not write 'Final Answer', 'FINAL ANSWER', or any other variation.\n"
            "End with exactly:\n"
            "FINAL_ANSWER: <number>\n\n"
            "Example (format only):\n"
            "Problem: A book has 10 pages and you read 3 pages. How many pages are left?\n"
            "Solution: 10 - 3 = 7\n"
            "FINAL_ANSWER: 7\n\n"
            f"Problem: {question}\n\n"
            "Solution:"
        )

    max_new_tokens = get_max_tokens(difficulty, "gsm8k", prompt_mode)

    # Do not use aggressive stop strings here; server can optionally stop after FINAL_ANSWER.
    stop_sequences: List[str] = []
    return system, user, max_new_tokens, stop_sequences


# ---------------------------------------------------------------------------
# Unified entry point used by load_generator.py and evaluation.py
# ---------------------------------------------------------------------------

def build_llama_formatted_prompt(
    example: Dict[str, Any],
    dataset_type: str,
    prompt_mode: str = "slo",
) -> Tuple[str, int, List[str]]:
    """
    Returns:
        formatted_prompt: plain text "system\\n\\nuser" (server can split and wrap with chat template)
        max_new_tokens: int
        stop_sequences: list[str]
    """
    dataset_type = (dataset_type or "").lower().strip()
    prompt_mode = _norm_prompt_mode(prompt_mode)

    if dataset_type == "mmlu":
        system, user, max_new_tokens, stops = build_mmlu_prompt(example, prompt_mode)
    elif dataset_type == "gsm8k":
        system, user, max_new_tokens, stops = build_gsm8k_prompt(example, prompt_mode)
    else:
        # fallback
        system = "You are a helpful assistant."
        user = example.get("prompt", "")
        max_new_tokens = get_max_tokens(example.get("difficulty", DEFAULT_DIFFICULTY), dataset_type, prompt_mode)
        stops = []

    formatted = f"{system}\n\n{user}".strip()
    return formatted, max_new_tokens, stops


def split_system_user(formatted_prompt: str) -> Tuple[str, str]:
    """
    Split "system\\n\\nuser" back into (system, user).
    If not splittable, returns ("", formatted_prompt).
    """
    if not formatted_prompt:
        return "", ""
    parts = formatted_prompt.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", formatted_prompt.strip()
