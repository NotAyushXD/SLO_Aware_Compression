# prompt_templates.py
"""Prompt templates for (MMLU + GSM8K) with two explicit modes.

- prompt_mode="accuracy": maximize correctness / instruction adherence.
- prompt_mode="slo":      shorter prompts / smaller token budgets (used for SLO-mode).

This file is intentionally model-agnostic: it returns plain-text prompts.
If you use an instruct/chat model, server.py will wrap (system,user) with the
model's chat template.

Important design constraints for *reliable evaluation*:
- Avoid prompts that end with bullet lists (models often "continue the list").
- Avoid placeholder traps like "FINAL_ANSWER: <number>" without an explicit example.
- Always include a clear "Answer:" / "Solution:" cue.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

PROMPT_MODES = ("slo", "accuracy")
DEFAULT_DIFFICULTY = "medium"

# ---------------------------------------------------------------------------
# Token budgets
# ---------------------------------------------------------------------------

# MMLU should be a single letter.
MMLU_MAX_NEW_TOKENS = {
    "slo": {"easy": 2, "medium": 2, "hard": 2},
    "accuracy": {"easy": 2, "medium": 2, "hard": 2},
}

# GSM8K needs room to finish. Accuracy mode budgets are intentionally larger.
# SLO mode should still be large enough to reach FINAL_ANSWER on multi-step items.
GSM8K_MAX_NEW_TOKENS = {
    "slo": {
        "easy": 160,
        "medium": 160,
        "hard": 160,
    },
    "accuracy": {
        "easy": 256,
        "medium": 384,
        "hard": 512,
    },
}


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


def get_max_tokens(arg1: str, arg2: str, prompt_mode: str = "slo") -> int:
    """Return max_new_tokens for a dataset+mode+difficulty.

    Backward compatible with older call sites:
      - get_max_tokens(difficulty, dataset_type, prompt_mode)
      - get_max_tokens(dataset_type, difficulty, prompt_mode)
    """

    a1 = (arg1 or "").lower().strip()
    a2 = (arg2 or "").lower().strip()

    # Detect argument order.
    if a1 in ("mmlu", "gsm8k") and a2 in ("easy", "medium", "hard"):
        dataset_type, difficulty = a1, a2
    else:
        difficulty, dataset_type = a1, a2

    difficulty = _norm_difficulty(difficulty)
    prompt_mode = _norm_prompt_mode(prompt_mode)
    dataset_type = (dataset_type or "").lower().strip()

    if dataset_type == "mmlu":
        return int(MMLU_MAX_NEW_TOKENS[prompt_mode][difficulty])
    if dataset_type == "gsm8k":
        return int(GSM8K_MAX_NEW_TOKENS[prompt_mode][difficulty])

    # Fallback
    return 128 if prompt_mode == "accuracy" else 64


# ---------------------------------------------------------------------------
# MMLU prompt building
# ---------------------------------------------------------------------------


def _parse_mmlu_prompt(raw_prompt: str) -> Tuple[str, Dict[str, str]]:
    """raw_prompt format from preprocessing:

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
    stop_sequences = ["\n"]
    return system, user, int(max_new_tokens), stop_sequences


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
        "Show your working as short equations (not long prose).\n"
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
        # SLO mode: do NOT remove reasoning entirely; just discourage verbosity.
        # The prior "<= 6 lines" constraint was too destructive for GSM8K.
        user = (
            f"{rules}\n"
            "Keep it compact: aim for <= ~8 short lines.\n"
            "Do not add explanations unrelated to solving.\n\n"
            "Example (format only):\n"
            "Problem: A book has 10 pages and you read 3 pages. How many pages are left?\n"
            "Solution: 10 - 3 = 7\n"
            "FINAL_ANSWER: 7\n\n"
            f"Problem: {question}\n\n"
            "Solution:"
        )

    max_new_tokens = get_max_tokens(difficulty, "gsm8k", prompt_mode)
    stop_sequences: List[str] = []
    return system, user, int(max_new_tokens), stop_sequences


# ---------------------------------------------------------------------------
# Unified entry point used by load_generator.py and evaluation.py
# ---------------------------------------------------------------------------


def build_llama_formatted_prompt(
    example: Dict[str, Any],
    dataset_type: str,
    prompt_mode: str = "slo",
) -> Tuple[str, int, List[str]]:
    """Return a plain-text prompt that is "system\n\nuser"."""

    dataset_type = (dataset_type or "").lower().strip()
    prompt_mode = _norm_prompt_mode(prompt_mode)

    if dataset_type == "mmlu":
        system, user, max_new_tokens, stops = build_mmlu_prompt(example, prompt_mode)
    elif dataset_type == "gsm8k":
        system, user, max_new_tokens, stops = build_gsm8k_prompt(example, prompt_mode)
    else:
        system = "You are a helpful assistant."
        user = example.get("prompt", "")
        max_new_tokens = get_max_tokens(example.get("difficulty", DEFAULT_DIFFICULTY), dataset_type, prompt_mode)
        stops = []

    formatted = f"{system}\n\n{user}".strip()
    return formatted, int(max_new_tokens), stops


def split_system_user(formatted_prompt: str) -> Tuple[str, str]:
    """Split "system\n\nuser" back into (system, user)."""

    if not formatted_prompt:
        return "", ""
    parts = formatted_prompt.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", formatted_prompt.strip()
