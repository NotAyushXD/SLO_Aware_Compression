"""
Prompt templates for the SLO-aware task-adaptive compression project.

We support two prompt modes:

- prompt_mode="accuracy": prioritize *answer correctness* (allow brief visible reasoning).
- prompt_mode="slo": prioritize *short outputs / low latency* (discourage visible reasoning).

You can switch modes from code by passing prompt_mode=... into build_llama_formatted_prompt(),
or by setting the environment variable PROMPT_MODE to "accuracy" or "slo".

Why this matters:
- Base (non-instruct) LLMs often underperform on math if we forbid visible reasoning.
- For a publishable story, we need both: a high-accuracy reference setting and an SLO-constrained setting.
"""

from __future__ import annotations

import os
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

PROMPT_MODE_CHOICES = ("accuracy", "slo")
DEFAULT_PROMPT_MODE = os.getenv("PROMPT_MODE", "accuracy").strip().lower()
if DEFAULT_PROMPT_MODE not in PROMPT_MODE_CHOICES:
    logger.warning(f"Unknown PROMPT_MODE='{DEFAULT_PROMPT_MODE}', falling back to 'accuracy'")
    DEFAULT_PROMPT_MODE = "accuracy"

# -----------------------------------------------------------------------------
# Conservative fallback budgets (used only when server.generate(max_tokens=None))
# -----------------------------------------------------------------------------

DIFFICULTY_TOKEN_BUDGETS = {
    "easy": 64,
    "medium": 96,
    "hard": 128,
}

# -----------------------------------------------------------------------------
# MMLU templates (same for both modes; output must be a single letter)
# -----------------------------------------------------------------------------

MMLU_TEMPLATES = {
    "easy": {
        "system": "You are a knowledgeable assistant.",
        "user_template": """Answer the following multiple-choice question.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Select the correct option.
Answer with ONLY the letter (A, B, C, or D).

Answer:""",
        # small budget; server enforces 1 token for mmlu anyway
        "max_tokens": 8,
        "stop_sequences": [],
    },
    "medium": {
        "system": "You are a knowledgeable assistant. Think carefully before answering.",
        "user_template": """Answer the following multiple-choice question.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Select the correct option.
Answer with ONLY the letter (A, B, C, or D).

Answer:""",
        "max_tokens": 8,
        "stop_sequences": [],
    },
    "hard": {
        "system": "You are a knowledgeable assistant. Be precise and avoid guessing.",
        "user_template": """Answer the following multiple-choice question.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Select the correct option.
Answer with ONLY the letter (A, B, C, or D).

Answer:""",
        "max_tokens": 8,
        "stop_sequences": [],
    },
}

# -----------------------------------------------------------------------------
# GSM8K templates
# -----------------------------------------------------------------------------
# NOTE: For accuracy, we allow brief reasoning and require a final answer line.
# For SLO mode, we encourage "answer-only" generation.

GSM8K_TEMPLATES_SLO = {
    "easy": {
        "system": "You are a careful math problem solver. Avoid arithmetic mistakes.",
        "user_template": """Solve the following math problem.

Problem: {question}

Return ONLY the final numeric answer (no words, no units).
Write it on a single line as:

FINAL_ANSWER:""",
        "max_tokens": 32,
        "stop_sequences": [],
    },
    "medium": {
        "system": "You are a careful math problem solver. Double-check calculations.",
        "user_template": """Solve the following math problem.

Problem: {question}

Return ONLY the final numeric answer (no words, no units).
Write it on a single line as:

FINAL_ANSWER:""",
        "max_tokens": 48,
        "stop_sequences": [],
    },
    "hard": {
        "system": "You are an expert math problem solver. Verify your result.",
        "user_template": """Solve the following multi-step math problem.

Problem: {question}

Return ONLY the final numeric answer (no words, no units).
Write it on a single line as:

FINAL_ANSWER:""",
        "max_tokens": 64,
        "stop_sequences": [],
    },
}

GSM8K_TEMPLATES_ACCURACY = {
    # Keep reasoning brief to avoid huge token blowups while still enabling correct arithmetic.
    "easy": {
        "system": "You are a careful math problem solver.",
        "user_template": """Solve the following math problem.

Example (format only):
Problem: A book has 10 pages and you read 3 pages. How many pages are left?
Solution: 10 - 3 = 7
FINAL_ANSWER: 7

Now solve:
Problem: {question}

Work it out step by step.
Then write the final answer on the last line exactly as:
FINAL_ANSWER: <number>

Rules:
- The final line MUST start with 'FINAL_ANSWER:'.
- After the colon, output ONLY the number (no commas, no units, no extra words).
""",
        "max_tokens": 128,
        "stop_sequences": [],
    },
    "medium": {
        "system": "You are a careful math problem solver. Show your calculations clearly.",
        "user_template": """Solve the following math problem.

Example (format only):
Problem: A book has 10 pages and you read 3 pages. How many pages are left?
Solution: 10 - 3 = 7
FINAL_ANSWER: 7

Now solve:
Problem: {question}

Work it out step by step and double-check intermediate arithmetic.
Then write the final answer on the last line exactly as:
FINAL_ANSWER: <number>

Rules:
- The final line MUST start with 'FINAL_ANSWER:'.
- After the colon, output ONLY the number (no commas, no units, no extra words).
""",
        "max_tokens": 192,
        "stop_sequences": [],
    },
    "hard": {
        "system": "You are an expert math problem solver. Reason carefully and verify the result.",
        "user_template": """Solve the following multi-step math problem.

Example (format only):
Problem: A book has 10 pages and you read 3 pages. How many pages are left?
Solution: 10 - 3 = 7
FINAL_ANSWER: 7

Now solve:
Problem: {question}

Work it out step by step. Check the final number with a quick sanity check.
Then write the final answer on the last line exactly as:
FINAL_ANSWER: <number>

Rules:
- The final line MUST start with 'FINAL_ANSWER:'.
- After the colon, output ONLY the number (no commas, no units, no extra words).
""",
        "max_tokens": 256,
        "stop_sequences": [],
    },
}

# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

def get_max_tokens(difficulty: str, dataset_type: str = "generic") -> int:
    """Conservative fallback used when max_tokens is not specified."""
    if difficulty not in DIFFICULTY_TOKEN_BUDGETS:
        logger.warning(f"Unknown difficulty '{difficulty}', using 'medium'")
        difficulty = "medium"
    return int(DIFFICULTY_TOKEN_BUDGETS[difficulty])


def build_llama_formatted_prompt(
    example: Dict[str, Any],
    dataset_type: str,
    return_tokens: bool = False,
    prompt_mode: Optional[str] = None,
) -> Tuple[str, int, list]:
    """Build a full prompt string + token budget.

    Returns:
      - if return_tokens=False: (formatted_prompt, max_tokens, stop_sequences)
      - if return_tokens=True: (system_prompt, user_prompt, answer, max_tokens)

    prompt_mode:
      - "accuracy": enables brief visible reasoning for GSM8K
      - "slo": encourages answer-only for GSM8K
    """
    if dataset_type not in ("mmlu", "gsm8k"):
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    difficulty = example.get("difficulty", "medium")
    if difficulty not in ("easy", "medium", "hard"):
        logger.warning(f"Unknown difficulty '{difficulty}', using 'medium'")
        difficulty = "medium"

    mode = (prompt_mode or DEFAULT_PROMPT_MODE).strip().lower()
    if mode not in PROMPT_MODE_CHOICES:
        logger.warning(f"Unknown prompt_mode '{mode}', using '{DEFAULT_PROMPT_MODE}'")
        mode = DEFAULT_PROMPT_MODE

    # Choose templates
    if dataset_type == "mmlu":
        template = MMLU_TEMPLATES[difficulty]
        # Parse question + choices from the raw prompt in preprocessing output.
        prompt_text = example.get("prompt", "")
        lines = prompt_text.split('\n') if isinstance(prompt_text, str) else []
        question = lines[0].strip() if lines else ""

        choices = {"A": "", "B": "", "C": "", "D": ""}
        for line in lines[1:]:
            line = line.strip()
            if line and len(line) > 2 and line[0] in "ABCD" and line[1] in ") :":
                letter = line[0]
                choice_text = line[3:].strip() if len(line) > 3 else ""
                choices[letter] = choice_text

        user_prompt = template["user_template"].format(
            question=question,
            choice_a=choices.get("A", ""),
            choice_b=choices.get("B", ""),
            choice_c=choices.get("C", ""),
            choice_d=choices.get("D", ""),
        )

    else:  # gsm8k
        gsm_templates = GSM8K_TEMPLATES_ACCURACY if mode == "accuracy" else GSM8K_TEMPLATES_SLO
        template = gsm_templates[difficulty]
        question = example.get("prompt", "")
        user_prompt = template["user_template"].format(question=question)

    system_prompt = template["system"]
    max_tokens = int(template.get("max_tokens", get_max_tokens(difficulty, dataset_type)))
    stop_sequences = list(template.get("stop_sequences", []))

    formatted_prompt = f"{system_prompt}\n\n{user_prompt}".strip()

    answer = str(example.get("answer", "")).strip()

    if return_tokens:
        return system_prompt, user_prompt, answer, max_tokens

    return formatted_prompt, max_tokens, stop_sequences
