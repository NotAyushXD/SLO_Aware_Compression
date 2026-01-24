"""
Prompt templates tuned for *serving-time* evaluation in our SLO-aware routing project.

Design goals (for baseline correctness + paper alignment):
- **Deterministic, short outputs** whenever possible (important for latency + easy parsing).
- Use dataset-specific formats:
  - MMLU: output ONLY A/B/C/D
  - GSM8K: output ONLY `FINAL_ANSWER: <number>`
- Difficulty influences token budget mainly for GSM8K (harder problems get a bit more budget),
  but we avoid huge budgets that destroy latency.

NOTE:
These templates are intentionally conservative to align with the research setting:
we care about *quality under latency constraints*, not maximum CoT verbosity.
"""

from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# DIFFICULTY-AWARE TOKEN BUDGETS (fallback if max_tokens not provided)
# ============================================================================

DIFFICULTY_TOKEN_BUDGETS = {
    "easy": 32,
    "medium": 64,
    "hard": 128,
}

# ============================================================================
# DATASET-SPECIFIC TEMPLATES
# ============================================================================

MMLU_TEMPLATES = {
    # For MMLU we always want a single-letter output; do not waste tokens.
    "easy": {
        "system": "You are a knowledgeable assistant.",
        "user_template": """Answer the following multiple-choice question.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Select the correct option.
Answer with ONLY the letter (A, B, C, or D).""",
        "max_tokens": 4,
        "stop_sequences": ["\n"],
    },
    "medium": {
        "system": "You are a knowledgeable assistant. Think carefully.",
        "user_template": """Answer the following multiple-choice question.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Select the correct option.
Answer with ONLY the letter (A, B, C, or D).""",
        "max_tokens": 4,
        "stop_sequences": ["\n"],
    },
    "hard": {
        "system": "You are a domain expert. Think carefully.",
        "user_template": """Answer the following multiple-choice question.

Question: {question}

A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Select the correct option.
Answer with ONLY the letter (A, B, C, or D).""",
        "max_tokens": 4,
        "stop_sequences": ["\n"],
    },
}

GSM8K_TEMPLATES = {
    # For GSM8K we allow a little budget for reasoning, but we require the final line format.
    "easy": {
        "system": "You are a careful math problem solver. Avoid arithmetic mistakes.",
        "user_template": """Solve the following math problem.

Problem: {question}

Think step by step internally, but DO NOT show your steps.
Return only the final answer in the exact format:

FINAL_ANSWER: <number>""",
        "max_tokens": 64,
        "stop_sequences": [],
    },
    "medium": {
        "system": "You are a careful math problem solver. Double-check calculations.",
        "user_template": """Solve the following math problem.

Problem: {question}

Think step by step internally, but DO NOT show your steps.
Return only the final answer in the exact format:

FINAL_ANSWER: <number>""",
        "max_tokens": 96,
        "stop_sequences": [],
    },
    "hard": {
        "system": "You are an expert math problem solver. Verify your result.",
        "user_template": """Solve the following multi-step math problem.

Problem: {question}

Think step by step internally, but DO NOT show your steps.
Return only the final answer in the exact format:

FINAL_ANSWER: <number>""",
        "max_tokens": 128,
        "stop_sequences": [],
    },
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_max_tokens(difficulty: str, dataset_type: str = "generic") -> int:
    if difficulty not in DIFFICULTY_TOKEN_BUDGETS:
        logger.warning(f"Unknown difficulty '{difficulty}', using 'medium'")
        difficulty = "medium"
    return int(DIFFICULTY_TOKEN_BUDGETS[difficulty])

def build_llama_formatted_prompt(
    example: Dict[str, Any],
    dataset_type: str,
    return_tokens: bool = False
) -> Tuple[str, int, list]:
    """Return (formatted_prompt, max_tokens, stop_sequences).

    If return_tokens=True, returns (system_prompt, user_prompt, answer, max_tokens).
    """
    if dataset_type not in ["mmlu", "gsm8k"]:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    difficulty = example.get("difficulty", "medium")
    if difficulty not in ["easy", "medium", "hard"]:
        logger.warning(f"Unknown difficulty '{difficulty}', using 'medium'")
        difficulty = "medium"

    templates = MMLU_TEMPLATES if dataset_type == "mmlu" else GSM8K_TEMPLATES
    template = templates[difficulty]
    system_prompt = template["system"]

    if dataset_type == "mmlu":
        prompt_text = example.get("prompt", "")
        lines = prompt_text.split('\n') if isinstance(prompt_text, str) else []
        question = lines[0].strip() if lines else ""

        choices = {"A": "", "B": "", "C": "", "D": ""}
        for line in lines[1:]:
            line = line.strip()
            if line and len(line) > 2 and line[0] in "ABCD" and line[1] in ") :":
                letter = line[0]
                choices[letter] = line[3:].strip() if len(line) > 3 else ""

        user_prompt = template["user_template"].format(
            question=question,
            choice_a=choices.get("A", ""),
            choice_b=choices.get("B", ""),
            choice_c=choices.get("C", ""),
            choice_d=choices.get("D", ""),
        )
    else:
        question = example.get("prompt", "")
        user_prompt = template["user_template"].format(question=question)

    answer = example.get("answer", "")
    max_tokens = int(template["max_tokens"])
    stop_sequences = list(template.get("stop_sequences", []))

    # Simple completion formatting (works for both base + instruct reasonably well)
    formatted_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

    if return_tokens:
        return system_prompt, user_prompt, answer, max_tokens
    return formatted_prompt, max_tokens, stop_sequences


def get_expected_format(dataset_type: str) -> str:
    if dataset_type == "mmlu":
        return "A|B|C|D"
    if dataset_type == "gsm8k":
        return "FINAL_ANSWER: <number>"
    return ""
