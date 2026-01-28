"""
prompt_templates.py

Centralized prompt templates + decoding parameters for the evaluation harness.

Paper-friendly changes:
- SLO-mode GSM8K prompt is now *concise but not brittle* (accuracy drop was too large).
- SLO-mode GSM8K max_new_tokens increased modestly to preserve correctness while still
  encouraging short outputs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Per-dataset max_new_tokens by prompt_mode
GSM8K_MAX_NEW_TOKENS: Dict[str, int] = {
    "accuracy": 256,
    # Increased from 96 -> 160 to avoid brittle truncation and large GSM8K accuracy drop.
    "slo": 160,
}

MMLU_MAX_NEW_TOKENS: Dict[str, int] = {
    "accuracy": 4,
    "slo": 4,
}


def build_gsm8k_prompt(question: str, prompt_mode: str = "accuracy") -> Tuple[str, List[str]]:
    prompt_mode = (prompt_mode or "accuracy").lower()

    if prompt_mode == "slo":
        user_prompt = f"""You are running under a strict latency SLO. Be concise but correct.

Rules:
- Do the math carefully.
- Write only brief calculations (no explanations). Keep it short (≤ 10 short lines).
- End with exactly one line: FINAL_ANSWER: <number>

Question:
{question}
"""
    else:
        # Accuracy mode: allows fuller reasoning
        user_prompt = f"""Solve the math word problem carefully.

Instructions:
- Show your work step-by-step.
- End with exactly one line: FINAL_ANSWER: <number>

Question:
{question}
"""

    stop_sequences: List[str] = []
    return user_prompt, stop_sequences


def build_mmlu_prompt(question: str, choices: Dict[str, str], prompt_mode: str = "accuracy") -> Tuple[str, List[str]]:
    prompt_mode = (prompt_mode or "accuracy").lower()

    choice_lines = "\n".join([f"{k}. {v}" for k, v in choices.items()])

    if prompt_mode == "slo":
        user_prompt = f"""Answer the multiple-choice question under a strict latency SLO.

Rules:
- Think silently.
- Output ONLY the single letter (A/B/C/D). No extra text.

Question:
{question}

Choices:
{choice_lines}
"""
    else:
        user_prompt = f"""Answer the multiple-choice question.

Instructions:
- Choose the best option.
- Output ONLY the single letter (A/B/C/D). No extra text.

Question:
{question}

Choices:
{choice_lines}
"""

    # Strong stop: newline or end-of-turn token.
    stop_sequences: List[str] = ["\n", "<|eot_id|>"]
    return user_prompt, stop_sequences


def get_max_tokens(dataset_type: str, prompt_mode: str) -> int:
    dataset_type = (dataset_type or "").lower()
    prompt_mode = (prompt_mode or "accuracy").lower()

    if dataset_type == "gsm8k":
        return GSM8K_MAX_NEW_TOKENS.get(prompt_mode, GSM8K_MAX_NEW_TOKENS["accuracy"])
    if dataset_type == "mmlu":
        return MMLU_MAX_NEW_TOKENS.get(prompt_mode, MMLU_MAX_NEW_TOKENS["accuracy"])

    return 128


def build_llama_formatted_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Minimal Llama-3 style chat formatting that works both with and without HF chat_template.

    Note: If you rely on tokenizer.apply_chat_template elsewhere, keep this consistent.
    """
    system_prompt = (system_prompt or "").strip()
    user_prompt = (user_prompt or "").strip()

    if system_prompt:
        return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def build_prompt(example: Dict, prompt_mode: str) -> Tuple[str, List[str], int]:
    dataset_type = (example.get("dataset_type") or "").lower()
    prompt_mode = (prompt_mode or "accuracy").lower()

    # A light system prompt helps standardize behavior across datasets.
    system_prompt = (
        "You are a helpful assistant. Follow the requested output format exactly. "
        "Do not include any additional commentary beyond what is requested."
    )

    if dataset_type == "gsm8k":
        user_prompt, stop_sequences = build_gsm8k_prompt(example["question"], prompt_mode=prompt_mode)
        max_new_tokens = get_max_tokens("gsm8k", prompt_mode)
    elif dataset_type == "mmlu":
        user_prompt, stop_sequences = build_mmlu_prompt(example["question"], example["choices"], prompt_mode=prompt_mode)
        max_new_tokens = get_max_tokens("mmlu", prompt_mode)
    else:
        user_prompt = example.get("question", "")
        stop_sequences = []
        max_new_tokens = 128

    formatted = build_llama_formatted_prompt(system_prompt, user_prompt)
    return formatted, stop_sequences, max_new_tokens
