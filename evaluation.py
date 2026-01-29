"""evaluation.py

Accuracy/format evaluation for GSM8K + MMLU for the SLO-aware compression harness.

Key design choice:
- We evaluate *task correctness* (GSM8K numeric answer, MMLU choice letter)
- We track *format compliance* separately (did the model output the expected format)

The evaluation runs against a `SingleVariantServer` (server.py) via:
    outputs, per_request_metrics = server.generate([...])

This keeps the evaluation logic identical across variants (base/med/cheap).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from prompt_templates import build_prompt


# -----------------------------
# Parsing helpers
# -----------------------------

_MMLU_LETTERS = {"A", "B", "C", "D"}


def _extract_mmlu_answer(text: str) -> Optional[str]:
    """Return first A/B/C/D letter found at start of output (robust)."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None

    # Common patterns: "A", "A.", "A)" etc.
    m = re.match(r"^\s*([A-D])\b", s)
    if m:
        return m.group(1)

    # As a fallback, find the first occurrence of an isolated letter.
    m = re.search(r"\b([A-D])\b", s)
    if m:
        return m.group(1)

    return None


def _normalize_number_string(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("%", "")
    return s


def _extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract a numeric final answer from GSM8K style output.

    We accept the last number in the output (covers many formats).
    """
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    # Prefer a line like "Answer: 42" or "Final: 42"
    m = re.search(r"(?i)(final|answer)\s*[:=]\s*([-+]?\d[\d,]*\.?\d*)", s)
    if m:
        return _normalize_number_string(m.group(2))

    # Otherwise take the last number in the entire output.
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", s)
    if not nums:
        return None
    return _normalize_number_string(nums[-1])


def _answers_match(pred: str, gt: str) -> bool:
    """Loose match used for GSM8K numeric answers."""
    if pred is None or gt is None:
        return False
    p = _normalize_number_string(str(pred))
    g = _normalize_number_string(str(gt))

    # Exact string match first
    if p == g:
        return True

    # Numeric fallback
    try:
        return float(p) == float(g)
    except Exception:
        return False


# -----------------------------
# Evaluation
# -----------------------------


def evaluate_dataset(
    server,
    examples: List[Dict],
    prompt_mode: str,
    max_tokens: int = 256,
    save_path: Optional[str] = None,
) -> Dict:
    """Evaluate a list of normalized examples.

    Each example must have keys:
      - dataset_type: "gsm8k"|"mmlu"
      - difficulty: "easy"|"medium"|"hard"
      - question: str
      - choices: Optional[List[str]]
      - answer: str
    """

    per_example_logs: List[Dict] = []

    totals = {"n": 0, "correct": 0, "format_ok": 0}
    by_dataset = {
        "gsm8k": {"n": 0, "correct": 0, "format_ok": 0},
        "mmlu": {"n": 0, "correct": 0, "format_ok": 0},
    }

    for idx, ex in enumerate(examples):
        dataset_type = ex["dataset_type"]
        difficulty = ex.get("difficulty", "easy")
        question = ex.get("question", "")
        choices = ex.get("choices")
        gt = ex.get("answer", "")

        prompt = build_prompt(
            dataset_type=dataset_type,
            question=question,
            choices=choices,
            prompt_mode=prompt_mode,
            difficulty=difficulty,
        )

        outputs, metrics_list = server.generate(
            [prompt],
            dataset_type=dataset_type,
            difficulty=difficulty,
            prompt_mode=prompt_mode,
            max_new_tokens=max_tokens,
            temperature=0.0,
        )

        text = outputs[0] if outputs else ""
        metrics = metrics_list[0] if metrics_list else {}

        if dataset_type == "mmlu":
            pred = _extract_mmlu_answer(text)
            format_ok = pred in _MMLU_LETTERS
            is_correct = (pred is not None) and (pred.strip().upper() == str(gt).strip().upper())
        else:
            pred = _extract_gsm8k_answer(text)
            format_ok = pred is not None
            is_correct = pred is not None and _answers_match(pred, gt)

        totals["n"] += 1
        totals["correct"] += int(is_correct)
        totals["format_ok"] += int(format_ok)

        if dataset_type in by_dataset:
            by_dataset[dataset_type]["n"] += 1
            by_dataset[dataset_type]["correct"] += int(is_correct)
            by_dataset[dataset_type]["format_ok"] += int(format_ok)

        per_example_logs.append(
            {
                "idx": idx,
                "dataset_type": dataset_type,
                "difficulty": difficulty,
                "prompt_mode": prompt_mode,
                "ground_truth": gt,
                "prediction": pred,
                "correct": bool(is_correct),
                "format_ok": bool(format_ok),
                "output_text": text,
                "metrics": metrics,
            }
        )

    summary = {
        "prompt_mode": prompt_mode,
        "overall": {
            "n": totals["n"],
            "accuracy": (totals["correct"] / totals["n"]) if totals["n"] else 0.0,
            "format_ok": (totals["format_ok"] / totals["n"]) if totals["n"] else 0.0,
            "correct": totals["correct"],
        },
        "by_dataset": {
            k: {
                "n": v["n"],
                "accuracy": (v["correct"] / v["n"]) if v["n"] else 0.0,
                "format_ok": (v["format_ok"] / v["n"]) if v["n"] else 0.0,
                "correct": v["correct"],
            }
            for k, v in by_dataset.items()
        },
        "logs": per_example_logs,
    }

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary
