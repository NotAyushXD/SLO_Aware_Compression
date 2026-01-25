# evaluation.py
"""
Evaluation utilities for GSM8K + MMLU with:
- strict, robust extraction (but tolerant to commas/whitespace)
- format_ok tracking
- per-example logging used to build performance_summary.xlsx

This version is aligned with:
- prompt_templates.py (accuracy vs slo mode)
- server.py (GSM8K early-stop + optional MMLU constrained decoding)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from prompt_templates import build_llama_formatted_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluation")


# -----------------------------------------------------------------------------
# Extraction helpers
# -----------------------------------------------------------------------------

_GSM8K_FINAL_RE = re.compile(
    r"FINAL_ANSWER\s*[:=\s]*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE
)

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    STRICT: Only accept answers that appear on a FINAL_ANSWER line.

    Returns:
        normalized numeric string (commas removed) or None
    """
    if not text:
        return None
    matches = _GSM8K_FINAL_RE.findall(text)
    if not matches:
        return None
    ans = matches[-1].strip().replace(",", "")
    # Validate parseable as float
    try:
        _ = float(ans)
        return ans
    except Exception:
        return None


_MMLU_ANSWER_RE = re.compile(r"ANSWER\s*[:=\s]*([ABCD])\b", re.IGNORECASE)
_MMLU_SINGLE_RE = re.compile(r"^\s*([ABCD])\s*$", re.IGNORECASE)

def extract_mmlu_answer(text: str) -> Optional[str]:
    """
    Prefer an explicit 'ANSWER: X' marker (accuracy mode),
    otherwise accept a single-letter output (SLO mode).
    """
    if not text:
        return None
    m = _MMLU_ANSWER_RE.findall(text)
    if m:
        return m[-1].upper()

    # Fallback: last non-empty line must be a single letter
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    m2 = _MMLU_SINGLE_RE.match(last)
    if m2:
        return m2.group(1).upper()

    # Final fallback: sometimes model outputs "B." etc
    if len(last) >= 1 and last[0].upper() in ("A", "B", "C", "D"):
        # Only accept if remaining chars are punctuation/whitespace
        tail = last[1:].strip()
        if tail == "" or all(ch in ".)]" for ch in tail):
            return last[0].upper()

    return None


# -----------------------------------------------------------------------------
# Exact match
# -----------------------------------------------------------------------------

def _to_float_str(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def exact_match_gsm8k(pred: Optional[str], truth: Any) -> bool:
    p = _to_float_str(pred)
    t = _to_float_str(truth)
    if p is None or t is None:
        return False
    # Exact for ints, tolerance for floats
    if abs(p - round(p)) < 1e-9 and abs(t - round(t)) < 1e-9:
        return int(round(p)) == int(round(t))
    return abs(p - t) < 1e-4


def exact_match_mmlu(pred: Optional[str], truth: Any) -> bool:
    if pred is None:
        return False
    t = str(truth).strip().upper()
    return pred.upper() == t


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------

class Evaluator:
    def __init__(self, server, prompt_mode: str = "slo"):
        self.server = server
        self.prompt_mode = (prompt_mode or "slo").lower()

    def evaluate(self, examples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluate on a list of examples that contain:
          - dataset: "gsm8k" or "mmlu"
          - prompt: prompt text (question or question+choices)
          - answer: ground truth (number for gsm8k; letter for mmlu)
          - difficulty: easy/medium/hard

        Returns:
          summary dict, detailed rows list (for Excel)
        """
        logger.info("=" * 70)
        logger.info(f"EVALUATING ON {len(examples)} EXAMPLES (prompt_mode={self.prompt_mode})")
        logger.info("=" * 70)

        detailed: List[Dict[str, Any]] = []
        counts = {
            "gsm8k": {"n": 0, "correct": 0, "format_ok": 0},
            "mmlu": {"n": 0, "correct": 0, "format_ok": 0},
        }

        for i, ex in enumerate(examples):
            ds = (ex.get("dataset") or ex.get("dataset_type") or "unknown").lower()
            diff = (ex.get("difficulty") or "medium").lower()

            messages, formatted_prompt, max_tokens = build_llama_formatted_prompt(
                ex,
                dataset_type=ds,
                prompt_mode=self.prompt_mode,
            )

            output_text, inf = self.server.generate(
                messages=messages,
                prompt=formatted_prompt,  # fallback if no chat template
                max_tokens=max_tokens,
                difficulty=diff,
                dataset_type=ds,
                prompt_mode=self.prompt_mode,
                do_sample=False,
            )

            gt = ex.get("answer")

            if ds == "gsm8k":
                extracted = extract_gsm8k_answer(output_text)
                format_ok = extracted is not None
                correct = exact_match_gsm8k(extracted, gt)
            elif ds == "mmlu":
                extracted = extract_mmlu_answer(output_text)
                format_ok = extracted is not None
                correct = exact_match_mmlu(extracted, gt)
            else:
                extracted = None
                format_ok = False
                correct = False

            if ds in counts:
                counts[ds]["n"] += 1
                counts[ds]["correct"] += int(correct)
                counts[ds]["format_ok"] += int(format_ok)

            detailed.append({
                "request_id": i,
                "dataset": ds,
                "difficulty": diff,
                "prompt_mode": self.prompt_mode,
                "prompt": formatted_prompt,
                "ground_truth": gt,
                "prediction": output_text,
                "extracted_answer": extracted,
                "format_ok": format_ok,
                "is_correct": correct,
                "binary_score": int(correct),
                "output_length": inf.get("output_length", 0),
                "ttft_ms": inf.get("ttft_ms", 0.0),
                "tpot_ms": inf.get("tpot_ms", 0.0),
                "throughput_tokens_per_sec": inf.get("throughput_tokens_per_sec", 0.0),
                "success": inf.get("success", False),
                "error": inf.get("error", ""),
            })

            if (i + 1) % 25 == 0:
                logger.info(f"Progress: {i+1}/{len(examples)}")

        # Aggregate metrics
        results = {}
        total_n = 0
        total_correct = 0
        total_format_ok = 0

        for ds, st in counts.items():
            n = st["n"]
            c = st["correct"]
            f = st["format_ok"]
            total_n += n
            total_correct += c
            total_format_ok += f
            results[ds.upper()] = {
                "n": n,
                "correct": c,
                "accuracy": (c / n) if n else 0.0,
                "format_ok": (f / n) if n else 0.0,
            }

        results["OVERALL"] = {
            "n": total_n,
            "correct": total_correct,
            "accuracy": (total_correct / total_n) if total_n else 0.0,
            "format_ok": (total_format_ok / total_n) if total_n else 0.0,
        }

        return results, detailed
