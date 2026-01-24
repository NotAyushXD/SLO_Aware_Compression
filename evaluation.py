"""
Evaluation utilities for baseline accuracy (MMLU + GSM8K).

This version focuses on *stable, correct scoring* (especially GSM8K) by fixing:
- Robust numeric extraction (handles commas like "40,000" and decimals).
- Ground-truth normalization (commas, whitespace).
- Prompt-mode support (accuracy vs SLO) via prompt_templates.build_llama_formatted_prompt(..., prompt_mode=...).

We intentionally keep evaluation simple (exact match / numeric match) and fast.
"""

from __future__ import annotations

import re
import json
import logging
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from prompt_templates import build_llama_formatted_prompt

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_NUM_WITH_COMMAS = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
_NUM_SIMPLE = r"[-+]?\d+(?:\.\d+)?"
NUM_REGEX = rf"(?:{_NUM_WITH_COMMAS}|{_NUM_SIMPLE})"

def _clean_number_str(s: str) -> str:
    s = (s or "").strip()
    # Strip trailing punctuation
    s = re.sub(r"[\s\]\)\}\.]+$", "", s)
    # Remove commas inside numbers (45,000 -> 45000)
    s = s.replace(",", "")
    return s.strip()

def _to_float(s: str) -> Optional[float]:
    s = _clean_number_str(s)
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

# -----------------------------------------------------------------------------
# EvaluationMetrics
# -----------------------------------------------------------------------------

class EvaluationMetrics:
    @staticmethod
    def extract_answer(response: str, dataset_type: str) -> str:
        response = (response or "").strip()
        dt = (dataset_type or "").lower().strip()
        if dt == "mmlu":
            return EvaluationMetrics.extract_mmlu_answer(response)
        if dt == "gsm8k":
            return EvaluationMetrics.extract_gsm8k_answer(response)
        return ""

    @staticmethod
    def extract_mmlu_answer(response: str) -> str:
        # Accept "B", "B)", "B.", "Answer: B", etc.
        if not response:
            return ""
        m = re.search(r"\b([ABCD])\b", response.strip(), flags=re.IGNORECASE)
        if not m:
            m = re.search(r"^\s*([ABCD])[\)\.:\s]*", response.strip(), flags=re.IGNORECASE)
        return (m.group(1).upper() if m else "")

    @staticmethod
    def extract_gsm8k_answer(response: str) -> str:
        """
        GSM8K numeric extraction (robust).

        Priority:
        1) FINAL_ANSWER marker (our intended format) — take the *last* match.
        2) GSM8K '####' marker (common in many GSM8K solutions)
        3) If the first non-empty line is just a number, use it (base models often do this).
        4) Otherwise, fall back to the last number in the output.

        Supports commas: "40,000" -> "40000"
        """
        if not response:
            return ""

        # 1) Explicit FINAL_ANSWER marker
        fa_matches = re.findall(
            rf"FINAL[_\s]*ANSWER\s*[:=\s]*({NUM_REGEX})",
            response,
            flags=re.IGNORECASE,
        )
        if fa_matches:
            return _clean_number_str(fa_matches[-1])

        # 2) GSM8K delimiter
        m = re.search(rf"####\s*({NUM_REGEX})", response)
        if m:
            return _clean_number_str(m.group(1))

        # 3) First-line number (common for base LMs that ignore tags)
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        if lines:
            if re.fullmatch(NUM_REGEX, lines[0]):
                return _clean_number_str(lines[0])

        # 4) Last number fallback (handles commas)
        nums = re.findall(NUM_REGEX, response)
        if nums:
            return _clean_number_str(nums[-1])

        return ""

    @staticmethod
    def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
        pred = EvaluationMetrics.extract_mmlu_answer(prediction)
        truth = (ground_truth or "").strip().upper()
        return bool(pred) and pred == truth

    @staticmethod
    def exact_match_gsm8k(prediction: str, ground_truth: str) -> bool:
        pred_str = EvaluationMetrics.extract_gsm8k_answer(prediction)
        if not pred_str:
            return False

        pred = _to_float(pred_str)
        truth = _to_float(ground_truth)
        if pred is None or truth is None:
            return False
        return abs(pred - truth) < 1e-6

    @staticmethod
    def evaluate_batch(predictions: List[str], ground_truths: List[str], dataset_type: str) -> Dict:
        dt = (dataset_type or "").lower().strip()
        total = len(predictions)
        correct = 0
        hallucinations = 0

        for pred, truth in zip(predictions, ground_truths):
            if dt == "mmlu":
                ok = EvaluationMetrics.exact_match_mmlu(pred, truth)
            elif dt == "gsm8k":
                ok = EvaluationMetrics.exact_match_gsm8k(pred, truth)
            else:
                ok = False

            correct += int(ok)

            # Optional hallucination heuristic (kept very light)
            if not ok and dt == "gsm8k":
                if any(k in (pred or "").lower() for k in ["attic", "bedroom", "linda", "mason"]):
                    hallucinations += 1

        acc = correct / total if total else 0.0
        hall_rate = hallucinations / total if total else 0.0
        return {
            "accuracy": acc,
            "em": acc,
            "correct_count": correct,
            "total_count": total,
            "hallucination_count": hallucinations,
            "hallucination_rate": hall_rate,
        }

# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------

class HeldOutEvaluator:
    def __init__(
        self,
        model,
        data_loader: List[Dict],
        batch_size: int = 32,
        verbose: bool = False,
        max_verbose: int = 5,
        prompt_mode: str = "accuracy",
    ):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_verbose = max_verbose
        self.prompt_mode = prompt_mode

    def evaluate(self) -> Tuple[Dict, List[Dict]]:
        logger.info("=" * 70)
        logger.info(f"EVALUATING ON {len(self.data_loader)} EXAMPLES")
        logger.info("=" * 70)

        all_predictions: List[str] = []
        all_ground_truths: List[str] = []
        all_dataset_types: List[str] = []
        all_prompts: List[str] = []
        all_hall: List[bool] = []
        detailed_results: List[Dict] = []

        for i, example in enumerate(self.data_loader):
            dataset_type = (example.get("dataset") or "mmlu").lower()
            difficulty = example.get("difficulty", "medium")

            formatted_prompt, max_tokens, _ = build_llama_formatted_prompt(
                example,
                dataset_type,
                prompt_mode=self.prompt_mode,
            )

            try:
                generated_text, _metrics = self.model.generate(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    difficulty=difficulty,
                    dataset_type=dataset_type,
                )
            except TypeError:
                # Backwards compatibility: older server.generate() without dataset_type
                generated_text, _metrics = self.model.generate(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    difficulty=difficulty,
                )

            predtext = (generated_text or "").strip()
            ttext = str(example.get("answer", "")).strip()
            dt = dataset_type
            ptext = formatted_prompt

            extracted = EvaluationMetrics.extract_answer(predtext, dt)
            if dt == "mmlu":
                is_correct = EvaluationMetrics.exact_match_mmlu(predtext, ttext)
            elif dt == "gsm8k":
                is_correct = EvaluationMetrics.exact_match_gsm8k(predtext, ttext)
            else:
                is_correct = False

            # Simple hallucination flag (not used for scoring)
            halluc = False
            if not is_correct and dt == "gsm8k":
                halluc = any(k in predtext.lower() for k in ["attic", "bedroom", "linda", "mason"])

            if self.verbose and i < self.max_verbose:
                print("\n--- PROMPT ---\n", ptext)
                print("\n--- OUTPUT ---\n", predtext)
                print("\n--- EXTRACTED ---\n", extracted)
                print("\n--- GT ---\n", ttext)

            detailed_results.append({
                "dataset": dt,
                "prompt": ptext,
                "ground_truth": ttext,
                "prediction": predtext,
                "extracted_answer": extracted,
                "is_correct": bool(is_correct),
                "is_hallucination": bool(halluc),
                "binary_score": int(bool(is_correct)),
            })

            all_predictions.append(predtext)
            all_ground_truths.append(ttext)
            all_dataset_types.append(dt)
            all_prompts.append(ptext)
            all_hall.append(halluc)

        # Aggregate
        gsm_preds = [p for p, dt in zip(all_predictions, all_dataset_types) if dt == "gsm8k"]
        gsm_truth = [t for t, dt in zip(all_ground_truths, all_dataset_types) if dt == "gsm8k"]
        mmlu_preds = [p for p, dt in zip(all_predictions, all_dataset_types) if dt == "mmlu"]
        mmlu_truth = [t for t, dt in zip(all_ground_truths, all_dataset_types) if dt == "mmlu"]

        gsm_res = EvaluationMetrics.evaluate_batch(gsm_preds, gsm_truth, "gsm8k")
        mmlu_res = EvaluationMetrics.evaluate_batch(mmlu_preds, mmlu_truth, "mmlu")

        overall_correct = gsm_res["correct_count"] + mmlu_res["correct_count"]
        overall_total = gsm_res["total_count"] + mmlu_res["total_count"]
        overall_hall = gsm_res["hallucination_count"] + mmlu_res["hallucination_count"]

        overall = {
            "accuracy": overall_correct / overall_total if overall_total else 0.0,
            "em": overall_correct / overall_total if overall_total else 0.0,
            "correct_count": overall_correct,
            "total_count": overall_total,
            "hallucination_count": overall_hall,
            "hallucination_rate": overall_hall / overall_total if overall_total else 0.0,
        }

        results = {
            "gsm8k": gsm_res,
            "mmlu": mmlu_res,
            "overall": overall,
        }

        logger.info("\nGSM8K Results")
        logger.info(f"  Accuracy: {gsm_res['accuracy']*100:.2f}%")
        logger.info(f"  Correct: {gsm_res['correct_count']}/{gsm_res['total_count']}")
        if gsm_res["hallucination_rate"] > 0:
            logger.info(f"  ⚠️ Hallucination Rate: {gsm_res['hallucination_rate']*100:.2f}%")

        logger.info("\nMMLU Results")
        logger.info(f"  Accuracy: {mmlu_res['accuracy']*100:.2f}%")
        logger.info(f"  Correct: {mmlu_res['correct_count']}/{mmlu_res['total_count']}")
        if mmlu_res["hallucination_rate"] > 0:
            logger.info(f"  ⚠️ Hallucination Rate: {mmlu_res['hallucination_rate']*100:.2f}%")

        logger.info("\n" + "=" * 70)
        logger.info("OVERALL RESULTS")
        logger.info(f"Accuracy: {overall['accuracy']*100:.2f}%")
        logger.info(f"Correct: {overall['correct_count']}/{overall['total_count']}")
        if overall["hallucination_rate"] > 0:
            logger.info(f"⚠️ HALLUCINATION RATE: {overall['hallucination_rate']*100:.2f}%")
        logger.info("=" * 70)

        return results, detailed_results
