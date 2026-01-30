# evaluation.py
"""Evaluation utilities for MMLU + GSM8K.

Key goals:
- Strict answer extraction (avoid "grab last number" fallbacks).
- Separate "format adherence" from correctness.
- Save per-example debug info (including inference timing components).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from prompt_templates import build_llama_formatted_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    @staticmethod
    def extract_mmlu_answer(text: str) -> str:
        if not text:
            return ""
        t = text.strip().upper()

        m = re.search(r"(?:FINAL\s*ANSWER|ANSWER|CORRECT\s*ANSWER)\s*[:=\s]*([A-D])\b", t)
        if m:
            return m.group(1)

        m = re.fullmatch(r"\s*([A-D])[\.\)]?\s*", t)
        if m:
            return m.group(1)

        first = t.splitlines()[0].strip() if t.splitlines() else t.strip()
        m = re.fullmatch(r"([A-D])[\.\)]?", first)
        if m:
            return m.group(1)

        return ""

    @staticmethod
    def extract_gsm8k_answer(text: str) -> str:
        if not text:
            return ""

        matches = re.findall(
            r"^\s*FINAL_ANSWER\s*[:=\s]*([-+]?\d[\d,]*(?:\.\d+)?)\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not matches:
            return ""
        return matches[-1].strip().replace(",", "")

    @staticmethod
    def _normalize_number_string(s: str) -> str:
        if s is None:
            return ""
        t = str(s).strip()
        t = t.replace(",", "")
        t = t.replace("$", "").replace("₹", "").replace("€", "").replace("£", "")
        return t

    @staticmethod
    def is_correct(pred_text: str, truth: str, dataset_type: str) -> Tuple[bool, str, bool]:
        dataset_type = (dataset_type or "").lower().strip()
        truth = (truth or "").strip()

        if dataset_type == "mmlu":
            extracted = EvaluationMetrics.extract_mmlu_answer(pred_text)
            format_ok = extracted != ""
            return (format_ok and extracted == truth.upper(), extracted, format_ok)

        if dataset_type == "gsm8k":
            extracted = EvaluationMetrics.extract_gsm8k_answer(pred_text)
            format_ok = extracted != ""
            if not format_ok:
                return (False, "", False)
            try:
                pred_val = float(EvaluationMetrics._normalize_number_string(extracted))
                truth_val = float(EvaluationMetrics._normalize_number_string(truth))
                return (abs(pred_val - truth_val) < 1e-6, extracted, True)
            except Exception:
                return (False, extracted, True)

        return (False, "", False)

    @staticmethod
    def evaluate_group(preds: List[str], truths: List[str], dataset_type: str) -> Dict[str, Any]:
        correct = 0
        format_ok = 0
        total = len(preds)

        for p, t in zip(preds, truths):
            ok, _ex, fmt = EvaluationMetrics.is_correct(p, t, dataset_type)
            correct += int(ok)
            format_ok += int(fmt)

        return {
            "accuracy": correct / max(total, 1),
            "correct_count": correct,
            "total_count": total,
            "format_ok_count": format_ok,
            "format_ok_rate": format_ok / max(total, 1),
            "format_fail_count": total - format_ok,
        }


class HeldOutEvaluator:
    def __init__(self, model, data_loader: List[Dict[str, Any]], batch_size: int = 32):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = int(batch_size)

    def evaluate(self, prompt_mode: str = "slo", verbose: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        logger.info("=" * 70)
        logger.info(f"EVALUATING ON {len(self.data_loader)} EXAMPLES (prompt_mode={prompt_mode})")
        logger.info("=" * 70)

        preds_by_type: Dict[str, List[str]] = {}
        truths_by_type: Dict[str, List[str]] = {}
        detailed: List[Dict[str, Any]] = []

        for i, ex in enumerate(self.data_loader):
            dataset_type = ex.get("dataset", "mmlu")
            formatted_prompt, max_tokens, _stops = build_llama_formatted_prompt(ex, dataset_type, prompt_mode=prompt_mode)

            pred_text, inf_metrics = self.model.generate(
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                difficulty=ex.get("difficulty", "medium"),
                dataset_type=dataset_type,
                prompt_mode=prompt_mode,
            )

            truth = ex.get("answer", "")
            ok, extracted, fmt_ok = EvaluationMetrics.is_correct(pred_text, truth, dataset_type)

            preds_by_type.setdefault(dataset_type, []).append(pred_text)
            truths_by_type.setdefault(dataset_type, []).append(truth)

            detailed.append(
                {
                    "dataset": dataset_type,
                    "difficulty": ex.get("difficulty", "medium"),
                    "prompt_mode": prompt_mode,
                    "prompt": formatted_prompt,
                    "ground_truth": truth,
                    "prediction": pred_text,
                    "extracted_answer": extracted,
                    "format_ok": fmt_ok,
                    "is_correct": ok,
                    "binary_score": int(ok),
                    # Useful timing components (may be absent depending on server version)
                    "output_length": inf_metrics.get("output_length"),
                    "ttft_ms": inf_metrics.get("ttft_ms"),
                    "ttft_infer_ms": inf_metrics.get("ttft_infer_ms"),
                    "ttft_model_ms": inf_metrics.get("ttft_model_ms"),
                    "tpot_ms": inf_metrics.get("tpot_ms"),
                    "scheduler_wait_ms": inf_metrics.get("scheduler_wait_ms"),
                    "queue_wait_ms": inf_metrics.get("queue_wait_ms"),
                    "tokenize_ms": inf_metrics.get("tokenize_ms"),
                    "total_latency_ms": inf_metrics.get("total_latency_ms"),
                    "throughput_tokens_per_sec": inf_metrics.get("throughput_tokens_per_sec"),
                }
            )

            if verbose and (i < 5):
                logger.info("-" * 70)
                logger.info(f"[{i}] {dataset_type} ({ex.get('difficulty','medium')})")
                logger.info(f"PRED:\n{pred_text}")
                logger.info(f"TRUTH: {truth} | EXTRACTED: {extracted} | OK={ok} | FORMAT_OK={fmt_ok}")

        results: Dict[str, Any] = {}
        total_correct = 0
        total = 0
        total_format_ok = 0

        for dtype in sorted(preds_by_type.keys()):
            group = EvaluationMetrics.evaluate_group(preds_by_type[dtype], truths_by_type[dtype], dtype)
            results[dtype] = group
            total_correct += group["correct_count"]
            total += group["total_count"]
            total_format_ok += group["format_ok_count"]

        results["overall"] = {
            "accuracy": total_correct / max(total, 1),
            "correct_count": total_correct,
            "total_count": total,
            "format_ok_rate": total_format_ok / max(total, 1),
        }

        logger.info("\n" + "=" * 70)
        for dtype in sorted(preds_by_type.keys()):
            g = results[dtype]
            logger.info(f"{dtype.upper()} Results")
            logger.info(f"  Accuracy: {g['accuracy']*100:.2f}% ({g['correct_count']}/{g['total_count']})")
            logger.info(f"  Format OK: {g['format_ok_rate']*100:.2f}% ({g['format_ok_count']}/{g['total_count']})")
        logger.info("=" * 70)
        o = results["overall"]
        logger.info("OVERALL RESULTS")
        logger.info(f"  Accuracy: {o['accuracy']*100:.2f}% ({o['correct_count']}/{o['total_count']})")
        logger.info(f"  Format OK: {o['format_ok_rate']*100:.2f}%")
        logger.info("=" * 70)

        return results, detailed
