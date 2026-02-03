# evaluation.py
"""Evaluation utilities for MMLU + GSM8K.

Paper-facing goals:
- Strict answer extraction as the *primary* metric (reproducible + hard to game).
- A second "parseable" metric for robustness/sensitivity analysis.
- Separate "format adherence" from correctness.
- Save per-example debug info (including timing components).

Strict vs Parseable (GSM8K):
- Strict: requires a standalone line: `FINAL_ANSWER: <number>`
- Parseable: if strict fails, uses conservative recovery rules (see answer_utils.py)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from answer_utils import (
    extract_gsm8k_parseable,
    extract_gsm8k_strict,
    extract_mmlu_answer,
    normalize_number_string,
    numbers_equal,
)
from prompt_templates import build_llama_formatted_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    @staticmethod
    def extract_mmlu_answer(text: str) -> str:
        return extract_mmlu_answer(text)

    @staticmethod
    def extract_gsm8k_answer(text: str) -> str:
        # Strict
        return extract_gsm8k_strict(text)

    @staticmethod
    def extract_gsm8k_answer_parseable(text: str) -> str:
        return extract_gsm8k_parseable(text)

    @staticmethod
    def is_correct(pred_text: str, truth: str, dataset_type: str) -> Tuple[bool, str, bool]:
        """Strict (paper primary) correctness + format adherence.

        Returns: (is_correct, extracted_answer, format_ok)
        """
        dataset_type = (dataset_type or "").lower().strip()

        if dataset_type == "mmlu":
            extracted = extract_mmlu_answer(pred_text)
            fmt_ok = bool(extracted)
            ok = bool(extracted and extracted == (truth or "").strip().upper())
            return ok, extracted, fmt_ok

        # GSM8K
        extracted = extract_gsm8k_strict(pred_text)
        fmt_ok = bool(extracted)
        ok = bool(extracted and numbers_equal(extracted, truth))
        return ok, extracted, fmt_ok

    @staticmethod
    def is_correct_parseable(pred_text: str, truth: str, dataset_type: str) -> Tuple[bool, str, bool]:
        """Parseable correctness + format adherence (sensitivity metric)."""
        dataset_type = (dataset_type or "").lower().strip()

        if dataset_type == "mmlu":
            # MMLU is already short; treat parseable == strict.
            extracted = extract_mmlu_answer(pred_text)
            fmt_ok = bool(extracted)
            ok = bool(extracted and extracted == (truth or "").strip().upper())
            return ok, extracted, fmt_ok

        extracted = extract_gsm8k_parseable(pred_text)
        fmt_ok = bool(extracted)
        ok = bool(extracted and numbers_equal(extracted, truth))
        return ok, extracted, fmt_ok

    @staticmethod
    def evaluate_group(preds: List[str], truths: List[str], dataset_type: str) -> Dict[str, Any]:
        total = len(preds)
        correct = 0
        fmt_ok = 0

        correct_p = 0
        fmt_ok_p = 0

        for p, t in zip(preds, truths):
            ok, _ex, f = EvaluationMetrics.is_correct(p, t, dataset_type)
            if ok:
                correct += 1
            if f:
                fmt_ok += 1

            okp, _exp, fp = EvaluationMetrics.is_correct_parseable(p, t, dataset_type)
            if okp:
                correct_p += 1
            if fp:
                fmt_ok_p += 1

        return {
            "total_count": total,
            # strict
            "correct_count": correct,
            "accuracy": (correct / total) if total else 0.0,
            "format_ok_count": fmt_ok,
            "format_ok_rate": (fmt_ok / total) if total else 0.0,
            # parseable
            "correct_parseable_count": correct_p,
            "accuracy_parseable": (correct_p / total) if total else 0.0,
            "format_ok_parseable_count": fmt_ok_p,
            "format_ok_parseable_rate": (fmt_ok_p / total) if total else 0.0,
            "strict_to_parseable_accuracy_gain": ((correct_p - correct) / total) if total else 0.0,
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
            if dataset_type == "gsm8k":
                truth = normalize_number_string(str(truth))

            ok, extracted, fmt_ok = EvaluationMetrics.is_correct(pred_text, truth, dataset_type)
            ok_p, extracted_p, fmt_ok_p = EvaluationMetrics.is_correct_parseable(pred_text, truth, dataset_type)

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
                    "raw_prediction": inf_metrics.get("raw_text", None),  # optional server-side field
                    # strict
                    "extracted_answer": extracted,
                    "format_ok": fmt_ok,
                    "is_correct": ok,
                    "binary_score": int(ok),
                    # parseable
                    "extracted_answer_parseable": extracted_p,
                    "format_ok_parseable": fmt_ok_p,
                    "is_correct_parseable": ok_p,
                    "parseable_score": int(ok_p),
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
                    "server_backend": inf_metrics.get("backend"),
                    "server_variant": inf_metrics.get("variant"),
                }
            )

            if verbose and (i < 5):
                logger.info("-" * 70)
                logger.info(f"[{i}] {dataset_type} ({ex.get('difficulty','medium')})")
                logger.info(f"PRED:\n{pred_text}")
                logger.info(f"TRUTH: {truth} | STRICT: {extracted} OK={ok} FMT={fmt_ok} | PARSE: {extracted_p} OK={ok_p} FMT={fmt_ok_p}")

        results: Dict[str, Any] = {}
        total_correct = 0
        total = 0
        total_format_ok = 0

        total_correct_p = 0
        total_format_ok_p = 0

        for dtype in sorted(preds_by_type.keys()):
            group = EvaluationMetrics.evaluate_group(preds_by_type[dtype], truths_by_type[dtype], dtype)
            results[dtype] = group
            total_correct += group["correct_count"]
            total += group["total_count"]
            total_format_ok += group["format_ok_count"]

            total_correct_p += group["correct_parseable_count"]
            total_format_ok_p += group["format_ok_parseable_count"]

        results["overall"] = {
            "total_count": total,
            # strict
            "correct_count": total_correct,
            "accuracy": (total_correct / total) if total else 0.0,
            "format_ok_rate": (total_format_ok / total) if total else 0.0,
            # parseable
            "correct_parseable_count": total_correct_p,
            "accuracy_parseable": (total_correct_p / total) if total else 0.0,
            "format_ok_parseable_rate": (total_format_ok_p / total) if total else 0.0,
        }

        # Console-friendly summary (keep strict primary, add parseable if helpful)
        def _fmt_pct(x: float) -> str:
            return f"{(x * 100.0):.2f}%"

        if "gsm8k" in results:
            g = results["gsm8k"]
            logger.info("\n" + "=" * 70)
            logger.info("GSM8K Results (strict primary)")
            logger.info(f"  Accuracy: {_fmt_pct(g['accuracy'])} ({g['correct_count']}/{g['total_count']})")
            logger.info(f"  Format OK: {_fmt_pct(g['format_ok_rate'])} ({g['format_ok_count']}/{g['total_count']})")
            logger.info("GSM8K Results (parseable sensitivity)")
            logger.info(f"  Accuracy: {_fmt_pct(g['accuracy_parseable'])} ({g['correct_parseable_count']}/{g['total_count']})")
            logger.info(f"  Format OK: {_fmt_pct(g['format_ok_parseable_rate'])} ({g['format_ok_parseable_count']}/{g['total_count']})")

        if "mmlu" in results:
            m = results["mmlu"]
            logger.info("MMLU Results")
            logger.info(f"  Accuracy: {_fmt_pct(m['accuracy'])} ({m['correct_count']}/{m['total_count']})")
            logger.info(f"  Format OK: {_fmt_pct(m['format_ok_rate'])} ({m['format_ok_count']}/{m['total_count']})")

        o = results["overall"]
        logger.info("=" * 70)
        logger.info("OVERALL RESULTS (strict primary)")
        logger.info(f"  Accuracy: {_fmt_pct(o['accuracy'])} ({o['correct_count']}/{o['total_count']})")
        logger.info(f"  Format OK: {_fmt_pct(o['format_ok_rate'])}")
        logger.info("OVERALL RESULTS (parseable sensitivity)")
        logger.info(f"  Accuracy: {_fmt_pct(o['accuracy_parseable'])} ({o['correct_parseable_count']}/{o['total_count']})")
        logger.info(f"  Format OK: {_fmt_pct(o['format_ok_parseable_rate'])}")
        logger.info("=" * 70)

        return results, detailed
