"""
Evaluation utilities for baseline accuracy (MMLU + GSM8K).

Key fixes vs previous version:
1) Robust extraction that matches our *serving* prompt formats:
   - MMLU: accept 'B', 'B)', 'B.' even if extra text follows (models sometimes violate format).
   - GSM8K: prefer explicit FINAL_ANSWER marker, but fall back to common patterns (####) and
     finally to a single-number / last-number heuristic (to avoid 0% due to formatting only).
2) No unconditional printing of prompts/outputs (printing slows evaluation dramatically).
   Use verbose=True to print a few examples for debugging.
"""

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


class EvaluationMetrics:
    @staticmethod
    def extract_answer(response: str, dataset_type: str) -> str:
        response = (response or "").strip()
        if dataset_type == "mmlu":
            return EvaluationMetrics.extract_mmlu_answer(response)
        if dataset_type == "gsm8k":
            return EvaluationMetrics.extract_gsm8k_answer(response)
        return ""

    @staticmethod
    def extract_mmlu_answer(response: str) -> str:
        """Extract A/B/C/D from model output (lenient to common format violations)."""
        if not response:
            return ""
        text = response.strip().upper()

        # 1) Explicit answer markers
        m = re.search(r'(?:FINAL\s+ANSWER|ANSWER|CORRECT\s+ANSWER)\s*[:=\s]*([A-D])\b', text)
        if m:
            return m.group(1)

        # 2) If the first non-empty line starts with 'B)' or 'C.' etc
        first_line = ""
        for line in text.splitlines():
            if line.strip():
                first_line = line.strip()
                break

        if first_line:
            m = re.match(r'^\s*([A-D])\s*[\)\.:\-]?\s*$', first_line)
            if m:
                return m.group(1)
            m = re.match(r'^\s*([A-D])\s*[\)\.:\-]\s+.*$', first_line)
            if m:
                return m.group(1)

        # 3) Entire output is just a single letter with punctuation
        m = re.fullmatch(r'\s*([A-D])[\)\.]?\s*', text)
        if m:
            return m.group(1)

        return ""

    @staticmethod
    def extract_gsm8k_answer(response: str) -> str:
        """Extract numeric answer for GSM8K.

        Priority:
        1) FINAL_ANSWER / FINAL ANSWER markers (our intended format)
        2) GSM8K '####' marker
        3) If output is just a number
        4) Otherwise, take the last number in the output (lenient fallback)

        This prevents artificial 0% accuracy due to formatting drift.
        """
        if not response:
            return ""

        # 1) Explicit FINAL_ANSWER marker (underscore or space)
        # 1) Explicit FINAL_ANSWER marker (underscore or space) — take the *last* match
        matches = re.findall(r'FINAL[_\s]*ANSWER\s*[:=\s]*([-+]?\d+(?:\.\d+)?)', response, flags=re.IGNORECASE)
        if matches:
            return matches[-1]

        # 2) GSM8K delimiter
        m = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)', response)
        if m:
            return m.group(1)

        # 3) Output is a single number
        stripped = response.strip()
        m = re.fullmatch(r'[-+]?\d+(?:\.\d+)?', stripped)
        if m:
            return m.group(0)

        # 4) Last number fallback
        nums = re.findall(r'[-+]?\d+(?:\.\d+)?', response)
        if nums:
            return nums[-1]

        return ""

    @staticmethod
    def detect_hallucination(response: str, dataset_type: str) -> bool:
        # Lightweight heuristic (optional)
        if not response:
            return False
        response_lower = response.lower()
        hallucination_keywords = {
            "mmlu": ["linda", "mason", "attic", "bedroom"],
            "gsm8k": ["linda", "attic", "teosinte"],
        }
        for kw in hallucination_keywords.get(dataset_type, []):
            if kw in response_lower:
                return True
        return False

    @staticmethod
    def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
        extracted = EvaluationMetrics.extract_mmlu_answer(prediction)
        return extracted != "" and extracted == (ground_truth or "").strip().upper()

    @staticmethod
    def exact_match_gsm8k(prediction: str, ground_truth: str) -> bool:
        extracted = EvaluationMetrics.extract_gsm8k_answer(prediction)
        if not extracted:
            return False
        try:
            pred = float(extracted)
            truth = float(ground_truth)
            return abs(pred - truth) < 1e-6
        except Exception:
            return False

    @staticmethod
    def evaluate_batch(predictions: List[str], ground_truths: List[str], dataset_type: str) -> Dict:
        assert len(predictions) == len(ground_truths)
        correct = 0
        halluc = 0

        for pred, truth in zip(predictions, ground_truths):
            if EvaluationMetrics.detect_hallucination(pred, dataset_type):
                halluc += 1

            if dataset_type == "mmlu":
                ok = EvaluationMetrics.exact_match_mmlu(pred, truth)
            elif dataset_type == "gsm8k":
                ok = EvaluationMetrics.exact_match_gsm8k(pred, truth)
            else:
                ok = False

            if ok:
                correct += 1

        total = len(predictions)
        acc = correct / max(total, 1)
        return {
            "accuracy": acc,
            "em": acc,
            "correct_count": correct,
            "total_count": total,
            "hallucination_count": halluc,
            "hallucination_rate": halluc / max(total, 1),
        }


class HeldOutEvaluator:
    def __init__(self, model, data_loader: List[Dict], batch_size: int = 32, verbose: bool = False, max_verbose: int = 5):
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.verbose = verbose
        self.max_verbose = max_verbose

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
            dataset_type = example.get("dataset", "mmlu")
            difficulty = example.get("difficulty", "medium")

            try:
                formatted_prompt, max_tokens, _ = build_llama_formatted_prompt(example, dataset_type)

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

                if self.verbose and i < self.max_verbose:
                    print("\n--- PROMPT ---\n", formatted_prompt)
                    print("\n--- OUTPUT ---\n", generated_text)
                    print("\n--- GT ---\n", example.get("answer", ""))

                all_predictions.append(generated_text)
                all_ground_truths.append(example.get("answer", ""))
                all_dataset_types.append(dataset_type)
                all_prompts.append(formatted_prompt)
                all_hall.append(EvaluationMetrics.detect_hallucination(generated_text, dataset_type))

            except Exception as e:
                logger.error(f"Failed to generate for example {i}: {e}")
                all_predictions.append("")
                all_ground_truths.append(example.get("answer", ""))
                all_dataset_types.append(dataset_type)
                all_prompts.append("")
                all_hall.append(False)

            if (i + 1) % self.batch_size == 0:
                logger.info(f"Generated {i + 1}/{len(self.data_loader)} predictions")

        results: Dict[str, Dict] = {}
        for dt in sorted(set(all_dataset_types)):
            idx = [j for j, x in enumerate(all_dataset_types) if x == dt]
            preds = [all_predictions[j] for j in idx]
            truths = [all_ground_truths[j] for j in idx]
            prompts = [all_prompts[j] for j in idx]
            halls = [all_hall[j] for j in idx]

            r = EvaluationMetrics.evaluate_batch(preds, truths, dt)
            results[dt] = r

            logger.info(f"\n{dt.upper()} Results")
            logger.info(f"  Accuracy: {r['accuracy']*100:.2f}%")
            logger.info(f"  Correct: {r['correct_count']}/{r['total_count']}")
            logger.info(f"  ⚠️ Hallucination Rate: {r['hallucination_rate']*100:.2f}%")

            for ptext, ttext, predtext, is_hall in zip(prompts, truths, preds, halls):
                extracted = EvaluationMetrics.extract_answer(predtext, dt)
                if dt == "mmlu":
                    is_correct = EvaluationMetrics.exact_match_mmlu(predtext, ttext)
                elif dt == "gsm8k":
                    is_correct = EvaluationMetrics.exact_match_gsm8k(predtext, ttext)
                else:
                    is_correct = False

                detailed_results.append({
                    "dataset": dt,
                    "prompt": ptext,
                    "ground_truth": ttext,
                    "prediction": predtext,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "is_hallucination": is_hall,
                    "binary_score": 1.0 if is_correct else 0.0,
                })

        overall_correct = sum(r["correct_count"] for r in results.values())
        overall_total = sum(r["total_count"] for r in results.values())
        results["overall"] = {
            "accuracy": overall_correct / max(overall_total, 1),
            "em": overall_correct / max(overall_total, 1),
            "correct_count": overall_correct,
            "total_count": overall_total,
            "hallucination_count": sum(r.get("hallucination_count", 0) for r in results.values()),
            "hallucination_rate": sum(1 for h in all_hall if h) / max(len(all_hall), 1),
        }

        logger.info("\n" + "=" * 70)
        logger.info("OVERALL RESULTS")
        logger.info(f"Accuracy: {results['overall']['accuracy']*100:.2f}%")
        logger.info(f"Correct: {results['overall']['correct_count']}/{results['overall']['total_count']}")
        logger.info(f"⚠️ HALLUCINATION RATE: {results['overall']['hallucination_rate']*100:.2f}%")
        logger.info("=" * 70)

        return results, detailed_results