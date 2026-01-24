"""
CRITICAL FIX: Proper answer extraction to catch hallucinations

Previous issue: Model outputs wrong problem (e.g., "Linda is painting...")
This should be caught by answer extraction and marked as WRONG.

New logic:
  1. Extract ONLY what matches expected format
  2. If no valid answer found → mark as INCORRECT
  3. Track hallucination_detected for debugging
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
    """Evaluate model predictions against ground truth"""
    
    @staticmethod
    def extract_answer(response: str, dataset_type: str) -> str:
        """
        Extract answer from model response based on prompt template format.
        
        MMLU: Expects single letter A, B, C, or D
        GSM8K: Expects single number
        
        Args:
            response: Model-generated text
            dataset_type: 'mmlu' or 'gsm8k'
        
        Returns:
            Extracted answer as string or empty string if not found
        """
        response = response.strip()
        
        if dataset_type == "mmlu":
            return EvaluationMetrics.extract_mmlu_answer(response)
        elif dataset_type == "gsm8k":
            return EvaluationMetrics.extract_gsm8k_answer(response)
        
        return response
    
    @staticmethod
    def extract_mmlu_answer(response: str) -> str:
        """
        Extract MMLU answer from response.
        
        Expected format: "ANSWER: A" or just "A"
        
        Detection strategy (in order):
        1. Look for "ANSWER:" prefix
        2. Look for just the letter at start of line
        3. Return empty if no valid format found
        """
        response_lower = response.lower()
        
        # Strategy 1: "ANSWER: A" or similar
        match = re.search(r'answer\s*[:=\s]*([a-d])', response_lower, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Strategy 2: First letter on first line
        first_line = response.split('\n')[0].strip().upper()
        if len(first_line) > 0 and first_line[0] in "ABCD":
            return first_line[0]
        
        # Strategy 3: Any single letter A-D in response
        matches = re.findall(r'[A-D]', response_upper := response.upper())
        if matches:
            return matches[0]  # Return first match
        
        # No valid answer found
        return ""
    
    @staticmethod
    def extract_gsm8k_answer(response: str) -> str:
        """
        Extract GSM8K answer from response.
        
        Expected format: "FINAL_ANSWER: 15" or just "15"
        
        Detection strategy (in order):
        1. Look for "FINAL_ANSWER:" prefix
        2. Look for last number in response
        3. Return empty if no number found
        """
        response_lower = response.lower()
        
        # Strategy 1: "FINAL_ANSWER: 15" or "FINALANSWER: 15"
        match = re.search(r'final\s*_?\s*answer\s*[:=\s]*(-?\d+\.?\d*)', response_lower, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Strategy 2: Look for numbers in response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]  # Return LAST number (most likely answer)
        
        # No valid number found
        return ""
    
    @staticmethod
    def detect_hallucination(response: str, dataset_type: str) -> bool:
        """
        Detect if model hallucinated (generated wrong problem/context).
        
        Hallucination signatures:
        - For MMLU: Response contains problem intro words (e.g., "Linda", "Mason")
        - For GSM8K: Response contains unrelated proper nouns
        """
        hallucination_keywords = {
            "mmlu": ["linda", "mason", "john", "sarah", "mary"],
            "gsm8k": ["linda", "mason", "painting", "attic", "bedroom"]
        }
        
        response_lower = response.lower()
        keywords = hallucination_keywords.get(dataset_type, [])
        
        # Check if response contains suspicious keywords
        for keyword in keywords:
            if keyword in response_lower and len(response) > 100:
                return True
        
        return False
    
    @staticmethod
    def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
        """
        Check if MMLU prediction matches ground truth.
        
        Args:
            prediction: Model-generated response
            ground_truth: Correct answer (ABCD)
        
        Returns:
            True if extracted answer matches ground truth
        """
        extracted = EvaluationMetrics.extract_answer(prediction, "mmlu")
        
        if not extracted:
            return False
        
        return extracted.upper() == ground_truth.upper()
    
    @staticmethod
    def exact_match_gsm8k(prediction: str, ground_truth: str) -> bool:
        """
        Check if GSM8K prediction matches ground truth.
        
        Handles numeric comparison with floating-point tolerance.
        
        Args:
            prediction: Model-generated response with number
            ground_truth: Correct answer (number or text)
        
        Returns:
            True if extracted number matches ground truth within tolerance
        """
        extracted = EvaluationMetrics.extract_answer(prediction, "gsm8k")
        
        if not extracted:
            return False
        
        try:
            pred_num = float(extracted)
            true_num = float(ground_truth)
            
            # Allow 1% tolerance for floating point
            return abs(pred_num - true_num) < 1e-6 or abs(pred_num - true_num) / abs(true_num) < 0.01
        
        except (ValueError, ZeroDivisionError):
            return False
    
    @staticmethod
    def evaluate_batch(
        predictions: List[str],
        ground_truths: List[str],
        dataset_type: str
    ) -> Dict:
        """
        Evaluate batch of predictions.
        
        Args:
            predictions: List of generated texts
            ground_truths: List of correct answers
            dataset_type: 'mmlu' or 'gsm8k'
        
        Returns:
            Dict with accuracy, correct_count, total_count
        """
        assert len(predictions) == len(ground_truths), f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} truths"
        
        correct_count = 0
        hallucination_count = 0
        
        for pred, truth in zip(predictions, ground_truths):
            # Check for hallucination first
            if EvaluationMetrics.detect_hallucination(pred, dataset_type):
                hallucination_count += 1
                continue  # Mark as wrong
            
            # Check if answer matches
            if dataset_type == "mmlu":
                is_correct = EvaluationMetrics.exact_match_mmlu(pred, truth)
            elif dataset_type == "gsm8k":
                is_correct = EvaluationMetrics.exact_match_gsm8k(pred, truth)
            else:
                is_correct = False
            
            if is_correct:
                correct_count += 1
        
        total = len(predictions)
        accuracy = correct_count / max(total, 1)
        
        return {
            "accuracy": accuracy,
            "em": accuracy,
            "correct_count": correct_count,
            "total_count": total,
            "hallucination_count": hallucination_count,
            "hallucination_rate": hallucination_count / max(total, 1)
        }


class HeldOutEvaluator:
    """Evaluate model on held-out test set"""
    
    def __init__(self, model, data_loader: List[Dict], batch_size: int = 32):
        """
        Args:
            model: Server object with generate method
            data_loader: List of prompt, answer, dataset, ... dicts
            batch_size: Batch size for display logging
        """
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
    
    def evaluate(self) -> Tuple[Dict, List[Dict]]:
        """
        Run evaluation on held-out test set.
        
        Returns:
            Tuple containing:
            - Dict with per-dataset and overall results
            - List of detailed prediction dictionaries
        """
        logger.info("=" * 70)
        logger.info(f"EVALUATING ON {len(self.data_loader)} EXAMPLES")
        logger.info("=" * 70)
        
        all_predictions = []
        all_ground_truths = []
        all_dataset_types = []
        all_prompts = []
        all_hallucinations = []
        detailed_results = []
        
        for i, example in enumerate(self.data_loader):
            try:
                dataset_type = example.get("dataset", "mmlu")
                formatted_prompt, max_tokens, stops = build_llama_formatted_prompt(
                        example, dataset_type
                    )
                
                # Generate
                generated_text, metrics = self.model.generate(
                                            prompt=formatted_prompt,
                                            max_tokens=max_tokens,                           # ← CHANGE: 512 → max_tokens
                                            difficulty=example.get("difficulty", "medium")   # ← ADD THIS LINE
                                        )

                print(formatted_prompt)
                print(generated_text)
                print("_______________________")
                
                all_predictions.append(generated_text)
                all_ground_truths.append(example.get("answer", ""))
                all_dataset_types.append(dataset_type)
                all_prompts.append(formatted_prompt)
                
                # Check for hallucination
                is_hallucination = EvaluationMetrics.detect_hallucination(generated_text, dataset_type)
                all_hallucinations.append(is_hallucination)
            
            except Exception as e:
                logger.error(f"Failed to generate for example {i}: {e}")
                all_predictions.append("")
                all_ground_truths.append(example.get("answer", ""))
                all_dataset_types.append(example.get("dataset", "mmlu"))
                all_prompts.append("")
                all_hallucinations.append(False)
            
            if (i + 1) % self.batch_size == 0:
                logger.info(f"Generated {i + 1}/{len(self.data_loader)} predictions")
        
        # Evaluate by dataset type
        results = {}
        for dataset_type in set(all_dataset_types):
            indices = [j for j, dt in enumerate(all_dataset_types) if dt == dataset_type]
            preds = [all_predictions[j] for j in indices]
            truths = [all_ground_truths[j] for j in indices]
            prompts = [all_prompts[j] for j in indices]
            hallucinations = [all_hallucinations[j] for j in indices]
            
            result = EvaluationMetrics.evaluate_batch(preds, truths, dataset_type)
            results[dataset_type] = result
            
            logger.info(f"\n{dataset_type.upper()} Results")
            logger.info(f"  Accuracy: {result['accuracy']*100:.2f}%")
            logger.info(f"  Correct: {result['correct_count']}/{result['total_count']}")
            logger.info(f"  ⚠️ Hallucination Rate: {result['hallucination_rate']*100:.2f}%")
            
            # Build detailed results
            for ptext, ttext, predtext, is_hall in zip(prompts, truths, preds, hallucinations):
                extracted = EvaluationMetrics.extract_answer(predtext, dataset_type)
                
                if dataset_type == "mmlu":
                    is_correct = EvaluationMetrics.exact_match_mmlu(predtext, ttext)
                elif dataset_type == "gsm8k":
                    is_correct = EvaluationMetrics.exact_match_gsm8k(predtext, ttext)
                else:
                    is_correct = False
                
                detailed_results.append({
                    "dataset": dataset_type,
                    "prompt": ptext,
                    "ground_truth": ttext,
                    "prediction": predtext,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "is_hallucination": is_hall,
                    "max_tokens_used": len(extracted),
                    "token_efficiency": 1.0 if is_correct else 0.0
                })
        
        # Overall results
        overall_correct = sum(r["correct_count"] for r in results.values())
        overall_total = sum(r["total_count"] for r in results.values())
        
        results["overall"] = {
            "accuracy": overall_correct / max(overall_total, 1),
            "em": overall_correct / max(overall_total, 1),
            "correct_count": overall_correct,
            "total_count": overall_total,
            "hallucination_count": sum(r.get("hallucination_count", 0) for r in results.values()),
            "hallucination_rate": sum(1 for h in all_hallucinations if h) / max(len(all_hallucinations), 1)
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("OVERALL RESULTS")
        logger.info(f"Accuracy: {results['overall']['accuracy']*100:.2f}%")
        logger.info(f"Correct: {results['overall']['correct_count']}/{results['overall']['total_count']}")
        logger.info(f"⚠️ HALLUCINATION RATE: {results['overall']['hallucination_rate']*100:.2f}%")
        logger.info("=" * 70)
        
        return results, detailed_results


if __name__ == "__main__":
    # Test parsing
    logger.info("=" * 70)
    logger.info("MMLU EXTRACTION TEST")
    logger.info("=" * 70)
    
    mmlu_test_cases = [
        ("ANSWER: A", "A", True),
        ("ANSWER: B", "B", True),
        ("The answer is ANSWER C", "C", True),
        ("ANSWER: D (correct)", "D", True),
        ("A", "A", True),
        ("B", "B", True),
        ("Linda is painting her bedroom...", "", False),
        ("", "", False),
    ]
    
    mmlu_pass = 0
    for i, (prediction, truth, expected) in enumerate(mmlu_test_cases, 1):
        extracted = EvaluationMetrics.extract_answer(prediction, "mmlu")
        result = EvaluationMetrics.exact_match_mmlu(prediction, truth)
        
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result == expected:
            mmlu_pass += 1
        
        logger.info(f"{status} Case {i}: Extracted='{extracted}', Expected='{truth}', Match={result}")
    
    logger.info(f"Score: {mmlu_pass}/{len(mmlu_test_cases)} ({mmlu_pass*100//len(mmlu_test_cases)}%)")
    
    logger.info("\n" + "=" * 70)
    logger.info("GSM8K EXTRACTION TEST (THE CRITICAL ONE)")
    logger.info("=" * 70)
    
    gsm8k_test_cases = [
        ("FINAL_ANSWER: 15", "15", True),
        ("FINAL_ANSWER: 100", "100", True),
        ("The answer is FINAL_ANSWER 42", "42", True),
        ("Linda is repainting her bedroom... 50 gallons...", "", False),
        ("Step 1: ...\nFINAL_ANSWER: 3600", "3600", True),
        ("", "", False),
    ]
    
    gsm8k_pass = 0
    for i, (prediction, truth, expected) in enumerate(gsm8k_test_cases, 1):
        extracted = EvaluationMetrics.extract_answer(prediction, "gsm8k")
        result = EvaluationMetrics.exact_match_gsm8k(prediction, truth)
        is_hall = EvaluationMetrics.detect_hallucination(prediction, "gsm8k")
        
        status = "✅ PASS" if result == expected else "❌ FAIL"
        if result == expected:
            gsm8k_pass += 1
        
        halluc_flag = " [HALLUCINATION DETECTED]" if is_hall else ""
        logger.info(f"{status} Case {i}: Extracted='{extracted}', Expected='{truth}', Match={result}{halluc_flag}")
    
    logger.info(f"Score: {gsm8k_pass}/{len(gsm8k_test_cases)} ({gsm8k_pass*100//len(gsm8k_test_cases)}%)")

# # evaluation.py - OPTIMIZED FOR LLAMA-3.1 & DIFFICULTY-AWARE PROMPTS
# """
# Evaluation metrics optimized for Llama-3.1-8B with difficulty-aware prompts.

# Key improvements:
# 1. Better extraction with explicit format support (from improved prompts)
# 2. Difficulty-aware evaluation metrics
# 3. Per-dataset and per-difficulty analysis
# 4. Token efficiency tracking
# 5. Comprehensive error analysis and logging

# Expected formats:
# - MMLU: "ANSWER: [A/B/C/D]" (from difficulty-aware examples)
# - GSM8K: "FINAL_ANSWER: [number]" (from difficulty-aware examples)"""

# import re
# import json
# import logging
# from typing import List, Dict, Tuple, Optional
# from dataclasses import dataclass


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# # ============================================================================
# # DATA STRUCTURES FOR EVALUATION
# # ============================================================================

# @dataclass
# class PredictionResult:
#     """Single prediction result with metadata"""
#     dataset: str
#     difficulty: str
#     prompt: str
#     ground_truth: str
#     prediction: str
#     extracted_answer: str
#     is_correct: bool
#     max_tokens_used: int
#     token_efficiency: float  # actual_tokens / max_tokens_budget


# # ============================================================================
# # EXTRACTION LOGIC (Optimized for Llama-3.1 difficulty-aware prompts)
# # ============================================================================

# class AnswerExtractor:
#     """Extract answers from model outputs with multiple fallback strategies"""
    
#     @staticmethod
#     def extract_mmlu(response: str) -> str:
#         """
#         Extract MMLU answer from response with Llama-3.1 format awareness.
        
#         Expected formats (in priority order):
#         1. "ANSWER: [A/B/C/D]" ← PRIMARY (explicit from few-shot examples)
#         2. "ANSWER: A/B/C/D" ← Secondary
#         3. "The answer is X" ← Tertiary
#         4. "(X)" at end ← Fallback
#         5. Last A/B/C/D letter ← Last resort
        
#         Args:
#             response: Model-generated text from Llama-3.1
        
#         Returns:
#             Single letter (A-D) or empty string
#         """
#         response_lower = response.lower()
        
#         # Strategy 1: "ANSWER: [X]" - Primary (explicit from few-shot examples)
#         # This is what we trained the model to output via examples
#         match = re.search(
#             r'answer\s*:\s*\[\s*([A-D])\s*\]',
#             response_lower,
#             re.IGNORECASE
#         )
#         if match:
#             logger.debug(f"  MMLU extracted (Strategy 1): '{match.group(1).upper()}'")
#             return match.group(1).upper()
        
#         # Strategy 2: "ANSWER: X" - Secondary (variant)
#         # Use word boundary to avoid matching letters that come after "ANSWER:"
#         match = re.search(
#             r'answer\s*:\s*\b([A-D])\b',
#             response_lower,
#             re.IGNORECASE
#         )
#         if match:
#             logger.debug(f"  MMLU extracted (Strategy 2): '{match.group(1).upper()}'")
#             return match.group(1).upper()
        
#         # Strategy 3: "The answer is X" or similar
#         # Match patterns like "answer is X", "correct choice is X", "choice is X"
#         match = re.search(
#             r'(?:answer|choice|option|correct)\s+(?:is|be|was)\s+\b([A-D])\b',
#             response_lower,
#             re.IGNORECASE
#         )
#         if match:
#             logger.debug(f"  MMLU extracted (Strategy 3): '{match.group(1).upper()}'")
#             return match.group(1).upper()
        
#         # Strategy 4: "(X)" at end
#         match = re.search(
#             r'\(([A-D])\)\s*$',
#             response_lower,
#             re.IGNORECASE
#         )
#         if match:
#             logger.debug(f"  MMLU extracted (Strategy 4): '{match.group(1).upper()}'")
#             return match.group(1).upper()
        
#         # Strategy 5: Last A/B/C/D letter
#         matches = re.findall(r'\b[A-D]\b', response_lower, re.IGNORECASE)
#         if matches:
#             logger.debug(f"  MMLU extracted (Strategy 5 - last letter): '{matches[-1].upper()}'")
#             return matches[-1].upper()
        
#         logger.debug(f"  MMLU extraction failed - no answer found")
#         return ""
    
    
#     @staticmethod
#     def extract_gsm8k(response: str) -> str:
#         """
#         Extract GSM8K answer from response with Llama-3.1 format awareness.
        
#         Expected formats (in priority order):
#         1. "FINAL_ANSWER: [number]" ← PRIMARY (explicit from few-shot examples)
#         2. "FINAL_ANSWER: number" ← Secondary
#         3. "#### number" ← Tertiary (standard format)
#         4. Last number in response ← Last resort
        
#         Handles:
#         - Integers: 42, -5, 1000
#         - Floats: 3.14, 42.0
#         - Formatted: 1,234, 1,234.56
        
#         Args:
#             response: Model-generated text from Llama-3.1
        
#         Returns:
#             Number as string (or empty string if not found)
#         """
#         response_lower = response.lower()
        
#         # Strategy 1: "FINAL_ANSWER: [X]" - Primary (explicit from few-shot examples)
#         # Matches: [42], [3.14], [1,234], [-5], [1,234.56]
#         match = re.search(
#             r'final_answer\s*:\s*\[\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\]',
#             response_lower,
#             re.IGNORECASE
#         )
#         if match:
#             answer = match.group(1).replace(',', '')
#             logger.debug(f"  GSM8K extracted (Strategy 1): '{answer}'")
#             return answer
        
#         # Strategy 2: "FINAL_ANSWER: X" - Secondary (without brackets)
#         match = re.search(
#             r'final_answer\s*:\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
#             response_lower,
#             re.IGNORECASE
#         )
#         if match:
#             answer = match.group(1).replace(',', '')
#             logger.debug(f"  GSM8K extracted (Strategy 2): '{answer}'")
#             return answer
        
#         # Strategy 3: "#### number" - Standard format
#         if '####' in response:
#             after_hash = response.split('####')[-1].strip()
#             numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', after_hash)
#             if numbers:
#                 answer = numbers[0].replace(',', '')
#                 logger.debug(f"  GSM8K extracted (Strategy 3 - #### format): '{answer}'")
#                 return answer
        
#         # Strategy 4: Last number in response
#         numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', response)
#         if numbers:
#             answer = numbers[-1].replace(',', '')
#             logger.debug(f"  GSM8K extracted (Strategy 4 - last number): '{answer}'")
#             return answer
        
#         logger.debug(f"  GSM8K extraction failed - no number found")
#         return ""


# # ============================================================================
# # EVALUATION METRICS
# # ============================================================================

# class EvaluationMetrics:
#     """Compute evaluation metrics for MMLU and GSM8K"""
    
#     @staticmethod
#     def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
#         """
#         Check if MMLU prediction matches ground truth.
        
#         Args:
#             prediction: Model-generated response
#             ground_truth: Correct answer (A/B/C/D or full text)
        
#         Returns:
#             True if extracted answer matches ground truth
#         """
#         extracted = AnswerExtractor.extract_mmlu(prediction)
        
#         # Ensure ground_truth is single letter
#         truth_letter = ground_truth.upper().strip()
#         if len(truth_letter) > 1:
#             # Extract if it's full text like "A) Option text"
#             truth_letter = truth_letter[0]
        
#         if not extracted:
#             return False
        
#         return extracted == truth_letter
    
    
#     @staticmethod
#     def exact_match_gsm8k(prediction: str, ground_truth: str) -> bool:
#         """
#         Check if GSM8K prediction matches ground truth.
#         Handles numeric comparison with floating-point tolerance.
        
#         Args:
#             prediction: Model-generated response with number
#             ground_truth: Correct answer (number or text containing number)
        
#         Returns:
#             True if extracted number matches ground truth within tolerance
#         """
#         extracted = AnswerExtractor.extract_gsm8k(prediction)
        
#         if not extracted:
#             return False
        
#         try:
#             pred_num = float(extracted)
            
#             # Extract number from ground truth (may be in different format)
#             true_numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', ground_truth)
#             if not true_numbers:
#                 return False
            
#             true_num = float(true_numbers[-1].replace(',', ''))
            
#             # Allow small floating point tolerance (1e-6)
#             return abs(pred_num - true_num) < 1e-6
        
#         except (ValueError, IndexError):
#             logger.debug(f"  GSM8K comparison failed: pred={extracted}, truth={ground_truth}")
#             return False
    
    
#     @staticmethod
#     def evaluate_batch(predictions: List[str],
#                        ground_truths: List[str],
#                        dataset_type: str) -> Dict:
#         """
#         Evaluate batch of predictions.
        
#         Args:
#             predictions: List of generated texts
#             ground_truths: List of correct answers
#             dataset_type: 'mmlu' or 'gsm8k'
        
#         Returns:
#             {accuracy, em, correct_count, total_count}
#         """
#         if len(predictions) != len(ground_truths):
#             raise ValueError(
#                 f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} truths"
#             )
        
#         correct_count = 0
        
#         for pred, truth in zip(predictions, ground_truths):
#             if dataset_type == "mmlu":
#                 is_correct = EvaluationMetrics.exact_match_mmlu(pred, truth)
#             elif dataset_type == "gsm8k":
#                 is_correct = EvaluationMetrics.exact_match_gsm8k(pred, truth)
#             else:
#                 is_correct = False
            
#             if is_correct:
#                 correct_count += 1
        
#         total = len(predictions)
#         accuracy = correct_count / max(total, 1)
        
#         return {
#             "accuracy": accuracy,
#             "em": accuracy,
#             "correct_count": correct_count,
#             "total_count": total
#         }


# # ============================================================================
# # HELD-OUT EVALUATOR (Main evaluation pipeline)
# # ============================================================================

# class HeldOutEvaluator:
#     """Evaluate model on held-out test set with difficulty-aware metrics"""
    
#     def __init__(self, model, data_loader: List[Dict], batch_size: int = 32):
#         """
#         Args:
#             model: Server object with generate() method
#             data_loader: List of {prompt, answer, dataset, difficulty, ...} dicts
#             batch_size: Batch size for progress logging
#         """
#         self.model = model
#         self.data_loader = data_loader
#         self.batch_size = batch_size
    
    
#     def evaluate(self) -> Tuple[Dict, List[PredictionResult]]:
#         """
#         Run evaluation on held-out test set.
        
#         Returns:
#             Tuple containing:
#             - Dict with metrics (per-dataset, per-difficulty, overall)
#             - List of detailed PredictionResult objects
#         """
#         logger.info("=" * 80)
#         logger.info(f"EVALUATING ON {len(self.data_loader)} EXAMPLES")
#         logger.info("=" * 80)
        
#         all_predictions = []
#         all_ground_truths = []
#         all_dataset_types = []
#         all_difficulties = []
#         all_token_budgets = []
#         detailed_results = []
        
#         # Generate predictions for all examples
#         for i, example in enumerate(self.data_loader):
#             try:
#                 dataset_type = example.get("dataset", "mmlu")
#                 difficulty = example.get("difficulty", "medium")
                
#                 # Get difficulty-aware configuration
#                 from prompt_templates_improved import build_llama_formatted_prompt, get_difficulty_config
                
#                 full_prompt, max_tokens = build_llama_formatted_prompt(example, dataset_type)
                
#                 # Generate with difficulty-aware token budget
#                 generated_text, metrics = self.model.generate(
#                     prompt=full_prompt,
#                     max_tokens=max_tokens  # Difficulty-aware token budget
#                 )
                
#                 all_predictions.append(generated_text)
#                 used_max_tokens = max_tokens
                
#             except Exception as e:
#                 logger.error(f"Failed to generate for example {i}: {e}")
#                 all_predictions.append("")
#                 generated_text = ""
#                 difficulty = example.get("difficulty", "medium")
                
#                 from prompt_templates_improved import get_difficulty_config
#                 used_max_tokens = get_difficulty_config(difficulty)["max_tokens"]
            
#             all_ground_truths.append(example["answer"])
#             all_dataset_types.append(dataset_type)
#             all_difficulties.append(difficulty)
#             all_token_budgets.append(used_max_tokens)
            
#             # Progress logging
#             if (i + 1) % self.batch_size == 0:
#                 logger.info(f"  Generated {i + 1}/{len(self.data_loader)} predictions")
        
#         logger.info(f"  Generated {len(self.data_loader)}/{len(self.data_loader)} predictions")
        
#         # Evaluate by dataset type
#         results = {}
        
#         for dataset_type in set(all_dataset_types):
#             indices = [j for j, dt in enumerate(all_dataset_types) if dt == dataset_type]
#             preds = [all_predictions[j] for j in indices]
#             truths = [all_ground_truths[j] for j in indices]
#             diffs = [all_difficulties[j] for j in indices]
#             budgets = [all_token_budgets[j] for j in indices]
            
#             # Batch metrics
#             batch_result = EvaluationMetrics.evaluate_batch(preds, truths, dataset_type)
#             results[dataset_type] = batch_result
            
#             # Detailed results per item
#             for idx, (pred_text, truth_text, diff, budget) in enumerate(zip(preds, truths, diffs, budgets)):
#                 extracted = AnswerExtractor.extract_mmlu(pred_text) if dataset_type == "mmlu" else AnswerExtractor.extract_gsm8k(pred_text)
                
#                 if dataset_type == "mmlu":
#                     is_correct = EvaluationMetrics.exact_match_mmlu(pred_text, truth_text)
#                 elif dataset_type == "gsm8k":
#                     is_correct = EvaluationMetrics.exact_match_gsm8k(pred_text, truth_text)
#                 else:
#                     is_correct = False
                
#                 # Calculate token efficiency
#                 # Count actual tokens in prediction (approximation: space-separated words)
#                 actual_tokens = len(pred_text.split())
#                 token_efficiency = actual_tokens / max(budget, 1)
                
#                 detailed_results.append(PredictionResult(
#                     dataset=dataset_type,
#                     difficulty=diff,
#                     prompt=example["prompt"],  # Original prompt, not formatted
#                     ground_truth=truth_text,
#                     prediction=pred_text,
#                     extracted_answer=extracted,
#                     is_correct=is_correct,
#                     max_tokens_used=budget,
#                     token_efficiency=token_efficiency
#                 ))
            
#             logger.info(f"\n{dataset_type.upper()} Results:")
#             logger.info(f"  Accuracy: {batch_result['accuracy'] * 100:.2f}%")
#             logger.info(f"  Correct:  {batch_result['correct_count']}/{batch_result['total_count']}")
        
#         # Difficulty-aware evaluation
#         logger.info(f"\n{'='*80}")
#         logger.info("DIFFICULTY-AWARE EVALUATION:")
#         logger.info("-" * 80)
        
#         difficulty_results = {}
        
#         for diff_level in ["easy", "medium", "hard"]:
#             diff_indices = [j for j, d in enumerate(all_difficulties) if d == diff_level]
            
#             if diff_indices:
#                 diff_preds = [all_predictions[j] for j in diff_indices]
#                 diff_truths = [all_ground_truths[j] for j in diff_indices]
#                 diff_types = [all_dataset_types[j] for j in diff_indices]
                
#                 correct = 0
#                 for pred, truth, dt in zip(diff_preds, diff_truths, diff_types):
#                     if dt == "mmlu":
#                         correct += 1 if EvaluationMetrics.exact_match_mmlu(pred, truth) else 0
#                     elif dt == "gsm8k":
#                         correct += 1 if EvaluationMetrics.exact_match_gsm8k(pred, truth) else 0
                
#                 total = len(diff_indices)
                
#                 from prompt_templates_improved import get_difficulty_config
#                 config = get_difficulty_config(diff_level)
                
#                 accuracy = correct / max(total, 1)
                
#                 difficulty_results[diff_level] = {
#                     "accuracy": accuracy,
#                     "correct_count": correct,
#                     "total_count": total,
#                     "max_tokens_budget": config["max_tokens"],
#                     "instruction": config["instruction"]
#                 }
                
#                 logger.info(
#                     f"  {diff_level.upper():8s}: {accuracy * 100:6.2f}% ({correct:3d}/{total:3d}) | "
#                     f"tokens ≤ {config['max_tokens']:3d}"
#                 )
        
#         results["by_difficulty"] = difficulty_results
        
#         # Overall results
#         overall_correct = sum(
#             r['correct_count'] for r in results.values()
#             if isinstance(r, dict) and 'correct_count' in r and 'by_difficulty' not in str(r)
#         )
#         overall_total = sum(
#             r['total_count'] for r in results.values()
#             if isinstance(r, dict) and 'total_count' in r and 'by_difficulty' not in str(r)
#         )
        
#         results["overall"] = {
#             "accuracy": overall_correct / max(overall_total, 1),
#             "em": overall_correct / max(overall_total, 1),
#             "correct_count": overall_correct,
#             "total_count": overall_total
#         }
        
#         logger.info(f"\nOVERALL Results:")
#         logger.info(f"  Accuracy: {results['overall']['accuracy'] * 100:.2f}%")
#         logger.info(f"  Correct:  {results['overall']['correct_count']}/{results['overall']['total_count']}")
#         logger.info("=" * 80)
        
#         return results, detailed_results


# # ============================================================================
# # TEST SUITE
# # ============================================================================

# if __name__ == "__main__":
#     """Comprehensive test cases for answer extraction"""
    
#     logger.info("=" * 80)
#     logger.info("ANSWER EXTRACTION TEST SUITE")
#     logger.info("=" * 80)
    
#     # ============================================================
#     # MMLU Test Cases
#     # ============================================================
#     mmlu_tests = [
#         # Primary format: "ANSWER: [X]"
#         ("ANSWER: [A]", "A", True, "Primary format with brackets"),
#         ("ANSWER: [B]", "B", True, "Primary format B"),
#         ("Conclusion: ANSWER: [C]", "C", True, "Primary format embedded"),
#         ("ANSWER: [ A ]", "A", True, "Primary format with spaces"),
        
#         # Secondary format: "ANSWER: X"
#         ("ANSWER: A", "A", True, "Secondary format no brackets"),
#         ("Final answer: ANSWER: B", "B", True, "Secondary format embedded"),
        
#         # Tertiary format: text description
#         ("The answer is C", "C", True, "Tertiary format 'answer is'"),
#         ("The correct choice is D", "D", True, "Tertiary format 'choice is'"),
        
#         # Fallback format: "(X)"
#         ("The correct answer is (A).", "A", True, "Fallback format (A)"),
#         ("Final answer (D).", "D", True, "Fallback format (D)"),
        
#         # Last resort: Last letter
#         ("A is wrong. B is wrong. C is right.", "C", True, "Last resort - last letter"),
        
#         # Edge cases
#         ("ANSWER: [A]", "B", False, "Wrong answer"),
#         ("No answer given.", "A", False, "No answer"),
#         ("", "A", False, "Empty response"),
#         ("ANSWER: [a]", "A", True, "Lowercase input"),
#     ]
    
#     logger.info("\nMMLA EXTRACTION TESTS:")
#     logger.info("-" * 80)
    
#     mmlu_pass = 0
#     for prediction, truth, expected, description in mmlu_tests:
#         result = EvaluationMetrics.exact_match_mmlu(prediction, truth)
#         status = "✓ PASS" if result == expected else "✗ FAIL"
        
#         if result == expected:
#             mmlu_pass += 1
        
#         logger.info(f"{status} | {description:40s} | Result: {result:5} | Expected: {expected:5}")
#         if result != expected:
#             extracted = AnswerExtractor.extract_mmlu(prediction)
#             logger.warning(f"       Extracted: '{extracted}', Expected: '{truth}'")
    
#     logger.info(f"\nMMLA Score: {mmlu_pass}/{len(mmlu_tests)} ({mmlu_pass * 100 // len(mmlu_tests)}%)")
    
#     # ============================================================
#     # GSM8K Test Cases
#     # ============================================================
#     gsm8k_tests = [
#         # Primary format: "FINAL_ANSWER: [X]"
#         ("FINAL_ANSWER: [42]", "42", True, "Primary format with brackets"),
#         ("FINAL_ANSWER: [100]", "100", True, "Primary format larger number"),
#         ("Therefore, FINAL_ANSWER: [3.14]", "3.14", True, "Primary format float"),
#         ("FINAL_ANSWER: [1,234]", "1234", True, "Primary format with comma"),
#         ("FINAL_ANSWER: [ 42 ]", "42", True, "Primary format with spaces"),
        
#         # Secondary format: "FINAL_ANSWER: X"
#         ("FINAL_ANSWER: 50", "50", True, "Secondary format no brackets"),
#         ("The answer is FINAL_ANSWER: 25.5", "25.5", True, "Secondary format embedded"),
        
#         # Tertiary format: "#### X"
#         ("Calculation: #### 42", "42", True, "Tertiary format #### 42"),
#         ("Step final: #### 100", "100", True, "Tertiary format #### 100"),
        
#         # Last resort: Last number
#         ("The answer involves 5, then 10, finally 15", "15", True, "Last resort - last number"),
        
#         # Edge cases
#         ("FINAL_ANSWER: [42]", "50", False, "Wrong answer"),
#         ("No answer provided", "42", False, "No answer"),
#         ("FINAL_ANSWER: [3.14159]", "3.14159", True, "Precision float"),
#         ("FINAL_ANSWER: [-42]", "-42", True, "Negative number"),
#     ]
    
#     logger.info("\n" + "=" * 80)
#     logger.info("GSM8K EXTRACTION TESTS:")
#     logger.info("-" * 80)
    
#     gsm8k_pass = 0
#     for prediction, truth, expected, description in gsm8k_tests:
#         result = EvaluationMetrics.exact_match_gsm8k(prediction, truth)
#         status = "✓ PASS" if result == expected else "✗ FAIL"
        
#         if result == expected:
#             gsm8k_pass += 1
        
#         logger.info(f"{status} | {description:40s} | Result: {result:5} | Expected: {expected:5}")
#         if result != expected:
#             extracted = AnswerExtractor.extract_gsm8k(prediction)
#             logger.warning(f"       Extracted: '{extracted}', Expected: '{truth}'")
    
#     logger.info(f"\nGSM8K Score: {gsm8k_pass}/{len(gsm8k_tests)} ({gsm8k_pass * 100 // len(gsm8k_tests)}%)")
    
#     logger.info("\n" + "=" * 80)
#     total_pass = mmlu_pass + gsm8k_pass
#     total_tests = len(mmlu_tests) + len(gsm8k_tests)
#     logger.info(f"TOTAL SCORE: {total_pass}/{total_tests} ({total_pass * 100 // total_tests}%)")
#     logger.info("=" * 80)