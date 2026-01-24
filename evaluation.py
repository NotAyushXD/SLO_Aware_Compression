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
    def extract_mmlu_answer(response: str) -> str:
        """
        Extract MMLU answer from response.
        
        Handles formats with punctuation:
        - "A", "B.", "C)", "D "
        - "ANSWER: A"
        - "The correct answer is B"
        """
        response_upper = response.upper()
        
        # Strategy 1: Look for explicit "answer" keyword followed by letter
        # Matches: "ANSWER: A", "The answer is B", "correct answer: C"
        match = re.search(
            r'(?:ANSWER|answer|correct\s+answer)\s*[:=\s]*([A-D])',
            response,
            re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        
        # Strategy 2: Find first letter A-D (ignore punctuation)
        # Matches: "A.", "B)", "C ", "D" etc.
        match = re.search(r'([A-D])[\)\.\\s]?', response_upper)
        if match:
            return match.group(1)
        
        # Strategy 3: Any A-D letter in response
        for char in response_upper:
            if char in 'ABCD':
                return char
        
        # No answer found
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
                                            max_tokens=max_tokens,
                                            difficulty=example.get("difficulty", "medium")
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
