# evaluation.py
"""
Evaluation metrics: accuracy, exact match, evaluation on held-out test set
"""

import re
import json
import logging
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Evaluate model predictions against ground truth"""
    
    @staticmethod
    def extract_answer(response: str, dataset_type: str) -> str:
        """
        Robustly extract answer from model response, handling CoT templates
        """
        response = response.strip()
        
        if dataset_type == "mmlu":
            # CoT Strategy: Explicit "ANSWER: A" pattern (High priority)
            # Handles: "ANSWER: A", "ANSWER: (A)", "ANSWER: **A**", "ANSWER: A."
            match = re.search(r'ANSWER[:\s]*\(?[\*]*([A-D])[\*]*\)?', response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            
            # Strategy 2: Look for "(X)" at the very end (allowing punctuation)
            match = re.search(r"\(([A-D])\)[\.\!]?\s*$", response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            
            # Strategy 3: Look for standalone X at the very end
            match = re.search(r"\b([A-D])[\.\!]?\s*$", response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
                
            # Strategy 4: Look for last standalone X in text
            matches = re.findall(r"\b([A-D])\b", response, re.IGNORECASE)
            if matches:
                return matches[-1].upper()
                
            return ""

        elif dataset_type == "gsm8k":
            # CoT Strategy: Explicit "FINAL_ANSWER: [number]" pattern (High priority)
            match = re.search(r'FINAL_ANSWER[:\s]*([-\d,]+\.?\d*)', response)
            if match:
                raw_num = match.group(1).replace(',', '')
                try:
                    float(raw_num)
                    return raw_num
                except ValueError:
                    pass

            # Strategy 2: Look for standard "####" separator
            if "####" in response:
                response = response.split("####")[-1]

            # Strategy 3: Extract last number
            numbers = re.findall(r'-?\d+\.?\d*', response.replace(',', ''))
            return numbers[-1] if numbers else ""
            
        return response

    @staticmethod
    def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
        """MMLU: Check if extracted answer matches ground truth"""
        extracted = EvaluationMetrics.extract_answer(prediction, "mmlu")
        return extracted == ground_truth.upper()
    
    @staticmethod
    def exact_match_gsm8k(prediction: str, ground_truth: str) -> bool:
        """GSM8K: Check if extracted number matches ground truth"""
        extracted = EvaluationMetrics.extract_answer(prediction, "gsm8k")
        if not extracted:
            return False
            
        try:
            pred_num = float(extracted)
            true_numbers = re.findall(r'-?\d+\.?\d*', ground_truth.replace(',', ''))
            true_num = float(true_numbers[-1])
            return abs(pred_num - true_num) < 1e-6
        except (ValueError, IndexError):
            return False
    
    @staticmethod
    def evaluate_batch(predictions: List[str],
                       ground_truths: List[str],
                       dataset_type: str) -> Dict:
        """
        Evaluate batch of predictions
        
        Args:
            predictions: List of generated texts
            ground_truths: List of correct answers
            dataset_type: 'mmlu' or 'gsm8k'
        
        Returns:
            {accuracy, em, correct_count, total_count}
        """
        assert len(predictions) == len(ground_truths), \
            f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} truths"
        
        correct_count = 0
        
        for pred, truth in zip(predictions, ground_truths):
            if dataset_type == "mmlu":
                is_correct = EvaluationMetrics.exact_match_mmlu(pred, truth)
            elif dataset_type == "gsm8k":
                is_correct = EvaluationMetrics.exact_match_gsm8k(pred, truth)
            else:
                is_correct = False
            
            if is_correct:
                correct_count += 1
        
        total = len(predictions)
        
        return {
            "accuracy": correct_count / max(total, 1),
            "em": correct_count / max(total, 1),
            "correct_count": correct_count,
            "total_count": total
        }


class HeldOutEvaluator:
    """Evaluate model on held-out test set"""
    
    def __init__(self, model, data_loader: List[Dict], batch_size: int = 32):
        """
        Args:
            model: Server object with generate() method
            data_loader: List of {prompt, answer, dataset, ...} dicts
            batch_size: Batch size for display logging
        """
        self.model = model
        self.data_loader = data_loader
        self.batch_size = batch_size
    
    def evaluate(self) -> Dict:
        """
        Run evaluation on held-out test set
        
        Returns:
            Dict with per-dataset and overall results
        """
        logger.info("="*70)
        logger.info(f"EVALUATING ON {len(self.data_loader)} EXAMPLES")
        logger.info("="*70)
        
        all_predictions = []
        all_ground_truths = []
        all_dataset_types = []
        
        # Generate predictions for all examples
        for i, example in enumerate(self.data_loader):
            try:
                generated_text, metrics = self.model.generate(
                    prompt=example["prompt"],
                    max_tokens=example.get("output_length", 128)
                )
                all_predictions.append(generated_text)
            except Exception as e:
                logger.error(f"Failed to generate for example {i}: {e}")
                all_predictions.append("")
            
            all_ground_truths.append(example["answer"])
            all_dataset_types.append(example.get("dataset", "mmlu"))
            
            # Progress logging
            if (i + 1) % self.batch_size == 0:
                logger.info(f"  Generated {i+1}/{len(self.data_loader)} predictions")
        
        logger.info(f"  Generated {len(self.data_loader)}/{len(self.data_loader)} predictions")
        
        # Evaluate by dataset type
        results = {}
        
        for dataset_type in set(all_dataset_types):
            indices = [j for j, dt in enumerate(all_dataset_types) if dt == dataset_type]
            preds = [all_predictions[j] for j in indices]
            truths = [all_ground_truths[j] for j in indices]
            
            result = EvaluationMetrics.evaluate_batch(preds, truths, dataset_type)
            results[dataset_type] = result
            
            logger.info(f"\n{dataset_type.upper()} Results:")
            logger.info(f"  Accuracy: {result['accuracy']*100:.2f}%")
            logger.info(f"  Correct:  {result['correct_count']}/{result['total_count']}")
        
        # Overall results
        overall_correct = sum(r['correct_count'] for r in results.values())
        overall_total = sum(r['total_count'] for r in results.values())
        
        results["overall"] = {
            "accuracy": overall_correct / max(overall_total, 1),
            "em": overall_correct / max(overall_total, 1),
            "correct_count": overall_correct,
            "total_count": overall_total
        }
        
        logger.info(f"\nOVERALL Results:")
        logger.info(f"  Accuracy: {results['overall']['accuracy']*100:.2f}%")
        logger.info(f"  Correct:  {results['overall']['correct_count']}/{results['overall']['total_count']}")
        
        logger.info("="*70)
        
        return results


if __name__ == "__main__":
    # Test exact match functions
    mmlu_examples = [
        ("The answer is B", "B", True),
        ("B is the correct answer", "B", True),
        ("ANSWER: C", "C", True),        # CoT format
        ("Therefore, ANSWER: D", "D", True), # CoT embedded
        ("ANSWER: **B**", "B", True),    # Markdown bold
        ("ANSWER: (A)", "A", True),      # Brackets in CoT
        ("The answer is A.", "A", True), # Trailing period
        ("The answer is A", "B", False),
        ("", "A", False),
        ("C", "C", True),
    ]
    
    logger.info("Testing MMLU exact match:")
    for pred, truth, expected in mmlu_examples:
        result = EvaluationMetrics.exact_match_mmlu(pred, truth)
        status = "✓" if result == expected else "✗"
        logger.info(f"  {status} '{pred}' vs '{truth}': {result}")
    
    gsm8k_examples = [
        ("The answer is 42", "42", True),
        ("Therefore, FINAL_ANSWER: 42", "42", True), # CoT format
        ("FINAL_ANSWER: 42.0", "42", True),          # CoT float
        ("FINAL_ANSWER: 1,234", "1234", True),       # Commas
        ("Calculation... #### 42.0", "42", True),    # Standard format
        ("The answer is 43", "42", False),
        ("No number here", "42", False),
        ("100", "100", True)
    ]
    
    logger.info("\nTesting GSM8K exact match:")
    for pred, truth, expected in gsm8k_examples:
        result = EvaluationMetrics.exact_match_gsm8k(pred, truth)
        status = "✓" if result == expected else "✗"
        logger.info(f"  {status} '{pred}' vs '{truth}': {result}")
