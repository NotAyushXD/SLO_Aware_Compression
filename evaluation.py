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
    def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
        """
        MMLU: Check if predicted answer matches (A/B/C/D)
        
        Args:
            prediction: Model's generated text
            ground_truth: Correct answer (A, B, C, or D)
        
        Returns:
            True if answer is correct
        """
        # Extract first letter from prediction
        pred = prediction.strip().upper()
        
        # Get first valid answer letter
        for char in pred:
            if char in ['A', 'B', 'C', 'D']:
                return char == ground_truth.upper()
        
        return False
    
    @staticmethod
    def exact_match_gsm8k(prediction: str, ground_truth: str) -> bool:
        """
        GSM8K: Extract and compare final numeric answer
        
        Args:
            prediction: Model's generated text
            ground_truth: Correct answer as string (e.g., "42")
        
        Returns:
            True if extracted number matches
        """
        # Extract all numbers from prediction
        pred_numbers = re.findall(r"[-+]?\d*\.?\d+", prediction)
        if not pred_numbers:
            return False
        
        # Get last number as final answer
        try:
            pred_num = float(pred_numbers[-1])
        except ValueError:
            return False
        
        # Extract from ground truth
        try:
            true_numbers = re.findall(r"[-+]?\d*\.?\d+", ground_truth)
            true_num = float(true_numbers[-1])
        except (ValueError, IndexError):
            return False
        
        # Compare with small tolerance for floating point
        return abs(pred_num - true_num) < 1e-6
    
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
        ("The answer is A", "B", False),
        ("", "A", False),
    ]
    
    logger.info("Testing MMLU exact match:")
    for pred, truth, expected in mmlu_examples:
        result = EvaluationMetrics.exact_match_mmlu(pred, truth)
        status = "✓" if result == expected else "✗"
        logger.info(f"  {status} '{pred}' vs '{truth}': {result}")
    
    gsm8k_examples = [
        ("The answer is 42", "42", True),
        ("Therefore, the answer is 42 apples", "42", True),
        ("The answer is 43", "42", False),
        ("No number here", "42", False),
    ]
    
    logger.info("\nTesting GSM8K exact match:")
    for pred, truth, expected in gsm8k_examples:
        result = EvaluationMetrics.exact_match_gsm8k(pred, truth)
        status = "✓" if result == expected else "✗"
        logger.info(f"  {status} '{pred}' vs '{truth}': {result}")
