# evaluation.py - OPTIMIZED FOR PROMPT TEMPLATES
"""
Evaluation metrics: accuracy, exact match, evaluation on held-out test set
Optimized to match explicit prompt template formats:
- MMLU: "ANSWER: [A/B/C/D]"
- GSM8K: "FINAL_ANSWER: [number]"
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

from prompt_templates import build_improved_prompt


class EvaluationMetrics:
    """Evaluate model predictions against ground truth"""
    
    @staticmethod
    def extract_answer(response: str, dataset_type: str) -> str:
        """
        Extract answer from model response based on prompt template format.
        
        MMLU: Expects "ANSWER: [A/B/C/D]"
        GSM8K: Expects "FINAL_ANSWER: [number]"
        
        Args:
            response: Model-generated text
            dataset_type: 'mmlu' or 'gsm8k'
        
        Returns:
            Extracted answer as string
        """
        response = response.strip()
        
        if dataset_type == "mmlu":
            return EvaluationMetrics._extract_mmlu_answer(response)
        elif dataset_type == "gsm8k":
            return EvaluationMetrics._extract_gsm8k_answer(response)
        
        return response
    
    
    @staticmethod
    def _extract_mmlu_answer(response: str) -> str:
        """
        Extract MMLU answer from response.
        Expected formats (in order of priority):
        1. "ANSWER: [A/B/C/D]"  ← Primary (matches prompt template)
        2. "ANSWER: A/B/C/D"     ← Secondary
        3. "(A/B/C/D)"           ← Tertiary
        4. Last A/B/C/D letter   ← Last resort
        
        Args:
            response: Model-generated text
        
        Returns:
            Single letter: A, B, C, or D (or empty string if not found)
        """
        response_lower = response.lower()
        
        # Strategy 1: "ANSWER: [X]" (PRIMARY - matches prompt template)
        # Most reliable - directly matches expected format from prompt
        match = re.search(
            r'answer\s*:\s*\[\s*([A-D])\s*\]',
            response_lower,
            re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        
        # Strategy 2: "ANSWER: X" (SECONDARY - variant without brackets)
        match = re.search(
            r'answer\s*:\s*([A-D])',
            response_lower,
            re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        
        # Strategy 3: "(X)" at end (TERTIARY - common in reasoning chains)
        match = re.search(
            r'\(([A-D])\)\s*$',
            response_lower,
            re.IGNORECASE
        )
        if match:
            return match.group(1).upper()
        
        # Strategy 4: Last A/B/C/D letter (LAST RESORT)
        matches = re.findall(r'\b[A-D]\b', response_lower, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
        
        return ""
    
    
    @staticmethod
    def _extract_gsm8k_answer(response: str) -> str:
        """
        Extract GSM8K answer from response.
        Expected formats (in order of priority):
        1. "FINAL_ANSWER: [number]"  ← Primary (matches prompt template)
        2. "FINAL_ANSWER: number"     ← Secondary
        3. "#### number"              ← Tertiary (standard format)
        4. Last number in response    ← Last resort
        
        Args:
            response: Model-generated text
        
        Returns:
            Number as string (or empty string if not found)
        """
        response_lower = response.lower()
        
        # Strategy 1: "FINAL_ANSWER: [X]" (PRIMARY - matches prompt template)
        # Handles: [42], [3.14], [1,234]
        match = re.search(
            r'final_answer\s*:\s*\[\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*\]',
            response_lower,
            re.IGNORECASE
        )
        if match:
            return match.group(1).replace(',', '')
        
        # Strategy 2: "FINAL_ANSWER: X" (SECONDARY - without brackets)
        match = re.search(
            r'final_answer\s*:\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
            response_lower,
            re.IGNORECASE
        )
        if match:
            return match.group(1).replace(',', '')
        
        # Strategy 3: "#### number" (TERTIARY - standard GSM8K format)
        if '####' in response:
            after_hash = response.split('####')[-1].strip()
            numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', after_hash)
            if numbers:
                return numbers[0].replace(',', '')
        
        # Strategy 4: Last number (LAST RESORT)
        numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', response)
        if numbers:
            return numbers[-1].replace(',', '')
        
        return ""
    
    
    @staticmethod
    def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
        """
        Check if MMLU prediction matches ground truth.
        
        Args:
            prediction: Model-generated response
            ground_truth: Correct answer (A/B/C/D)
        
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
            ground_truth: Correct answer (number or text containing number)
        
        Returns:
            True if extracted number matches ground truth within tolerance
        """
        extracted = EvaluationMetrics.extract_answer(prediction, "gsm8k")
        if not extracted:
            return False
        
        try:
            pred_num = float(extracted)
            
            # Extract number from ground truth (may be in different format)
            true_numbers = re.findall(r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?', ground_truth)
            if not true_numbers:
                return False
            
            true_num = float(true_numbers[-1].replace(',', ''))
            
            # Allow small floating point tolerance
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
    
    
    def evaluate(self) -> Tuple[Dict, List[Dict]]:
        """
        Run evaluation on held-out test set
        
        Returns:
            Tuple containing:
            - Dict with per-dataset and overall results
            - List of detailed prediction dictionaries
        """
        logger.info("="*70)
        logger.info(f"EVALUATING ON {len(self.data_loader)} EXAMPLES")
        logger.info("="*70)
        
        all_predictions = []
        all_ground_truths = []
        all_dataset_types = []
        all_prompts = []
        detailed_results = []
        
        # Generate predictions for all examples
        for i, example in enumerate(self.data_loader):
            try:
                # Build improved prompt with Llama 2 formatting
                dataset_type = example.get("dataset", "mmlu")
                system_prompt, user_prompt, _ = build_improved_prompt(example, dataset_type)
                
                # Apply Llama 2 Chat structured formatting
                formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
                
                generated_text, metrics = self.model.generate(
                    prompt=formatted_prompt,
                    max_tokens=512
                )
                all_predictions.append(generated_text)
                final_prompt = formatted_prompt
                
            except Exception as e:
                logger.error(f"Failed to generate for example {i}: {e}")
                all_predictions.append("")
                generated_text = ""
                final_prompt = example["prompt"]
            
            all_ground_truths.append(example["answer"])
            all_dataset_types.append(dataset_type)
            all_prompts.append(final_prompt)
            
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
            prompts = [all_prompts[j] for j in indices]
            
            # Batch metrics
            result = EvaluationMetrics.evaluate_batch(preds, truths, dataset_type)
            results[dataset_type] = result
            
            # Detailed results per item
            for p_text, t_text, pred_text in zip(prompts, truths, preds):
                is_correct = False
                extracted = EvaluationMetrics.extract_answer(pred_text, dataset_type)
                
                if dataset_type == "mmlu":
                    is_correct = EvaluationMetrics.exact_match_mmlu(pred_text, t_text)
                elif dataset_type == "gsm8k":
                    is_correct = EvaluationMetrics.exact_match_gsm8k(pred_text, t_text)
                
                detailed_results.append({
                    "dataset": dataset_type,
                    "prompt": p_text,
                    "ground_truth": t_text,
                    "prediction": pred_text,
                    "extracted_answer": extracted,
                    "is_correct": is_correct
                })
            
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
        
        return results, detailed_results


if __name__ == "__main__":
    """Test parsing with comprehensive test cases"""
    
    # ============================================================
    # MMLU Test Cases
    # ============================================================
    mmlu_test_cases = [
        # Primary format: "ANSWER: [X]"
        ("ANSWER: [A]", "A", True),
        ("ANSWER: [B]", "B", True),
        ("The answer is: ANSWER: [C]", "C", True),
        ("Conclusion: ANSWER: [D] Might sometimes...", "D", True),
        ("ANSWER: [ A ]", "A", True),  # With spaces
        
        # Secondary format: "ANSWER: X"
        ("ANSWER: A", "A", True),
        ("Final answer: ANSWER: B", "B", True),
        
        # Tertiary format: "(X)"
        ("The correct answer is (C).", "C", True),
        ("Final answer (D).", "D", True),
        
        # Last resort: Last letter
        ("Option A is wrong. Option B is wrong. Option C is correct.", "C", True),
        
        # Edge cases
        ("ANSWER: [A]", "B", False),
        ("No answer given.", "A", False),
        ("", "A", False),
    ]
    
    logger.info("="*70)
    logger.info("MMLU EXTRACTION TEST CASES")
    logger.info("="*70)
    
    mmlu_pass = 0
    for i, (prediction, truth, expected) in enumerate(mmlu_test_cases, 1):
        extracted = EvaluationMetrics.extract_answer(prediction, "mmlu")
        result = EvaluationMetrics.exact_match_mmlu(prediction, truth)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            mmlu_pass += 1
        
        logger.info(f"{status} | Case {i:2d} | Extracted: '{extracted:>1}' | Expected: '{truth}' | Match: {result}")
        if result != expected:
            logger.warning(f"        Input: {prediction[:60]}...")
    
    logger.info(f"\nMMLA Score: {mmlu_pass}/{len(mmlu_test_cases)} ({mmlu_pass*100//len(mmlu_test_cases)}%)")
    
    # ============================================================
    # GSM8K Test Cases
    # ============================================================
    gsm8k_test_cases = [
        # Primary format: "FINAL_ANSWER: [X]"
        ("FINAL_ANSWER: [42]", "42", True),
        ("FINAL_ANSWER: [100]", "100", True),
        ("Therefore, FINAL_ANSWER: [3.14]", "3.14", True),
        ("FINAL_ANSWER: [1,234]", "1234", True),
        ("FINAL_ANSWER: [ 42 ]", "42", True),  # With spaces
        
        # Secondary format: "FINAL_ANSWER: X"
        ("FINAL_ANSWER: 50", "50", True),
        ("The answer is FINAL_ANSWER: 25.5", "25.5", True),
        
        # Tertiary format: "#### X"
        ("Calculation: #### 42", "42", True),
        ("Step final: #### 100", "100", True),
        
        # Last resort: Last number
        ("The answer involves 5, then 10, finally 15", "15", True),
        
        # Edge cases
        ("FINAL_ANSWER: [42]", "50", False),
        ("No answer provided", "42", False),
        ("FINAL_ANSWER: [3.14159]", "3.14159", True),
    ]
    
    logger.info("\n" + "="*70)
    logger.info("GSM8K EXTRACTION TEST CASES")
    logger.info("="*70)
    
    gsm8k_pass = 0
    for i, (prediction, truth, expected) in enumerate(gsm8k_test_cases, 1):
        extracted = EvaluationMetrics.extract_answer(prediction, "gsm8k")
        result = EvaluationMetrics.exact_match_gsm8k(prediction, truth)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            gsm8k_pass += 1
        
        logger.info(f"{status} | Case {i:2d} | Extracted: '{extracted:>6}' | Expected: '{truth:>6}' | Match: {result}")
        if result != expected:
            logger.warning(f"        Input: {prediction[:60]}...")
    
    logger.info(f"\nGSM8K Score: {gsm8k_pass}/{len(gsm8k_test_cases)} ({gsm8k_pass*100//len(gsm8k_test_cases)}%)")
    
    logger.info("\n" + "="*70)
    total_pass = mmlu_pass + gsm8k_pass
    total_tests = len(mmlu_test_cases) + len(gsm8k_test_cases)
    logger.info(f"TOTAL SCORE: {total_pass}/{total_tests} ({total_pass*100//total_tests}%)")
    logger.info("="*70)
# # evaluation.py
# """
# Evaluation metrics: accuracy, exact match, evaluation on held-out test set
# """

# import re
# import json
# import logging
# from typing import List, Dict, Tuple

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# from prompt_templates import build_improved_prompt

# class EvaluationMetrics:
#     """Evaluate model predictions against ground truth"""
    
#     @staticmethod
#     def extract_answer(response: str, dataset_type: str) -> str:
#         """
#         Robustly extract answer from model response, handling CoT templates
#         """
#         response = response.strip()
        
#         if dataset_type == "mmlu":
#             # Strategy 1: CoT "ANSWER: X" pattern (handles [], (), **, plain)
#             match = re.search(r'ANSWER[:\s]*[\[\(]?[\*]*([A-D])[\*]*[\]\)]?', response, re.IGNORECASE)
#             if match:
#                 return match.group(1).upper()
            
#             # Strategy 2: "(X)" at end
#             match = re.search(r"\(([A-D])\)[\.\!]?\s*$", response, re.IGNORECASE)
#             if match:
#                 return match.group(1).upper()
            
#             # Strategy 3: Standalone X at end
#             match = re.search(r"\b([A-D])[\.\!]?\s*$", response, re.IGNORECASE)
#             if match:
#                 return match.group(1).upper()
            
#             # Strategy 4: Last standalone X
#             matches = re.findall(r"\b([A-D])\b", response, re.IGNORECASE)
#             if matches:
#                 return matches[-1].upper()
            
#             return ""


#         elif dataset_type == "gsm8k":
#             # CoT Strategy: Explicit "FINAL_ANSWER: [number]" pattern (High priority)
#             match = re.search(r'FINAL_ANSWER[:\s]*([-\d,]+\.?\d*)', response)
#             if match:
#                 raw_num = match.group(1).replace(',', '')
#                 try:
#                     float(raw_num)
#                     return raw_num
#                 except ValueError:
#                     pass

#             # Strategy 2: Look for standard "####" separator
#             if "####" in response:
#                 response = response.split("####")[-1]

#             # Strategy 3: Extract last number
#             numbers = re.findall(r'-?\d+\.?\d*', response.replace(',', ''))
#             return numbers[-1] if numbers else ""
            
#         return response

#     @staticmethod
#     def exact_match_mmlu(prediction: str, ground_truth: str) -> bool:
#         """MMLU: Check if extracted answer matches ground truth"""
#         extracted = EvaluationMetrics.extract_answer(prediction, "mmlu")
#         return extracted == ground_truth.upper()
    
#     @staticmethod
#     def exact_match_gsm8k(prediction: str, ground_truth: str) -> bool:
#         """GSM8K: Check if extracted number matches ground truth"""
#         extracted = EvaluationMetrics.extract_answer(prediction, "gsm8k")
#         if not extracted:
#             return False
            
#         try:
#             pred_num = float(extracted)
#             true_numbers = re.findall(r'-?\d+\.?\d*', ground_truth.replace(',', ''))
#             true_num = float(true_numbers[-1])
#             return abs(pred_num - true_num) < 1e-6
#         except (ValueError, IndexError):
#             return False
    
#     @staticmethod
#     def evaluate_batch(predictions: List[str],
#                        ground_truths: List[str],
#                        dataset_type: str) -> Dict:
#         """
#         Evaluate batch of predictions
        
#         Args:
#             predictions: List of generated texts
#             ground_truths: List of correct answers
#             dataset_type: 'mmlu' or 'gsm8k'
        
#         Returns:
#             {accuracy, em, correct_count, total_count}
#         """
#         assert len(predictions) == len(ground_truths), \
#             f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} truths"
        
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
        
#         return {
#             "accuracy": correct_count / max(total, 1),
#             "em": correct_count / max(total, 1),
#             "correct_count": correct_count,
#             "total_count": total
#         }


# class HeldOutEvaluator:
#     """Evaluate model on held-out test set"""
    
#     def __init__(self, model, data_loader: List[Dict], batch_size: int = 32):
#         """
#         Args:
#             model: Server object with generate() method
#             data_loader: List of {prompt, answer, dataset, ...} dicts
#             batch_size: Batch size for display logging
#         """
#         self.model = model
#         self.data_loader = data_loader
#         self.batch_size = batch_size
    
#     def evaluate(self) -> Tuple[Dict, List[Dict]]:
#         """
#         Run evaluation on held-out test set
        
#         Returns:
#             Tuple containing:
#             - Dict with per-dataset and overall results
#             - List of detailed prediction dictionaries
#         """
#         logger.info("="*70)
#         logger.info(f"EVALUATING ON {len(self.data_loader)} EXAMPLES")
#         logger.info("="*70)
        
#         all_predictions = []
#         all_ground_truths = []
#         all_dataset_types = []
#         all_prompts = []
#         detailed_results = []
        
#         # Generate predictions for all examples
#         for i, example in enumerate(self.data_loader):
#             try:
#                 # Build improved prompt with Llama 2 formatting
#                 dataset_type = example.get("dataset", "mmlu")
#                 system_prompt, user_prompt, _ = build_improved_prompt(example, dataset_type)
                
#                 # Apply Llama 2 Chat structured formatting
#                 formatted_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
                
#                 generated_text, metrics = self.model.generate(
#                     prompt=formatted_prompt,
#                     max_tokens=512
#                 )
#                 all_predictions.append(generated_text)
#                 final_prompt = formatted_prompt # Store the actual prompt sent to model
                
#             except Exception as e:
#                 logger.error(f"Failed to generate for example {i}: {e}")
#                 all_predictions.append("")
#                 generated_text = ""
#                 final_prompt = example["prompt"]
            
#             all_ground_truths.append(example["answer"])
#             all_dataset_types.append(dataset_type)
#             all_prompts.append(final_prompt)
            
#             # Progress logging
#             if (i + 1) % self.batch_size == 0:
#                 logger.info(f"  Generated {i+1}/{len(self.data_loader)} predictions")
        
#         logger.info(f"  Generated {len(self.data_loader)}/{len(self.data_loader)} predictions")
        
#         # Evaluate by dataset type
#         results = {}
        
#         for dataset_type in set(all_dataset_types):
#             indices = [j for j, dt in enumerate(all_dataset_types) if dt == dataset_type]
#             preds = [all_predictions[j] for j in indices]
#             truths = [all_ground_truths[j] for j in indices]
#             prompts = [all_prompts[j] for j in indices]
            
#             # Batch metrics
#             result = EvaluationMetrics.evaluate_batch(preds, truths, dataset_type)
#             results[dataset_type] = result
            
#             # Detailed results per item
#             for p_text, t_text, pred_text in zip(prompts, truths, preds):
#                 is_correct = False
#                 extracted = EvaluationMetrics.extract_answer(pred_text, dataset_type)
                
#                 if dataset_type == "mmlu":
#                     is_correct = extracted == t_text.upper()
#                 elif dataset_type == "gsm8k":
#                     # Re-use logic from exact_match_gsm8k
#                     try:
#                         if extracted:
#                             pred_num = float(extracted)
#                             true_numbers = re.findall(r'-?\d+\.?\d*', t_text.replace(',', ''))
#                             true_num = float(true_numbers[-1])
#                             is_correct = abs(pred_num - true_num) < 1e-6
#                     except:
#                         pass
                
#                 detailed_results.append({
#                     "dataset": dataset_type,
#                     "prompt": p_text,
#                     "ground_truth": t_text,
#                     "prediction": pred_text,
#                     "extracted_answer": extracted,
#                     "is_correct": is_correct
#                 })
            
#             logger.info(f"\n{dataset_type.upper()} Results:")
#             logger.info(f"  Accuracy: {result['accuracy']*100:.2f}%")
#             logger.info(f"  Correct:  {result['correct_count']}/{result['total_count']}")
        
#         # Overall results
#         overall_correct = sum(r['correct_count'] for r in results.values())
#         overall_total = sum(r['total_count'] for r in results.values())
        
#         results["overall"] = {
#             "accuracy": overall_correct / max(overall_total, 1),
#             "em": overall_correct / max(overall_total, 1),
#             "correct_count": overall_correct,
#             "total_count": overall_total
#         }
        
#         logger.info(f"\nOVERALL Results:")
#         logger.info(f"  Accuracy: {results['overall']['accuracy']*100:.2f}%")
#         logger.info(f"  Correct:  {results['overall']['correct_count']}/{results['overall']['total_count']}")
        
#         logger.info("="*70)
        
#         return results, detailed_results


# if __name__ == "__main__":
#     # Test exact match functions
#     mmlu_examples = [
#         ("The answer is B", "B", True),
#         ("B is the correct answer", "B", True),
#         ("ANSWER: C", "C", True),        # CoT format
#         ("Therefore, ANSWER: D", "D", True), # CoT embedded
#         ("ANSWER: **B**", "B", True),    # Markdown bold
#         ("ANSWER: (A)", "A", True),      # Brackets in CoT
#         ("The answer is A.", "A", True), # Trailing period
#         ("The answer is A", "B", False),
#         ("", "A", False),
#         ("C", "C", True),
#     ]
    
#     logger.info("Testing MMLU exact match:")
#     for pred, truth, expected in mmlu_examples:
#         result = EvaluationMetrics.exact_match_mmlu(pred, truth)
#         status = "✓" if result == expected else "✗"
#         logger.info(f"  {status} '{pred}' vs '{truth}': {result}")
    
#     gsm8k_examples = [
#         ("The answer is 42", "42", True),
#         ("Therefore, FINAL_ANSWER: 42", "42", True), # CoT format
#         ("FINAL_ANSWER: 42.0", "42", True),          # CoT float
#         ("FINAL_ANSWER: 1,234", "1234", True),       # Commas
#         ("Calculation... #### 42.0", "42", True),    # Standard format
#         ("The answer is 43", "42", False),
#         ("No number here", "42", False),
#         ("100", "100", True)
#     ]
    
#     logger.info("\nTesting GSM8K exact match:")
#     for pred, truth, expected in gsm8k_examples:
#         result = EvaluationMetrics.exact_match_gsm8k(pred, truth)
#         status = "✓" if result == expected else "✗"
#         logger.info(f"  {status} '{pred}' vs '{truth}': {result}")
