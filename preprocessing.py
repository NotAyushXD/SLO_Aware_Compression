# preprocessing.py
"""
Data pipeline: Download, preprocess, and split MMLU + GSM8K datasets
Outputs: train/val/test splits with difficulty labels
"""

import json
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
import tiktoken
import logging
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Unified preprocessing for MMLU and GSM8K"""
    
    def __init__(self, data_dir="data/raw", output_dir="data/processed"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized DataPreprocessor")
        logger.info(f"  Data dir: {data_dir}")
        logger.info(f"  Output dir: {output_dir}")
    
    def process_mmlu(self) -> List[Dict]:
        """
        Process MMLU dataset
        Returns list of {prompt, answer, subject, difficulty, input_length, output_length}
        
        MMLU characteristics:
        - 15,908 questions across 57 subjects
        - Multiple choice (A/B/C/D)
        - Subject-based difficulty: STEM/Law/Medicine=hard, Humanities/Social=easy
        """
        logger.info("Processing MMLU dataset...")
        
        try:
            mmlu_dataset = load_dataset("cais/mmlu", "all")
        except Exception as e:
            logger.error(f"Failed to load MMLU: {e}")
            return []
        
        # Subject-based difficulty mapping
        easy_subjects = [
            "abstract_algebra", "anatomy", "astronomy", "prehistory",
            "philosophy", "psychology", "sociology", "high_school_world_history",
            "high_school_us_history", "us_foreign_policy"
        ]
        
        hard_subjects = [
            "college_chemistry", "college_physics", "college_computer_science",
            "medical_genetics", "organic_chemistry", "professional_law",
            "professional_medicine", "clinical_knowledge", "anatomy"
        ]
        
        processed = []
        example_count = 0
        
        for split in ["validation", "test"]:
            if split not in mmlu_dataset:
                logger.warning(f"Split '{split}' not found in MMLU")
                continue
            
            data = mmlu_dataset[split]
            logger.info(f"  Processing MMLU {split} split: {len(data)} examples")
            
            for example in data:
                try:
                    question = example["question"]
                    choices = example["choices"]
                    answer_idx = example["answer"]
                    subject = example["subject"]
                    answer = chr(ord('A') + answer_idx)
                    
                    # Determine difficulty
                    if subject in hard_subjects:
                        difficulty = "hard"
                    elif subject in easy_subjects:
                        difficulty = "easy"
                    else:
                        difficulty = "medium"
                    
                    # Format prompt
                    prompt = f"{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
                    
                    # Count tokens
                    input_tokens = len(self.tokenizer.encode(question))
                    
                    processed.append({
                        "dataset": "mmlu",
                        "prompt": prompt,
                        "answer": answer,
                        "subject": subject,
                        "difficulty": difficulty,
                        "input_length": input_tokens,
                        "output_length": 1
                    })
                    
                    example_count += 1
                except Exception as e:
                    logger.warning(f"Error processing MMLU example: {e}")
                    continue
        
        logger.info(f"Processed {example_count} MMLU examples")
        
        # Save to JSONL
        output_file = os.path.join(self.output_dir, "mmlu_processed.jsonl")
        with open(output_file, 'w') as f:
            for item in processed:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved MMLU to {output_file}")
        return processed
    
    def process_gsm8k(self) -> List[Dict]:
        """
        Process GSM8K dataset
        Returns list of {prompt, answer, steps, difficulty, input_length, output_length}
        
        GSM8K characteristics:
        - 1,319 grade school math problems
        - Multi-step solutions (2-8 steps typical)
        - Difficulty by solution complexity
        """
        logger.info("Processing GSM8K dataset...")
        
        try:
            gsm8k_dataset = load_dataset("openai/gsm8k", "main")
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []
        
        processed = []
        example_count = 0
        
        for split in ["train", "test"]:
            if split not in gsm8k_dataset:
                logger.warning(f"Split '{split}' not found in GSM8K")
                continue
            
            data = gsm8k_dataset[split]
            logger.info(f"  Processing GSM8K {split} split: {len(data)} examples")
            
            for example in data:
                try:
                    question = example["question"]
                    full_answer = example["answer"]
                    
                    # Extract final answer (after ####)
                    if "####" in full_answer:
                        answer = full_answer.split("####")[-1].strip()
                    else:
                        answer = full_answer.strip()
                    
                    # Count solution steps
                    steps = len([line for line in full_answer.split('\n') if line.strip()])
                    
                    # Determine difficulty by steps
                    if steps <= 3:
                        difficulty = "easy"
                    elif steps <= 6:
                        difficulty = "medium"
                    else:
                        difficulty = "hard"
                    
                    # Count tokens
                    input_tokens = len(self.tokenizer.encode(question))
                    output_tokens = len(self.tokenizer.encode(answer))
                    
                    processed.append({
                        "dataset": "gsm8k",
                        "prompt": question,
                        "answer": answer,
                        "steps": steps,
                        "difficulty": difficulty,
                        "input_length": input_tokens,
                        "output_length": min(output_tokens, 256)  # Cap output tokens
                    })
                    
                    example_count += 1
                except Exception as e:
                    logger.warning(f"Error processing GSM8K example: {e}")
                    continue
        
        logger.info(f"Processed {example_count} GSM8K examples")
        
        # Save to JSONL
        output_file = os.path.join(self.output_dir, "gsm8k_processed.jsonl")
        with open(output_file, 'w') as f:
            for item in processed:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved GSM8K to {output_file}")
        return processed
    
    def combine_and_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Combine all datasets and create stratified train/val/test split
        
        Returns:
            (train_data, val_data, test_data)
        """
        logger.info("Combining datasets and creating splits...")
        
        all_data = []
        
        # Load processed datasets
        for dataset_name in ["mmlu_processed", "gsm8k_processed"]:
            path = os.path.join(self.output_dir, f"{dataset_name}.jsonl")
            if os.path.exists(path):
                logger.info(f"Loading {dataset_name}...")
                with open(path, 'r') as f:
                    for line in f:
                        all_data.append(json.loads(line))
        
        total = len(all_data)
        logger.info(f"Total examples loaded: {total}")
        
        if total == 0:
            logger.error("No data loaded!")
            return [], [], []
        
        # Stratified split by difficulty
        train, val, test = [], [], []
        
        for difficulty in ["easy", "medium", "hard"]:
            diff_data = [d for d in all_data if d["difficulty"] == difficulty]
            logger.info(f"  {difficulty:6s}: {len(diff_data)} examples")
            
            np.random.shuffle(diff_data)
            n = len(diff_data)
            
            train.extend(diff_data[:int(0.6 * n)])
            val.extend(diff_data[int(0.6 * n):int(0.8 * n)])
            test.extend(diff_data[int(0.8 * n):])
        
        # Save splits
        splits = [("train", train), ("val", val), ("test", test)]
        for split_name, split_data in splits:
            output_file = os.path.join(self.output_dir, f"{split_name}_data.jsonl")
            with open(output_file, 'w') as f:
                for item in split_data:
                    f.write(json.dumps(item) + '\n')
            
            # Statistics
            difficulty_counts = {}
            dataset_counts = {}
            for item in split_data:
                diff = item["difficulty"]
                ds = item["dataset"]
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
            
            logger.info(f"{split_name:5s} split: {len(split_data):5d} examples | "
                       f"Easy: {difficulty_counts.get('easy', 0):5d} | "
                       f"Medium: {difficulty_counts.get('medium', 0):5d} | "
                       f"Hard: {difficulty_counts.get('hard', 0):5d}")
        
        return train, val, test
    
    def run_pipeline(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Run complete preprocessing pipeline"""
        logger.info("="*70)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # Step 1: Process MMLU
        logger.info("\n[STEP 1] Processing MMLU")
        logger.info("-"*70)
        mmlu_data = self.process_mmlu()
        
        # Step 2: Process GSM8K
        logger.info("\n[STEP 2] Processing GSM8K")
        logger.info("-"*70)
        gsm8k_data = self.process_gsm8k()
        
        # Step 3: Combine and split
        logger.info("\n[STEP 3] Combining and creating splits")
        logger.info("-"*70)
        train, val, test = self.combine_and_split()
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*70)
        
        return train, val, test


if __name__ == "__main__":
    preprocessor = DataPreprocessor(
        data_dir="data/raw",
        output_dir="data/processed"
    )
    train, val, test = preprocessor.run_pipeline()
    
    logger.info(f"\nFinal data split:")
    logger.info(f"  Train: {len(train)} examples")
    logger.info(f"  Val:   {len(val)} examples")
    logger.info(f"  Test:  {len(test)} examples")
