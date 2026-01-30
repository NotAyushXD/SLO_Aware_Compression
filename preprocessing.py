# preprocessing.py
"""Data pipeline: download, preprocess, and split MMLU + GSM8K.

Paper-facing reliability changes:
- Preserve *official* test splits (do NOT leak test into train/val).
- Create an internal train/val split from non-test data for load tests and SLO calibration.
- Deterministic shuffling (seeded) and stratification by difficulty.

Outputs (JSONL):
  - train_data.jsonl
  - val_data.jsonl
  - test_data.jsonl

Each record contains:
  {dataset, source_split, prompt, answer, difficulty, input_length, output_length, ...}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np

try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Unified preprocessing for MMLU and GSM8K."""

    def __init__(self, data_dir: str = "data/raw", output_dir: str = "data/processed", seed: int = 0):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Token counting is best-effort.
        if tiktoken is not None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            self.tokenizer = None

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Initialized DataPreprocessor")
        logger.info(f"  Data dir: {self.data_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Seed: {self.seed}")

    def _count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            try:
                return int(len(self.tokenizer.encode(text or "")))
            except Exception:
                pass
        return int(len((text or "").split()))

    # ------------------------------------------------------------------
    # MMLU
    # ------------------------------------------------------------------

    def process_mmlu(self) -> List[Dict]:
        """Process MMLU validation+test splits."""

        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Please `pip install datasets`.")

        logger.info("Processing MMLU dataset...")

        try:
            mmlu_dataset = load_dataset("cais/mmlu", "all")
        except Exception as e:
            logger.error(f"Failed to load MMLU: {e}")
            return []

        # Lightweight subject-based difficulty mapping.
        easy_subjects = {
            "abstract_algebra",
            "astronomy",
            "high_school_world_history",
            "high_school_us_history",
            "prehistory",
            "philosophy",
            "psychology",
            "sociology",
            "us_foreign_policy",
        }
        hard_subjects = {
            "college_chemistry",
            "college_physics",
            "college_computer_science",
            "medical_genetics",
            "organic_chemistry",
            "professional_law",
            "professional_medicine",
            "clinical_knowledge",
        }

        processed: List[Dict] = []
        for split in ["validation", "test"]:
            if split not in mmlu_dataset:
                logger.warning(f"Split '{split}' not found in MMLU")
                continue

            data = mmlu_dataset[split]
            logger.info(f"  Processing MMLU {split} split: {len(data)} examples")

            for ex in data:
                try:
                    question = ex["question"]
                    choices = ex["choices"]
                    answer_idx = int(ex["answer"])
                    subject = ex.get("subject", "")

                    answer = chr(ord("A") + answer_idx)

                    if subject in hard_subjects:
                        difficulty = "hard"
                    elif subject in easy_subjects:
                        difficulty = "easy"
                    else:
                        difficulty = "medium"

                    prompt = (
                        f"{question}\n"
                        f"A) {choices[0]}\n"
                        f"B) {choices[1]}\n"
                        f"C) {choices[2]}\n"
                        f"D) {choices[3]}"
                    )

                    processed.append(
                        {
                            "dataset": "mmlu",
                            "source_split": split,
                            "prompt": prompt,
                            "answer": answer,
                            "subject": subject,
                            "difficulty": difficulty,
                            "input_length": self._count_tokens(question),
                            "output_length": 1,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing MMLU example: {e}")

        out_path = os.path.join(self.output_dir, "mmlu_processed.jsonl")
        with open(out_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Saved MMLU to {out_path} ({len(processed)} examples)")
        return processed

    # ------------------------------------------------------------------
    # GSM8K
    # ------------------------------------------------------------------

    def process_gsm8k(self) -> List[Dict]:
        """Process GSM8K train+test splits."""

        if load_dataset is None:
            raise RuntimeError("datasets is not installed. Please `pip install datasets`.")

        logger.info("Processing GSM8K dataset...")

        try:
            gsm8k_dataset = load_dataset("openai/gsm8k", "main")
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []

        processed: List[Dict] = []
        for split in ["train", "test"]:
            if split not in gsm8k_dataset:
                logger.warning(f"Split '{split}' not found in GSM8K")
                continue

            data = gsm8k_dataset[split]
            logger.info(f"  Processing GSM8K {split} split: {len(data)} examples")

            for ex in data:
                try:
                    question = ex["question"]
                    full_answer = ex["answer"]

                    # Extract final answer (after ####)
                    if "####" in full_answer:
                        answer = full_answer.split("####")[-1].strip()
                    else:
                        answer = full_answer.strip()

                    # Steps ~= number of non-empty lines in the rationale
                    steps = len([ln for ln in full_answer.split("\n") if ln.strip()])

                    if steps <= 3:
                        difficulty = "easy"
                    elif steps <= 6:
                        difficulty = "medium"
                    else:
                        difficulty = "hard"

                    processed.append(
                        {
                            "dataset": "gsm8k",
                            "source_split": split,
                            "prompt": question,
                            "answer": answer,
                            "steps": steps,
                            "difficulty": difficulty,
                            "input_length": self._count_tokens(question),
                            "output_length": min(self._count_tokens(answer), 256),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error processing GSM8K example: {e}")

        out_path = os.path.join(self.output_dir, "gsm8k_processed.jsonl")
        with open(out_path, "w") as f:
            for item in processed:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Saved GSM8K to {out_path} ({len(processed)} examples)")
        return processed

    # ------------------------------------------------------------------
    # Split logic
    # ------------------------------------------------------------------

    def _load_processed(self, filename: str) -> List[Dict]:
        path = os.path.join(self.output_dir, filename)
        data: List[Dict] = []
        if not os.path.exists(path):
            return data
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    def combine_and_split(
        self,
        train_frac: float = 0.75,
        val_frac: float = 0.25,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Combine datasets and create train/val with *official* test held out."""

        assert abs(train_frac + val_frac - 1.0) < 1e-6

        all_data = []
        all_data.extend(self._load_processed("mmlu_processed.jsonl"))
        all_data.extend(self._load_processed("gsm8k_processed.jsonl"))

        if not all_data:
            logger.error("No processed data found. Run process_mmlu/process_gsm8k first.")
            return [], [], []

        test_data = [d for d in all_data if (d.get("source_split") == "test")]
        pool_data = [d for d in all_data if (d.get("source_split") != "test")]

        logger.info(f"Total examples: {len(all_data)}")
        logger.info(f"  Non-test pool: {len(pool_data)}")
        logger.info(f"  Official test: {len(test_data)}")

        # Stratified split by difficulty for the non-test pool.
        train_data: List[Dict] = []
        val_data: List[Dict] = []

        for difficulty in ["easy", "medium", "hard"]:
            diff_items = [d for d in pool_data if d.get("difficulty") == difficulty]
            self.rng.shuffle(diff_items)

            n = len(diff_items)
            n_train = int(round(train_frac * n))

            train_data.extend(diff_items[:n_train])
            val_data.extend(diff_items[n_train:])

        # Final shuffle for each split.
        self.rng.shuffle(train_data)
        self.rng.shuffle(val_data)
        self.rng.shuffle(test_data)

        # Save
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            out_path = os.path.join(self.output_dir, f"{split_name}_data.jsonl")
            with open(out_path, "w") as f:
                for item in split_data:
                    f.write(json.dumps(item) + "\n")

            diff_counts: Dict[str, int] = {}
            ds_counts: Dict[str, int] = {}
            for item in split_data:
                diff_counts[item.get("difficulty", "medium")] = diff_counts.get(item.get("difficulty", "medium"), 0) + 1
                ds_counts[item.get("dataset", "unknown")] = ds_counts.get(item.get("dataset", "unknown"), 0) + 1

            logger.info(
                f"{split_name:5s} split: {len(split_data):5d} | "
                f"easy={diff_counts.get('easy', 0):5d}, "
                f"medium={diff_counts.get('medium', 0):5d}, "
                f"hard={diff_counts.get('hard', 0):5d} | "
                f"mmlu={ds_counts.get('mmlu', 0):5d}, gsm8k={ds_counts.get('gsm8k', 0):5d}"
            )

        return train_data, val_data, test_data

    def run_pipeline(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        logger.info("=" * 70)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("=" * 70)

        logger.info("\n[STEP 1] Processing MMLU")
        logger.info("-" * 70)
        _ = self.process_mmlu()

        logger.info("\n[STEP 2] Processing GSM8K")
        logger.info("-" * 70)
        _ = self.process_gsm8k()

        logger.info("\n[STEP 3] Combining and creating splits")
        logger.info("-" * 70)
        train, val, test = self.combine_and_split()

        logger.info("\n" + "=" * 70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 70)

        return train, val, test


if __name__ == "__main__":
    pre = DataPreprocessor(data_dir="data/raw", output_dir="data/processed", seed=0)
    train, val, test = pre.run_pipeline()

    logger.info("\nFinal data split:")
    logger.info(f"  Train: {len(train)} examples")
    logger.info(f"  Val:   {len(val)} examples")
    logger.info(f"  Test:  {len(test)} examples")
