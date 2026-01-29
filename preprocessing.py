"""preprocessing.py

Lightweight preprocessing helpers used by the evaluation harness.

This file is intentionally self-contained so you can:
1) Use already-preprocessed JSONL files in `data/processed/` (fast path), OR
2) (Optionally) regenerate them from HuggingFace datasets if missing.

Expected processed JSONL schema (one example per line):
- dataset: "gsm8k" | "mmlu"
- prompt: str  (GSM8K question OR MMLU question + options A-D)
- answer: str  (GSM8K numeric answer OR MMLU letter A-D)
- difficulty: "easy" | "medium" | "hard"

The harness converts these into a normalized format used by prompt templates.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# -----------------------------
# Utilities: read/write JSONL
# -----------------------------

def _read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _file_exists(p: Path) -> bool:
    return p.exists() and p.is_file() and p.stat().st_size > 0


# -----------------------------
# Fast-path loader (preferred)
# -----------------------------

def load_processed_data(processed_dir: str = "data/processed") -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load train/val/test JSONL splits from `processed_dir`.

    If files are missing, we raise a clear error rather than silently downloading
    large datasets. (If you want auto-regeneration, call `generate_processed_data`).
    """

    pdir = Path(processed_dir)
    train_p = pdir / "train.jsonl"
    val_p = pdir / "val.jsonl"
    test_p = pdir / "test.jsonl"

    if not (_file_exists(train_p) and _file_exists(val_p) and _file_exists(test_p)):
        raise FileNotFoundError(
            f"Processed splits not found in '{processed_dir}'. Expected train.jsonl / val.jsonl / test.jsonl. "
            "If you don't have these yet, generate them once and commit/copy them into the repo."
        )

    train = _read_jsonl(str(train_p))
    val = _read_jsonl(str(val_p))
    test = _read_jsonl(str(test_p))

    logger.info(f"Loaded processed data: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# -----------------------------
# Optional: regeneration hook
# -----------------------------

def generate_processed_data(*_args, **_kwargs) -> None:  # pragma: no cover
    """Optional placeholder.

    Your repo may already contain a heavier preprocessing pipeline.
    If you want, you can integrate it here.

    We keep this stub to avoid accidental HF downloads in Kaggle smoke tests.
    """

    raise NotImplementedError(
        "Auto-generation is disabled in this lightweight harness. "
        "Provide data/processed/train.jsonl, val.jsonl, test.jsonl."
    )


# -----------------------------
# Normalize examples
# -----------------------------

_MMLU_CHOICE_RE = re.compile(r"^[A-D]\)\s*(.*)$")
_MMLU_CHOICE_RE2 = re.compile(r"^[A-D]\s*\)\s*(.*)$")


def _parse_mmlu_prompt(prompt: str) -> Tuple[str, List[str]]:
    """Parse an MMLU prompt into (question, [A,B,C,D]).

    We support common formats:
    - "Q\nA) ...\nB) ...\nC) ...\nD) ..."
    """

    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if len(lines) < 5:
        # Fallback: return entire prompt as question and empty choices.
        return prompt.strip(), []

    question = lines[0]
    choices: List[str] = []
    for ln in lines[1:]:
        # Match "A) foo" / "A ) foo" / etc.
        m = re.match(r"^[A-D]\s*\)\s*(.*)$", ln)
        if m:
            choices.append(m.group(1).strip())

    return question, choices


def format_example_for_evaluation(example: Dict) -> Dict:
    """Convert a processed example into the normalized format used by the harness."""

    dataset = str(example.get("dataset", "")).lower().strip()
    difficulty = str(example.get("difficulty", "easy")).lower().strip()

    prompt = str(example.get("prompt", "")).strip()
    answer = str(example.get("answer", "")).strip()

    if dataset == "mmlu":
        question, choices = _parse_mmlu_prompt(prompt)
        return {
            "dataset_type": "mmlu",
            "difficulty": difficulty,
            "question": question,
            "choices": choices,
            "answer": answer,
        }

    # Default to GSM8K style.
    return {
        "dataset_type": "gsm8k",
        "difficulty": difficulty,
        "question": prompt,
        "choices": None,
        "answer": answer,
    }
