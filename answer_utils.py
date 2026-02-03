# answer_utils.py
"""Answer extraction + postprocessing utilities.

This module supports two levels of GSM8K answer handling:

1) Strict (paper primary):
   - Output must contain a line that matches:
       FINAL_ANSWER: <number>

2) Parseable (robustness / sensitivity):
   - If strict fails, attempt a conservative recovery using common patterns:
       - "FINAL_ANSWER: #### 42"
       - "#### 42"
       - "Answer: 42"
       - last line is a number
       - "= 42" near the end

Server-side postprocessing uses the same conservative recovery to append a
canonical strict line when safe to do so, without changing the model's content.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


# -----------------------------
# Normalization helpers
# -----------------------------


_NUM_RE = r"[-+]?\d[\d,]*(?:\.\d+)?"

_CURRENCY_RE = re.compile(r"[\$₹€£]", flags=re.UNICODE)


def normalize_number_string(s: str) -> str:
    """Normalize a numeric string for comparison and formatting."""
    if s is None:
        return ""
    t = str(s).strip()
    t = _CURRENCY_RE.sub("", t)
    t = t.replace(",", "")
    return t.strip()


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(normalize_number_string(s))
    except Exception:
        return None


def numbers_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """Compare numbers with a small tolerance (handles integer/decimal)."""
    fa = _safe_float(a)
    fb = _safe_float(b)
    if fa is None or fb is None:
        return normalize_number_string(a) == normalize_number_string(b)
    return abs(fa - fb) <= tol


# -----------------------------
# MMLU extraction (strict-ish)
# -----------------------------


def extract_mmlu_answer(text: str) -> str:
    """Extract an MMLU option letter (A-D)."""
    if not text:
        return ""
    t = text.strip().upper()

    m = re.search(r"(?:FINAL\s*ANSWER|ANSWER|CORRECT\s*ANSWER)\s*[:=\s]*([A-D])\b", t)
    if m:
        return m.group(1)

    m = re.fullmatch(r"\s*([A-D])[\.\)]?\s*", t)
    if m:
        return m.group(1)

    first = t.splitlines()[0].strip() if t.splitlines() else t.strip()
    m = re.fullmatch(r"([A-D])[\.\)]?", first)
    if m:
        return m.group(1)

    return ""


# -----------------------------
# GSM8K extraction
# -----------------------------


_STRICT_LINE_RE = re.compile(
    rf"^\s*FINAL_ANSWER\s*[:=\s]*({_NUM_RE})\s*[\.\)]?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

_FINAL_ANSWER_ANYWHERE_RE = re.compile(
    rf"FINAL_ANSWER\s*[:=\s]*({_NUM_RE})",
    flags=re.IGNORECASE,
)

_HASH_RE = re.compile(
    rf"####\s*({_NUM_RE})\b",
    flags=re.IGNORECASE,
)

_ANSWER_LINE_RE = re.compile(
    rf"^\s*(?:ANSWER|FINAL\s*ANSWER)\s*[:=\s]*({_NUM_RE})\s*[\.\)]?\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

_LAST_LINE_NUMBER_RE = re.compile(
    rf"^\s*({_NUM_RE})\s*[\.\)]?\s*$",
    flags=re.IGNORECASE,
)

_EQUALS_TAIL_RE = re.compile(
    rf"=\s*({_NUM_RE})\s*$",
    flags=re.IGNORECASE,
)


def extract_gsm8k_strict(text: str) -> str:
    """Strict GSM8K extraction: requires a standalone FINAL_ANSWER line."""
    if not text:
        return ""
    matches = _STRICT_LINE_RE.findall(text)
    if not matches:
        return ""
    return normalize_number_string(matches[-1])


def extract_gsm8k_parseable(text: str) -> str:
    """Conservative GSM8K recovery when strict fails."""
    if not text:
        return ""

    # 1) FINAL_ANSWER anywhere
    m = _FINAL_ANSWER_ANYWHERE_RE.findall(text)
    if m:
        return normalize_number_string(m[-1])

    # 2) GSM8K-style ####
    m = _HASH_RE.findall(text)
    if m:
        return normalize_number_string(m[-1])

    # 3) Explicit Answer:/Final Answer: line
    m = _ANSWER_LINE_RE.findall(text)
    if m:
        return normalize_number_string(m[-1])

    # 4) Last non-empty line is just a number
    lines = [ln.strip() for ln in (text or "").splitlines()]
    for ln in reversed(lines):
        if not ln:
            continue
        m2 = _LAST_LINE_NUMBER_RE.match(ln)
        if m2:
            return normalize_number_string(m2.group(1))
        break  # only consider the last non-empty line

    # 5) "= 42" at the very end
    t = (text or "").strip()
    m = _EQUALS_TAIL_RE.search(t)
    if m:
        return normalize_number_string(m.group(1))

    return ""


def enforce_strict_gsm8k_final_answer(text: str) -> Tuple[str, Optional[str], bool]:
    """Append a canonical strict FINAL_ANSWER line if safe.

    Returns:
        (new_text, candidate_answer, did_modify)
    """
    if not text:
        return text, None, False

    strict = extract_gsm8k_strict(text)
    if strict:
        return text, strict, False

    cand = extract_gsm8k_parseable(text)
    if not cand:
        return text, None, False

    # Avoid double-appending if some 'FINAL_ANSWER' already exists but doesn't pass strict.
    # In that case, prefer to append a clean final line at the end.
    new_text = text.rstrip() + f"\nFINAL_ANSWER: {cand}\n"
    return new_text, cand, True
