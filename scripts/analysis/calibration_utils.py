"""Calibration utilities (ECE, reliability curves, simple post-hoc calibration).

This file supports E7 (Calibration & threshold sensitivity) from the blueprint/execution plan.
We intentionally keep dependencies minimal and avoid SciPy optimizers by using a
lightweight temperature grid-search.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import numpy as np


def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray):
    """Fit an isotonic regression calibrator (sklearn)."""
    from sklearn.isotonic import IsotonicRegression

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob, y_true)
    return ir


def apply_isotonic(calibrator, y_prob: np.ndarray) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    return np.asarray(calibrator.predict(y_prob), dtype=float)


def _clip_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(p, eps, 1.0 - eps)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = _clip_prob(p, eps)
    return np.log(p) - np.log(1.0 - p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    """Return binned reliability stats.

    Output dict keys:
      - bin_lower, bin_upper
      - bin_count
      - bin_confidence (mean predicted prob)
      - bin_accuracy (mean y_true)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if y_true.size == 0:
        return {
            "bin_lower": np.zeros((0,), dtype=float),
            "bin_upper": np.zeros((0,), dtype=float),
            "bin_count": np.zeros((0,), dtype=int),
            "bin_confidence": np.zeros((0,), dtype=float),
            "bin_accuracy": np.zeros((0,), dtype=float),
        }

    n_bins = int(max(1, n_bins))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index in [0, n_bins-1]
    idx = np.digitize(y_prob, edges, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    bin_count = np.zeros((n_bins,), dtype=int)
    bin_conf = np.zeros((n_bins,), dtype=float)
    bin_acc = np.zeros((n_bins,), dtype=float)

    for b in range(n_bins):
        m = idx == b
        c = int(np.sum(m))
        bin_count[b] = c
        if c > 0:
            bin_conf[b] = float(np.mean(y_prob[m]))
            bin_acc[b] = float(np.mean(y_true[m]))
        else:
            bin_conf[b] = float((edges[b] + edges[b + 1]) / 2.0)
            bin_acc[b] = float("nan")

    return {
        "bin_lower": edges[:-1],
        "bin_upper": edges[1:],
        "bin_count": bin_count,
        "bin_confidence": bin_conf,
        "bin_accuracy": bin_acc,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bins = reliability_bins(y_true, y_prob, n_bins=n_bins)
    counts = bins["bin_count"].astype(float)
    conf = bins["bin_confidence"].astype(float)
    acc = bins["bin_accuracy"].astype(float)
    # Ignore empty bins (acc may be nan)
    mask = counts > 0
    if not np.any(mask):
        return 0.0
    counts = counts[mask]
    conf = conf[mask]
    acc = acc[mask]
    return float(np.sum((counts / np.sum(counts)) * np.abs(acc - conf)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = _clip_prob(np.asarray(y_prob, dtype=float), eps)
    if y_true.size == 0:
        return 0.0
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def fit_temperature(y_true: np.ndarray, y_prob: np.ndarray, grid: Optional[List[float]] = None) -> float:
    """Fit a single temperature T by grid-search on NLL.

    Temperature scaling is applied as:
        p_T = sigmoid(logit(p) / T)
    """

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.size == 0:
        return 1.0
    if grid is None:
        # Reasonable range for probability calibration
        grid = [0.25, 0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
    z = logit(y_prob)
    best_T = 1.0
    best = float("inf")
    for T in grid:
        T = float(T)
        if T <= 0:
            continue
        pT = sigmoid(z / T)
        loss = nll(y_true, pT)
        if loss < best:
            best = loss
            best_T = T
    return float(best_T)


def apply_temperature(y_prob: np.ndarray, T: float) -> np.ndarray:
    z = logit(np.asarray(y_prob, dtype=float))
    return sigmoid(z / float(T))
