"""Risk-controlled router for task-adaptive variant selection under latency SLOs.

This module adds a *calibration / guarantees* layer on top of the existing
"learned_router" scaffolding.

Goal
----
Given a request (x) and system state (s), select the *cheapest* model variant
that is likely to be correct *and* meets latency SLOs with a tunable risk.

We support two latency constraints simultaneously:
  1) TTFT (time-to-first-token)
  2) Total latency

Guarantee mechanism (finite-sample, under standard exchangeability assumptions)
---------------------------------------------------------------------------
We train latency predictors and then apply one-sided conformal calibration
per variant:

    upper(x,s) = pred(x,s) + q_delta

Where q_delta is the (1-delta) quantile of residuals (true - pred) on a
calibration set.

Quality control uses a selective-prediction style threshold on the predicted
probability of correctness. We choose the smallest threshold that satisfies
a high-confidence upper bound on error <= epsilon on a calibration set.

This is intentionally lightweight and uses sklearn models already depended on
by the repo.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _quantile_higher(values: np.ndarray, q: float) -> float:
    """Quantile with 'higher' method for conservative one-sided conformal bounds."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    q = float(min(max(q, 0.0), 1.0))

    # Numpy API changed: interpolation -> method.
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def binom_upper_confidence_bound(num_errors: int, num_total: int, alpha: float) -> float:
    """Clopper-Pearson one-sided upper confidence bound for a binomial proportion."""
    n = int(num_total)
    e = int(num_errors)
    if n <= 0:
        return 0.0
    if e <= 0:
        # Upper bound when e=0
        # beta.ppf(1-alpha, 1, n) is valid
        return float(beta.ppf(1.0 - float(alpha), 1.0, float(n)))
    if e >= n:
        return 1.0
    return float(beta.ppf(1.0 - float(alpha), float(e + 1), float(n - e)))


@dataclass
class RiskRouterDecision:
    """Decision + metadata for logging and analysis."""

    variant: str
    reason: str

    # Predicted quality and gating
    predicted_quality: float
    quality_threshold: float
    quality_pass: bool

    # Latency predictions (ms)
    predicted_ttft_ms: float
    predicted_total_ms: float

    # Conformal upper bounds (ms)
    upper_ttft_ms: float
    upper_total_ms: float

    # Budgets (ms)
    budget_ttft_ms: float
    budget_total_ms: float

    # Pass/fail
    ttft_pass: bool
    total_pass: bool
    latency_pass: bool

    # Slack (ms)
    slack_ttft_ms: float
    slack_total_ms: float
    slack_min_ms: float

    # Per-variant detailed scores
    per_variant: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "reason": self.reason,
            "predicted_quality": float(self.predicted_quality),
            "quality_threshold": float(self.quality_threshold),
            "quality_pass": bool(self.quality_pass),
            "predicted_ttft_ms": float(self.predicted_ttft_ms),
            "predicted_total_ms": float(self.predicted_total_ms),
            "upper_ttft_ms": float(self.upper_ttft_ms),
            "upper_total_ms": float(self.upper_total_ms),
            "budget_ttft_ms": float(self.budget_ttft_ms),
            "budget_total_ms": float(self.budget_total_ms),
            "ttft_pass": bool(self.ttft_pass),
            "total_pass": bool(self.total_pass),
            "latency_pass": bool(self.latency_pass),
            "slack_ttft_ms": float(self.slack_ttft_ms),
            "slack_total_ms": float(self.slack_total_ms),
            "slack_min_ms": float(self.slack_min_ms),
            "per_variant": self.per_variant,
        }


class RiskRouter:
    """Risk-controlled router (latency + quality)."""

    VARIANT_COSTS = {"base": 1.0, "med": 0.6, "cheap": 0.3}

    def __init__(
        self,
        *,
        quality_models: Dict[str, Any],
        ttft_models: Dict[str, Any],
        total_models: Dict[str, Any],
        calibration: Dict[str, np.ndarray],
        variants: Optional[List[str]] = None,
        default_slo: Optional[Dict[str, Dict[str, float]]] = None,
        quality_alpha: float = 0.05,
        base_always_accept: bool = True,
    ):
        self.quality_models = quality_models
        self.ttft_models = ttft_models
        self.total_models = total_models
        self.calibration = calibration
        self.variants = variants or ["cheap", "med", "base"]
        self.quality_alpha = float(quality_alpha)
        self.base_always_accept = bool(base_always_accept)

        self.default_slo = default_slo or {
            "easy": {"ttft_ms": 250.0, "tpot_ms": 10.0},
            "medium": {"ttft_ms": 350.0, "tpot_ms": 12.0},
            "hard": {"ttft_ms": 450.0, "tpot_ms": 15.0},
        }

        # Small caches because delta/epsilon are knobs swept in experiments.
        self._q_cache: Dict[Tuple[str, str, float], float] = {}
        self._tau_cache: Dict[Tuple[str, float, float], float] = {}

    # -------------------------
    # Features / prediction
    # -------------------------

    @staticmethod
    def extract_features(
        dataset_type: str,
        difficulty: str,
        max_tokens: int,
        prompt_tokens: int,
        concurrency: int,
        queue_depths: Dict[str, int],
        # Adapter-aware routing inputs (optional)
        adapter_id: str = "",
        adapter_rank: Optional[int] = None,
        adapter_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        # Reuse the exact same feature map as learned_router.py
        from learned_router import LearnedRouter

        return LearnedRouter.extract_features(
            dataset_type=dataset_type,
            difficulty=difficulty,
            max_tokens=max_tokens,
            prompt_tokens=prompt_tokens,
            concurrency=concurrency,
            queue_depths=queue_depths,
            adapter_id=str(adapter_id or ""),
            adapter_rank=int(adapter_rank) if adapter_rank is not None else None,
            adapter_state=adapter_state,
        )

    def predict_quality(self, variant: str, features: np.ndarray) -> float:
        model = self.quality_models[variant]
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(features)[0, 1])
        return float(model.predict(features)[0])

    def predict_ttft_ms(self, variant: str, features: np.ndarray) -> float:
        pred = float(self.ttft_models[variant].predict(features)[0])
        return max(0.0, pred)

    def predict_total_ms(self, variant: str, features: np.ndarray) -> float:
        pred = float(self.total_models[variant].predict(features)[0])
        return max(0.0, pred)

    # -------------------------
    # Calibration helpers
    # -------------------------

    def _get_slo(self, slo_dict: Optional[Dict[str, Dict[str, float]]], difficulty: str) -> Dict[str, float]:
        d = (difficulty or "easy").lower()
        src = slo_dict or self.default_slo
        return src.get(d) or src.get("default") or self.default_slo["easy"]

    def _get_latency_budget_ms(self, slo: Dict[str, float], max_tokens: int) -> Tuple[float, float]:
        """Return (ttft_budget_ms, total_budget_ms)."""
        ttft_budget = _as_float(slo.get("ttft_ms"), 1e9)

        # If a direct total_ms is provided, use it; else derive from (ttft + tpot * max_tokens)
        if "total_ms" in slo:
            total_budget = _as_float(slo.get("total_ms"), 1e9)
        else:
            tpot_budget = _as_float(slo.get("tpot_ms"), 1e9)
            total_budget = ttft_budget + tpot_budget * float(max_tokens)
        return float(ttft_budget), float(total_budget)

    def conformal_q(self, variant: str, metric: str, delta: float) -> float:
        """Return q_delta for one-sided conformal upper bound."""
        metric = metric.lower().strip()
        key = (variant, metric, float(delta))
        if key in self._q_cache:
            return self._q_cache[key]
        arr_key = f"resid_{metric}__{variant}"
        resid = self.calibration.get(arr_key)
        if resid is None:
            qv = 0.0
        else:
            qv = _quantile_higher(np.asarray(resid, dtype=float), 1.0 - float(delta))
        self._q_cache[key] = float(qv)
        return float(qv)

    def quality_threshold(self, variant: str, epsilon: float, alpha: Optional[float] = None) -> float:
        """Select a score threshold s.t. error among accepted <= epsilon (high-confidence)."""
        eps = float(epsilon)
        a = float(self.quality_alpha if alpha is None else alpha)
        cache_key = (variant, eps, a)
        if cache_key in self._tau_cache:
            return self._tau_cache[cache_key]

        if self.base_always_accept and variant == "base":
            self._tau_cache[cache_key] = 0.0
            return 0.0

        scores = self.calibration.get(f"qscore__{variant}")
        labels = self.calibration.get(f"qlabel__{variant}")
        if scores is None or labels is None or len(scores) == 0:
            # If we can't calibrate, be conservative: require high confidence.
            self._tau_cache[cache_key] = 0.9
            return 0.9

        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        labels = (labels > 0).astype(int)

        # Candidate thresholds: include endpoints.
        cand = np.unique(scores)
        cand = np.concatenate([np.array([0.0], dtype=float), cand, np.array([1.0], dtype=float)])
        cand = np.unique(np.clip(cand, 0.0, 1.0))
        cand.sort()  # increasing => larger acceptance first

        best_tau = 1.0
        best_acc = -1

        for tau in cand:
            mask = scores >= float(tau)
            n = int(mask.sum())
            if n <= 0:
                # Accept none => trivially safe but useless.
                continue
            e = int((labels[mask] == 0).sum())
            ub = binom_upper_confidence_bound(e, n, alpha=a)
            if ub <= eps:
                # First passing tau is the *smallest* tau that passes => max acceptance.
                best_tau = float(tau)
                best_acc = n
                break

        # If nothing passes, keep tau=1.0 (near reject-all).
        self._tau_cache[cache_key] = float(best_tau)
        return float(best_tau)

    # -------------------------
    # Routing policy
    # -------------------------

    def route(
        self,
        *,
        dataset_type: str,
        difficulty: str,
        max_tokens: int,
        prompt_tokens: int,
        concurrency: int,
        queue_depths: Dict[str, int],
        # Adapter-aware routing inputs (optional)
        adapter_id: str = "",
        adapter_rank: Optional[int] = None,
        adapter_state: Optional[Dict[str, Dict[str, Any]]] = None,
        slo_dict: Optional[Dict[str, Dict[str, float]]],
        latency_delta: float,
        quality_epsilon: float,
        allowed_variants: Optional[List[str]] = None,
    ) -> RiskRouterDecision:
        dataset_type = (dataset_type or "gsm8k").lower().strip()
        difficulty = (difficulty or "easy").lower().strip()
        allowed = allowed_variants or self.variants
        allowed = [v for v in ["cheap", "med", "base"] if v in allowed]
        if not allowed:
            allowed = ["base"]

        slo = self._get_slo(slo_dict, difficulty)
        budget_ttft, budget_total = self._get_latency_budget_ms(slo, max_tokens=int(max_tokens))

        features = self.extract_features(
            dataset_type=dataset_type,
            difficulty=difficulty,
            max_tokens=int(max_tokens),
            prompt_tokens=int(prompt_tokens),
            concurrency=int(concurrency),
            queue_depths=queue_depths,
            adapter_id=str(adapter_id or ""),
            adapter_rank=int(adapter_rank) if adapter_rank is not None else None,
            adapter_state=adapter_state,
        )

        per_v: Dict[str, Dict[str, Any]] = {}
        candidates: List[Tuple[float, str]] = []  # (cost, variant)

        for v in allowed:
            q_pred = self.predict_quality(v, features)
            ttft_pred = self.predict_ttft_ms(v, features)
            total_pred = self.predict_total_ms(v, features)

            q_ttft = self.conformal_q(v, "ttft", float(latency_delta))
            q_total = self.conformal_q(v, "total", float(latency_delta))

            upper_ttft = ttft_pred + q_ttft
            upper_total = total_pred + q_total

            ttft_pass = upper_ttft <= budget_ttft
            total_pass = upper_total <= budget_total
            latency_pass = bool(ttft_pass and total_pass)

            tau = self.quality_threshold(v, float(quality_epsilon))
            quality_pass = bool(q_pred >= tau)

            slack_ttft = budget_ttft - upper_ttft
            slack_total = budget_total - upper_total
            slack_min = float(min(slack_ttft, slack_total))

            per_v[v] = {
                "predicted_quality": float(q_pred),
                "quality_threshold": float(tau),
                "quality_pass": bool(quality_pass),
                "predicted_ttft_ms": float(ttft_pred),
                "predicted_total_ms": float(total_pred),
                "q_ttft_ms": float(q_ttft),
                "q_total_ms": float(q_total),
                "upper_ttft_ms": float(upper_ttft),
                "upper_total_ms": float(upper_total),
                "budget_ttft_ms": float(budget_ttft),
                "budget_total_ms": float(budget_total),
                "ttft_pass": bool(ttft_pass),
                "total_pass": bool(total_pass),
                "latency_pass": bool(latency_pass),
                "slack_ttft_ms": float(slack_ttft),
                "slack_total_ms": float(slack_total),
                "slack_min_ms": float(slack_min),
                "cost": float(self.VARIANT_COSTS.get(v, 1.0)),
            }

            if latency_pass and quality_pass:
                candidates.append((float(self.VARIANT_COSTS.get(v, 1.0)), v))

        if candidates:
            # Cheapest among passing
            candidates.sort(key=lambda t: (t[0], {"cheap": 0, "med": 1, "base": 2}.get(t[1], 99)))
            chosen = candidates[0][1]
            reason = "risk_pass"
        else:
            # Fallback: choose base for quality safety.
            chosen = "base" if "base" in allowed else allowed[-1]
            reason = "risk_fallback"

        ch = per_v.get(chosen) or {}

        return RiskRouterDecision(
            variant=chosen,
            reason=reason,
            predicted_quality=float(ch.get("predicted_quality", 0.0)),
            quality_threshold=float(ch.get("quality_threshold", 0.0)),
            quality_pass=bool(ch.get("quality_pass", False)),
            predicted_ttft_ms=float(ch.get("predicted_ttft_ms", 0.0)),
            predicted_total_ms=float(ch.get("predicted_total_ms", 0.0)),
            upper_ttft_ms=float(ch.get("upper_ttft_ms", 0.0)),
            upper_total_ms=float(ch.get("upper_total_ms", 0.0)),
            budget_ttft_ms=float(ch.get("budget_ttft_ms", budget_ttft)),
            budget_total_ms=float(ch.get("budget_total_ms", budget_total)),
            ttft_pass=bool(ch.get("ttft_pass", False)),
            total_pass=bool(ch.get("total_pass", False)),
            latency_pass=bool(ch.get("latency_pass", False)),
            slack_ttft_ms=float(ch.get("slack_ttft_ms", 0.0)),
            slack_total_ms=float(ch.get("slack_total_ms", 0.0)),
            slack_min_ms=float(ch.get("slack_min_ms", 0.0)),
            per_variant=per_v,
        )

    # -------------------------
    # Bundle I/O
    # -------------------------

    def save_bundle(self, out_dir: str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "quality_models.pkl"), "wb") as f:
            pickle.dump(self.quality_models, f)
        with open(os.path.join(out_dir, "ttft_models.pkl"), "wb") as f:
            pickle.dump(self.ttft_models, f)
        with open(os.path.join(out_dir, "total_models.pkl"), "wb") as f:
            pickle.dump(self.total_models, f)

        # Save calibration arrays
        np.savez_compressed(os.path.join(out_dir, "calibration.npz"), **self.calibration)

        meta: Dict[str, Any] = {
            "variants": list(self.variants),
            "quality_alpha": float(self.quality_alpha),
            "base_always_accept": bool(self.base_always_accept),
        }
        if extra_metadata:
            meta["extra_metadata"] = extra_metadata
        with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load_bundle(cls, in_dir: str) -> "RiskRouter":
        with open(os.path.join(in_dir, "quality_models.pkl"), "rb") as f:
            quality = pickle.load(f)
        with open(os.path.join(in_dir, "ttft_models.pkl"), "rb") as f:
            ttft = pickle.load(f)
        with open(os.path.join(in_dir, "total_models.pkl"), "rb") as f:
            total = pickle.load(f)
        cal_path = os.path.join(in_dir, "calibration.npz")
        calibration: Dict[str, np.ndarray] = {}
        if os.path.exists(cal_path):
            data = np.load(cal_path, allow_pickle=False)
            for k in data.files:
                calibration[k] = data[k]

        meta = {}
        meta_path = os.path.join(in_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        return cls(
            quality_models=quality,
            ttft_models=ttft,
            total_models=total,
            calibration=calibration,
            variants=meta.get("variants") or ["cheap", "med", "base"],
            quality_alpha=float(meta.get("quality_alpha", 0.05)),
            base_always_accept=bool(meta.get("base_always_accept", True)),
        )
