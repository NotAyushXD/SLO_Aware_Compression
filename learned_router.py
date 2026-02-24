"""Learned router for selecting model variants under SLO constraints.

Supports two learned policies (exposed via MultiVariantService.router_mode):
  - learned_ttft  : optimize using TTFT SLO only
  - learned_total : optimize using derived total latency SLO

Derived total latency SLO (as requested):
  total_slo_ms = ttft_slo_ms + tpot_slo_ms * max_tokens
  total_pred_ms = ttft_pred_ms + tpot_pred_ms * max_tokens

Models:
  - Quality: per-variant logistic regression (P(correct | features))
  - Latency: per-variant ridge regression for TTFT and TPOT

Features (22D + adapter features):
  - dataset one-hot (gsm8k, mmlu) => 2
  - difficulty one-hot (easy, medium, hard) => 3
  - max_tokens, sqrt(max_tokens), log1p(max_tokens), max_tokens transforms => 3
  - prompt_tokens: raw, sqrt, log1p => 3
  - concurrency: raw, log1p => 2
  - queue_depth per variant (cheap, med, base): raw, sqrt, log1p => 9

Adapter-aware features (paper polish track):
  - request has_adapter, adapter_rank, log1p(adapter_rank)
  - per-variant adapter cache/hotness state (cheap/med/base):
      resident, hot, num_loaded, log1p(num_loaded), setup_est_ms, log1p(setup_est_ms)
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LearnedRouterDecision:
    variant: str
    predicted_quality: float
    predicted_ttft_ms: float
    predicted_tpot_ms: float
    predicted_total_ms: float
    slo_target_ms: float
    score: float
    slo_compliant_pred: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "predicted_quality": float(self.predicted_quality),
            "predicted_ttft_ms": float(self.predicted_ttft_ms),
            "predicted_tpot_ms": float(self.predicted_tpot_ms),
            "predicted_total_ms": float(self.predicted_total_ms),
            "slo_target_ms": float(self.slo_target_ms),
            "score": float(self.score),
            "slo_compliant_pred": bool(self.slo_compliant_pred),
        }


class LearnedRouter:
    VARIANT_COSTS = {"base": 1.0, "med": 0.6, "cheap": 0.3}

    def __init__(
        self,
        quality_models: Dict[str, Any],
        ttft_models: Dict[str, Any],
        tpot_models: Dict[str, Any],
        lambda_slo: float = 1.5,
        mu_quality: float = 1.5,
        default_slo: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.quality_models = quality_models
        self.ttft_models = ttft_models
        self.tpot_models = tpot_models
        self.lambda_slo = float(lambda_slo)
        self.mu_quality = float(mu_quality)
        self.default_slo = default_slo or {
            "easy": {"ttft_ms": 250.0, "tpot_ms": 10.0},
            "medium": {"ttft_ms": 350.0, "tpot_ms": 12.0},
            "hard": {"ttft_ms": 450.0, "tpot_ms": 15.0},
        }

    @staticmethod
    def extract_features(
        dataset_type: str,
        difficulty: str,
        max_tokens: int,
        prompt_tokens: int,
        concurrency: int,
        queue_depths: Dict[str, int],
        # Adapter-aware routing (optional)
        adapter_id: str = "",
        adapter_rank: Optional[int] = None,
        adapter_state: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> np.ndarray:
        dataset_type = (dataset_type or "").lower()
        difficulty = (difficulty or "easy").lower()

        # dataset one-hot (2)
        dataset_gsm8k = 1.0 if dataset_type == "gsm8k" else 0.0
        dataset_mmlu = 1.0 if dataset_type == "mmlu" else 0.0

        # difficulty one-hot (3)
        diff_easy = 1.0 if difficulty == "easy" else 0.0
        diff_medium = 1.0 if difficulty == "medium" else 0.0
        diff_hard = 1.0 if difficulty == "hard" else 0.0

        # max_tokens transforms (3)
        max_tok = float(max_tokens)
        sqrt_max_tok = float(np.sqrt(max_tok))
        log_max_tok = float(np.log1p(max_tok))

        # prompt_tokens transforms (3)
        ptok = float(prompt_tokens)
        sqrt_ptok = float(np.sqrt(ptok))
        log_ptok = float(np.log1p(ptok))

        # concurrency transforms (2)
        conc = float(concurrency)
        log_conc = float(np.log1p(conc))

        feats: List[float] = [
            dataset_gsm8k,
            dataset_mmlu,
            diff_easy,
            diff_medium,
            diff_hard,
            max_tok,
            sqrt_max_tok,
            log_max_tok,
            ptok,
            sqrt_ptok,
            log_ptok,
            conc,
            log_conc,
        ]

        # queue depths per variant (9)
        for v in ["cheap", "med", "base"]:
            qd = float(queue_depths.get(v, 0))
            feats.extend([qd, float(np.sqrt(qd)), float(np.log1p(qd))])

        # ------------------------------------------------------------------
        # Adapter hotness / setup-cost features
        # ------------------------------------------------------------------
        aid = str(adapter_id or "").strip()
        has_adapter = 1.0 if aid else 0.0
        try:
            rnk = float(int(adapter_rank) if adapter_rank is not None else 0)
        except Exception:
            rnk = 0.0
        feats.extend([has_adapter, rnk, float(np.log1p(max(0.0, rnk)))])

        ast = adapter_state if isinstance(adapter_state, dict) else {}
        for v in ["cheap", "med", "base"]:
            st = ast.get(v) if isinstance(ast.get(v), dict) else {}
            try:
                resident = float(int(st.get("resident", 0) or 0))
            except Exception:
                resident = 0.0
            # "hot" means the last served adapter key matches this request.
            # If missing, fall back to 'active'.
            try:
                hot = float(int(st.get("hot", st.get("active", 0)) or 0))
            except Exception:
                hot = 0.0
            try:
                num_loaded = float(int(st.get("num_loaded", 0) or 0))
            except Exception:
                num_loaded = 0.0
            try:
                setup_est = float(st.get("setup_est_ms", 0.0) or 0.0)
            except Exception:
                setup_est = 0.0

            feats.extend(
                [
                    resident,
                    hot,
                    num_loaded,
                    float(np.log1p(max(0.0, num_loaded))),
                    setup_est,
                    float(np.log1p(max(0.0, setup_est))),
                ]
            )

        return np.array(feats, dtype=np.float32).reshape(1, -1)

    def _get_slo(self, slo_dict: Optional[Dict[str, Dict[str, float]]], difficulty: str) -> Dict[str, float]:
        d = (difficulty or "easy").lower()
        src = slo_dict or self.default_slo
        return src.get(d) or src.get("default") or self.default_slo["easy"]

    def predict_quality(self, variant: str, features: np.ndarray) -> float:
        model = self.quality_models[variant]
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(features)[0, 1]
            return float(p)
        # fallback
        p = model.predict(features)[0]
        return float(p)

    def predict_ttft_ms(self, variant: str, features: np.ndarray) -> float:
        model = self.ttft_models[variant]
        pred = float(model.predict(features)[0])
        return max(0.0, pred)

    def predict_tpot_ms(self, variant: str, features: np.ndarray) -> float:
        model = self.tpot_models[variant]
        pred = float(model.predict(features)[0])
        return max(0.0, pred)

    def _score(self, variant: str, quality: float, pred_latency_ms: float, slo_target_ms: float) -> float:
        cost = float(self.VARIANT_COSTS.get(variant, 1.0))
        latency_violation = (pred_latency_ms - slo_target_ms) / max(1e-6, slo_target_ms)
        latency_penalty = self.lambda_slo * max(0.0, latency_violation)
        quality_penalty = self.mu_quality * max(0.0, 1.0 - quality)
        return cost + latency_penalty + quality_penalty

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
        slo_dict: Optional[Dict[str, Dict[str, float]]] = None,
        mode: str = "ttft",
        allowed_variants: Optional[List[str]] = None,
    ) -> LearnedRouterDecision:
        mode = (mode or "ttft").lower()
        allowed = allowed_variants or ["cheap", "med", "base"]
        allowed = [v for v in ["cheap", "med", "base"] if v in allowed]
        if not allowed:
            allowed = ["base"]

        slo = self._get_slo(slo_dict, difficulty)
        ttft_slo = float(slo.get("ttft_ms", 1e9))
        tpot_slo = float(slo.get("tpot_ms", 1e9))

        # derived total slo
        total_slo = ttft_slo + tpot_slo * float(max_tokens)

        features = self.extract_features(
            dataset_type,
            difficulty,
            max_tokens,
            prompt_tokens,
            concurrency,
            queue_depths,
            adapter_id=str(adapter_id or ""),
            adapter_rank=int(adapter_rank) if adapter_rank is not None else None,
            adapter_state=adapter_state,
        )

        best: Optional[LearnedRouterDecision] = None
        for v in allowed:
            q = self.predict_quality(v, features)
            ttft = self.predict_ttft_ms(v, features)
            tpot = self.predict_tpot_ms(v, features)
            total = ttft + tpot * float(max_tokens)

            if mode == "total":
                slo_target = total_slo
                pred_lat = total
            else:
                slo_target = ttft_slo
                pred_lat = ttft

            score = self._score(v, q, pred_lat, slo_target)
            dec = LearnedRouterDecision(
                variant=v,
                predicted_quality=q,
                predicted_ttft_ms=ttft,
                predicted_tpot_ms=tpot,
                predicted_total_ms=total,
                slo_target_ms=slo_target,
                score=score,
                slo_compliant_pred=(pred_lat <= slo_target),
            )
            if best is None or dec.score < best.score:
                best = dec

        assert best is not None
        return best

    def save(self, out_dir: str, extra_metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "quality_models.pkl"), "wb") as f:
            pickle.dump(self.quality_models, f)
        with open(os.path.join(out_dir, "ttft_models.pkl"), "wb") as f:
            pickle.dump(self.ttft_models, f)
        with open(os.path.join(out_dir, "tpot_models.pkl"), "wb") as f:
            pickle.dump(self.tpot_models, f)
        with open(os.path.join(out_dir, "weights.json"), "w") as f:
            json.dump({"lambda_slo": self.lambda_slo, "mu_quality": self.mu_quality}, f, indent=2)
        # Keep metadata consistent with the actual feature extractor.
        try:
            feature_dim = int(
                self.extract_features(
                    dataset_type="gsm8k",
                    difficulty="easy",
                    max_tokens=1,
                    prompt_tokens=1,
                    concurrency=1,
                    queue_depths={"cheap": 0, "med": 0, "base": 0},
                ).shape[1]
            )
        except Exception:
            feature_dim = 22

        meta: Dict[str, Any] = {"feature_dim": int(feature_dim), "variants": ["cheap", "med", "base"]}
        if extra_metadata:
            meta["extra_metadata"] = extra_metadata
        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, in_dir: str) -> "LearnedRouter":
        with open(os.path.join(in_dir, "quality_models.pkl"), "rb") as f:
            quality = pickle.load(f)
        with open(os.path.join(in_dir, "ttft_models.pkl"), "rb") as f:
            ttft = pickle.load(f)
        with open(os.path.join(in_dir, "tpot_models.pkl"), "rb") as f:
            tpot = pickle.load(f)
        lambda_slo = 1.5
        mu_quality = 1.5
        weights_path = os.path.join(in_dir, "weights.json")
        if os.path.exists(weights_path):
            try:
                w = json.load(open(weights_path, "r"))
                lambda_slo = float(w.get("lambda_slo", lambda_slo))
                mu_quality = float(w.get("mu_quality", mu_quality))
            except Exception:
                pass
        return cls(quality_models=quality, ttft_models=ttft, tpot_models=tpot, lambda_slo=lambda_slo, mu_quality=mu_quality)
