"""SLO-safe contextual bandit router.

Implements the blueprint algorithm:
  - Action space: (variant, adapter_id, rank_tier)
  - Online models for quality and SLO-violation risk
  - Primal-dual virtual queue for risk budget enforcement
  - Conservative deviation screen vs. a baseline policy

This module is intentionally dependency-light (numpy only) so it can run in the
same environment as the rest of the codebase.

Design notes
------------
* We use a lightweight online logistic model with an approximate second-order
  update (online IRLS / Newton step) and a Sherman–Morrison covariance update.
* Uncertainty is computed as sqrt(x^T A^{-1} x). Bounds are formed as
  mean ± beta * uncertainty (then clipped to [0, 1]).
* The router can operate with a dynamic action set per request (e.g., when
  adapters are not available). Models are instantiated lazily per action key.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else x)


def _stable_uniform_0_1(seed: int, key: str) -> float:
    """Deterministic U(0,1) from (seed, key)."""

    h = hashlib.md5(f"{int(seed)}::{key}".encode("utf-8")).hexdigest()
    # 32-bit prefix
    u32 = int(h[:8], 16)
    return float(u32) / float(2**32)


@dataclass(frozen=True)
class BanditAction:
    variant: str
    adapter_id: str = ""
    adapter_rank: Optional[int] = None

    def key(self) -> str:
        r = "none" if self.adapter_rank is None else str(int(self.adapter_rank))
        aid = str(self.adapter_id or "")
        return f"{self.variant}||{aid}||{r}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "adapter_id": str(self.adapter_id or ""),
            "adapter_rank": int(self.adapter_rank) if self.adapter_rank is not None else None,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BanditAction":
        return cls(
            variant=str(d.get("variant") or "base"),
            adapter_id=str(d.get("adapter_id") or ""),
            adapter_rank=int(d["adapter_rank"]) if d.get("adapter_rank") is not None else None,
        )


@dataclass
class BanditRouterConfig:
    # Risk budget (target long-run violation rate)
    delta: float = 0.05

    # Trade-off: higher alpha values weight quality more strongly.
    alpha: float = 1.0

    # Confidence multipliers for risk/quality bounds used in conservative screen.
    beta_r: float = 2.0
    beta_q: float = 2.0

    # Conservative deviation tolerances vs. baseline.
    eps_r: float = 0.0
    eps_q: float = 0.0

    # Exploration bonus multiplier (subtracted from score).
    beta_u: float = 0.2

    # Convert overhead milliseconds into "cost units" (token-equivalent units).
    overhead_ms_to_cost_units: float = 0.1

    # Online model hyperparameters
    l2_reg: float = 10.0
    step_size: float = 1.0

    # Whether to enforce conservative fallback screen.
    use_conservative_fallback: bool = True

    # Whether to maintain / use the primal-dual queue Q.
    use_primal_dual: bool = True

    # Optional additional guard provided by the caller (e.g., conformal latency-safe).
    require_action_latency_safe: bool = True

    # Seed for deterministic label subsampling and any internal randomness.
    seed: int = 0

    # Fraction of requests for which quality labels are "observed" (simulate delayed labels).
    label_budget_p: float = 1.0

    # Delayed-label support: store (action, x) when the quality label is not yet available.
    # This enables later ingestion via `ingest_quality_label(join_key, y)`.
    max_pending_labels: int = 200000
    store_pending_when_no_label: bool = True

    # If set, periodically save state here.
    checkpoint_path: Optional[str] = None
    checkpoint_every: int = 500

    # Safety: if True, never update on requests that escalated to another variant.
    skip_update_on_escalation: bool = True

    # Feature ablations
    use_adapter_features: bool = True
    use_system_features: bool = True
    use_overhead_cost: bool = True


class _OnlineLogisticModel:
    """Online logistic regression with approximate uncertainty."""

    def __init__(self, dim: int, l2_reg: float = 10.0, step_size: float = 1.0):
        self.dim = int(dim)
        self.l2_reg = float(max(1e-6, l2_reg))
        self.step_size = float(max(1e-6, step_size))

        self.w = np.zeros((self.dim,), dtype=np.float32)
        self.A_inv = (1.0 / self.l2_reg) * np.eye(self.dim, dtype=np.float32)
        self.n = 0

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        z = float(np.dot(self.w, x))
        z = max(-20.0, min(20.0, z))
        p = _sigmoid(z)
        # Uncertainty proxy
        try:
            unc = float(math.sqrt(max(0.0, float(x @ (self.A_inv @ x)))))
        except Exception:
            unc = 0.0
        return float(p), float(unc)

    def update(self, x: np.ndarray, y: int) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = 1 if int(y) != 0 else 0

        p, _ = self.predict(x)
        # Curvature weight (<= 0.25)
        w = float(max(1e-6, min(0.25, p * (1.0 - p))))

        # Sherman–Morrison update on A_inv for A := A + w * x x^T
        u = math.sqrt(w) * x
        Au = self.A_inv @ u
        denom = 1.0 + float(u @ Au)
        if denom > 1e-6:
            self.A_inv = self.A_inv - np.outer(Au, Au) / float(denom)

        # Approximate Newton step
        g = (float(y) - float(p)) * x
        self.w = self.w + (self.step_size * (self.A_inv @ g))
        self.n += 1

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dim": int(self.dim),
            "l2_reg": float(self.l2_reg),
            "step_size": float(self.step_size),
            "n": int(self.n),
        }


class BanditRouter:
    """Risk-constrained contextual bandit router (online)."""

    def __init__(self, feature_dim: int, config: Optional[BanditRouterConfig] = None):
        self.config = config or BanditRouterConfig()
        self.feature_dim = int(feature_dim)

        self._lock = threading.Lock()
        self._q_models: Dict[str, _OnlineLogisticModel] = {}
        self._r_models: Dict[str, _OnlineLogisticModel] = {}

        # Pending quality labels: join_key -> (action_key, x)
        self._pending: Dict[str, Tuple[str, np.ndarray]] = {}
        self._pending_fifo: List[str] = []
        self._quality_updates_from_ingest: int = 0

        self.Q: float = 0.0
        self.t: int = 0

    def _ensure_models(self, action_key: str) -> None:
        if action_key in self._q_models and action_key in self._r_models:
            return
        self._q_models.setdefault(
            action_key,
            _OnlineLogisticModel(self.feature_dim, l2_reg=self.config.l2_reg, step_size=self.config.step_size),
        )
        self._r_models.setdefault(
            action_key,
            _OnlineLogisticModel(self.feature_dim, l2_reg=self.config.l2_reg, step_size=self.config.step_size),
        )

    def predict_quality(self, action_key: str, x: np.ndarray) -> Tuple[float, float, float, float]:
        self._ensure_models(action_key)
        p, unc = self._q_models[action_key].predict(x)
        lcb = _clip01(float(p) - float(self.config.beta_q) * float(unc))
        ucb = _clip01(float(p) + float(self.config.beta_q) * float(unc))
        return float(p), float(unc), float(lcb), float(ucb)

    def predict_risk(self, action_key: str, x: np.ndarray) -> Tuple[float, float, float, float]:
        self._ensure_models(action_key)
        p, unc = self._r_models[action_key].predict(x)
        lcb = _clip01(float(p) - float(self.config.beta_r) * float(unc))
        ucb = _clip01(float(p) + float(self.config.beta_r) * float(unc))
        return float(p), float(unc), float(lcb), float(ucb)

    def score_action(self, *, action_key: str, x: np.ndarray, cost_hat: float, Q: float) -> Dict[str, float]:
        q_mean, q_unc, q_lcb, q_ucb = self.predict_quality(action_key, x)
        r_mean, r_unc, r_lcb, r_ucb = self.predict_risk(action_key, x)

        u_bonus = float(self.config.beta_u) * float(q_unc + r_unc)
        score = float(cost_hat) + float(Q) * float(r_mean) - float(self.config.alpha) * float(q_mean) - float(u_bonus)
        return {
            "score": float(score),
            "cost_hat": float(cost_hat),
            "q_mean": float(q_mean),
            "q_unc": float(q_unc),
            "q_lcb": float(q_lcb),
            "q_ucb": float(q_ucb),
            "r_mean": float(r_mean),
            "r_unc": float(r_unc),
            "r_lcb": float(r_lcb),
            "r_ucb": float(r_ucb),
            "u_bonus": float(u_bonus),
        }

    def route(
        self,
        *,
        actions: List[BanditAction],
        features_by_action: Dict[str, np.ndarray],
        cost_hat_by_action: Dict[str, float],
        baseline_action: BanditAction,
        action_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[BanditAction, Dict[str, Any]]:
        if not actions:
            return baseline_action, {"fallback_used": True, "fallback_reason": "no_actions"}

        baseline_key = baseline_action.key()

        # Optional latency-safe filter
        if self.config.require_action_latency_safe and isinstance(action_info, dict):
            filtered: List[BanditAction] = []
            for a in actions:
                info = action_info.get(a.key(), {}) if isinstance(action_info.get(a.key(), {}), dict) else {}
                if info.get("latency_safe", True):
                    filtered.append(a)
            if filtered:
                actions = filtered

        with self._lock:
            Q_now = float(self.Q)

            # Ensure baseline is present
            if baseline_key not in {a.key() for a in actions}:
                actions = list(actions) + [baseline_action]

            scored: List[Tuple[float, BanditAction, Dict[str, float]]] = []
            for a in actions:
                k = a.key()
                x = features_by_action.get(k)
                if x is None:
                    continue
                c = float(cost_hat_by_action.get(k, 0.0))
                comps = self.score_action(action_key=k, x=x, cost_hat=c, Q=Q_now)
                scored.append((float(comps["score"]), a, comps))

            if not scored:
                return baseline_action, {"fallback_used": True, "fallback_reason": "no_scored_actions"}

            scored.sort(key=lambda t: t[0])
            cand_score, cand_action, cand_comps = scored[0]

            fallback_used = False
            fallback_reason = ""
            chosen_action = cand_action
            chosen_comps = cand_comps

            if self.config.use_conservative_fallback:
                bx = features_by_action.get(baseline_key)
                if bx is None:
                    fallback_used = True
                    fallback_reason = "baseline_features_missing"
                else:
                    bc = float(cost_hat_by_action.get(baseline_key, 0.0))
                    base_comps = self.score_action(action_key=baseline_key, x=bx, cost_hat=bc, Q=Q_now)

                    ok_r = float(cand_comps["r_ucb"]) <= float(base_comps["r_ucb"]) + float(self.config.eps_r)
                    ok_q = float(cand_comps["q_lcb"]) >= float(base_comps["q_lcb"]) - float(self.config.eps_q)

                    if not (ok_r and ok_q):
                        fallback_used = True
                        fallback_reason = f"conservative_screen_failed(r={ok_r},q={ok_q})"
                    else:
                        chosen_action = cand_action
                        chosen_comps = cand_comps

            if fallback_used:
                chosen_action = baseline_action
                bx = features_by_action.get(baseline_key)
                bc = float(cost_hat_by_action.get(baseline_key, 0.0))
                if bx is not None:
                    chosen_comps = self.score_action(action_key=baseline_key, x=bx, cost_hat=bc, Q=Q_now)

            meta: Dict[str, Any] = {
                "bandit": {
                    "Q": float(Q_now),
                    "delta": float(self.config.delta),
                    "alpha": float(self.config.alpha),
                    "beta_r": float(self.config.beta_r),
                    "beta_q": float(self.config.beta_q),
                    "eps_r": float(self.config.eps_r),
                    "eps_q": float(self.config.eps_q),
                    "beta_u": float(self.config.beta_u),
                    "fallback_used": bool(fallback_used),
                    "fallback_reason": str(fallback_reason),
                    "baseline_action": baseline_action.to_dict(),
                    "candidate_action": cand_action.to_dict(),
                    "chosen_action": chosen_action.to_dict(),
                    "chosen_score": float(chosen_comps.get("score", cand_score)),
                    "chosen_components": {k: float(v) for k, v in chosen_comps.items()},
                }
            }

            topk = []
            for s, a, comps in scored[: min(5, len(scored))]:
                topk.append(
                    {
                        "action": a.to_dict(),
                        "score": float(s),
                        "cost_hat": float(comps.get("cost_hat", 0.0)),
                        "q_mean": float(comps.get("q_mean", 0.0)),
                        "r_mean": float(comps.get("r_mean", 0.0)),
                        "u_bonus": float(comps.get("u_bonus", 0.0)),
                    }
                )
            meta["bandit"]["topk"] = topk

            return chosen_action, meta

    def update(
        self,
        *,
        action: BanditAction,
        x: np.ndarray,
        cost: float,
        risk_violation: int,
        quality_label: Optional[int],
        label_key: str = "",
        escalated: bool = False,
    ) -> Dict[str, Any]:
        if escalated and self.config.skip_update_on_escalation:
            return {"updated": False, "skip_reason": "escalated"}

        k = action.key()
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.feature_dim:
            raise ValueError(f"BanditRouter.update: feature_dim mismatch ({x.shape[0]} != {self.feature_dim})")

        with self._lock:
            self._ensure_models(k)

            v = 1 if int(risk_violation) != 0 else 0
            self._r_models[k].update(x, v)

            # ------------------------------------------------------------
            # Quality label update (may be sparse and/or delayed)
            # ------------------------------------------------------------
            label_used = False
            y_used: Optional[int] = None

            pending_stored = False
            pending_evicted: Optional[str] = None

            if quality_label is not None:
                # Sparse-label budget (simulate that only p% of requests are judged).
                p = float(self.config.label_budget_p)
                if p >= 1.0:
                    label_used = True
                elif p <= 0.0:
                    label_used = False
                else:
                    u = _stable_uniform_0_1(int(self.config.seed), str(label_key or k))
                    label_used = bool(u < p)

                if label_used:
                    y_used = 1 if int(quality_label) != 0 else 0
                    self._q_models[k].update(x, y_used)
            else:
                # No label yet → store a pending record for later ingestion.
                if bool(self.config.store_pending_when_no_label) and str(label_key or ""):
                    max_p = int(max(0, int(self.config.max_pending_labels)))
                    if max_p > 0:
                        join = str(label_key)
                        if join not in self._pending:
                            # FIFO eviction (paper-friendly + deterministic)
                            if len(self._pending_fifo) >= max_p:
                                old = self._pending_fifo.pop(0)
                                self._pending.pop(old, None)
                                pending_evicted = old
                            self._pending[join] = (k, x.copy())
                            self._pending_fifo.append(join)
                            pending_stored = True

            Q_before = float(self.Q)
            Q_after = Q_before
            if self.config.use_primal_dual:
                Q_after = max(0.0, Q_before + (float(v) - float(self.config.delta)))
                self.Q = float(Q_after)

            self.t += 1

            ckpt_written = False
            ckpt_path = self.config.checkpoint_path
            if ckpt_path and self.config.checkpoint_every > 0 and (self.t % int(self.config.checkpoint_every) == 0):
                try:
                    self.save(ckpt_path)
                    ckpt_written = True
                except Exception:
                    ckpt_written = False

            return {
                "updated": True,
                "action_key": k,
                "t": int(self.t),
                "cost": float(cost),
                "risk_violation": int(v),
                "quality_label": (int(quality_label) if quality_label is not None else None),
                "quality_label_used": bool(label_used),
                "quality_label_used_value": (int(y_used) if y_used is not None else None),
                "pending_stored": bool(pending_stored),
                "pending_evicted": pending_evicted,
                "pending_size": int(len(self._pending)),
                "Q_before": float(Q_before),
                "Q_after": float(Q_after),
                "checkpoint_written": bool(ckpt_written),
            }

    # ------------------------------------------------------------------
    # Delayed label ingestion
    # ------------------------------------------------------------------

    def ingest_quality_label(self, join_key: str, quality_label: int) -> Dict[str, Any]:
        """Apply a delayed quality label.

        This updates *only* the quality model for the stored (action, x) pair.
        It does **not** update the risk model nor the primal-dual queue Q.
        """

        join_key = str(join_key or "")
        if not join_key:
            return {"updated": False, "error": "empty_join_key"}

        y = 1 if int(quality_label) != 0 else 0

        with self._lock:
            rec = self._pending.pop(join_key, None)
            if rec is None:
                return {"updated": False, "error": "join_key_not_found", "join_key": join_key}

            # Remove from FIFO list (O(n), but bounded and only used in offline replay).
            try:
                self._pending_fifo.remove(join_key)
            except ValueError:
                pass

            action_key, x = rec
            self._ensure_models(action_key)
            self._q_models[action_key].update(np.asarray(x, dtype=np.float32).reshape(-1), y)
            self._quality_updates_from_ingest += 1

            return {
                "updated": True,
                "join_key": join_key,
                "action_key": action_key,
                "quality_label": int(y),
                "pending_size": int(len(self._pending)),
                "ingest_quality_updates": int(self._quality_updates_from_ingest),
            }

    def ingest_quality_label_direct(
        self,
        *,
        join_key: str,
        action: BanditAction,
        x: np.ndarray,
        quality_label: int,
    ) -> Dict[str, Any]:
        """Ingest a delayed label when (action, x) is provided externally.

        This is useful when replaying from logs without relying on the pending buffer.
        """

        join_key = str(join_key or "")
        y = 1 if int(quality_label) != 0 else 0
        k = action.key()
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.feature_dim:
            return {"updated": False, "error": "feature_dim_mismatch", "got": int(x.shape[0]), "exp": int(self.feature_dim)}

        with self._lock:
            self._ensure_models(k)
            self._q_models[k].update(x, y)
            self._quality_updates_from_ingest += 1
            return {
                "updated": True,
                "join_key": join_key,
                "action_key": k,
                "quality_label": int(y),
                "ingest_quality_updates": int(self._quality_updates_from_ingest),
            }

    @staticmethod
    def _safe_key(name: str) -> str:
        s = "".join(ch if ch.isalnum() else "_" for ch in str(name))
        h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        return f"{s}__{h}"

    def save(self, out_path: str) -> None:
        out_path = str(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        with self._lock:
            meta: Dict[str, Any] = {
                "feature_dim": int(self.feature_dim),
                "t": int(self.t),
                "Q": float(self.Q),
                "config": json.loads(json.dumps(self.config.__dict__)),
                "actions": sorted(list(set(list(self._q_models.keys()) + list(self._r_models.keys())))),
                "pending": {
                    # Preserve FIFO order for deterministic offline replay.
                    "keys": [k for k in self._pending_fifo if k in self._pending],
                    "action_keys": [self._pending[k][0] for k in self._pending_fifo if k in self._pending],
                    "n": int(len(self._pending)),
                },
                "ingest_quality_updates": int(self._quality_updates_from_ingest),
            }

            arrays: Dict[str, np.ndarray] = {}
            for k, m in self._q_models.items():
                sk = self._safe_key(k)
                arrays[f"q_w__{sk}"] = np.asarray(m.w, dtype=np.float32)
                arrays[f"q_Ainv__{sk}"] = np.asarray(m.A_inv, dtype=np.float32)
                meta.setdefault("q_state", {})[k] = m.state_dict()
            for k, m in self._r_models.items():
                sk = self._safe_key(k)
                arrays[f"r_w__{sk}"] = np.asarray(m.w, dtype=np.float32)
                arrays[f"r_Ainv__{sk}"] = np.asarray(m.A_inv, dtype=np.float32)
                meta.setdefault("r_state", {})[k] = m.state_dict()

            # Pending label feature matrix
            try:
                if self._pending_fifo:
                    xs = []
                    for jk in self._pending_fifo:
                        rec = self._pending.get(jk)
                        if rec is None:
                            continue
                        xs.append(np.asarray(rec[1], dtype=np.float32).reshape(-1))
                    X = np.stack(xs, axis=0) if xs else np.zeros((0, int(self.feature_dim)), dtype=np.float32)
                else:
                    X = np.zeros((0, int(self.feature_dim)), dtype=np.float32)
                arrays["pending_X"] = X
            except Exception:
                arrays["pending_X"] = np.zeros((0, int(self.feature_dim)), dtype=np.float32)

        np.savez_compressed(out_path + ".npz", **arrays)
        with open(out_path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, base_path: str) -> "BanditRouter":
        meta_path = base_path + ".json"
        npz_path = base_path + ".npz"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cfg = BanditRouterConfig(**(meta.get("config") or {}))
        router = cls(feature_dim=int(meta.get("feature_dim", 0) or 0), config=cfg)
        router.t = int(meta.get("t", 0) or 0)
        router.Q = float(meta.get("Q", 0.0) or 0.0)
        try:
            router._quality_updates_from_ingest = int(meta.get("ingest_quality_updates", 0) or 0)
        except Exception:
            router._quality_updates_from_ingest = 0

        arrays = np.load(npz_path, allow_pickle=False)
        actions = meta.get("actions") or []
        for k in actions:
            router._ensure_models(k)
            sk = router._safe_key(k)
            qw = arrays.get(f"q_w__{sk}")
            qA = arrays.get(f"q_Ainv__{sk}")
            rw = arrays.get(f"r_w__{sk}")
            rA = arrays.get(f"r_Ainv__{sk}")
            if qw is not None:
                router._q_models[k].w = np.asarray(qw, dtype=np.float32)
            if qA is not None:
                router._q_models[k].A_inv = np.asarray(qA, dtype=np.float32)
            if rw is not None:
                router._r_models[k].w = np.asarray(rw, dtype=np.float32)
            if rA is not None:
                router._r_models[k].A_inv = np.asarray(rA, dtype=np.float32)

        # Restore pending labels (best-effort)
        try:
            pend = meta.get("pending") or {}
            keys = list(pend.get("keys") or [])
            action_keys = list(pend.get("action_keys") or [])
            X = arrays.get("pending_X")
            if X is None:
                X = np.zeros((0, int(router.feature_dim)), dtype=np.float32)
            X = np.asarray(X, dtype=np.float32)
            router._pending = {}
            router._pending_fifo = []
            n = min(len(keys), len(action_keys), int(X.shape[0]))
            for i in range(n):
                jk = str(keys[i])
                ak = str(action_keys[i])
                router._pending[jk] = (ak, np.asarray(X[i], dtype=np.float32).reshape(-1))
                router._pending_fifo.append(jk)
        except Exception:
            router._pending = {}
            router._pending_fifo = []

        return router
