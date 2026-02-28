#!/usr/bin/env python
"""Train the RiskRouter (latency + quality predictors) and fit calibration data.

This script is meant to be run after collecting traces with:

  python scripts/train_learned_router.py --collect_only ...

That collector writes a JSONL where each row corresponds to running a specific
variant (cheap/med/base) on a specific example under a specific concurrency.

We then:
  1) Train predictors on TRAIN split.
  2) Use VAL split as the calibration set.
  3) Save a "router bundle" that the server can load at runtime.

The bundle contains:
  - sklearn models for quality / TTFT / total latency
  - calibration arrays (residuals + quality scores/labels)
"""

from __future__ import annotations

# Allow running as a script from the scripts/ directory (Kaggle notebooks often do this).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace JSONL not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Allow resume/append files where the last line might be partial.
                continue
    return rows


def _write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _make_splits(records: List[Dict[str, Any]], seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (train_records, calib_records).

    Prefer the explicit 'split' field (train/val). If missing, do an 80/20 split.
    """

    has_split = any(str(r.get("split", "")).strip() for r in records)
    if has_split:
        train = [r for r in records if str(r.get("split", "")).lower().strip() == "train"]
        calib = [r for r in records if str(r.get("split", "")).lower().strip() == "val"]
        if train and calib:
            return train, calib

    rng = random.Random(int(seed))
    idx = list(range(len(records)))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_idx = set(idx[:cut])
    train = [records[i] for i in range(len(records)) if i in train_idx]
    calib = [records[i] for i in range(len(records)) if i not in train_idx]
    return train, calib


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_jsonl", type=str, required=True, help="Path to JSONL traces (trainval_traces.jsonl).")
    ap.add_argument("--output_dir", type=str, required=True, help="Output dir for router bundle.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_rows_per_variant", type=int, default=50)
    ap.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help=(
            "Optional list of variants to train (e.g., --variants cheap base). "
            "If omitted, variants are inferred from the trace file."
        ),
    )
    ap.add_argument(
        "--success_only",
        action="store_true",
        help=(
            "If set, train using only rows where success==1. "
            "Default behavior is best-effort: if a variant has too few successful rows, "
            "we include failed rows with conservative labels so training can proceed."
        ),
    )
    ap.add_argument(
        "--strict_min_rows",
        action="store_true",
        help=(
            "If set, error out when a variant has fewer than --min_rows_per_variant usable rows. "
            "Default is to proceed with a warning."
        ),
    )
    return ap.parse_args()


def _norm_variant(v: str) -> str:
    v = str(v or "").lower().strip()
    if v in {"medium", "mid"}:
        return "med"
    if v in {"small", "lite", "light"}:
        return "cheap"
    if v in {"large", "full"}:
        return "base"
    return v


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _coerce_latency_ms(r: Dict[str, Any], key: str, *, fail_value_ms: float) -> float:
    """Return a latency label (ms). If success==0 or value is missing/zero, return a large fail value."""
    ok = _safe_int(r.get("success", 1), 1) == 1
    try:
        v = float(r.get(key, 0.0) or 0.0)
    except Exception:
        v = 0.0
    if (not ok) or (v <= 0.0):
        return float(fail_value_ms)
    return float(v)


def _summarize_trace(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"total_rows": len(records), "by_variant": {}}
    by_v: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        v = _norm_variant(r.get("variant"))
        if not v:
            continue
        by_v.setdefault(v, []).append(r)
    for v, rows in by_v.items():
        succ = sum(1 for r in rows if _safe_int(r.get("success", 1), 1) == 1)
        out["by_variant"][v] = {
            "rows": len(rows),
            "success_rows": succ,
            "success_rate": float(succ / max(1, len(rows))),
        }
    return out


def _extract_features(r: Dict[str, Any]) -> np.ndarray:
    from risk_router import RiskRouter

    qds = r.get("queue_depths") or {}
    return RiskRouter.extract_features(
        dataset_type=str(r.get("dataset_type") or r.get("dataset") or "gsm8k"),
        difficulty=str(r.get("difficulty") or "easy"),
        max_tokens=int(r.get("max_tokens") or 0),
        prompt_tokens=int(r.get("prompt_tokens") or 0),
        concurrency=int(r.get("concurrency") or 1),
        queue_depths={k: int(v or 0) for k, v in dict(qds).items()},
        adapter_id=str(r.get("adapter_id") or ""),
        adapter_rank=(r.get("adapter_rank") if r.get("adapter_rank") is not None else None),
        adapter_state=(r.get("adapter_state") if isinstance(r.get("adapter_state"), dict) else None),
    )


def train_models(
    train_records: List[Dict[str, Any]],
    variants: List[str],
    min_rows: int,
    success_only: bool = False,
    strict_min_rows: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train per-variant models: quality, TTFT, total latency."""

    quality_models: Dict[str, Any] = {}
    ttft_models: Dict[str, Any] = {}
    total_models: Dict[str, Any] = {}

    # A conservative default for "failed" requests. We intentionally use a very
    # large value so the resulting router prefers safer variants.
    FAIL_TTFT_MS = 1e6
    FAIL_TOTAL_MS = 1e6

    for v in variants:
        # Prefer successful rows, but fall back to including failed rows (with
        # conservative labels) so training can proceed on constrained hardware.
        rows_succ = [
            r
            for r in train_records
            if _norm_variant(r.get("variant")) == v and _safe_int(r.get("success", 1), 1) == 1
        ]
        rows_all = [r for r in train_records if _norm_variant(r.get("variant")) == v]

        rows = rows_succ
        if (not bool(success_only)) and len(rows) < int(min_rows) and len(rows_all) > 0:
            rows = rows_all

        if len(rows) == 0:
            raise RuntimeError(
                f"No rows found for variant={v}. "
                "Check that your trace JSONL contains 'variant' entries matching this variant."
            )

        if len(rows) < int(min_rows):
            msg = (
                f"variant={v}: usable_rows={len(rows)} < min_rows={min_rows}. "
                f"(success_rows={len(rows_succ)}, total_rows={len(rows_all)}, success_only={success_only})"
            )
            if bool(strict_min_rows):
                raise RuntimeError(msg)
            print(
                "[warn] " + msg + "\n"
                "       Proceeding anyway (set --strict_min_rows to fail-fast)."
            )

        X = np.concatenate([_extract_features(r) for r in rows], axis=0)
        # Conservative label handling:
        # - If success==0, treat the request as incorrect and assign a very large
        #   latency so the router learns to avoid that action.
        yq = np.array(
            [int(r.get("correct", 0)) if (_safe_int(r.get("success", 1), 1) == 1) else 0 for r in rows],
            dtype=int,
        )
        yttft = np.array([_coerce_latency_ms(r, "ttft_ms", fail_value_ms=FAIL_TTFT_MS) for r in rows], dtype=float)
        ytot = np.array(
            [_coerce_latency_ms(r, "total_latency_ms", fail_value_ms=FAIL_TOTAL_MS) for r in rows],
            dtype=float,
        )

        # Quality model: LogisticRegression when we have both classes, else a
        # constant prior DummyClassifier (keeps pipeline runnable when a variant
        # never succeeds / never answers correctly in traces).
        if len(np.unique(yq)) >= 2:
            q_model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            penalty="l2",
                            C=1.0,
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            )
            q_model.fit(X, yq)
        else:
            q_model = DummyClassifier(strategy="prior")
            # DummyClassifier returns probabilities only for seen classes.
            # If a variant's traces contain a single class (all correct or all incorrect),
            # we add one tiny synthetic example of the opposite class so predict_proba()
            # always has 2 columns and RiskRouter can safely index [:, 1].
            y0 = int(yq[0]) if len(yq) > 0 else 0
            X_aug = np.concatenate([X, X[:1]], axis=0)
            y_aug = np.concatenate([yq, np.array([1 - y0], dtype=int)], axis=0)
            q_model.fit(X_aug, y_aug)
        quality_models[v] = q_model

        tt_model = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
        tt_model.fit(X, yttft)
        ttft_models[v] = tt_model

        tot_model = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
        tot_model.fit(X, ytot)
        total_models[v] = tot_model

    return quality_models, ttft_models, total_models


def build_calibration_arrays(
    calib_records: List[Dict[str, Any]],
    variants: List[str],
    quality_models: Dict[str, Any],
    ttft_models: Dict[str, Any],
    total_models: Dict[str, Any],
    success_only: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute residual arrays and quality score arrays on the calibration set."""

    out: Dict[str, np.ndarray] = {}
    FAIL_TTFT_MS = 1e6
    FAIL_TOTAL_MS = 1e6
    for v in variants:
        rows_succ = [
            r
            for r in calib_records
            if _norm_variant(r.get("variant")) == v and _safe_int(r.get("success", 1), 1) == 1
        ]
        rows_all = [r for r in calib_records if _norm_variant(r.get("variant")) == v]

        if bool(success_only):
            rows = rows_succ
        else:
            rows = rows_succ if len(rows_succ) > 0 else rows_all
        if not rows:
            # Keep empty arrays; router will degrade gracefully.
            out[f"resid_ttft__{v}"] = np.zeros((0,), dtype=float)
            out[f"resid_total__{v}"] = np.zeros((0,), dtype=float)
            out[f"qscore__{v}"] = np.zeros((0,), dtype=float)
            out[f"qlabel__{v}"] = np.zeros((0,), dtype=int)
            continue

        X = np.concatenate([_extract_features(r) for r in rows], axis=0)
        yq = np.array(
            [int(r.get("correct", 0)) if (_safe_int(r.get("success", 1), 1) == 1) else 0 for r in rows],
            dtype=int,
        )
        yttft = np.array([_coerce_latency_ms(r, "ttft_ms", fail_value_ms=FAIL_TTFT_MS) for r in rows], dtype=float)
        ytot = np.array(
            [_coerce_latency_ms(r, "total_latency_ms", fail_value_ms=FAIL_TOTAL_MS) for r in rows],
            dtype=float,
        )

        # Predictions
        qm = quality_models[v]
        if hasattr(qm, "predict_proba"):
            qscore = qm.predict_proba(X)[:, 1]
        else:
            qscore = qm.predict(X)

        tt_pred = ttft_models[v].predict(X)
        tot_pred = total_models[v].predict(X)

        resid_ttft = yttft - tt_pred
        resid_total = ytot - tot_pred

        out[f"resid_ttft__{v}"] = np.asarray(resid_ttft, dtype=float)
        out[f"resid_total__{v}"] = np.asarray(resid_total, dtype=float)
        out[f"qscore__{v}"] = np.asarray(qscore, dtype=float)
        out[f"qlabel__{v}"] = np.asarray(yq, dtype=int)

    return out


def main() -> None:
    args = parse_args()

    records = _read_jsonl(args.trace_jsonl)
    if not records:
        raise RuntimeError("Trace JSONL is empty; did trace collection run?")

    # Helpful for debugging on Kaggle/Colab.
    try:
        print("[trace-summary]", json.dumps(_summarize_trace(records), indent=2))
    except Exception:
        pass

    # Variants to train.
    order = ["cheap", "med", "base"]
    if args.variants is not None and len(args.variants) > 0:
        requested = [_norm_variant(v) for v in args.variants if str(v).strip()]
        variants = [v for v in order if v in requested]
    else:
        variants = sorted({_norm_variant(r.get("variant")) for r in records if str(r.get("variant"))})
        variants = [v for v in order if v in variants]
    if not variants:
        raise RuntimeError("Could not infer variants from trace file.")

    train_records, calib_records = _make_splits(records, seed=args.seed)

    quality_models, ttft_models, total_models = train_models(
        train_records,
        variants,
        min_rows=args.min_rows_per_variant,
        success_only=bool(args.success_only),
        strict_min_rows=bool(args.strict_min_rows),
    )
    calib = build_calibration_arrays(
        calib_records,
        variants,
        quality_models,
        ttft_models,
        total_models,
        success_only=bool(args.success_only),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "quality_models.pkl", "wb") as f:
        pickle.dump(quality_models, f)
    with open(out_dir / "ttft_models.pkl", "wb") as f:
        pickle.dump(ttft_models, f)
    with open(out_dir / "total_models.pkl", "wb") as f:
        pickle.dump(total_models, f)

    np.savez_compressed(out_dir / "calibration.npz", **calib)

    # Metadata
    try:
        from risk_router import RiskRouter

        feat_dim = int(
            RiskRouter.extract_features(
                dataset_type="gsm8k",
                difficulty="easy",
                max_tokens=1,
                prompt_tokens=1,
                concurrency=1,
                queue_depths={"cheap": 0, "med": 0, "base": 0},
            ).shape[1]
        )
    except Exception:
        feat_dim = 22

    _write_json(
        str(out_dir / "metadata.json"),
        {
            "variants": variants,
            "feature_dim": int(feat_dim),
            "train_rows": int(len(train_records)),
            "calib_rows": int(len(calib_records)),
            "seed": int(args.seed),
            "bundle_version": 1,
        },
    )

    # Small summary for sanity checks
    summary: Dict[str, Any] = {
        "variants": variants,
        "train_rows": len(train_records),
        "calib_rows": len(calib_records),
        "by_variant": {},
    }
    for v in variants:
        summary["by_variant"][v] = {
            "train": sum(1 for r in train_records if _norm_variant(r.get("variant")) == v),
            "calib": sum(1 for r in calib_records if _norm_variant(r.get("variant")) == v),
        }
    _write_json(str(out_dir / "summary.json"), summary)

    print(f"Saved risk-router bundle to: {out_dir}")


if __name__ == "__main__":
    main()
