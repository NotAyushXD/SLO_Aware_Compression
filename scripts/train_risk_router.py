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
    return ap.parse_args()


def _extract_features(r: Dict[str, Any]) -> np.ndarray:
    from risk_router import RiskRouter

    qds = r.get("queue_depths") or {}
    return RiskRouter.extract_features(
        dataset_type=str(r.get("dataset") or "gsm8k"),
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
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train per-variant models: quality, TTFT, total latency."""

    quality_models: Dict[str, Any] = {}
    ttft_models: Dict[str, Any] = {}
    total_models: Dict[str, Any] = {}

    for v in variants:
        rows = [r for r in train_records if r.get("variant") == v and int(r.get("success", 1))]
        if len(rows) < int(min_rows):
            raise RuntimeError(f"Not enough training rows for variant={v}: {len(rows)} < {min_rows}")

        X = np.concatenate([_extract_features(r) for r in rows], axis=0)
        yq = np.array([int(r.get("correct", 0)) for r in rows], dtype=int)
        yttft = np.array([float(r.get("ttft_ms", 0.0) or 0.0) for r in rows], dtype=float)
        ytot = np.array([float(r.get("total_latency_ms", 0.0) or 0.0) for r in rows], dtype=float)

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
) -> Dict[str, np.ndarray]:
    """Compute residual arrays and quality score arrays on the calibration set."""

    out: Dict[str, np.ndarray] = {}
    for v in variants:
        rows = [r for r in calib_records if r.get("variant") == v and int(r.get("success", 1))]
        if not rows:
            # Keep empty arrays; router will degrade gracefully.
            out[f"resid_ttft__{v}"] = np.zeros((0,), dtype=float)
            out[f"resid_total__{v}"] = np.zeros((0,), dtype=float)
            out[f"qscore__{v}"] = np.zeros((0,), dtype=float)
            out[f"qlabel__{v}"] = np.zeros((0,), dtype=int)
            continue

        X = np.concatenate([_extract_features(r) for r in rows], axis=0)
        yq = np.array([int(r.get("correct", 0)) for r in rows], dtype=int)
        yttft = np.array([float(r.get("ttft_ms", 0.0) or 0.0) for r in rows], dtype=float)
        ytot = np.array([float(r.get("total_latency_ms", 0.0) or 0.0) for r in rows], dtype=float)

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

    variants = sorted({str(r.get("variant")) for r in records if str(r.get("variant"))})
    # Normalize and keep canonical ordering
    order = ["cheap", "med", "base"]
    variants = [v for v in order if v in variants]
    if not variants:
        raise RuntimeError("Could not infer variants from trace file.")

    train_records, calib_records = _make_splits(records, seed=args.seed)

    quality_models, ttft_models, total_models = train_models(train_records, variants, min_rows=args.min_rows_per_variant)
    calib = build_calibration_arrays(calib_records, variants, quality_models, ttft_models, total_models)

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
            "train": sum(1 for r in train_records if r.get("variant") == v),
            "calib": sum(1 for r in calib_records if r.get("variant") == v),
        }
    _write_json(str(out_dir / "summary.json"), summary)

    print(f"Saved risk-router bundle to: {out_dir}")


if __name__ == "__main__":
    main()
