#!/usr/bin/env python
"""Train learned router predictors (quality + latency) and tune routing weights.

Protocol (paper-facing, per problem statement):
  1) Collect training traces by running *all 3 variants* (cheap/med/base) per example.
  2) Split examples across concurrencies (default: 1,2,4,8) so predictors learn queue regimes.
     Each example appears under exactly one concurrency.
  3) Train predictors on Train+Val.
  4) Tune (lambda_slo, mu_quality) on Val only.
  5) Save two router modes:
       - router_models/learned_ttft/
       - router_models/learned_total/

Notes:
  - We intentionally disable router escalation during trace collection (max_retries=0)
    so each variant is measured independently.
  - This script uses the HF-based MultiVariantService dispatcher for concurrency-safe
    multi-variant execution and realistic queue-depth snapshots.
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
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

from evaluation import EvaluationMetrics
from learned_router import LearnedRouter
from prompt_templates import build_llama_formatted_prompt
from server import MultiVariantService


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Can happen if a run was interrupted mid-line.
                continue
    return rows


def _write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_slo_dict(slo_path: Optional[str], profile_key: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Load a difficulty->(ttft_ms,tpot_ms) dict.

    Accepts either:
      - the calibration JSON written by run_baseline_evaluation.py (with "profiles")
      - a direct dict with keys easy/medium/hard
    """

    default = {
        "easy": {"ttft_ms": 168.0, "tpot_ms": 10.0},
        "medium": {"ttft_ms": 251.0, "tpot_ms": 12.0},
        "hard": {"ttft_ms": 341.0, "tpot_ms": 15.0},
    }

    if not slo_path:
        return default

    if not os.path.exists(slo_path):
        raise FileNotFoundError(f"SLO thresholds file not found: {slo_path}")

    with open(slo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "profiles" in data and isinstance(data.get("profiles"), dict):
        profiles: Dict[str, Any] = data.get("profiles") or {}
        primary = profile_key or data.get("primary")
        if primary and primary in profiles:
            return profiles[primary]
        if profiles:
            return next(iter(profiles.values()))
        return default

    if isinstance(data, dict) and any(k in data for k in ("easy", "medium", "hard")):
        # Direct slo dict.
        return data  # type: ignore[return-value]

    return default


def _split_across_concurrencies(
    examples: List[Dict[str, Any]],
    concurrencies: List[int],
    seed: int,
) -> Dict[int, List[Dict[str, Any]]]:
    rng = random.Random(int(seed))
    exs = list(examples)
    rng.shuffle(exs)

    parts = np.array_split(np.array(exs, dtype=object), len(concurrencies))
    out: Dict[int, List[Dict[str, Any]]] = {}
    for c, p in zip(concurrencies, parts):
        out[int(c)] = [x for x in p.tolist()]
    return out


def _save_split_map(path: Path, *, seed: int, concurrencies: List[int], assignments: Dict[int, int], example_ids: List[int]) -> None:
    obj = {
        "seed": int(seed),
        "concurrencies": [int(c) for c in concurrencies],
        "num_examples": int(len(example_ids)),
        "example_ids": [int(x) for x in example_ids],
        "assignments": {str(int(k)): int(v) for k, v in assignments.items()},
    }
    _write_json(str(path), obj)


def _load_split_map(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_split_map_stable(
    *,
    trainval: List[Dict[str, Any]],
    concurrencies: List[int],
    seed: int,
    out_root: Path,
) -> Dict[int, List[Dict[str, Any]]]:
    """Create (or reuse) a deterministic example->concurrency assignment.

    We persist the assignment so you can resume long trace-collection runs safely
    across Kaggle sessions without changing the queue-regime labels.
    """

    split_map_path = out_root / "split_map.json"
    example_ids = [int(ex.get("_global_index", i)) for i, ex in enumerate(trainval)]

    existing = _load_split_map(split_map_path)
    if existing:
        # Verify compatibility to avoid mixing regimes.
        if int(existing.get("seed", -1)) != int(seed) or [int(c) for c in existing.get("concurrencies", [])] != [int(c) for c in concurrencies]:
            raise RuntimeError(
                "split_map.json exists but was created with different --seed/--concurrencies. "
                "Use the same flags to resume, or delete router_models/split_map.json and trainval_traces.jsonl to restart."
            )
        if [int(x) for x in existing.get("example_ids", [])] != [int(x) for x in example_ids]:
            raise RuntimeError(
                "split_map.json exists but the Train+Val example set differs (maybe different --max_examples or processed data). "
                "Use the same data/flags to resume, or delete router_models/split_map.json and trainval_traces.jsonl to restart."
            )

        assignments = {int(k): int(v) for k, v in (existing.get("assignments") or {}).items()}
    else:
        # Create a new split map.
        parts = _split_across_concurrencies(trainval, concurrencies, seed=seed)
        assignments = {}
        for c, exs in parts.items():
            for ex in exs:
                gid = int(ex.get("_global_index"))
                assignments[gid] = int(c)
        _save_split_map(split_map_path, seed=seed, concurrencies=concurrencies, assignments=assignments, example_ids=example_ids)

    # Materialize split_map from persisted assignments.
    out: Dict[int, List[Dict[str, Any]]] = {int(c): [] for c in concurrencies}
    for ex in trainval:
        gid = int(ex.get("_global_index"))
        c = int(assignments[gid])
        out.setdefault(c, []).append(ex)
    return out


def _extract_queue_depths(metrics: Dict[str, Any], variants: List[str]) -> Dict[str, int]:
    qd = metrics.get("router_queue_depths")
    if isinstance(qd, dict):
        out = {k: int(v or 0) for k, v in qd.items()}
    else:
        out = {}
    for v in variants:
        out.setdefault(v, 0)
    return out


def _collect_traces_for_concurrency(
    service: MultiVariantService,
    examples: List[Dict[str, Any]],
    concurrency: int,
    variants: List[str],
    prompt_mode: str,
    seed: int,
    tokenizer: Any,
    skip_keys: Optional[set] = None,
    time_budget_s: float = 0.0,
    append_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run all (example, variant) pairs under a fixed concurrency."""

    skip_keys = skip_keys or set()
    start_t = time.time()

    tasks: List[Tuple[int, str, Dict[str, Any]]] = []
    for i, ex in enumerate(examples):
        for v in variants:
            gid = int(ex.get("_global_index", i))
            if (gid, v) in skip_keys:
                continue
            tasks.append((i, v, ex))

    rng = random.Random(int(seed) + int(concurrency) * 1009)
    rng.shuffle(tasks)

    records: List[Dict[str, Any]] = []

    # We avoid concurrent list writes by collecting per-thread and merging.
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    task_q: "queue.Queue[Tuple[int, str, Dict[str, Any]]]" = queue.Queue()
    for t in tasks:
        task_q.put(t)

    rec_lock = threading.Lock()

    # Optional append-to-disk for long-running collection jobs.
    write_lock = threading.Lock()
    trace_f = None
    if append_path:
        Path(append_path).parent.mkdir(parents=True, exist_ok=True)
        trace_f = open(append_path, "a", encoding="utf-8")

    stop_event = threading.Event()

    def _budget_exhausted() -> bool:
        if not time_budget_s or time_budget_s <= 0:
            return False
        return (time.time() - start_t) >= float(time_budget_s)

    def _append_one(rec: Dict[str, Any]) -> None:
        if trace_f is None:
            return
        with write_lock:
            trace_f.write(json.dumps(rec) + "\n")
            trace_f.flush()

    def worker() -> None:
        local: List[Dict[str, Any]] = []
        while True:
            if stop_event.is_set() or _budget_exhausted():
                stop_event.set()
                break
            try:
                ex_idx, variant, ex = task_q.get_nowait()
            except queue.Empty:
                break
            dataset_type = (ex.get("dataset") or "gsm8k").lower().strip()
            difficulty = (ex.get("difficulty") or "medium").lower().strip()

            formatted_prompt, max_new_tokens, _stops = build_llama_formatted_prompt(
                ex, dataset_type=dataset_type, prompt_mode=prompt_mode
            )

            pred_text, metrics = service.generate(
                prompt=formatted_prompt,
                max_tokens=int(max_new_tokens),
                temperature=0.0,
                top_p=1.0,
                dataset_type=dataset_type,
                difficulty=difficulty,
                prompt_mode=prompt_mode,
                concurrency=int(concurrency),
                force_variant=variant,
            )

            truth = ex.get("answer", "")
            ok, extracted, fmt_ok = EvaluationMetrics.is_correct(pred_text, str(truth), dataset_type)
            ok_p, extracted_p, fmt_ok_p = EvaluationMetrics.is_correct_parseable(pred_text, str(truth), dataset_type)

            qdepths = _extract_queue_depths(metrics, variants)

            rec = {
                "global_example_idx": int(ex.get("_global_index", ex_idx)),
                "split": str(ex.get("_split", "")),
                "dataset": dataset_type,
                "difficulty": difficulty,
                "prompt_mode": prompt_mode,
                "max_tokens": int(max_new_tokens),
                "prompt_tokens": int(len(tokenizer(formatted_prompt, add_special_tokens=False).input_ids))
                if tokenizer is not None
                else int(len(formatted_prompt.split())),
                "variant": variant,
                "variant_effective": metrics.get("variant_effective", metrics.get("variant")),
                "concurrency": int(concurrency),
                "queue_depths": qdepths,
                # outputs
                "success": int(bool(metrics.get("success", False))),
                "ground_truth": str(truth),
                "prediction": pred_text,
                # strict
                "correct": int(bool(ok)),
                "format_ok": int(bool(fmt_ok)),
                "extracted": extracted,
                # parseable
                "correct_parseable": int(bool(ok_p)),
                "format_ok_parseable": int(bool(fmt_ok_p)),
                "extracted_parseable": extracted_p,
                # timing
                "ttft_ms": float(metrics.get("ttft_ms", 0.0) or 0.0),
                "tpot_ms": float(metrics.get("tpot_ms", 0.0) or 0.0),
                "total_latency_ms": float(metrics.get("total_latency_ms", 0.0) or 0.0),
                "queue_wait_ms": float(metrics.get("queue_wait_ms", 0.0) or 0.0),
                "lock_wait_ms": float(metrics.get("lock_wait_ms", 0.0) or 0.0),
            }
            local.append(rec)
            _append_one(rec)

            try:
                task_q.task_done()
            except Exception:
                pass

        with rec_lock:
            records.extend(local)

    try:
        with ThreadPoolExecutor(max_workers=int(max(1, concurrency))) as ex:
            futs = [ex.submit(worker) for _ in range(int(max(1, concurrency)))]
            for f in as_completed(futs):
                _ = f.result()
    finally:
        if trace_f is not None:
            try:
                trace_f.close()
            except Exception:
                pass

    return records


def _train_predictors(
    records: List[Dict[str, Any]],
    variants: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train per-variant quality, TTFT, and TPOT predictors."""

    X_by_v: Dict[str, List[List[float]]] = {v: [] for v in variants}
    yq_by_v: Dict[str, List[int]] = {v: [] for v in variants}
    yttft_by_v: Dict[str, List[float]] = {v: [] for v in variants}
    ytpot_by_v: Dict[str, List[float]] = {v: [] for v in variants}

    for r in records:
        if not r.get("success", 1):
            continue
        v = r["variant"]
        qds = r.get("queue_depths") or {}
        feats = LearnedRouter.extract_features(
            dataset_type=r["dataset"],
            difficulty=r["difficulty"],
            max_tokens=int(r["max_tokens"]),
            prompt_tokens=int(r["prompt_tokens"]),
            concurrency=int(r.get("concurrency", 1)),
            queue_depths=qds,
        )[0].tolist()
        X_by_v[v].append(feats)
        yq_by_v[v].append(int(r.get("correct", 0)))
        yttft_by_v[v].append(float(r.get("ttft_ms", 0.0) or 0.0))
        ytpot_by_v[v].append(float(r.get("tpot_ms", 0.0) or 0.0))

    quality_models: Dict[str, Any] = {}
    ttft_models: Dict[str, Any] = {}
    tpot_models: Dict[str, Any] = {}

    for v in variants:
        X = np.array(X_by_v[v], dtype=float)
        yq = np.array(yq_by_v[v], dtype=int)
        yttft = np.array(yttft_by_v[v], dtype=float)
        ytpot = np.array(ytpot_by_v[v], dtype=float)

        if len(X) == 0:
            raise RuntimeError(f"No training rows collected for variant '{v}'.")

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

        tp_model = Pipeline(steps=[("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
        tp_model.fit(X, ytpot)
        tpot_models[v] = tp_model

    return quality_models, ttft_models, tpot_models


def _evaluate_predictors(
    *,
    records: List[Dict[str, Any]],
    variants: List[str],
    quality_models: Dict[str, Any],
    ttft_models: Dict[str, Any],
    tpot_models: Dict[str, Any],
    split_name: str,
) -> Dict[str, Any]:
    """Compute lightweight predictor diagnostics.

    These are not used for model selection; they are printed/saved to help debug
    collection quality and feature/label issues.
    """

    out: Dict[str, Any] = {"split": split_name, "by_variant": {}}
    for v in variants:
        rows = [r for r in records if str(r.get("variant")) == v and bool(r.get("success", 1))]
        if not rows:
            out["by_variant"][v] = {"n": 0}
            continue

        X = []
        yq = []
        yttft = []
        ytpot = []
        for r in rows:
            qds = r.get("queue_depths") or {}
            feats = LearnedRouter.extract_features(
                dataset_type=r.get("dataset", "gsm8k"),
                difficulty=r.get("difficulty", "easy"),
                max_tokens=int(r.get("max_tokens", 0) or 0),
                prompt_tokens=int(r.get("prompt_tokens", 0) or 0),
                concurrency=int(r.get("concurrency", 1) or 1),
                queue_depths=qds,
            )[0].tolist()
            X.append(feats)
            yq.append(int(r.get("correct", 0) or 0))
            yttft.append(float(r.get("ttft_ms", 0.0) or 0.0))
            ytpot.append(float(r.get("tpot_ms", 0.0) or 0.0))

        X_np = np.array(X, dtype=float)
        yq_np = np.array(yq, dtype=int)
        yttft_np = np.array(yttft, dtype=float)
        ytpot_np = np.array(ytpot, dtype=float)

        # Quality
        qm = quality_models[v]
        try:
            probs = qm.predict_proba(X_np)[:, 1]
        except Exception:
            probs = qm.predict(X_np)
        preds = (probs >= 0.5).astype(int)

        q_acc = float(accuracy_score(yq_np, preds))
        q_pos_rate = float(np.mean(yq_np))
        try:
            q_auc = float(roc_auc_score(yq_np, probs)) if len(set(yq_np.tolist())) > 1 else None
        except Exception:
            q_auc = None

        # Latency
        tt_pred = np.array(ttft_models[v].predict(X_np), dtype=float)
        tp_pred = np.array(tpot_models[v].predict(X_np), dtype=float)

        tt_mae = float(mean_absolute_error(yttft_np, tt_pred))
        tp_mae = float(mean_absolute_error(ytpot_np, tp_pred))
        tt_rmse = float(math.sqrt(mean_squared_error(yttft_np, tt_pred)))
        tp_rmse = float(math.sqrt(mean_squared_error(ytpot_np, tp_pred)))

        out["by_variant"][v] = {
            "n": int(len(rows)),
            "quality_pos_rate": q_pos_rate,
            "quality_acc@0.5": q_acc,
            "quality_auc": q_auc,
            "ttft_mae": tt_mae,
            "ttft_rmse": tt_rmse,
            "tpot_mae": tp_mae,
            "tpot_rmse": tp_rmse,
        }

    return out


def _group_by_example(records: List[Dict[str, Any]], variants: List[str]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    grouped: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        gid = int(r.get("global_example_idx", -1))
        if gid < 0:
            continue
        grouped.setdefault(gid, {})[r["variant"]] = r

    # Keep only complete triplets.
    keep: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for gid, d in grouped.items():
        if all(v in d for v in variants):
            keep[gid] = d
    return keep


def _tune_lambda_mu(
    mode: str,
    router: LearnedRouter,
    val_grouped: Dict[int, Dict[str, Dict[str, Any]]],
    slo_dict: Dict[str, Dict[str, float]],
    variants: List[str],
    target_slo_compliance: float = 0.0,
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    """Grid-search lambda/mu on VAL only.

    Returns:
      (best_lambda, best_mu, best_stats, full_grid)
    """

    lambdas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    mus = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    grid: List[Dict[str, Any]] = []

    for lam in lambdas:
        for mu in mus:
            router.lambda_slo = float(lam)
            router.mu_quality = float(mu)

            n = 0
            correct = 0
            slo_ok = 0
            cost_sum = 0.0

            for _gid, by_v in val_grouped.items():
                # Use BASE record as the representative system snapshot for queue depths.
                state = by_v.get("base") or next(iter(by_v.values()))
                qds = dict(state.get("queue_depths") or {})
                for vv in variants:
                    qds.setdefault(vv, 0)

                dec = router.route(
                    dataset_type=state["dataset"],
                    difficulty=state["difficulty"],
                    max_tokens=int(state["max_tokens"]),
                    prompt_tokens=int(state["prompt_tokens"]),
                    concurrency=int(state.get("concurrency", 1)),
                    queue_depths=qds,
                    slo_dict=slo_dict,
                    mode=mode,
                    allowed_variants=variants,
                )

                chosen = dec.variant
                true = by_v.get(chosen) or state

                slo = slo_dict.get(state["difficulty"], {})
                ttft_slo = float(slo.get("ttft_ms", 1e9))
                tpot_slo = float(slo.get("tpot_ms", 1e9))

                if mode == "ttft":
                    violation = float(true.get("ttft_ms", 0.0) or 0.0) > ttft_slo
                else:
                    total_slo = ttft_slo + tpot_slo * float(state["max_tokens"])
                    total_true = float(true.get("ttft_ms", 0.0) or 0.0) + float(true.get("tpot_ms", 0.0) or 0.0) * float(
                        state["max_tokens"]
                    )
                    violation = total_true > total_slo

                is_correct = bool(true.get("correct", 0))
                cost = float(router.VARIANT_COSTS.get(chosen, 1.0))

                n += 1
                correct += int(is_correct)
                slo_ok += int(not violation)
                cost_sum += cost

            if n == 0:
                continue

            stats = {
                "accuracy": float(correct / n),
                "slo_compliance": float(slo_ok / n),
                "avg_cost": float(cost_sum / n),
                "lambda": float(lam),
                "mu": float(mu),
            }
            grid.append(stats)

    if not grid:
        raise RuntimeError("No validation examples available for tuning.")

    target = float(target_slo_compliance or 0.0)
    if target > 0.0:
        eligible = [s for s in grid if float(s.get("slo_compliance", 0.0)) >= target]
        if eligible:
            best = max(eligible, key=lambda s: (float(s["accuracy"]), float(s["slo_compliance"]), -float(s["avg_cost"])))
            best = dict(best)
            best["selection"] = {
                "rule": "compliance_first_then_accuracy",
                "target_slo_compliance": target,
                "eligible": len(eligible),
                "total": len(grid),
            }
            return float(best["lambda"]), float(best["mu"]), best, grid

        # Fallback: no config meets target → maximize compliance, then accuracy, then lower cost.
        best = max(grid, key=lambda s: (float(s["slo_compliance"]), float(s["accuracy"]), -float(s["avg_cost"])))
        best = dict(best)
        best["selection"] = {
            "rule": "fallback_max_compliance",
            "target_slo_compliance": target,
            "eligible": 0,
            "total": len(grid),
        }
        return float(best["lambda"]), float(best["mu"]), best, grid

    # Default: maximize compliance, then accuracy, then lower cost.
    best = max(grid, key=lambda s: (float(s["slo_compliance"]), float(s["accuracy"]), -float(s["avg_cost"])))
    best = dict(best)
    best["selection"] = {
        "rule": "max_compliance_then_accuracy",
        "target_slo_compliance": 0.0,
        "total": len(grid),
    }
    return float(best["lambda"]), float(best["mu"]), best, grid


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", dest="model_name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--prompt_mode", default="slo", choices=["slo", "accuracy"])
    ap.add_argument("--output_root", default="router_models")
    ap.add_argument("--slo_thresholds_path", default=None)
    ap.add_argument(
        "--slo_profile",
        default=None,
        help="Optional key for SLO profile (e.g., p95). If omitted, uses file's 'primary'.",
    )
    ap.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max_batch_size", type=int, default=8)
    ap.add_argument("--batch_wait_ms", type=int, default=8)
    ap.add_argument("--load_strategy", type=str, default="auto")
    ap.add_argument("--max_loaded_variants", type=int, default=None)
    ap.add_argument("--preload_variants", nargs="*", default=None)
    ap.add_argument("--warmup", action="store_true")

    ap.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Debug: limit Train+Val examples (0 = full).",
    )
    ap.add_argument(
        "--reuse_traces",
        action="store_true",
        help="If set and cached trace JSONL exists under output_root/, reuse it.",
    )
    ap.add_argument(
        "--collect_only",
        action="store_true",
        help="If set, only collect (or reuse) traces and exit before training predictors.",
    )
    ap.add_argument(
        "--time_budget_hours",
        type=float,
        default=0.0,
        help=(
            "Optional wall-clock budget for trace collection. "
            "If > 0, the script will stop collecting when the budget is reached, "
            "save partial progress to trainval_traces.jsonl, and exit. "
            "Re-run the same command to resume."
        ),
    )
    ap.add_argument(
        "--eval_on_test",
        action="store_true",
        help="If set, run a quick held-out accuracy eval on TEST for each learned mode (no load/concurrency sweep).",
    )

    ap.add_argument(
        "--target_slo_compliance",
        type=float,
        default=0.0,
        help=(
            "Compliance-first tuning target in [0,1]. If > 0, tuning selects (lambda,mu) "
            "by: (1) keep configs with observed VAL compliance >= target, (2) maximize VAL accuracy, "
            "(3) tie-break by lower avg cost. If no configs meet the target, falls back to maximizing compliance."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Tokenizer for prompt length features (CPU-side)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    processed = Path(args.processed_dir)
    train_path = processed / "train_data.jsonl"
    val_path = processed / "val_data.jsonl"
    test_path = processed / "test_data.jsonl"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Missing processed data. Expected at least: {train_path} and {val_path}. "
            "Run preprocessing first (see run_baseline_evaluation.py --preprocess)."
        )

    train = _read_jsonl(str(train_path))
    val = _read_jsonl(str(val_path))

    # Tag examples so we can tune weights on VAL only.
    for i, ex in enumerate(train):
        ex["_global_index"] = int(i)
        ex["_split"] = "train"
    for j, ex in enumerate(val):
        ex["_global_index"] = int(len(train) + j)
        ex["_split"] = "val"

    trainval = train + val
    if args.max_examples and int(args.max_examples) > 0:
        trainval = trainval[: int(args.max_examples)]

    concs = [int(c) for c in args.concurrencies]
    if not concs:
        raise ValueError("--concurrencies must be non-empty")

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    trace_path = out_root / "trainval_traces.jsonl"

    slo_dict = _load_slo_dict(args.slo_thresholds_path, profile_key=args.slo_profile)

    variants = ["cheap", "med", "base"]

    # -----------------------------
    # 1) Collect traces (train+val)
    # -----------------------------
    records: List[Dict[str, Any]] = []

    # Always load any existing traces so we can resume safely.
    if trace_path.exists():
        records = _read_jsonl(str(trace_path))
        print(f"[resume] Found existing traces: {trace_path} (rows={len(records)})")

    if args.reuse_traces and trace_path.exists():
        print("[reuse] Skipping trace collection (will train from cached traces).")
    else:
        split_map = _build_split_map_stable(trainval=trainval, concurrencies=concs, seed=int(args.seed), out_root=out_root)

        done_keys = {(int(r.get("global_example_idx", -1)), str(r.get("variant"))) for r in records}
        expected = len(trainval) * len(variants)
        print(f"[resume] already_done={len(done_keys)}/{expected} (example-variant pairs)")

        time_budget_s = float(args.time_budget_hours or 0.0) * 3600.0
        start_time = time.time()

        service = MultiVariantService(
            model_name=args.model_name,
            variants=variants,
            device=args.device,
            dtype=args.dtype,
            router_mode="always_base",  # ignored during collection (we force variants)
            max_retries=0,  # IMPORTANT: no escalation when collecting per-variant ground truth
            enable_batching=True,
            max_batch_size=int(args.max_batch_size),
            batch_wait_ms=int(args.batch_wait_ms),
            load_strategy=str(args.load_strategy),
            max_loaded_variants=args.max_loaded_variants,
            preload_variants=args.preload_variants,
            warmup=bool(args.warmup),
        )

        try:
            for conc, exs in split_map.items():
                if not exs:
                    continue
                if time_budget_s and (time.time() - start_time) >= time_budget_s:
                    print("[budget] Time budget reached before next concurrency bucket; stopping collection.")
                    break
                print(f"\n[collect] concurrency={conc} examples={len(exs)}")
                recs = _collect_traces_for_concurrency(
                    service=service,
                    examples=exs,
                    concurrency=int(conc),
                    variants=variants,
                    prompt_mode=str(args.prompt_mode),
                    seed=int(args.seed),
                    tokenizer=tokenizer,
                    skip_keys=done_keys,
                    time_budget_s=max(0.0, time_budget_s - (time.time() - start_time)) if time_budget_s else 0.0,
                    append_path=str(trace_path),
                )
                if recs:
                    records.extend(recs)
                    done_keys.update({(int(r.get("global_example_idx", -1)), str(r.get("variant"))) for r in recs})
        finally:
            try:
                service.cleanup()
            except Exception:
                pass

        # We appended during collection; re-load to ensure file is consistent.
        records = _read_jsonl(str(trace_path))
        print(f"\n[saved] traces -> {trace_path} (rows={len(records)})")

        # If we were running under a wall-clock budget, exit so the user can resume later.
        if time_budget_s and (time.time() - start_time) >= time_budget_s:
            print(
                f"[budget] Collection stopped at wall-clock budget (~{args.time_budget_hours}h). "
                f"Re-run the same command to resume. Current rows={len(records)}"
            )
            return

    # Sanity: expect ~3 rows per example (one per variant).
    grouped_all = _group_by_example(records, variants)
    print(f"[sanity] complete triplets={len(grouped_all)} examples")
    if len(grouped_all) == 0:
        raise RuntimeError("No complete (cheap,med,base) triplets found in collected traces.")

    # Useful for long/expensive trace collection runs: allow exiting right after saving traces.
    if args.collect_only:
        print(f"[collect-only] Done. Traces at: {trace_path}")
        return

    # -----------------------------
    # 2) Train predictors on Train+Val
    # -----------------------------
    quality_models, ttft_models, tpot_models = _train_predictors(records, variants)
    router = LearnedRouter(quality_models, ttft_models, tpot_models, lambda_slo=1.5, mu_quality=1.5)

    # Lightweight diagnostics (helps catch "fishy" training runs early).
    trainval_metrics = _evaluate_predictors(
        records=records,
        variants=variants,
        quality_models=quality_models,
        ttft_models=ttft_models,
        tpot_models=tpot_models,
        split_name="train+val",
    )

    # -----------------------------
    # 3) Tune weights on VAL only
    # -----------------------------
    val_records = [r for r in records if str(r.get("split")) == "val"]
    val_grouped = _group_by_example(val_records, variants)
    print(f"[tune] validation examples={len(val_grouped)}")
    if len(val_grouped) == 0:
        raise RuntimeError("No validation examples found; cannot tune lambda/mu.")

    val_metrics = _evaluate_predictors(
        records=val_records,
        variants=variants,
        quality_models=quality_models,
        ttft_models=ttft_models,
        tpot_models=tpot_models,
        split_name="val",
    )

    def _print_metrics(m: Dict[str, Any]) -> None:
        print(f"\n[predictor-metrics] split={m.get('split')}")
        for vv in variants:
            r = (m.get('by_variant') or {}).get(vv) or {}
            if int(r.get('n', 0)) == 0:
                print(f"  {vv:5s}: n=0")
                continue
            auc = r.get('quality_auc', None)
            auc_s = f"{auc:.3f}" if isinstance(auc, (int, float)) else "NA"
            print(
                f"  {vv:5s}: n={int(r['n'])} pos={r['quality_pos_rate']:.3f} "
                f"q_acc@0.5={r['quality_acc@0.5']:.3f} auc={auc_s} "
                f"ttft_mae={r['ttft_mae']:.1f} ttft_rmse={r['ttft_rmse']:.1f} "
                f"tpot_mae={r['tpot_mae']:.2f} tpot_rmse={r['tpot_rmse']:.2f}"
            )

    _print_metrics(trainval_metrics)
    _print_metrics(val_metrics)

    lam_ttft, mu_ttft, stats_ttft, grid_ttft = _tune_lambda_mu(
        "ttft",
        router,
        val_grouped,
        slo_dict,
        variants,
        target_slo_compliance=float(args.target_slo_compliance or 0.0),
    )
    lam_total, mu_total, stats_total, grid_total = _tune_lambda_mu(
        "total",
        router,
        val_grouped,
        slo_dict,
        variants,
        target_slo_compliance=float(args.target_slo_compliance or 0.0),
    )

    # -----------------------------
    # 4) Save artifacts
    # -----------------------------
    meta_common = {
        "model": args.model_name,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_mode": args.prompt_mode,
        "concurrencies": concs,
        "seed": int(args.seed),
        "slo_thresholds_path": args.slo_thresholds_path,
        "slo_profile": args.slo_profile,
    }

    # Persist predictor diagnostics for debugging and paper appendices.
    _write_json(
        str(out_root / "predictor_metrics.json"),
        {"trainval": trainval_metrics, "val": val_metrics, **meta_common},
    )

    out_ttft = out_root / "learned_ttft"
    out_total = out_root / "learned_total"
    out_ttft.mkdir(parents=True, exist_ok=True)
    out_total.mkdir(parents=True, exist_ok=True)

    router.lambda_slo = float(lam_ttft)
    router.mu_quality = float(mu_ttft)
    router.save(str(out_ttft), extra_metadata={"mode": "ttft", "tuned": stats_ttft, **meta_common})
    _write_json(str(out_ttft / "tuning.json"), {"mode": "ttft", "tuned": stats_ttft, **meta_common})
    _write_json(str(out_ttft / "tuning_grid.json"), {"mode": "ttft", "grid": grid_ttft, **meta_common})

    router.lambda_slo = float(lam_total)
    router.mu_quality = float(mu_total)
    router.save(str(out_total), extra_metadata={"mode": "total", "tuned": stats_total, **meta_common})
    _write_json(str(out_total / "tuning.json"), {"mode": "total", "tuned": stats_total, **meta_common})
    _write_json(str(out_total / "tuning_grid.json"), {"mode": "total", "grid": grid_total, **meta_common})

    _write_json(str(out_root / "train_summary.json"), {"num_trace_rows": len(records), "num_examples": len(grouped_all)})

    print("\n[saved routers]")
    print(f"  Learned-TTFT            -> {out_ttft} (lambda={lam_ttft}, mu={mu_ttft})")
    print(f"  Learned-Total (Derived) -> {out_total} (lambda={lam_total}, mu={mu_total})")

    # -----------------------------
    # 5) Optional: quick held-out accuracy eval on TEST
    # -----------------------------
    if args.eval_on_test:
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test split: {test_path}")
        test = _read_jsonl(str(test_path))
        from evaluation import HeldOutEvaluator

        for mode in ["learned_ttft", "learned_total"]:
            service = MultiVariantService(
                model_name=args.model_name,
                variants=variants,
                device=args.device,
                dtype=args.dtype,
                router_mode=mode,
                learned_router_dir=str(out_root),
                max_retries=1,
                enable_batching=True,
                max_batch_size=int(args.max_batch_size),
                batch_wait_ms=int(args.batch_wait_ms),
                load_strategy=str(args.load_strategy),
                max_loaded_variants=args.max_loaded_variants,
                preload_variants=args.preload_variants,
                warmup=bool(args.warmup),
            )
            try:
                service.set_slo_dict(slo_dict)
                ev = HeldOutEvaluator(service, test, batch_size=1)
                summary, _detailed = ev.evaluate(prompt_mode=str(args.prompt_mode), verbose=False)
                out = out_root / f"test_eval_{mode}.json"
                _write_json(str(out), summary)
                print(f"[test-eval] {mode} -> {out}")
            finally:
                try:
                    service.cleanup()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
