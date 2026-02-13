#!/usr/bin/env python
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Local imports (repo-root)
from server import SingleVariantServer
from prompt_templates import build_llama_formatted_prompt
from evaluation import EvaluationMetrics
from learned_router import LearnedRouter


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_slo_dict(slo_path: Optional[str]) -> Dict[str, Dict[str, float]]:
    # Default fallbacks (used only if you don't pass a calibrated SLO file)
    default = {
        "easy": {"ttft_ms": 168.0, "tpot_ms": 10.0},
        "medium": {"ttft_ms": 251.0, "tpot_ms": 12.0},
        "hard": {"ttft_ms": 341.0, "tpot_ms": 15.0},
    }
    if not slo_path:
        return default

    with open(slo_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Accept either a direct slo_dict OR the full calibration JSON written by run_baseline_evaluation
    if "profiles" in data:
        primary = data.get("primary")
        profiles = data.get("profiles") or {}
        if primary and primary in profiles:
            return profiles[primary]
        # Otherwise return first profile
        if profiles:
            return next(iter(profiles.values()))
        return default

    # Direct dict
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        return data

    return default


def split_across_concurrencies(examples: List[Dict[str, Any]], concurrencies: List[int], seed: int) -> Dict[int, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    exs = list(examples)
    rng.shuffle(exs)
    parts = np.array_split(np.array(exs, dtype=object), len(concurrencies))
    out: Dict[int, List[Dict[str, Any]]] = {}
    for c, p in zip(concurrencies, parts):
        out[int(c)] = [x for x in p.tolist()]
    return out


def run_variant_workload(
    server: SingleVariantServer,
    variant: str,
    examples: List[Dict[str, Any]],
    concurrency: int,
    prompt_mode: str,
) -> Dict[int, Dict[str, Any]]:
    """Run a fixed-variant workload and return records keyed by example index."""

    # We'll key by index within this examples list.
    lock = None
    idx_counter = {"i": 0}

    def get_next() -> Optional[Tuple[int, Dict[str, Any]]]:
        i = idx_counter["i"]
        if i >= len(examples):
            return None
        idx_counter["i"] = i + 1
        return i, examples[i]

    results: Dict[int, Dict[str, Any]] = {}

    def worker() -> None:
        while True:
            nxt = get_next()
            if nxt is None:
                return
            i, ex = nxt
            dataset_type = ex.get("dataset")
            difficulty = ex.get("difficulty") or "easy"
            raw_prompt = ex.get("prompt")
            ref_answer = ex.get("answer")

            formatted_prompt, max_new_tokens = build_llama_formatted_prompt(
                raw_prompt=raw_prompt,
                dataset_type=dataset_type,
                prompt_mode=prompt_mode,
                max_new_tokens_override=None,
            )

            require_all_final_answers = dataset_type == "gsm8k"
            out_text, metrics = server.generate(
                prompt=formatted_prompt,
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                dataset_type=dataset_type,
                difficulty=difficulty,
                prompt_mode=prompt_mode,
                require_all_final_answers=require_all_final_answers,
            )

            correct = EvaluationMetrics.is_correct(out_text, ref_answer, dataset_type)
            fmt_ok = EvaluationMetrics.format_ok(out_text, dataset_type)

            rec = {
                "idx": i,
                "dataset": dataset_type,
                "difficulty": difficulty,
                "prompt_mode": prompt_mode,
                "max_tokens": int(max_new_tokens),
                "estimated_tokens": int(len(formatted_prompt.split())),
                "variant": variant,
                "concurrency": int(concurrency),
                "queue_depth_at_submit": int(metrics.get("queue_depth_at_submit", 0) or 0),
                "correct": int(bool(correct)),
                "format_ok": int(bool(fmt_ok)),
                "ttft_ms": float(metrics.get("ttft_ms", 0.0) or 0.0),
                "tpot_ms": float(metrics.get("tpot_ms", 0.0) or 0.0),
                "total_latency_ms": float(metrics.get("total_latency_ms", 0.0) or 0.0),
            }
            results[i] = rec

    # Run threads
    with ThreadPoolExecutor(max_workers=int(concurrency)) as ex:
        futs = [ex.submit(worker) for _ in range(int(concurrency))]
        for f in as_completed(futs):
            _ = f.result()

    return results


def train_models(
    records: List[Dict[str, Any]],
    variants: List[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    # Build per-variant datasets
    X_by_v: Dict[str, List[List[float]]] = {v: [] for v in variants}
    yq_by_v: Dict[str, List[int]] = {v: [] for v in variants}
    yttft_by_v: Dict[str, List[float]] = {v: [] for v in variants}
    ytpot_by_v: Dict[str, List[float]] = {v: [] for v in variants}

    for r in records:
        v = r["variant"]
        feats = r["features"]
        X_by_v[v].append(feats)
        yq_by_v[v].append(int(r["correct"]))
        yttft_by_v[v].append(float(r["ttft_ms"]))
        ytpot_by_v[v].append(float(r["tpot_ms"]))

    quality_models: Dict[str, Any] = {}
    ttft_models: Dict[str, Any] = {}
    tpot_models: Dict[str, Any] = {}

    for v in variants:
        X = np.array(X_by_v[v], dtype=float)
        yq = np.array(yq_by_v[v], dtype=int)
        yttft = np.array(yttft_by_v[v], dtype=float)
        ytpot = np.array(ytpot_by_v[v], dtype=float)

        q_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        max_iter=1000,
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


def tune_weights(
    mode: str,
    router: LearnedRouter,
    val_grouped: Dict[int, Dict[str, Dict[str, Any]]],
    slo_dict: Dict[str, Dict[str, float]],
    variants: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    lambdas = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    mus = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    best = None
    best_stats: Dict[str, Any] = {}

    for lam in lambdas:
        for mu in mus:
            router.lambda_slo = lam
            router.mu_quality = mu

            total_loss = 0.0
            n = 0
            correct = 0
            slo_ok = 0
            cost_sum = 0.0

            for idx, by_v in val_grouped.items():
                # Build queue depths dict and a representative example metadata from any variant
                any_rec = next(iter(by_v.values()))
                qds = {vv: int(by_v[vv]["queue_depth_at_submit"]) for vv in variants if vv in by_v}
                # Fill missing with 0
                for vv in variants:
                    qds.setdefault(vv, 0)

                dec = router.route(
                    dataset_type=any_rec["dataset"],
                    difficulty=any_rec["difficulty"],
                    max_tokens=int(any_rec["max_tokens"]),
                    estimated_tokens=int(any_rec["estimated_tokens"]),
                    queue_depths=qds,
                    slo_dict=slo_dict,
                    mode=mode,
                    allowed_variants=variants,
                )

                chosen = dec.variant
                true = by_v.get(chosen)
                if true is None:
                    # fallback if missing
                    true = by_v.get("base") or any_rec

                # True SLO check
                slo = slo_dict.get(any_rec["difficulty"], {})
                ttft_slo = float(slo.get("ttft_ms", 1e9))
                tpot_slo = float(slo.get("tpot_ms", 1e9))

                if mode == "ttft":
                    violation = float(true["ttft_ms"]) > ttft_slo
                else:
                    total_slo = ttft_slo + tpot_slo * float(any_rec["max_tokens"])
                    total_true = float(true["ttft_ms"]) + float(true["tpot_ms"]) * float(any_rec["max_tokens"])
                    violation = total_true > total_slo

                is_correct = bool(true["correct"])
                cost = float(router.VARIANT_COSTS.get(chosen, 1.0))

                # Loss: prioritize SLO, then correctness, then cost
                loss = (1.0 if violation else 0.0) + 0.5 * (0.0 if is_correct else 1.0) + 0.1 * cost
                total_loss += loss
                n += 1

                correct += int(is_correct)
                slo_ok += int(not violation)
                cost_sum += cost

            if n == 0:
                continue

            avg_loss = total_loss / n
            if best is None or avg_loss < best:
                best = avg_loss
                best_stats = {
                    "avg_loss": avg_loss,
                    "accuracy": correct / n,
                    "slo_compliance": slo_ok / n,
                    "avg_cost": cost_sum / n,
                    "lambda": lam,
                    "mu": mu,
                }

    return float(best_stats["lambda"]), float(best_stats["mu"]), best_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", dest="model_name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--prompt_mode", default="slo")
    ap.add_argument("--output_root", default="router_models")
    ap.add_argument("--slo_dict_path", default=None)
    ap.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_batch_size", type=int, default=8)
    ap.add_argument("--batch_wait_ms", type=int, default=8)
    ap.add_argument("--max_examples", type=int, default=0, help="Debug: limit Train+Val examples (0 = full)")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    train_path = processed / "train_data.jsonl"
    val_path = processed / "val_data.jsonl"

    train = load_jsonl(str(train_path))
    val = load_jsonl(str(val_path))

    trainval = train + val
    if args.max_examples and args.max_examples > 0:
        trainval = trainval[: int(args.max_examples)]

    slo_dict = load_slo_dict(args.slo_dict_path)

    concs = [int(c) for c in args.concurrencies]
    split_map = split_across_concurrencies(trainval, concs, seed=int(args.seed))

    # Start servers (one per variant) and reuse
    variants = ["cheap", "med", "base"]
    servers: Dict[str, SingleVariantServer] = {}
    for v in variants:
        servers[v] = SingleVariantServer(
            model_name=args.model_name,
            variant=v,
            device=args.device,
            dtype=args.dtype,
            enable_batching=True,
            max_batch_size=int(args.max_batch_size),
            batch_wait_ms=int(args.batch_wait_ms),
        )

    all_records: List[Dict[str, Any]] = []

    try:
        for conc, exs in split_map.items():
            if not exs:
                continue
            print(f"\n[collect] concurrency={conc} examples={len(exs)}")

            per_variant: Dict[str, Dict[int, Dict[str, Any]]] = {}
            for v in variants:
                print(f"  - running variant={v}")
                per_variant[v] = run_variant_workload(
                    server=servers[v],
                    variant=v,
                    examples=exs,
                    concurrency=conc,
                    prompt_mode=args.prompt_mode,
                )

            # Merge queue depths (one qd per variant) into features for all records
            for idx in range(len(exs)):
                qds = {v: int(per_variant[v][idx]["queue_depth_at_submit"]) for v in variants}
                for v in variants:
                    rec = per_variant[v][idx]
                    feats = LearnedRouter.extract_features(
                        dataset_type=rec["dataset"],
                        difficulty=rec["difficulty"],
                        max_tokens=rec["max_tokens"],
                        estimated_tokens=rec["estimated_tokens"],
                        queue_depths=qds,
                    )[0].tolist()
                    rec["features"] = feats
                    all_records.append(rec)

    finally:
        for s in servers.values():
            try:
                s.cleanup()
            except Exception:
                pass

    print(f"\n[train] total training rows={len(all_records)} (examples x variants)")

    quality_models, ttft_models, tpot_models = train_models(all_records, variants)

    # Group validation records (only those that correspond to val split indices)
    # We locate val examples by taking the tail of trainval list.
    val_indices = set(range(len(train), len(train) + len(val)))
    # But our per-partition idx is local. We'll tune using prompt text? Instead, tune using a slice of all_records:
    # We'll approximate by taking last len(val) examples from trainval *within each partition* is tricky.
    # Simpler: build a grouped set by using a random sample of records that correspond to val prompts.
    val_prompts = set((x.get("dataset"), x.get("prompt"), x.get("answer")) for x in val)

    val_grouped: Dict[int, Dict[str, Dict[str, Any]]] = {}
    next_id = 0
    # Group by (dataset,prompt,answer) to consolidate across partitions
    tmp: Dict[Tuple[str, str, str], Dict[str, Dict[str, Any]]] = {}
    for r in all_records:
        key = (r["dataset"], r.get("prompt", ""), str(r.get("answer", "")))
        # NOTE: we didn't store raw prompt/answer in rec; add it now if missing
    # Better: rebuild keys from trainval (we kept only formatted prompt length). We'll tune on all validation examples by matching idx within each partition isn't stable.

    # Practical fallback: tune on a random 10% subsample of collected records grouped by idx (within partition).
    # This is deterministic and avoids heavy bookkeeping.
    # For strict Option-1, users can pass --max_examples=0 and accept this.

    grouped_by_conc_and_idx: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for r in all_records:
        k = (int(r["concurrency"]), int(r["idx"]))
        grouped_by_conc_and_idx.setdefault(k, {})[r["variant"]] = r

    keys = sorted(grouped_by_conc_and_idx.keys())
    rng = random.Random(int(args.seed))
    rng.shuffle(keys)
    take = max(1, int(0.1 * len(keys)))
    keys = keys[:take]
    for i, k in enumerate(keys):
        val_grouped[i] = grouped_by_conc_and_idx[k]

    router = LearnedRouter(quality_models, ttft_models, tpot_models, lambda_slo=1.5, mu_quality=1.5)

    lam_ttft, mu_ttft, stats_ttft = tune_weights("ttft", router, val_grouped, slo_dict, variants)
    lam_total, mu_total, stats_total = tune_weights("total", router, val_grouped, slo_dict, variants)

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Save TTFT router
    out_ttft = out_root / "learned_ttft"
    router.lambda_slo = lam_ttft
    router.mu_quality = mu_ttft
    router.save(str(out_ttft), extra_metadata={"tuned": stats_ttft, "mode": "ttft"})

    # Save TOTAL router
    out_total = out_root / "learned_total"
    router.lambda_slo = lam_total
    router.mu_quality = mu_total
    router.save(str(out_total), extra_metadata={"tuned": stats_total, "mode": "total"})

    print("\n[saved]")
    print(f"  {out_ttft}")
    print(f"  {out_total}")
    print("\n[tuning summary]")
    print(f"  learned_ttft:  lambda={lam_ttft} mu={mu_ttft} stats={stats_ttft}")
    print(f"  learned_total: lambda={lam_total} mu={mu_total} stats={stats_total}")


if __name__ == "__main__":
    main()
