#!/usr/bin/env python
"""run_baseline_evaluation.py

Single entry point used for:
  1) Accuracy smoke (correctness + formatting)
  2) SLO-mode accuracy guardrail (SLO prompt should still answer correctly)
  3) Serving/load smoke (TTFT/TPOT/queue + dynamic SLO calibration)

Paper-facing reliability features in this patch bundle:
- TTFT definition for calibration is Option A (queue-inclusive), implemented in server.py:

    TTFT_A = scheduler_wait_ms + ttft_infer_ms

  where:
    scheduler_wait_ms : time from enqueue -> dequeue (micro-batching scheduler)
    ttft_infer_ms     : tokenize_ms + lock_wait_ms + model_prefill+first_decode

  (For non-batched calls, scheduler_wait_ms=0 and ttft_ms == ttft_infer_ms.)

- SLO thresholds are calibrated from concurrency=1 and saved as percentile profiles:
    p90 / p95 / p99 (for paper sensitivity analysis).

Outputs written to --output_dir:
  config.json
  accuracy_results.json
  accuracy_detailed.jsonl
  slo_thresholds.json            (if load tests + calibration enabled)
  metrics_concurrency_<N>.json
  requests_concurrency_<N>.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from evaluation import HeldOutEvaluator
from load_generator import ClosedLoopLoadGenerator
from metrics import MetricsCalculator, calibrate_slo_profiles
from preprocessing import DataPreprocessor
from server import SingleVariantServer, MultiVariantService

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _write_json(path: str, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_slo_profiles(path: str) -> Optional[Dict[str, Dict]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)

    # Preferred format
    if isinstance(data, dict) and "profiles" in data and isinstance(data["profiles"], dict):
        return data["profiles"]

    # Backward compatible: top-level percentile keys
    if isinstance(data, dict) and any(k.startswith("p") for k in data.keys()):
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    # Backward compatible: single slo dict
    if isinstance(data, dict) and any(k in data for k in ("easy", "medium", "hard")):
        return {"p95": data}

    return None


def _effective_enable_batching(prompt_mode: str, enable_batching_flag: Optional[bool]) -> bool:
    if enable_batching_flag is None:
        return prompt_mode == "slo"
    return bool(enable_batching_flag)


def _parse_concurrency_schedule(spec: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """Parse a nonstationary concurrency schedule.

    Format: "<conc>:<num_requests>,<conc>:<num_requests>,..." (commas or semicolons).
    Returns a list of (concurrency, num_requests) phases.
    """

    if not spec:
        return None
    spec = str(spec).strip()
    if not spec:
        return None

    phases: List[Tuple[int, int]] = []
    parts = [p.strip() for p in spec.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        a, b = p.split(":", 1)
        try:
            conc = int(a.strip())
            nreq = int(b.strip())
        except Exception:
            continue
        if conc <= 0 or nreq <= 0:
            continue
        phases.append((conc, nreq))
    return phases or None


def _parse_data_schedule(spec: Optional[str]) -> Optional[List[Tuple[str, str, int]]]:
    """Parse a nonstationary *data* schedule (domain/length shift).

    Format: "<selector>:<num_requests>,<selector>:<num_requests>,..." where
    selector is "dataset=<name>" or "length=<short|long>".

    Examples:
      --data_schedule "dataset=gsm8k:200,dataset=mmlu:200"
      --data_schedule "length=short:200,length=long:200"

    Returns a list of (kind, value, num_requests) phases.
    """

    if not spec:
        return None
    spec = str(spec).strip()
    if not spec:
        return None

    phases: List[Tuple[str, str, int]] = []
    parts = [p.strip() for p in spec.replace(";", ",").split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        sel, nreq_s = p.split(":", 1)
        sel = sel.strip()
        nreq_s = nreq_s.strip()
        if "=" not in sel:
            continue
        kind, value = sel.split("=", 1)
        kind = kind.strip().lower()
        value = value.strip().lower()
        try:
            nreq = int(nreq_s)
        except Exception:
            continue
        if nreq <= 0:
            continue
        if kind not in {"dataset", "length"}:
            continue
        phases.append((kind, value, nreq))
    return phases or None


def _length_buckets(examples: List[Dict[str, Any]], seed: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """Split examples into (short, long) buckets by median input_length.

    If input_length is missing, we approximate it by whitespace token count.
    Returns (short, long, threshold).
    """

    if not examples:
        return [], [], 0

    lens: List[int] = []
    for ex in examples:
        try:
            l = int(ex.get("input_length", 0) or 0)
        except Exception:
            l = 0
        if l <= 0:
            # fallback: whitespace tokens (best-effort)
            try:
                l = int(len(str(ex.get("prompt", "") or "").split()))
            except Exception:
                l = 0
        lens.append(max(0, l))

    if not lens:
        return list(examples), [], 0

    # Median threshold
    thr = int(sorted(lens)[len(lens) // 2])
    short: List[Dict[str, Any]] = []
    long: List[Dict[str, Any]] = []
    for ex, l in zip(examples, lens):
        if int(l) <= int(thr):
            short.append(ex)
        else:
            long.append(ex)

    # If degenerate, fall back to deterministic split.
    if not short or not long:
        rng = random.Random(int(seed))
        exs = list(examples)
        rng.shuffle(exs)
        mid = len(exs) // 2
        short = exs[:mid]
        long = exs[mid:]
        thr = int(thr) if thr > 0 else int(mid)

    return short, long, int(thr)




def _build_stratified_pool(
    examples: List[Dict[str, Any]],
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return a deterministic stratified subset of size k.

    Stratification key: (dataset, difficulty) with an explicit 6-bucket target:
      (gsm8k|mmlu) × (easy|medium|hard)

    If a bucket is sparse, we take as many as available and redistribute the
    remainder to other buckets deterministically (seeded).
    """
    if k <= 0 or (not examples):
        return list(examples)

    if k >= len(examples):
        return list(examples)

    # Canonical buckets (explicit 6)
    canonical_keys: List[Tuple[str, str]] = [
        ("gsm8k", "easy"),
        ("gsm8k", "medium"),
        ("gsm8k", "hard"),
        ("mmlu", "easy"),
        ("mmlu", "medium"),
        ("mmlu", "hard"),
    ]

    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {k: [] for k in canonical_keys}
    extras: List[Dict[str, Any]] = []

    for ex in examples:
        ds = (ex.get("dataset") or "unknown").lower().strip()
        diff = (ex.get("difficulty") or "medium").lower().strip()
        key = (ds, diff)
        if key in buckets:
            buckets[key].append(ex)
        else:
            extras.append(ex)

    rng = random.Random(int(seed))
    for key in canonical_keys:
        rng.shuffle(buckets[key])
    rng.shuffle(extras)

    # Start with equal targets across 6 buckets
    base = k // len(canonical_keys)
    rem = k % len(canonical_keys)
    targets = {key: base for key in canonical_keys}
    for i in range(rem):
        targets[canonical_keys[i]] += 1

    selected: List[Dict[str, Any]] = []
    remaining_capacity: List[Tuple[str, str]] = []

    # First pass: take up to target from each bucket
    for key in canonical_keys:
        take = min(targets[key], len(buckets[key]))
        if take:
            selected.extend(buckets[key][:take])
            buckets[key] = buckets[key][take:]
        # Track buckets that still have capacity for top-ups
        if buckets[key]:
            remaining_capacity.append(key)

    # Compute remaining needed
    need = k - len(selected)
    if need > 0:
        # Prefer to top up from canonical buckets that still have items, in a round-robin seeded order
        rng.shuffle(remaining_capacity)
        while need > 0 and remaining_capacity:
            progressed = False
            for key in list(remaining_capacity):
                if need <= 0:
                    break
                if buckets[key]:
                    selected.append(buckets[key].pop(0))
                    need -= 1
                    progressed = True
                if not buckets[key]:
                    remaining_capacity.remove(key)
            if not progressed:
                break

    # If still short (all canonical buckets exhausted), top up from extras
    if need > 0 and extras:
        selected.extend(extras[:need])
        need = k - len(selected)

    # Final deterministic shuffle
    rng.shuffle(selected)
    return selected[:k]


def _log_strata_counts(tag: str, examples: List[Dict[str, Any]]) -> None:
    counts: Dict[str, int] = {}
    for ex in examples:
        ds = (ex.get("dataset") or "unknown").lower().strip()
        diff = (ex.get("difficulty") or "medium").lower().strip()
        key = f"{ds}:{diff}"
        counts[key] = counts.get(key, 0) + 1
    parts = ", ".join([f"{k}={v}" for k, v in sorted(counts.items())])
    logger.info(f"{tag} strata counts: {parts}")


def _log_env() -> None:
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--variant", type=str, default="med", choices=["base", "med", "cheap"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16"])

    # Optional per-variant overrides.
    # A common use-case is making CHEAP a smaller fp16 model (e.g., Llama-3B) rather than
    # a 4-bit quantized version of the same base model.
    p.add_argument(
        "--cheap_model",
        type=str,
        default=None,
        help="Optional override HF model id for variant=cheap (e.g., meta-llama/Llama-3.2-3B-Instruct).",
    )
    p.add_argument(
        "--cheap_quantization",
        type=str,
        default=None,
        choices=["fp16", "bf16", "int8", "int4", "none"],
        help="Optional override quantization for variant=cheap. If --cheap_model is set and this is omitted, defaults to fp16.",
    )


    # Serving mode
    p.add_argument(
        "--service",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Inference service type: single-variant or multi-variant (task-adaptive routing).",
    )

    # Router knobs (used when --service multi)
    p.add_argument(
        "--router_mode",
        type=str,
        default="difficulty",
        choices=[
            "difficulty",
            "slo_aware",
            "fixed",
            "always_cheap",
            "always_base",
            "learned_ttft",
            "learned_total",
            "risk",
            "bandit",
        ],
        help="Routing policy for multi-variant serving.",
    )

    # Reproducibility / cost model knobs
    p.add_argument(
        "--router_seed",
        type=int,
        default=0,
        help="Deterministic seed for routing-related randomness (bandit label subsampling, etc.).",
    )
    p.add_argument(
        "--overhead_ms_to_cost_units",
        type=float,
        default=0.1,
        help="Convert overhead milliseconds into token-equivalent cost units (applied to adapter/swap overheads).",
    )
    p.add_argument(
        "--router_fixed_variant",
        type=str,
        default=None,
        choices=["cheap", "med", "base"],
        help="If --router_mode fixed, always route to this variant.",
    )
    p.add_argument(
        "--learned_router_dir",
        type=str,
        default=None,
        help="Path to learned-router artifacts. For learned_* modes, this is REQUIRED. You may pass either a root folder containing subfolders learned_ttft/learned_total, or a mode-specific folder.",
    )

    # Risk router artifacts + knobs
    p.add_argument(
        "--risk_router_dir",
        type=str,
        default=None,
        help="Path to risk-router bundle artifacts (trained with scripts/train_risk_router.py). Required for --router_mode risk or bandit.",
    )
    p.add_argument(
        "--risk_latency_delta",
        type=float,
        default=0.05,
        help="Latency risk level δ for conformal upper bounds (violation rate target).",
    )
    p.add_argument(
        "--risk_quality_epsilon",
        type=float,
        default=0.25,
        help="Quality risk target ε (upper bound on error among accepted predictions, per variant).",
    )
    p.add_argument(
        "--risk_quality_alpha",
        type=float,
        default=0.05,
        help="Confidence level α for the quality risk bound (Clopper-Pearson).",
    )

    # ------------------------------------------------------------------
    # Bandit router knobs (SLO-safe contextual bandit)
    # ------------------------------------------------------------------
    p.add_argument("--bandit_delta", type=float, default=0.05, help="Risk budget δ (target violation rate) for the bandit.")
    p.add_argument("--bandit_alpha", type=float, default=1.0, help="Quality weight α in the bandit objective.")
    p.add_argument("--bandit_beta_r", type=float, default=2.0, help="Risk bound multiplier β_r (conservative screen).")
    p.add_argument("--bandit_beta_q", type=float, default=2.0, help="Quality bound multiplier β_q (conservative screen).")
    p.add_argument("--bandit_eps_r", type=float, default=0.0, help="Risk tolerance ε_r vs baseline (conservative screen).")
    p.add_argument("--bandit_eps_q", type=float, default=0.0, help="Quality tolerance ε_q vs baseline (conservative screen).")
    p.add_argument("--bandit_beta_u", type=float, default=0.2, help="Exploration bonus multiplier β_u.")
    p.add_argument(
        "--bandit_label_budget_p",
        type=float,
        default=1.0,
        help="Fraction of requests with observed quality labels (simulated delayed labels).",
    )
    p.add_argument(
        "--bandit_checkpoint_path",
        type=str,
        default=None,
        help="If set, periodically checkpoint bandit state to this path prefix.",
    )
    p.add_argument(
        "--bandit_checkpoint_every",
        type=int,
        default=500,
        help="Checkpoint every N bandit updates.",
    )
    p.add_argument(
        "--bandit_state_path",
        type=str,
        default=None,
        help="Load an existing bandit state (path prefix used by BanditRouter.save).",
    )
    p.add_argument(
        "--bandit_learn_requests",
        type=int,
        default=0,
        help="Number of labeled requests to run for the bandit learning phase before evaluation.",
    )
    p.add_argument(
        "--bandit_learn_concurrency",
        type=int,
        default=None,
        help="Concurrency for the bandit learning phase (defaults to --slo_calibration_concurrency).",
    )
    p.add_argument(
        "--bandit_keep_learning_during_eval",
        action="store_true",
        help="If set, do NOT freeze bandit updates after the learning phase.",
    )
    p.add_argument(
        "--bandit_force_freeze",
        action="store_true",
        help=(
            "Freeze bandit updates from the start of load tests (frozen-policy baseline / ablation). "
            "This is useful for E5 (domain/length shift) to compare online adaptation vs a frozen policy."
        ),
    )
    p.add_argument(
        "--bandit_adapter_ids",
        type=str,
        default=None,
        help="Comma-separated adapter_ids to include in bandit action space (plus implicit none).",
    )
    p.add_argument(
        "--bandit_rank_tiers",
        type=str,
        default=None,
        help="Comma-separated rank tiers to include in bandit action space.",
    )
    p.add_argument(
        "--bandit_variant_load_synthetic_ms",
        type=float,
        default=1000.0,
        help="Synthetic prior for variant swap/load overhead (ms) before EWMA is populated.",
    )

    # Bandit safety/feature toggles
    p.add_argument(
        "--bandit_disable_latency_guard",
        action="store_true",
        help="If set, do not require actions to be latency-safe under conformal bounds.",
    )
    p.add_argument(
        "--bandit_disable_conservative_fallback",
        action="store_true",
        help="If set, disable conservative fallback to the baseline policy.",
    )
    p.add_argument(
        "--bandit_disable_primal_dual",
        action="store_true",
        help="If set, disable the primal-dual virtual queue update.",
    )
    p.add_argument(
        "--bandit_disable_overhead_cost",
        action="store_true",
        help="If set, do not include adapter/swap overhead in the bandit cost estimate.",
    )
    p.add_argument(
        "--bandit_disable_system_features",
        action="store_true",
        help="If set, do not append system-load features to the bandit context.",
    )
    p.add_argument(
        "--bandit_disable_adapter_features",
        action="store_true",
        help="If set, zero out adapter-related features in the bandit context.",
    )

    # Dispatcher policy (scheduler)
    p.add_argument(
        "--dispatcher_policy",
        type=str,
        default="age",
        choices=["age", "edf", "lstf", "setup_aware", "setup_edf", "setup_lstf"],
        help="How the multi-variant dispatcher picks the next queue: age (default), edf, or lstf.",
    )

    # ------------------------------------------------------------------
    # PEFT / LoRA adapters (optional)
    # ------------------------------------------------------------------
    p.add_argument(
        "--enable_adapters",
        action="store_true",
        help="Enable PEFT/LoRA adapters (shared-base) in the serving stack.",
    )
    p.add_argument(
        "--adapter_root",
        type=str,
        default=None,
        help="Root directory containing PEFT adapters. Convention: <adapter_root>/<adapter_id>/adapter_config.json",
    )
    p.add_argument(
        "--adapter_policy",
        type=str,
        default="none",
        choices=["none", "dataset", "fixed"],
        help="How to choose adapter_id per request: none, dataset (adapter_id=dataset_type), or fixed.",
    )
    p.add_argument(
        "--adapter_fixed",
        type=str,
        default=None,
        help="If --adapter_policy fixed, the adapter_id to use.",
    )
    p.add_argument(
        "--adapter_rank_policy",
        type=str,
        default="max",
        choices=["max", "difficulty", "load", "fixed"],
        help="Nested-rank tier policy (effective LoRA rank at runtime).",
    )
    p.add_argument(
        "--adapter_rank_tiers",
        type=str,
        default="8,16,32",
        help="Comma-separated list of nested-rank tiers (e.g., '8,16,32').",
    )
    p.add_argument(
        "--adapter_fixed_rank",
        type=int,
        default=None,
        help="If --adapter_rank_policy fixed, the active LoRA rank to use.",
    )
    p.add_argument(
        "--max_loaded_adapters",
        type=int,
        default=8,
        help="Max number of adapters to keep resident per variant (LRU eviction).",
    )
    p.add_argument(
        "--adapter_eviction_policy",
        type=str,
        default="lru",
        choices=["lru"],
        help="Adapter cache eviction policy.",
    )
    p.add_argument(
        "--adapter_synthetic_load_ms",
        type=float,
        default=0.0,
        help="Optional extra (simulated) adapter load overhead in ms per cache miss.",
    )
    p.add_argument(
        "--adapter_synthetic_switch_ms",
        type=float,
        default=0.0,
        help="Optional extra (simulated) adapter switch overhead in ms per activation.",
    )
    p.add_argument(
        "--adapter_allow_missing",
        action="store_true",
        help="Allow 'synthetic' adapters without PEFT installed and/or without adapter dirs on disk (for churn/cache experiments).",
    )
    p.add_argument(
        "--dispatcher_max_sticky_adapter_batches",
        type=int,
        default=4,
        help="For setup_* dispatcher policies, limit how many consecutive batches to keep the same adapter active.",
    )

    p.add_argument(
        "--router_max_retries",
        type=int,
        default=2,
        help="Max escalation retries on format/error (0 disables retries).",
    )
    p.add_argument(
        "--router_ema_alpha",
        type=float,
        default=0.2,
        help="EWMA smoothing for per-variant latency tracking (0<alpha<=1).",
    )
    p.add_argument(
        "--router_calibration_mode",
        type=str,
        default="base",
        choices=["base", "router"],
        help="When calibrating SLO thresholds under multi-variant serving, calibrate on the base model only (recommended) or on router outputs.",
    )

    p.add_argument(
        "--multi_variants",
        type=str,
        nargs="+",
        default=["cheap", "med", "base"],
        choices=["cheap", "med", "base"],
        help="Variants to load when --service multi. Use fewer variants if you are memory constrained.",
    )
    p.add_argument(
        "--router_lazy_load_base",
        action="store_true",
        help="Lazy-load the base (fp16) variant on first use. Useful to avoid initial GPU OOM.",
    )
    p.add_argument(
        "--router_allow_quality_downgrade_for_slo",
        action="store_true",
        help="In --router_mode slo_aware, allow routing to cheaper variants than the difficulty-based minimum to try to meet latency SLOs.",
    )

    p.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "vllm"],
        help="Inference backend. 'hf' uses Transformers; 'vllm' uses optional vLLM async backend.",
    )
    # vLLM-specific options (only used when --backend vllm)
    p.add_argument("--vllm_model_override", type=str, default=None,
                   help="Optional model path/name override for vLLM (REQUIRED for --backend vllm --variant med).")
    p.add_argument("--vllm_quantization", type=str, default=None,
                   help="vLLM quantization mode for pre-quantized checkpoints (e.g., awq, gptq, bitsandbytes).")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--vllm_max_model_len", type=int, default=4096)
    p.add_argument("--vllm_max_num_seqs", type=int, default=128)
    p.add_argument("--vllm_enforce_eager", action="store_true",
                   help="Best-effort flag for vLLM to disable CUDA graphs (may help debugging).")
    p.add_argument("--prompt_mode", type=str, default="accuracy", choices=["accuracy", "slo"])

    # Paper: delayed / partial feedback experiments.
    # - gold  : send gold labels to the server (enables online bandit quality updates).
    # - none  : do NOT send labels to the server; correctness is still computed client-side
    #           and can be ingested later via scripts/replay_delayed_labels.py.
    p.add_argument(
        "--server_label_mode",
        type=str,
        default="gold",
        choices=["gold", "none"],
        help="Whether to send gold quality labels to the server during load tests (default: gold).",
    )

    p.add_argument("--preprocess", action="store_true", help="Run dataset preprocessing")
    p.add_argument("--data_dir", type=str, default="data/raw")
    p.add_argument("--processed_dir", type=str, default="data/processed")

    p.add_argument("--data_subset", type=int, default=0, help="If >0, truncate val+test to N examples")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument(
        "--stratify_difficulty",
        action="store_true",
        help="If set, load-test/calibration request sampling is stratified by (dataset, difficulty) so that each bucket is represented (avoids skew if val_data is ordered).",
    )


    p.add_argument("--output_dir", type=str, default="outputs")

    p.add_argument("--skip_load_test", action="store_true")
    p.add_argument("--num_requests", type=int, default=20)

    p.add_argument(
        "--load_test_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which split to use for load testing / SLO calibration requests (default: val). Use 'test' for final paper-grade evaluation.",
    )
    p.add_argument(
        "--load_test_jsonl",
        type=str,
        default=None,
        help="Optional explicit JSONL path for load-test/calibration requests. Overrides --load_test_split if provided.",
    )
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])

    # Nonstationary load: phase schedule (E2). If set, overrides --concurrencies.
    # Format: "<conc>:<num_requests>,<conc>:<num_requests>,..." (commas or semicolons).
    # Example: --concurrency_schedule "1:100,8:200,2:100"
    p.add_argument(
        "--concurrency_schedule",
        type=str,
        default=None,
        help="Optional nonstationary phase schedule; overrides --concurrencies.",
    )

    # Domain/length shift schedule (E5). Mutually exclusive with --concurrency_schedule.
    # Format: "dataset=<gsm8k|mmlu>:<nreq>,dataset=<...>:<nreq>" OR
    #         "length=<short|long>:<nreq>,length=<...>:<nreq>".
    p.add_argument(
        "--data_schedule",
        type=str,
        default=None,
        help="Optional nonstationary data schedule for domain/length shift (E5).",
    )
    p.add_argument(
        "--data_schedule_concurrency",
        type=int,
        default=None,
        help="Concurrency to use when running --data_schedule phases (defaults to the first value in --concurrencies).",
    )

    p.add_argument("--skip_accuracy_eval", action="store_true")

    # Batching flags: allow explicit on/off, else default to (prompt_mode == slo)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--enable_batching", dest="enable_batching", action="store_true")
    g.add_argument("--disable_batching", dest="enable_batching", action="store_false")
    p.set_defaults(enable_batching=None)

    p.add_argument("--max_batch_size", type=int, default=8)
    p.add_argument("--batch_wait_ms", type=int, default=8)

    # Dynamic SLO calibration
    p.add_argument("--disable_slo_calibration", action="store_true")
    p.add_argument("--slo_calibration_concurrency", type=int, default=1)
    p.add_argument("--slo_calibration_percentiles", type=float, nargs="+", default=[90.0, 95.0, 99.0])
    p.add_argument("--slo_primary_percentile", type=float, default=95.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # learned router artifacts are required for learned_* modes
    if args.router_mode in {"learned_ttft", "learned_total"}:
        if args.service != "multi":
            raise ValueError("learned_* router modes require --service multi.")
        if args.learned_router_dir is None:
            raise ValueError("--learned_router_dir is required for learned router modes.")

    # risk router artifacts are required for risk + bandit modes
    if args.router_mode in {"risk", "bandit"}:
        if args.service != "multi":
            raise ValueError("risk/bandit router modes require --service multi.")
        if args.risk_router_dir is None:
            raise ValueError("--risk_router_dir is required for --router_mode risk or bandit.")


    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log_env()

    # ------------------------------------------------------------------
    # CONFIGURATION LOG
    # ------------------------------------------------------------------
    effective_device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    effective_enable_batching = _effective_enable_batching(args.prompt_mode, args.enable_batching)
    if args.backend == "vllm":
        # vLLM does its own internal (continuous) batching; the HF micro-batching scheduler is not used.
        if effective_enable_batching:
            logger.info("NOTE: --backend vllm ignores HF micro-batching flags; vLLM uses internal continuous batching.")
        effective_enable_batching = False

    logger.info("=" * 80)
    logger.info("CONFIGURATION:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Backend: {args.backend}")
    if args.backend == "vllm":
        logger.info(f"  vLLM model override: {args.vllm_model_override}")
        logger.info(f"  vLLM quantization: {args.vllm_quantization}")
    logger.info(f"  Variant: {args.variant}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Prompt mode: {args.prompt_mode}")
    logger.info(f"  Data subset: {args.data_subset}")
    logger.info(f"  Output dir: {str(out_dir)}")
    logger.info(f"  Skip load test: {args.skip_load_test}")
    logger.info(f"  Skip accuracy eval: {args.skip_accuracy_eval}")
    logger.info(f"  Batching enabled (effective): {effective_enable_batching}")
    logger.info(f"  Max batch size: {args.max_batch_size}")
    logger.info(f"  Batch wait (ms): {args.batch_wait_ms}")
    logger.info(f"  SLO calibration disabled: {args.disable_slo_calibration}")
    logger.info("  TTFT definition: Option A (queue-inclusive)" )
    logger.info("=" * 80)

    # Save config for reproducibility
    _write_json(
        str(out_dir / "config.json"),
        {
            "model": args.model,
            "backend": args.backend,
            "vllm_model_override": args.vllm_model_override,
            "vllm_quantization": args.vllm_quantization,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_model_len": args.vllm_max_model_len,
            "vllm_max_num_seqs": args.vllm_max_num_seqs,
            "vllm_enforce_eager": bool(args.vllm_enforce_eager),
            "variant": args.variant,
            "device_arg": args.device,
            "device_effective": effective_device,
            "dtype": args.dtype,
            "prompt_mode": args.prompt_mode,
            "seed": args.seed,
            "data_dir": args.data_dir,
            "processed_dir": args.processed_dir,
            "data_subset": args.data_subset,
            "skip_load_test": args.skip_load_test,
            "num_requests": args.num_requests,
            "concurrencies": args.concurrencies,
            "concurrency_schedule": args.concurrency_schedule,
            "data_schedule": getattr(args, "data_schedule", None),
            "data_schedule_concurrency": getattr(args, "data_schedule_concurrency", None),
            "skip_accuracy_eval": args.skip_accuracy_eval,
            "enable_batching": effective_enable_batching,
            "max_batch_size": args.max_batch_size,
            "batch_wait_ms": args.batch_wait_ms,
            "disable_slo_calibration": args.disable_slo_calibration,
            "slo_calibration_concurrency": args.slo_calibration_concurrency,
            "slo_calibration_percentiles": args.slo_calibration_percentiles,
            "slo_primary_percentile": args.slo_primary_percentile,
            "ttft_definition": "OptionA_queue_inclusive",
            "bandit_force_freeze": bool(getattr(args, "bandit_force_freeze", False)),
        },
    )

    # ------------------------------------------------------------------
    # STEP 1: LOAD DATA
    # ------------------------------------------------------------------
    logger.info("\n[STEP 1] LOADING DATA")
    logger.info("-" * 80)

    if args.preprocess:
        dp = DataPreprocessor(data_dir=args.data_dir, output_dir=args.processed_dir, seed=args.seed)
        dp.run_pipeline()

    train_path = str(Path(args.processed_dir) / "train_data.jsonl")
    val_path = str(Path(args.processed_dir) / "val_data.jsonl")
    test_path = str(Path(args.processed_dir) / "test_data.jsonl")

    train_data = _read_jsonl(train_path)
    val_data = _read_jsonl(val_path)
    test_data = _read_jsonl(test_path)

    logger.info(f"Loaded data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    if args.data_subset and args.data_subset > 0:
        val_data = val_data[: args.data_subset]
        test_data = test_data[: args.data_subset]
        logger.info(f"Using subset: val={len(val_data)}, test={len(test_data)}")

    # ------------------------------------------------------------------
    # STEP 2: INITIALIZE SERVER
    # ------------------------------------------------------------------
    logger.info("\n[STEP 2] INITIALIZING SERVER")
    logger.info("-" * 80)

    if args.backend == "hf":
        if args.service == "multi":
            # Optional per-variant overrides (e.g., make CHEAP a smaller fp16 model).
            variant_models = {}
            variant_quant = {}
            if args.cheap_model:
                variant_models["cheap"] = str(args.cheap_model)
                # Convenience: if a separate cheap model is provided, default to fp16 unless
                # the user explicitly requests int8/int4.
                if args.cheap_quantization is None:
                    variant_quant["cheap"] = "fp16"
            if args.cheap_quantization:
                variant_quant["cheap"] = str(args.cheap_quantization)

            server = MultiVariantService(
                model_name=args.model,
                variants=tuple(args.multi_variants),
                router_mode=args.router_mode,
                learned_router_dir=args.learned_router_dir,
                risk_router_dir=args.risk_router_dir,
                risk_latency_delta=float(args.risk_latency_delta),
                risk_quality_epsilon=float(args.risk_quality_epsilon),
                risk_quality_alpha=float(args.risk_quality_alpha),
                fixed_variant=args.router_fixed_variant,
                allow_quality_downgrade_for_slo=bool(args.router_allow_quality_downgrade_for_slo),
                dispatcher_policy=str(args.dispatcher_policy),
                device=effective_device,
                dtype=args.dtype,
                variant_models=variant_models or None,
                variant_quantization=variant_quant or None,
                enable_batching=effective_enable_batching,
                max_batch_size=args.max_batch_size,
                batch_wait_ms=args.batch_wait_ms,
                ema_alpha=args.router_ema_alpha,
                max_retries=args.router_max_retries,
                lazy_load_base=bool(args.router_lazy_load_base),
                enable_adapters=bool(args.enable_adapters),
                adapter_root=args.adapter_root,
                adapter_policy=str(args.adapter_policy),
                adapter_fixed=args.adapter_fixed,
                adapter_rank_policy=str(args.adapter_rank_policy),
                adapter_rank_tiers=str(args.adapter_rank_tiers),
                adapter_fixed_rank=args.adapter_fixed_rank,
                max_loaded_adapters=int(args.max_loaded_adapters),
                adapter_eviction_policy=str(args.adapter_eviction_policy),
                adapter_synthetic_load_ms=float(args.adapter_synthetic_load_ms),
                adapter_synthetic_switch_ms=float(args.adapter_synthetic_switch_ms),
                adapter_allow_missing=bool(args.adapter_allow_missing),
                dispatcher_max_sticky_adapter_batches=int(args.dispatcher_max_sticky_adapter_batches),
                overhead_ms_to_cost_units=float(args.overhead_ms_to_cost_units),
                router_seed=int(args.router_seed),
                # Bandit router knobs (only used when router_mode=bandit)
                bandit_delta=float(args.bandit_delta),
                bandit_alpha=float(args.bandit_alpha),
                bandit_beta_r=float(args.bandit_beta_r),
                bandit_beta_q=float(args.bandit_beta_q),
                bandit_eps_r=float(args.bandit_eps_r),
                bandit_eps_q=float(args.bandit_eps_q),
                bandit_beta_u=float(args.bandit_beta_u),
                bandit_label_budget_p=float(args.bandit_label_budget_p),
                bandit_checkpoint_path=(str(args.bandit_checkpoint_path) if args.bandit_checkpoint_path else None),
                bandit_checkpoint_every=int(args.bandit_checkpoint_every),
                bandit_state_path=(str(args.bandit_state_path) if args.bandit_state_path else None),
                bandit_require_latency_safe=(not bool(args.bandit_disable_latency_guard)),
                bandit_use_conservative_fallback=(not bool(args.bandit_disable_conservative_fallback)),
                bandit_use_primal_dual=(not bool(args.bandit_disable_primal_dual)),
                bandit_use_overhead_cost=(not bool(args.bandit_disable_overhead_cost)),
                bandit_use_system_features=(not bool(args.bandit_disable_system_features)),
                bandit_use_adapter_features=(not bool(args.bandit_disable_adapter_features)),
                bandit_variant_load_synthetic_ms=float(args.bandit_variant_load_synthetic_ms),
                bandit_adapter_ids=args.bandit_adapter_ids,
                bandit_rank_tiers=args.bandit_rank_tiers,
                bandit_update_enabled=True,
            )
        else:
            # Optional CHEAP override for single-variant runs.
            model_name = args.model
            quant_override = None
            if str(args.variant).lower() == "cheap":
                if args.cheap_model:
                    model_name = str(args.cheap_model)
                    if args.cheap_quantization is None:
                        quant_override = "fp16"
                if args.cheap_quantization:
                    quant_override = str(args.cheap_quantization)

            server = SingleVariantServer(
                model_name=model_name,
                variant=args.variant,
                device=effective_device,
                dtype=args.dtype,
                quantization_override=quant_override,
                enable_batching=effective_enable_batching,
                max_batch_size=args.max_batch_size,
                batch_wait_ms=args.batch_wait_ms,
                enable_adapters=bool(args.enable_adapters),
                adapter_root=args.adapter_root,
                adapter_policy=str(args.adapter_policy),
                adapter_fixed=args.adapter_fixed,
                adapter_rank_policy=str(args.adapter_rank_policy),
                adapter_rank_tiers=str(args.adapter_rank_tiers),
                adapter_fixed_rank=args.adapter_fixed_rank,
                max_loaded_adapters=int(args.max_loaded_adapters),
                adapter_eviction_policy=str(args.adapter_eviction_policy),
                adapter_synthetic_load_ms=float(args.adapter_synthetic_load_ms),
                adapter_synthetic_switch_ms=float(args.adapter_synthetic_switch_ms),
                adapter_allow_missing=bool(args.adapter_allow_missing),
                overhead_ms_to_cost_units=float(args.overhead_ms_to_cost_units),
            )
    else:
        if args.service == "multi":
            raise ValueError("--service multi is currently supported only for --backend hf")
        from vllm_server import VLLMConfig, VLLMVariantServer
        vllm_model = args.vllm_model_override or args.model
        if args.variant == "med" and not args.vllm_model_override:
            raise ValueError(
                "For --backend vllm --variant med, please pass --vllm_model_override pointing to a pre-quantized checkpoint."
            )
        vllm_dtype = "float16" if args.dtype in ["auto", "float16"] else args.dtype
        vcfg = VLLMConfig(
            model=vllm_model,
            dtype=vllm_dtype,
            quantization=args.vllm_quantization,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            max_num_seqs=args.vllm_max_num_seqs,
            enforce_eager=bool(args.vllm_enforce_eager),
        )
        server = VLLMVariantServer(model_name=vllm_model, variant=args.variant, config=vcfg)

    # Optional: force-freeze bandit updates (frozen-policy baseline / ablation).
    # Must happen before any learning/load phases.
    if bool(getattr(args, "bandit_force_freeze", False)) and isinstance(server, MultiVariantService):
        try:
            server.set_bandit_update_enabled(False)
            logger.info("[BANDIT] bandit_force_freeze=True -> bandit updates disabled.")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # STEP 3: RUN LOAD TESTS
    # ------------------------------------------------------------------
    logger.info("\n[STEP 3] RUNNING LOAD TESTS")
    logger.info("-" * 80)

    slo_thresholds_path = str(out_dir / "slo_thresholds.json")
    slo_profiles = None if args.disable_slo_calibration else _load_slo_profiles(slo_thresholds_path)

    primary_key = f"p{int(args.slo_primary_percentile)}"

    # Optionally build a stratified load-test/calibration pool from a chosen split.
    if args.load_test_jsonl:
        load_pool = _read_jsonl(str(args.load_test_jsonl))
        logger.info(f"Loaded load-test pool from --load_test_jsonl: {args.load_test_jsonl} (n={len(load_pool)})")
    else:
        split = (args.load_test_split or "val").lower().strip()
        load_pool = test_data if split == "test" else val_data
        logger.info(f"Using load-test split: {split} (n={len(load_pool)})")

    if args.stratify_difficulty:
        load_pool = _build_stratified_pool(load_pool, k=int(args.num_requests), seed=int(args.seed))
        _log_strata_counts("[STRATIFY]", load_pool)



    # ------------------------------------------------------------------
    # Multi-variant: optional base-only SLO calibration
    # ------------------------------------------------------------------
    if (
        (not args.skip_load_test)
        and (not args.disable_slo_calibration)
        and (slo_profiles is None)
        and isinstance(server, MultiVariantService)
        and (args.router_calibration_mode == "base" or args.router_mode == "bandit")
    ):
        calib_conc = int(args.slo_calibration_concurrency)
        logger.info(
            f"[CALIBRATION] Multi-variant service: calibrating SLO profiles using BASE variant only at concurrency={calib_conc}"
        )
        base_server = server.get_variant_server("base")
        lg_calib = ClosedLoopLoadGenerator(
            inference_func=base_server.generate,
            max_concurrency=calib_conc,
            num_requests=args.num_requests,
            data_loader=load_pool,
            prompt_mode=args.prompt_mode,
            seed=args.seed,
            send_labels_to_server=(args.server_label_mode == "gold"),
        )
        req_metrics_calib = lg_calib.run_load_test()
        slo_profiles = calibrate_slo_profiles(req_metrics_calib, percentiles=args.slo_calibration_percentiles)
        _write_json(
            slo_thresholds_path,
            {
                    "definition": "TTFT_OptionA_queue_inclusive + E2E(total_latency_ms)",
                "calibration_concurrency": calib_conc,
                "percentiles": args.slo_calibration_percentiles,
                "primary": args.slo_primary_percentile,
                "profiles": slo_profiles,
                "calibration_mode": "base_only",
            },
        )
        # Save calibration request traces for debugging / appendix.
        lg_calib.save_results(str(out_dir / f"requests_calibration_base_concurrency_{calib_conc}.jsonl"))

    # If we have calibrated (or loaded) SLO profiles, apply them to the server
    # before any bandit learning/evaluation.
    if isinstance(server, MultiVariantService) and slo_profiles is not None:
        try:
            slo_for_report = slo_profiles.get(primary_key) or next(iter(slo_profiles.values()))
            server.set_slo_dict(slo_for_report)
        except Exception:
            pass

    # Fallback calibration for single-variant runs (or if profiles are still missing).
    if (
        (not args.skip_load_test)
        and (not args.disable_slo_calibration)
        and (slo_profiles is None)
        and (not isinstance(server, MultiVariantService))
    ):
        calib_conc = int(args.slo_calibration_concurrency)
        logger.info(
            f"[CALIBRATION] Single-variant: calibrating SLO profiles at concurrency={calib_conc}"
        )
        lg_calib = ClosedLoopLoadGenerator(
            inference_func=server.generate,
            max_concurrency=calib_conc,
            num_requests=args.num_requests,
            data_loader=load_pool,
            prompt_mode=args.prompt_mode,
            seed=args.seed,
            send_labels_to_server=(args.server_label_mode == "gold"),
        )
        req_metrics_calib = lg_calib.run_load_test()
        slo_profiles = calibrate_slo_profiles(req_metrics_calib, percentiles=args.slo_calibration_percentiles)
        _write_json(
            slo_thresholds_path,
            {
                "definition": "TTFT_OptionA_queue_inclusive + E2E(total_latency_ms)",
                "calibration_concurrency": calib_conc,
                "percentiles": args.slo_calibration_percentiles,
                "primary": args.slo_primary_percentile,
                "profiles": slo_profiles,
                "calibration_mode": "single_variant_fallback",
            },
        )
        lg_calib.save_results(str(out_dir / f"requests_calibration_concurrency_{calib_conc}.jsonl"))

    # ------------------------------------------------------------------
    # Bandit learning phase (online updates), then freeze for evaluation.
    # ------------------------------------------------------------------
    if args.router_mode == "bandit" and isinstance(server, MultiVariantService):
        learn_n = int(getattr(args, "bandit_learn_requests", 0) or 0)
        if learn_n > 0:
            learn_conc = int(args.bandit_learn_concurrency or args.slo_calibration_concurrency or 1)
            logger.info(
                f"[LEARNING] Bandit learning: {learn_n} requests at concurrency={learn_conc} "
                f"(label_budget_p={args.bandit_label_budget_p})."
            )
            try:
                server.set_bandit_update_enabled(True)
            except Exception:
                pass

            learn_dir = out_dir / "bandit_learning"
            learn_dir.mkdir(parents=True, exist_ok=True)
            lg_learn = ClosedLoopLoadGenerator(
                inference_func=server.generate,
                max_concurrency=learn_conc,
                num_requests=learn_n,
                data_loader=train_data if train_data is not None else load_pool,
                prompt_mode=args.prompt_mode,
                seed=args.seed,
                send_labels_to_server=(args.server_label_mode == "gold"),
            )
            # ClosedLoopLoadGenerator.run_load_test() takes no arguments.
            # (The dataset mix and difficulty come from the provided data pool.)
            learn_metrics = lg_learn.run_load_test()
            lg_learn.save_results(str(learn_dir / f"requests_bandit_learn_concurrency_{learn_conc}.jsonl"))
            try:
                learn_report = MetricsCalculator(learn_metrics, slo_dict=(server.slo_dict or {})).compute_all_metrics()
                _write_json(str(learn_dir / f"metrics_bandit_learn_concurrency_{learn_conc}.json"), learn_report)
            except Exception:
                pass

            # Save bandit router state for reproducibility
            try:
                save_prefix = args.bandit_checkpoint_path or str(learn_dir / "bandit_state")
                if server.save_bandit_state(save_prefix):
                    logger.info(f"[LEARNING] Saved bandit state to: {save_prefix}.json/.npz")
            except Exception:
                pass

        # Freeze updates during evaluation unless explicitly requested.
        if not bool(getattr(args, "bandit_keep_learning_during_eval", False)):
            if args.bandit_state_path is not None or int(getattr(args, "bandit_learn_requests", 0) or 0) > 0:
                try:
                    server.set_bandit_update_enabled(False)
                    logger.info("[EVAL] Bandit updates frozen for evaluation.")
                except Exception:
                    pass
            else:
                logger.warning(
                    "[EVAL] router_mode=bandit with no --bandit_state_path and no --bandit_learn_requests; "
                    "bandit will keep learning during evaluation unless you provide a learning phase or a saved state."
                )

    if args.skip_load_test:
        logger.info("Skipping load tests (--skip_load_test).")
    else:
        if not val_data:
            logger.warning("val_data is empty. Load tests will run against a tiny dummy pool.")
            val_data = [{"dataset": "gsm8k", "prompt": "1+1?", "answer": "2", "difficulty": "easy"}]
            load_pool = val_data

        # Choose SLO dict for reporting (fixed across phases/concurrencies).
        slo_for_report = None
        if not args.disable_slo_calibration:
            if slo_profiles is not None:
                slo_for_report = slo_profiles.get(primary_key) or next(iter(slo_profiles.values()))

        # If using multi-variant routing, update the router's SLO targets.
        if isinstance(server, MultiVariantService) and slo_for_report is not None:
            server.set_slo_dict(slo_for_report)

        schedule = _parse_concurrency_schedule(args.concurrency_schedule)
        data_schedule = _parse_data_schedule(getattr(args, "data_schedule", None))

        if schedule is not None and data_schedule is not None:
            raise ValueError("--concurrency_schedule and --data_schedule are mutually exclusive. Use one at a time.")

        if schedule is not None:
            logger.info(f"Running nonstationary load schedule with {len(schedule)} phases: {schedule}")
            all_req_metrics: List[Any] = []
            all_req_rows: List[Dict[str, Any]] = []

            for phase_idx, (conc, nreq) in enumerate(schedule, 1):
                conc = int(conc)
                nreq = int(nreq)
                print(f"\n>>> Phase {phase_idx}/{len(schedule)}: concurrency={conc} num_requests={nreq}")

                lg = ClosedLoopLoadGenerator(
                    inference_func=server.generate,
                    max_concurrency=conc,
                    num_requests=nreq,
                    data_loader=load_pool,
                    prompt_mode=args.prompt_mode,
                    seed=args.seed + phase_idx,  # diversify per phase while remaining deterministic
                    send_labels_to_server=(args.server_label_mode == "gold"),
                )
                req_metrics = lg.run_load_test()
                all_req_metrics.extend(req_metrics)

                # Add phase metadata for time-series analysis.
                for m in req_metrics:
                    try:
                        d = m.to_dict()
                    except Exception:
                        continue
                    d["phase"] = int(phase_idx)
                    d["phase_concurrency"] = int(conc)
                    d["phase_num_requests"] = int(nreq)
                    all_req_rows.append(d)

                mc = MetricsCalculator(req_metrics, slo_dict=slo_for_report)
                report = mc.print_report(title=f"LOAD TEST RESULTS (Phase {phase_idx}, Concurrency {conc})")

                # Sensitivity: report compliance under p90/p95/p99
                sensitivity: Dict[str, float] = {}
                if (not args.disable_slo_calibration) and slo_profiles is not None:
                    for k, slo in slo_profiles.items():
                        sensitivity[k] = float(
                            MetricsCalculator(req_metrics, slo_dict=slo).compute_all_metrics()["summary"]["slo_compliance"]
                        )
                report["slo_profile_used"] = primary_key
                report["slo_sensitivity"] = sensitivity
                report["phase"] = phase_idx
                report["phase_concurrency"] = conc
                report["phase_num_requests"] = nreq

                metrics_path = str(out_dir / f"metrics_phase_{phase_idx}_concurrency_{conc}.json")
                _write_json(metrics_path, report)
                lg.save_results(str(out_dir / f"requests_phase_{phase_idx}_concurrency_{conc}.jsonl"))

            # Combined artifacts for time-series analysis
            try:
                all_req_metrics = sorted(all_req_metrics, key=lambda m: float(getattr(m, "end_time", 0.0) or 0.0))
            except Exception:
                pass
            # Combined artifacts for time-series analysis
            try:
                all_req_rows = sorted(all_req_rows, key=lambda r: float(r.get("end_time", 0.0) or 0.0))
            except Exception:
                pass
            combined_path = str(out_dir / "requests_schedule.jsonl")
            with open(combined_path, "w") as f:
                for r in all_req_rows:
                    f.write(json.dumps(r) + "\n")
            logger.info(f"Saved combined schedule trace to: {combined_path}")

            try:
                overall = MetricsCalculator(all_req_metrics, slo_dict=slo_for_report).compute_all_metrics()
                overall["slo_profile_used"] = primary_key
                _write_json(str(out_dir / "metrics_schedule.json"), overall)
            except Exception:
                pass
        elif data_schedule is not None:
            # ------------------------------------------------------------------
            # Nonstationary data schedule (E5): fixed concurrency, changing request mix.
            # ------------------------------------------------------------------
            conc = int(getattr(args, "data_schedule_concurrency", None) or (args.concurrencies[0] if args.concurrencies else 1))

            # Precompute pools for selectors.
            dataset_pools: Dict[str, List[Dict[str, Any]]] = {}
            for ex in load_pool:
                ds = str(ex.get("dataset") or "unknown").lower().strip()
                dataset_pools.setdefault(ds, []).append(ex)

            short_pool, long_pool, thr = _length_buckets(load_pool, seed=int(args.seed))
            logger.info(
                f"Running data schedule with {len(data_schedule)} phases @ concurrency={conc}. "
                f"Length split threshold (median input_length) ~ {thr}."
            )

            all_req_metrics: List[Any] = []
            all_req_rows: List[Dict[str, Any]] = []

            for phase_idx, (kind, value, nreq) in enumerate(data_schedule, 1):
                kind = str(kind)
                value = str(value)
                nreq = int(nreq)

                # Phase-specific request pool.
                if kind == "dataset":
                    phase_pool = dataset_pools.get(value) or []
                    if not phase_pool:
                        logger.warning(f"[DATA_SCHEDULE] No examples for dataset='{value}'. Falling back to full pool.")
                        phase_pool = load_pool
                elif kind == "length":
                    if value == "short":
                        phase_pool = short_pool or load_pool
                    elif value == "long":
                        phase_pool = long_pool or load_pool
                    else:
                        logger.warning(f"[DATA_SCHEDULE] Unknown length bucket '{value}'. Using full pool.")
                        phase_pool = load_pool
                else:
                    phase_pool = load_pool

                print(f"\n>>> Phase {phase_idx}/{len(data_schedule)}: {kind}={value} num_requests={nreq} (conc={conc})")

                lg = ClosedLoopLoadGenerator(
                    inference_func=server.generate,
                    max_concurrency=conc,
                    num_requests=nreq,
                    data_loader=phase_pool,
                    prompt_mode=args.prompt_mode,
                    seed=args.seed + phase_idx,
                    send_labels_to_server=(args.server_label_mode == "gold"),
                )
                req_metrics = lg.run_load_test()
                all_req_metrics.extend(req_metrics)

                for m in req_metrics:
                    try:
                        d = m.to_dict()
                    except Exception:
                        continue
                    d["phase"] = int(phase_idx)
                    d["phase_concurrency"] = int(conc)
                    d["phase_num_requests"] = int(nreq)
                    d["phase_selector_kind"] = str(kind)
                    d["phase_selector_value"] = str(value)
                    d["phase_length_threshold"] = int(thr)
                    all_req_rows.append(d)

                mc = MetricsCalculator(req_metrics, slo_dict=slo_for_report)
                report = mc.print_report(title=f"LOAD TEST RESULTS (Phase {phase_idx}, {kind}={value}, Concurrency {conc})")

                sensitivity: Dict[str, float] = {}
                if (not args.disable_slo_calibration) and slo_profiles is not None:
                    for k, slo in slo_profiles.items():
                        sensitivity[k] = float(
                            MetricsCalculator(req_metrics, slo_dict=slo).compute_all_metrics()["summary"]["slo_compliance"]
                        )

                report["slo_profile_used"] = primary_key
                report["slo_sensitivity"] = sensitivity
                report["phase"] = phase_idx
                report["phase_concurrency"] = conc
                report["phase_num_requests"] = nreq
                report["phase_selector_kind"] = str(kind)
                report["phase_selector_value"] = str(value)
                report["phase_length_threshold"] = int(thr)

                metrics_path = str(out_dir / f"metrics_phase_{phase_idx}_{kind}_{value}_concurrency_{conc}.json")
                _write_json(metrics_path, report)
                lg.save_results(str(out_dir / f"requests_phase_{phase_idx}_{kind}_{value}_concurrency_{conc}.jsonl"))

            try:
                all_req_rows = sorted(all_req_rows, key=lambda r: float(r.get("end_time", 0.0) or 0.0))
            except Exception:
                pass
            combined_path = str(out_dir / "requests_schedule.jsonl")
            with open(combined_path, "w") as f:
                for r in all_req_rows:
                    f.write(json.dumps(r) + "\n")
            logger.info(f"Saved combined schedule trace to: {combined_path}")

            try:
                overall = MetricsCalculator(all_req_metrics, slo_dict=slo_for_report).compute_all_metrics()
                overall["slo_profile_used"] = primary_key
                overall["schedule_type"] = "data"
                overall["data_schedule"] = [
                    {"kind": k, "value": v, "num_requests": int(n)} for (k, v, n) in data_schedule
                ]
                _write_json(str(out_dir / "metrics_schedule.json"), overall)
            except Exception:
                pass

        else:
            for conc in args.concurrencies:
                conc = int(conc)
                print(f"\n>>> Testing with concurrency={conc}")

                lg = ClosedLoopLoadGenerator(
                    inference_func=server.generate,
                    max_concurrency=conc,
                    num_requests=args.num_requests,
                    data_loader=load_pool,
                    prompt_mode=args.prompt_mode,
                    seed=args.seed,
                    send_labels_to_server=(args.server_label_mode == "gold"),
                )
                req_metrics = lg.run_load_test()

                mc = MetricsCalculator(req_metrics, slo_dict=slo_for_report)
                report = mc.print_report(title=f"LOAD TEST RESULTS (Concurrency {conc})")

                # Sensitivity: report compliance under p90/p95/p99
                sensitivity: Dict[str, float] = {}
                if (not args.disable_slo_calibration) and slo_profiles is not None:
                    for k, slo in slo_profiles.items():
                        sensitivity[k] = float(
                            MetricsCalculator(req_metrics, slo_dict=slo).compute_all_metrics()["summary"]["slo_compliance"]
                        )

                report["slo_profile_used"] = primary_key
                report["slo_sensitivity"] = sensitivity

                metrics_path = str(out_dir / f"metrics_concurrency_{conc}.json")
                _write_json(metrics_path, report)

                requests_path = str(out_dir / f"requests_concurrency_{conc}.jsonl")
                lg.save_results(requests_path)

    # ------------------------------------------------------------------
    # STEP 4: EVALUATE ACCURACY
    # ------------------------------------------------------------------
    logger.info("\n[STEP 4] EVALUATING ACCURACY")
    logger.info("-" * 80)

    if args.skip_accuracy_eval:
        logger.info("Skipping accuracy evaluation.")
    else:
        if not test_data:
            logger.error("test_data is empty. Did preprocessing run?")
        else:
            evaluator = HeldOutEvaluator(server, test_data)
            results, detailed = evaluator.evaluate(prompt_mode=args.prompt_mode, verbose=False)

            _write_json(str(out_dir / "accuracy_results.json"), results)
            _write_jsonl(str(out_dir / "accuracy_detailed.jsonl"), detailed)

    # ------------------------------------------------------------------
    # STEP 5: DONE
    # ------------------------------------------------------------------
    logger.info("\n[STEP 5] SAVING OUTPUTS")
    logger.info("-" * 80)

    logger.info("\n" + "=" * 80)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info(f"Results saved to: {str(out_dir)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Interrupted")
        sys.exit(1)
