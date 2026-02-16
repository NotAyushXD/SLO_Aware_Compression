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



def _build_stratified_pool(
    examples: List[Dict[str, Any]],
    k: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Return a deterministic, stratified subset of size k.

    Stratification key: (dataset, difficulty). This prevents calibration/load-test runs
    from accidentally selecting only one bucket when processed val_data.jsonl is ordered.
    """
    if k <= 0 or (not examples):
        return list(examples)

    # If k >= pool size, nothing to sample.
    if k >= len(examples):
        return list(examples)

    # Group by (dataset, difficulty)
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ex in examples:
        ds = (ex.get("dataset") or "unknown").lower().strip()
        diff = (ex.get("difficulty") or "medium").lower().strip()
        key = (ds, diff)
        buckets.setdefault(key, []).append(ex)

    # Deterministic shuffle inside each bucket
    rng = random.Random(int(seed))
    for key in buckets:
        rng.shuffle(buckets[key])

    # Prefer a stable bucket ordering: dataset asc, then difficulty in easy/medium/hard/other.
    diff_order = {"easy": 0, "medium": 1, "hard": 2}
    ordered_keys = sorted(buckets.keys(), key=lambda t: (t[0], diff_order.get(t[1], 99), t[1]))

    # Equal allocation across buckets, then top-up from remaining.
    num_buckets = max(1, len(ordered_keys))
    base = k // num_buckets
    rem = k % num_buckets

    targets: Dict[Tuple[str, str], int] = {}
    for i, key in enumerate(ordered_keys):
        targets[key] = base + (1 if i < rem else 0)

    selected: List[Dict[str, Any]] = []
    for key in ordered_keys:
        take = min(targets[key], len(buckets[key]))
        if take > 0:
            selected.extend(buckets[key][:take])
            buckets[key] = buckets[key][take:]

    # Top up if any buckets were short.
    need = k - len(selected)
    if need > 0:
        remaining: List[Dict[str, Any]] = []
        for key in ordered_keys:
            remaining.extend(buckets[key])
        rng.shuffle(remaining)
        selected.extend(remaining[:need])

    # Final deterministic shuffle so ordering doesn't bias selection.
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
        choices=["difficulty", "slo_aware", "fixed", "always_cheap", "always_base", "learned_ttft", "learned_total"],
        help="Routing policy for multi-variant serving.",
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

    p.add_argument(
        "--router_max_retries",
        type=int,
        default=1,
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
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4])

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
            "skip_accuracy_eval": args.skip_accuracy_eval,
            "enable_batching": effective_enable_batching,
            "max_batch_size": args.max_batch_size,
            "batch_wait_ms": args.batch_wait_ms,
            "disable_slo_calibration": args.disable_slo_calibration,
            "slo_calibration_concurrency": args.slo_calibration_concurrency,
            "slo_calibration_percentiles": args.slo_calibration_percentiles,
            "slo_primary_percentile": args.slo_primary_percentile,
            "ttft_definition": "OptionA_queue_inclusive",
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
            server = MultiVariantService(
                model_name=args.model,
                variants=tuple(args.multi_variants),
                router_mode=args.router_mode,
                learned_router_dir=args.learned_router_dir,
                fixed_variant=args.router_fixed_variant,
                allow_quality_downgrade_for_slo=bool(args.router_allow_quality_downgrade_for_slo),
                device=effective_device,
                dtype=args.dtype,
                enable_batching=effective_enable_batching,
                max_batch_size=args.max_batch_size,
                batch_wait_ms=args.batch_wait_ms,
                ema_alpha=args.router_ema_alpha,
                max_retries=args.router_max_retries,
                lazy_load_base=bool(args.router_lazy_load_base),
            )
        else:
            server = SingleVariantServer(
                model_name=args.model,
                variant=args.variant,
                device=effective_device,
                dtype=args.dtype,
                enable_batching=effective_enable_batching,
                max_batch_size=args.max_batch_size,
                batch_wait_ms=args.batch_wait_ms,
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

    # ------------------------------------------------------------------
    # STEP 3: RUN LOAD TESTS
    # ------------------------------------------------------------------
    logger.info("\n[STEP 3] RUNNING LOAD TESTS")
    logger.info("-" * 80)

    slo_thresholds_path = str(out_dir / "slo_thresholds.json")
    slo_profiles = None if args.disable_slo_calibration else _load_slo_profiles(slo_thresholds_path)

    primary_key = f"p{int(args.slo_primary_percentile)}"

    # Optionally build a stratified load-test/calibration pool from val_data.
    val_pool = val_data
    if args.stratify_difficulty:
        val_pool = _build_stratified_pool(val_data, k=int(args.num_requests), seed=int(args.seed))
        _log_strata_counts("[STRATIFIED] val_pool", val_pool)


    # ------------------------------------------------------------------
    # Multi-variant: optional base-only SLO calibration
    # ------------------------------------------------------------------
    if (
        (not args.skip_load_test)
        and (not args.disable_slo_calibration)
        and (slo_profiles is None)
        and isinstance(server, MultiVariantService)
        and (args.router_calibration_mode == "base")
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
            data_loader=val_pool,
            prompt_mode=args.prompt_mode,
            seed=args.seed,
        )
        req_metrics_calib = lg_calib.run_load_test()
        slo_profiles = calibrate_slo_profiles(req_metrics_calib, percentiles=args.slo_calibration_percentiles)
        _write_json(
            slo_thresholds_path,
            {
                "definition": "TTFT_OptionA_queue_inclusive",
                "calibration_concurrency": calib_conc,
                "percentiles": args.slo_calibration_percentiles,
                "primary": args.slo_primary_percentile,
                "profiles": slo_profiles,
                "calibration_mode": "base_only",
            },
        )
        # Save calibration request traces for debugging / appendix.
        lg_calib.save_results(str(out_dir / f"requests_calibration_base_concurrency_{calib_conc}.jsonl"))

    if args.skip_load_test:
        logger.info("Skipping load tests (--skip_load_test).")
    else:
        if not val_data:
            logger.warning("val_data is empty. Load tests will run against a tiny dummy pool.")
            val_data = [{"dataset": "gsm8k", "prompt": "1+1?", "answer": "2", "difficulty": "easy"}]
            val_pool = val_data

        for conc in args.concurrencies:
            conc = int(conc)
            print(f"\n>>> Testing with concurrency={conc}")

            lg = ClosedLoopLoadGenerator(
                inference_func=server.generate,
                max_concurrency=conc,
                num_requests=args.num_requests,
                data_loader=val_pool,
                prompt_mode=args.prompt_mode,
                seed=args.seed,
            )
            req_metrics = lg.run_load_test()

            # Calibrate thresholds from a specific baseline concurrency.
            if (
                (not args.disable_slo_calibration)
                and (slo_profiles is None)
                and (conc == int(args.slo_calibration_concurrency))
            ):
                logger.info(
                    f"Calibrating SLOs from concurrency={conc} at percentiles {args.slo_calibration_percentiles}"
                )
                slo_profiles = calibrate_slo_profiles(req_metrics, percentiles=args.slo_calibration_percentiles)
                _write_json(
                    slo_thresholds_path,
                    {
                        "definition": "TTFT_OptionA_queue_inclusive",
                        "calibration_concurrency": conc,
                        "percentiles": args.slo_calibration_percentiles,
                        "primary": args.slo_primary_percentile,
                        "profiles": slo_profiles,
                    },
                )

            # Choose SLO dict for reporting
            slo_for_report = None
            if args.disable_slo_calibration:
                slo_for_report = None
            else:
                if slo_profiles is not None:
                    slo_for_report = slo_profiles.get(primary_key) or next(iter(slo_profiles.values()))


            # If using multi-variant routing, update the router's SLO targets.
            if isinstance(server, MultiVariantService) and slo_for_report is not None:
                server.set_slo_dict(slo_for_report)

            mc = MetricsCalculator(req_metrics, slo_dict=slo_for_report)
            report = mc.print_report(title=f"LOAD TEST RESULTS (Concurrency {conc})")

            # Sensitivity: report compliance under p90/p95/p99
            sensitivity: Dict[str, float] = {}
            if (not args.disable_slo_calibration) and slo_profiles is not None:
                for k, slo in slo_profiles.items():
                    sensitivity[k] = float(MetricsCalculator(req_metrics, slo_dict=slo).compute_all_metrics()["summary"]["slo_compliance"])

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
