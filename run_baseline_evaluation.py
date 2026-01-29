"""run_baseline_evaluation.py

Unified harness to:
1) Run accuracy/format evaluation (GSM8K + MMLU) under either:
   - prompt_mode=accuracy (normal answers)
   - prompt_mode=slo      (short, SLO-friendly answers)
2) Run serving/load smoke tests to measure TTFT/TPOT/queue/E2E under concurrency.
3) Optionally calibrate per-difficulty SLO thresholds from a baseline run
   (typically concurrency=1) and then report SLO compliance at other concurrencies.

This script is intentionally "single-variant" for the ablation story:
- base  = fp16
- med   = int8
- cheap = int4

Router logic lives outside this file; here we focus on clean, reproducible
measurements that you can put in the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from evaluation import Evaluator
from load_generator import ClosedLoopLoadGenerator
from metrics import MetricsCalculator, calibrate_slos
from preprocessing import format_example_for_evaluation, load_processed_data
from server import SingleVariantServer


LOGGER = logging.getLogger("__main__")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _resolve_device(device: str) -> str:
    device = str(device).lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be one of: auto|cuda|cpu")
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    return device


def _maybe_write_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _print_banner(title: str) -> None:
    LOGGER.info("=" * 80)
    LOGGER.info(title)
    LOGGER.info("=" * 80)


def _format_config(args: argparse.Namespace, device: str) -> str:
    return (
        "CONFIGURATION:\n"
        f"  Model: {args.model}\n"
        f"  Variant: {args.variant}\n"
        f"  Device: {device}\n"
        f"  Dtype: {args.dtype}\n"
        f"  Prompt mode: {args.prompt_mode}\n"
        f"  Requests per test: {args.num_requests}\n"
        f"  Concurrency levels: {args.concurrencies}\n"
        f"  Data subset: {args.data_subset}\n"
        f"  Output dir: {args.output_dir}\n"
        f"  SLO calibration disabled: {args.disable_slo_calibration}\n"
        f"  Skip load test: {args.skip_load_test}\n"
        f"  Batching enabled: {args.batching_enabled}\n"
        f"  Max batch size: {args.max_batch_size}\n"
        f"  Batch wait (ms): {args.batch_wait_ms}\n"
        f"  Skip accuracy eval: {args.skip_accuracy_eval}"
    )


def _load_examples(data_dir: str, data_subset: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    train_raw, val_raw, test_raw = load_processed_data(data_dir=data_dir)

    def _subset(xs: List[Dict]) -> List[Dict]:
        if not data_subset or data_subset <= 0:
            return xs
        return xs[: min(data_subset, len(xs))]

    train = [format_example_for_evaluation(x) for x in _subset(train_raw)]
    val = [format_example_for_evaluation(x) for x in _subset(val_raw)]
    test = [format_example_for_evaluation(x) for x in _subset(test_raw)]
    return train, val, test


def _run_load_tests(
    *,
    server: SingleVariantServer,
    data_pool: List[Dict],
    output_dir: str,
    prompt_mode: str,
    num_requests: int,
    concurrencies: List[int],
    max_new_tokens: int,
    disable_slo_calibration: bool,
    slo_calibration_percentile: float,
) -> Dict:
    """Run load tests and return a summary dict."""

    os.makedirs(output_dir, exist_ok=True)

    all_results = {
        "variant": server.variant,
        "prompt_mode": prompt_mode,
        "num_requests": num_requests,
        "concurrencies": concurrencies,
        "runs": {},
    }

    slo_profiles = None

    for concurrency in concurrencies:
        LOGGER.info("\n>>> Testing with concurrency=%s", concurrency)
        gen = ClosedLoopLoadGenerator(
            server=server,
            concurrency=concurrency,
            total_requests=num_requests,
            data_pool=data_pool,
            prompt_mode=prompt_mode,
            max_new_tokens=max_new_tokens,
        )
        metrics = gen.run()

        # Save raw request traces.
        req_path = os.path.join(output_dir, f"requests_concurrency_{concurrency}.jsonl")
        gen.save_requests(req_path)

        # Calibrate SLOs (optional) using the first concurrency that equals 1.
        if (
            not disable_slo_calibration
            and slo_profiles is None
            and concurrency == 1
            and len(metrics) > 0
        ):
            LOGGER.info(
                "Calibrating SLOs from concurrency=1 at p%.1f", slo_calibration_percentile
            )
            slo_profiles = calibrate_slos(metrics, percentile=slo_calibration_percentile)
            LOGGER.info("Calibrated SLOs at p%.1f: %s", slo_calibration_percentile, slo_profiles)

        calc = MetricsCalculator(slo_profiles=slo_profiles)
        summary = calc.summarize(metrics)

        # Save per-concurrency summary.
        met_path = os.path.join(output_dir, f"metrics_concurrency_{concurrency}.json")
        _maybe_write_json(met_path, summary)

        # Print a compact report (matches the style you pasted).
        calc.pretty_print(summary, title=f"LOAD TEST RESULTS (Concurrency {concurrency})")

        all_results["runs"][str(concurrency)] = summary

    return all_results


def _run_accuracy_eval(
    *,
    server: SingleVariantServer,
    examples: List[Dict],
    output_dir: str,
    prompt_mode: str,
    max_new_tokens: int,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator(server)

    summary_path = os.path.join(output_dir, "accuracy_summary.json")
    logs_path = os.path.join(output_dir, "accuracy_per_example.jsonl")

    summary = evaluator.run(
        examples,
        prompt_mode=prompt_mode,
        max_new_tokens=max_new_tokens,
        output_json_path=summary_path,
        per_example_log_jsonl_path=logs_path,
        description=f"EVALUATING ON {len(examples)} EXAMPLES (prompt_mode={prompt_mode})",
    )

    # Also write a convenience summary file named after the output directory.
    try:
        out_name = Path(output_dir).name.rstrip("/")
        if out_name:
            _maybe_write_json(os.path.join(Path(output_dir).parent, f"{out_name}_summary.json"), summary)
    except Exception:
        pass

    return summary


def main(args: argparse.Namespace) -> None:
    _setup_logging()

    device = _resolve_device(args.device)

    _print_banner("BASELINE EVALUATION")
    LOGGER.info("\n%s", _format_config(args, device))

    # ---------------------------------------------------------------------
    # STEP 1: Load data
    # ---------------------------------------------------------------------
    LOGGER.info("\n[STEP 1] LOADING DATA")
    LOGGER.info("-" * 80)
    train, val, test = _load_examples(args.data_dir, args.data_subset)
    LOGGER.info("Loaded data: train=%d, val=%d, test=%d", len(train), len(val), len(test))

    # Use val split by default for both load tests and evaluation (matches your logs).
    eval_split = val

    # ---------------------------------------------------------------------
    # STEP 2: Initialize server
    # ---------------------------------------------------------------------
    LOGGER.info("\n[STEP 2] INITIALIZING SERVER")
    LOGGER.info("-" * 80)

    server = SingleVariantServer(
        model_name=args.model,
        variant=args.variant,
        device=device,
        dtype=args.dtype,
        batching_enabled=args.batching_enabled,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        seed=args.seed,
    )

    # ---------------------------------------------------------------------
    # STEP 3: Load tests
    # ---------------------------------------------------------------------
    LOGGER.info("\n[STEP 3] RUNNING LOAD TESTS")
    LOGGER.info("-" * 80)

    load_summary = None
    if args.skip_load_test:
        LOGGER.info("Skipping load tests (--skip_load_test).")
    else:
        load_summary = _run_load_tests(
            server=server,
            data_pool=eval_split,
            output_dir=args.output_dir,
            prompt_mode=args.prompt_mode,
            num_requests=args.num_requests,
            concurrencies=args.concurrencies,
            max_new_tokens=args.max_new_tokens,
            disable_slo_calibration=args.disable_slo_calibration,
            slo_calibration_percentile=args.slo_calibration_percentile,
        )

    # ---------------------------------------------------------------------
    # STEP 4: Accuracy eval
    # ---------------------------------------------------------------------
    LOGGER.info("\n[STEP 4] EVALUATING ACCURACY")
    LOGGER.info("-" * 80)

    acc_summary = None
    if args.skip_accuracy_eval:
        LOGGER.info("Skipping accuracy evaluation.")
    else:
        acc_summary = _run_accuracy_eval(
            server=server,
            examples=eval_split,
            output_dir=args.output_dir,
            prompt_mode=args.prompt_mode,
            max_new_tokens=args.max_new_tokens,
        )

    # ---------------------------------------------------------------------
    # STEP 5: Save top-level summary
    # ---------------------------------------------------------------------
    LOGGER.info("\n[STEP 5] SAVING OUTPUTS")
    LOGGER.info("-" * 80)

    combined = {
        "config": {
            "model": args.model,
            "variant": args.variant,
            "device": device,
            "dtype": args.dtype,
            "prompt_mode": args.prompt_mode,
            "num_requests": args.num_requests,
            "concurrencies": args.concurrencies,
            "data_subset": args.data_subset,
            "batching_enabled": args.batching_enabled,
            "max_batch_size": args.max_batch_size,
            "batch_wait_ms": args.batch_wait_ms,
            "disable_slo_calibration": args.disable_slo_calibration,
            "slo_calibration_percentile": args.slo_calibration_percentile,
        },
        "accuracy": acc_summary,
        "load": load_summary,
    }

    _maybe_write_json(os.path.join(args.output_dir, "run_summary.json"), combined)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("BASELINE EVALUATION COMPLETE")
    LOGGER.info("Results saved to: %s", args.output_dir)
    LOGGER.info("=" * 80)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline evaluation for SLO-aware compression")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "med", "cheap"],
        help="base=fp16, med=int8, cheap=int4",
    )

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )

    parser.add_argument("--prompt_mode", type=str, default="accuracy", choices=["accuracy", "slo"])

    # Accuracy eval
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--data_subset", type=int, default=0, help="0 means full split")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--skip_accuracy_eval", action="store_true")

    # Load tests
    parser.add_argument("--skip_load_test", action="store_true")
    parser.add_argument("--num_requests", type=int, default=20)
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="one or more concurrency levels",
    )

    parser.add_argument("--disable_slo_calibration", action="store_true")
    parser.add_argument("--slo_calibration_percentile", type=float, default=95.0)

    # Server batching (renamed from the old 'enable_batching' to avoid the crash you hit)
    parser.add_argument("--batching_enabled", action="store_true")
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument("--batch_wait_ms", type=int, default=8)

    parser.add_argument("--output_dir", type=str, default="router_logs_smoke")
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


if __name__ == "__main__":
    try:
        main(_parse_args())
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        sys.exit(130)
