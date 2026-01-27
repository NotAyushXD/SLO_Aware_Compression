# run_baseline_evaluation.py
"""
End-to-end baseline evaluation script.

Supports:
- Load tests (closed-loop concurrency) + metric export
- Held-out accuracy evaluation (MMLU + GSM8K)
- Two prompt modes:
    * slo      : shorter prompts / smaller token budgets (for later SLO work)
    * accuracy : stronger prompts / few-shot for GSM8K / higher token budgets

New CLI flags (v7):
  --prompt_mode {slo,accuracy}
  --disable_slo_calibration
  --skip_load_test
  --model (alias for --model_name)

If you see "unrecognized arguments: --prompt_mode ...", you are running an older
copy of this file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import logging
from typing import Dict, Any, List

from preprocessing import DataPreprocessor
from server import SingleVariantServer
from load_generator import ClosedLoopLoadGenerator
from metrics import MetricsCalculator, calibrate_slos
from evaluation import HeldOutEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main(args: argparse.Namespace) -> None:
    ensure_dir(args.output_dir)

    logger.info("=" * 80)
    logger.info("CONFIGURATION:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Variant: {args.variant}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Prompt mode: {args.prompt_mode}")
    logger.info(f"  Requests per test: {args.num_requests}")
    logger.info(f"  Concurrency levels: {args.concurrencies}")
    logger.info(f"  Data subset: {args.data_subset}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  SLO calibration disabled: {args.disable_slo_calibration}")
    logger.info(f"  Skip load test: {args.skip_load_test}")
    logger.info(
        f"  Batching enabled: {False if getattr(args, 'disable_batching', False) else True if getattr(args, 'enable_batching', False) else (args.prompt_mode == 'slo') }"
    )
    logger.info(f"  Max batch size: {args.max_batch_size}")
    logger.info(f"  Batch wait (ms): {args.batch_wait_ms}")
    logger.info(f"  Skip accuracy eval: {args.skip_accuracy_eval}")
    logger.info("=" * 80)

    # Step 0: Preprocess (optional)
    if args.preprocess:
        logger.info("\n[STEP 0] PREPROCESSING DATA")
        logger.info("-" * 80)
        pre = DataPreprocessor(data_dir=args.data_dir, output_dir=args.processed_dir)
        train_data, val_data, test_data = pre.run_full_pipeline()
    else:
        # Step 1: Load data
        logger.info("\n[STEP 1] LOADING DATA")
        logger.info("-" * 80)

        train_path = os.path.join(args.processed_dir, "train_data.jsonl")
        val_path = os.path.join(args.processed_dir, "val_data.jsonl")
        test_path = os.path.join(args.processed_dir, "test_data.jsonl")

        logger.info(f"Loading train data from {train_path}")
        train_data = load_jsonl(train_path)
        logger.info(f"Loading val data from {val_path}")
        val_data = load_jsonl(val_path)
        logger.info(f"Loading test data from {test_path}")
        test_data = load_jsonl(test_path)

    logger.info(f"Loaded data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Use subset for faster iteration (optional)
    if args.data_subset and args.data_subset > 0:
        val_data = val_data[: args.data_subset]
        test_data = test_data[: args.data_subset]
        logger.info(f"Using subset: val={len(val_data)}, test={len(test_data)}")

    # Step 2: Initialize server
    logger.info("\n[STEP 2] INITIALIZING SERVER")
    logger.info("-" * 80)

    server = SingleVariantServer(
        model_name=args.model_name,
        variant=args.variant,
        device=args.device,
        dtype=args.dtype,
        enable_batching=(
            False if getattr(args, "disable_batching", False) else
            True if getattr(args, "enable_batching", False) else
            (args.prompt_mode == "slo")
        ),
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
    )

    # Load/Calibrate SLOs (optional)
    slo_file = os.path.join(args.output_dir, "slo_thresholds.json")
    current_slos = None

    if args.slo_thresholds_path:
        try:
            with open(args.slo_thresholds_path, "r") as f:
                current_slos = json.load(f)
            logger.info(f"Loaded SLOs from {args.slo_thresholds_path}")
        except Exception as e:
            logger.warning(f"Failed to load SLOs from {args.slo_thresholds_path}: {e}")

    if (current_slos is None) and (not args.disable_slo_calibration):
        # Try to load existing thresholds
        if os.path.exists(slo_file):
            try:
                with open(slo_file, "r") as f:
                    current_slos = json.load(f)
                logger.info(f"Loaded existing SLOs from {slo_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing SLOs: {e}")

    # Step 3: Run load tests (optional)
    logger.info("\n[STEP 3] RUNNING LOAD TESTS")
    logger.info("-" * 80)

    load_test_results: Dict[int, Any] = {}
    all_metrics_summary: List[Dict[str, Any]] = []
    all_raw_metrics = []

    if args.skip_load_test:
        logger.info("Skipping load tests (--skip_load_test).")
    else:
        for concurrency in args.concurrencies:
            logger.info(f"\n>>> Testing with concurrency={concurrency}")

            load_gen = ClosedLoopLoadGenerator(
                inference_func=server.generate,
                max_concurrency=concurrency,
                num_requests=args.num_requests,
                data_loader=val_data,
                prompt_mode=args.prompt_mode,
            )

            start_time = time.time()
            request_metrics = load_gen.run()
            duration = time.time() - start_time

            all_raw_metrics.extend(request_metrics)

            # If enabled, calibrate SLO thresholds from a chosen baseline concurrency run,
            # then use those thresholds for compliance/violation reporting.
            if (not args.disable_slo_calibration) and (current_slos is None) and (concurrency == args.slo_calibrate_from_concurrency):
                logger.info(f"Calibrating SLOs from concurrency={concurrency} at p{args.slo_calibrate_percentile:.1f}...")
                current_slos = calibrate_slos(request_metrics, percentile=args.slo_calibrate_percentile)
            try:
                with open(slo_file, "w") as f:
                    json.dump(current_slos, f, indent=2)
                logger.info(f"Saved calibrated SLOs to {slo_file}")
            except Exception as e:
                logger.warning(f"Failed to save calibrated SLOs: {e}")

            calc = MetricsCalculator(request_metrics, slo_dict=current_slos)
            test_metrics = calc.compute_all_metrics()
            load_test_results[concurrency] = test_metrics

            # Print report
            calc.print_report(title=f"LOAD TEST RESULTS (Concurrency {concurrency})")

            # Save metrics JSON
            metrics_file = os.path.join(args.output_dir, f"metrics_concurrency_{concurrency}.json")
            calc.save_metrics(metrics_file)

            # Save raw request logs JSONL
            requests_file = os.path.join(args.output_dir, f"requests_concurrency_{concurrency}.jsonl")
            load_gen.save_metrics(requests_file)

            all_metrics_summary.append({
                "concurrency": concurrency,
                "num_requests": args.num_requests,
                "duration_sec": duration,
                "success_rate": test_metrics["summary"]["success_rate"],
                "throughput_tokens_per_sec": test_metrics["summary"]["throughput_tokens_per_sec"],
                "ttft_p99_ms": test_metrics["ttft"]["p99"],
                "tpot_p95_ms": test_metrics["tpot"]["p95"],
                "e2e_p99_ms": test_metrics["e2e_latency"]["p99"],
                "slo_compliance": test_metrics["summary"]["slo_compliance"],
                "slo_violations": test_metrics["summary"]["slo_violations"],
            })

    # Optional SLO calibration: only if enabled and we don't already have SLOs
    if (not args.disable_slo_calibration) and (current_slos is None) and all_raw_metrics:
        logger.info("\n[STEP 3.5] CALIBRATING SLOs (measurement-only)")
        logger.info("-" * 80)

        current_slos = calibrate_slos(all_raw_metrics, percentile=args.slo_calibrate_percentile)

        # Save calibrated SLOs
        try:
            with open(slo_file, "w") as f:
                json.dump(current_slos, f, indent=2)
            logger.info(f"Saved calibrated SLOs to {slo_file}")
        except Exception as e:
            logger.warning(f"Failed to write {slo_file}: {e}")

    # Step 4: Evaluate accuracy (optionally skipped)
    eval_results, detailed_predictions = {}, []
    if args.skip_accuracy_eval:
        logger.info("\n[STEP 4] EVALUATING ACCURACY")
        logger.info("-" * 80)
        logger.info("Skipping accuracy evaluation (--skip_accuracy_eval).")
    else:
        logger.info("\n[STEP 4] EVALUATING ACCURACY")
        logger.info("-" * 80)

        evaluator = HeldOutEvaluator(server, test_data, batch_size=32)
        eval_results, detailed_predictions = evaluator.evaluate(
            prompt_mode=args.prompt_mode,
            verbose=args.verbose_eval,
        )

    # Step 5: Save results
    logger.info("\n[STEP 5] SAVING OUTPUTS")
    logger.info("-" * 80)

    # Save eval results json
    eval_out_file = os.path.join(args.output_dir, "eval_results.json")
    with open(eval_out_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"Saved evaluation results to {eval_out_file}")

    # Save summary json
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "config": {
                "model_name": args.model_name,
                "variant": args.variant,
                "device": args.device,
                "dtype": args.dtype,
                "prompt_mode": args.prompt_mode,
                "num_requests": args.num_requests,
                "concurrencies": args.concurrencies,
                "data_subset": args.data_subset,
                "disable_slo_calibration": args.disable_slo_calibration,
                "skip_load_test": args.skip_load_test,
                "skip_accuracy_eval": args.skip_accuracy_eval,
                "enable_batching": (
                    False if getattr(args, "disable_batching", False) else
                    True if getattr(args, "enable_batching", False) else
                    (args.prompt_mode == "slo")
                ),
                "max_batch_size": args.max_batch_size,
                "batch_wait_ms": args.batch_wait_ms,
            },
            "slo_thresholds": current_slos,
            "load_test_results": load_test_results,
            "eval_results": eval_results,
        }, f, indent=2)
    logger.info(f"Saved overall summary to {summary_file}")

    # Save Excel report
    try:
        import pandas as pd

        excel_file = os.path.join(args.output_dir, "performance_summary.xlsx")

        df_load = pd.DataFrame(all_metrics_summary)
        df_eval = pd.DataFrame([
            {"dataset": k, **v} for k, v in eval_results.items()
        ]) if eval_results else pd.DataFrame()
        df_details = pd.DataFrame(detailed_predictions)

        with pd.ExcelWriter(excel_file) as writer:
            df_load.to_excel(writer, sheet_name="Load Test Metrics", index=False)
            if not df_eval.empty:
                df_eval.to_excel(writer, sheet_name="Accuracy Metrics", index=False)
            if not df_details.empty:
                df_details.to_excel(writer, sheet_name="Model Outputs", index=False)

        logger.info(f"Saved Excel summary to {excel_file}")

    except ImportError:
        logger.warning("pandas/openpyxl not installed, skipping Excel export")
    except Exception as e:
        logger.error(f"Failed to save Excel report: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end baseline evaluation")

    p.add_argument("--preprocess", action="store_true", help="Run preprocessing to create processed jsonl files.")
    p.add_argument("--data_dir", type=str, default="data/raw", help="Raw data directory")
    p.add_argument("--processed_dir", type=str, default="data/processed", help="Processed data directory")
    p.add_argument("--data_subset", type=int, default=0, help="Use first N examples from val/test for faster runs (0 = all)")

    # Model/server config
    p.add_argument("--model_name", "--model", dest="model_name", type=str, default="meta-llama/Llama-3.1-8B",
                   help="HF model name (alias: --model)")
    p.add_argument("--device", type=str, default="auto", help="auto|cuda|cpu|mps")
    p.add_argument("--dtype", type=str, default="auto", help="auto|float16|bfloat16")
    p.add_argument("--variant", type=str, default="med", choices=["base", "med", "cheap"],
                   help="base=fp16/bf16, med=8-bit, cheap=4-bit")

    # Benchmark controls
    p.add_argument("--num_requests", type=int, default=10, help="Requests per load test")
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4], help="Concurrency levels to test")
    p.add_argument("--output_dir", type=str, default="outputs", help="Where to write metrics and reports")

    # New (v7)
    p.add_argument("--prompt_mode", type=str, choices=["slo", "accuracy"], default="slo",
                   help="Prompt mode: accuracy (best correctness) vs slo (shorter outputs for later SLO work)")
    p.add_argument("--disable_slo_calibration", action="store_true",
                   help="Disable SLO calibration and ignore slo_thresholds.json.")
    p.add_argument("--slo_calibrate_percentile", type=float, default=95.0,
                   help="Percentile to use when calibrating SLO thresholds (default: 95.0)")
    p.add_argument("--slo_calibrate_from_concurrency", type=int, default=1,
                   help="Which concurrency run to use for SLO calibration (default: 1)")
    p.add_argument("--slo_thresholds_path", type=str, default=None,
                   help="Path to slo_thresholds.json to load and use for compliance checks")

    p.add_argument("--skip_load_test", action="store_true",
                   help="Skip load tests and only run accuracy evaluation.")
    p.add_argument("--skip_accuracy_eval", action="store_true",
                   help="Skip held-out accuracy evaluation (run only load tests/SLO calibration).")
    p.add_argument("--enable_batching", action="store_true",
                   help="Enable dynamic request batching (recommended for prompt_mode=slo).")
    p.add_argument("--disable_batching", action="store_true",
                   help="Force-disable dynamic batching (debugging / apples-to-apples).")
    p.add_argument("--max_batch_size", type=int, default=8,
                   help="Maximum batch size for the dynamic batcher.")
    p.add_argument("--batch_wait_ms", type=int, default=8,
                   help="How long to wait (ms) to form a batch before running it.")
    p.add_argument("--verbose_eval", action="store_true",
                   help="Print a few example prompts/outputs during evaluation.")

    return p


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
