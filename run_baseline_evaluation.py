# run_baseline_evaluation.py
"""
End-to-end baseline evaluation orchestration.

Pipeline:
  preprocessing → server → (optional SLO calibration) → load tests → accuracy eval

Key fixes vs your current script:
- Enables SLO calibration safely (measurement-only, does NOT stop generation).
- Calibration is done on a *small* run first, then SLOs are applied to the main runs.
- Works with the fixed ClosedLoopLoadGenerator that truly executes num_requests.
"""

import json
import argparse
import os
import logging
import time
import math
import random
from typing import Dict, Any, List

from preprocessing import DataPreprocessor
from server import SingleVariantServer
from load_generator import ClosedLoopLoadGenerator
from metrics import MetricsCalculator, calibrate_slos
from evaluation import HeldOutEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: str = "data/processed"):
    train_data, val_data, test_data = [], [], []
    for split_name, split_list in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = os.path.join(data_dir, f"{split_name}_data.jsonl")
        if os.path.exists(path):
            logger.info(f"Loading {split_name} data from {path}")
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        split_list.append(json.loads(line))
        else:
            logger.warning(f"File not found: {path}")
    logger.info(f"Loaded data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    return train_data, val_data, test_data


def apply_slo_slack(slos: Dict[str, Dict[str, float]], slack: float) -> Dict[str, Dict[str, float]]:
    if slack is None:
        return slos
    out = {}
    for diff, d in slos.items():
        out[diff] = {
            "ttft_ms": float(math.ceil(d.get("ttft_ms", 0) * slack)),
            "tpot_ms": float(math.ceil(d.get("tpot_ms", 0) * slack)),
        }
    return out


def maybe_calibrate_slos(
    server: SingleVariantServer,
    val_data: List[Dict[str, Any]],
    output_dir: str,
    disable: bool,
    calibration_requests: int,
    calibration_concurrency: int,
    percentile: float,
    slack: float,
) -> Dict[str, Dict[str, float]]:
    slo_file = os.path.join(output_dir, "slo_thresholds.json")

    # Load if exists
    if os.path.exists(slo_file):
        try:
            with open(slo_file, 'r') as f:
                current_slos = json.load(f)
            logger.info(f"Loaded existing SLOs from {slo_file}")
            logger.info(f"Using SLOs: {current_slos}")
            return current_slos
        except Exception as e:
            logger.warning(f"Failed to load existing SLOs: {e}")

    if disable:
        logger.info("SLO calibration disabled; using default SLOs from MetricsCalculator.")
        return None

    # Run a small calibration load test (measurement-only)
    n = min(int(calibration_requests), len(val_data))
    if n <= 0:
        logger.warning("No validation data available for calibration.")
        return None

    logger.info("\n[STEP 3.5] CALIBRATING SLOs (measurement-only)")
    logger.info("-" * 80)
    logger.info(f"Calibration run: {n} requests @ concurrency={calibration_concurrency}")

    # Stratified sampling by difficulty improves calibration stability
    by_diff = {'easy': [], 'medium': [], 'hard': []}
    for ex in val_data:
        d = (ex.get('difficulty') or 'medium').lower()
        if d not in by_diff:
            d = 'medium'
        by_diff[d].append(ex)

    calib_data: List[Dict[str, Any]] = []
    per = max(n // 3, 1)
    for d in ['easy', 'medium', 'hard']:
        take = min(per, len(by_diff[d]))
        if take > 0:
            calib_data.extend(random.sample(by_diff[d], take))

    # Fill remainder (if any) from the overall pool
    if len(calib_data) < n:
        need = n - len(calib_data)
        remaining = [ex for ex in val_data if ex not in calib_data]
        if remaining:
            calib_data.extend(random.sample(remaining, min(need, len(remaining))))

    calib_data = calib_data[:n]

    load_gen = ClosedLoopLoadGenerator(
        inference_func=server.generate,
        max_concurrency=calibration_concurrency,
        num_requests=n,
        data_loader=calib_data,
        prompt_mode=args.prompt_mode,
    )
    raw_metrics = load_gen.run()

    calibrated = calibrate_slos(raw_metrics, percentile=float(percentile))
    calibrated = apply_slo_slack(calibrated, float(slack))

    with open(slo_file, 'w') as f:
        json.dump(calibrated, f, indent=2)
    logger.info(f"Saved calibrated SLOs to {slo_file}")
    logger.info(f"Calibrated SLOs (p{percentile} with slack x{slack}): {calibrated}")

    return calibrated


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("END-TO-END BASELINE EVALUATION: MED-ONLY SERVER (8-BIT QUANTIZATION)")
    logger.info("=" * 80)

    # Step 0: preprocessing
    if args.preprocess:
        logger.info("\n[STEP 0] PREPROCESSING DATASETS")
        logger.info("-" * 80)
        preprocessor = DataPreprocessor(data_dir=args.data_dir, output_dir=args.processed_dir)
        _train, _val, _test = preprocessor.run_pipeline()

    # Step 1: load data
    logger.info("\n[STEP 1] LOADING DATA")
    logger.info("-" * 80)
    train_data, val_data_full, test_data_full = load_data(args.processed_dir)

    val_data = val_data_full
    test_data = test_data_full

    if not val_data_full or not test_data_full:
        logger.error("No validation or test data found!")
        return

    if args.data_subset > 0:
        val_data = val_data_full[:args.data_subset]
        test_data = test_data_full[:args.data_subset]
        logger.info(f"Using subset: val={len(val_data)}, test={len(test_data)}")

    # Step 2: initialize server
    logger.info("\n[STEP 2] INITIALIZING SERVER")
    logger.info("-" * 80)
    try:
        server = SingleVariantServer(
            model_name=args.model_name,
            variant=args.variant,
            device=args.device,
            dtype=args.dtype,
        )
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        return

    # Step 3.5: calibrate slos (optional)
    current_slos = maybe_calibrate_slos(
        server=server,
        val_data=val_data_full,
        output_dir=args.output_dir,
        disable=args.disable_slo_calibration,
        calibration_requests=args.calibration_requests,
        calibration_concurrency=args.calibration_concurrency,
        percentile=args.slo_percentile,
        slack=args.slo_slack,
    )

    # Step 3: load tests
    logger.info("\n[STEP 3] RUNNING LOAD TESTS")
    logger.info("-" * 80)

    load_test_results = {}
    all_metrics_summary = []
    all_raw_metrics = []

    for concurrency in args.concurrencies:
        logger.info(f"\n>>> Testing with concurrency={concurrency}")

        load_gen = ClosedLoopLoadGenerator(
            inference_func=server.generate,
            max_concurrency=concurrency,
            num_requests=args.num_requests,
            data_loader=val_data,
        )

        t0 = time.time()
        raw = load_gen.run()
        duration = time.time() - t0

        all_raw_metrics.extend(raw)

        calc = MetricsCalculator(raw, slo_dict=current_slos)
        test_metrics = calc.compute_all_metrics()
        load_test_results[concurrency] = test_metrics

        calc.print_report(title=f"LOAD TEST RESULTS (Concurrency {concurrency})")

        metrics_file = os.path.join(args.output_dir, f"metrics_concurrency_{concurrency}.json")
        calc.save_metrics(metrics_file)

        requests_file = os.path.join(args.output_dir, f"requests_concurrency_{concurrency}.jsonl")
        load_gen.save_metrics(requests_file)

        summary_entry = {
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
        }
        all_metrics_summary.append(summary_entry)

    # Step 4: accuracy evaluation
    logger.info("\n[STEP 4] EVALUATING ACCURACY")
    logger.info("-" * 80)
    try:
        evaluator = HeldOutEvaluator(
            model=server,
            data_loader=test_data,
            batch_size=32,
            verbose=args.verbose_eval,
            max_verbose=args.max_verbose_eval,
            prompt_mode=args.prompt_mode,
        )
        eval_results, detailed_predictions = evaluator.evaluate()

        eval_file = os.path.join(args.output_dir, "eval_results.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_file}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        eval_results = {}
        detailed_predictions = []

    # Step 5: summary report
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY REPORT")
    logger.info("=" * 80)

    print("\nLoad Test Results by Concurrency:")
    print(f"{'Concurrency':<12} {'Throughput':<18} {'TTFT P99':<12} {'TPOT P95':<12} {'E2E P99':<12} {'SLO Compl':<10}")
    print("-" * 80)

    for summary in all_metrics_summary:
        print(
            f"{summary['concurrency']:<12} "
            f"{summary['throughput_tokens_per_sec']:<18.2f} "
            f"{summary['ttft_p99_ms']:<12.2f} "
            f"{summary['tpot_p95_ms']:<12.2f} "
            f"{summary['e2e_p99_ms']:<12.2f} "
            f"{summary['slo_compliance']*100:<10.1f}%"
        )

    print("\nAccuracy Results:")
    if eval_results:
        for dataset_type in sorted(eval_results.keys()):
            if dataset_type not in ["overall", "by_difficulty"]:
                res = eval_results[dataset_type]
                if isinstance(res, dict):
                    acc = res.get("accuracy", res.get("em", 0.0))
                    correct = res.get("correct_count", 0)
                    total = res.get("total_count", 0)
                    print(f"  {dataset_type.upper():<10s}: {acc*100:6.2f}% ({correct}/{total})")

        overall = eval_results.get("overall", {})
        if isinstance(overall, dict):
            acc = overall.get("accuracy", overall.get("em", 0.0))
            correct = overall.get("correct_count", 0)
            total = overall.get("total_count", 0)
            print(f"  {'OVERALL':<10s}: {acc*100:6.2f}% ({correct}/{total})")
    else:
        print("  No evaluation results available")

    # Save summary.json
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump(
            {
                "load_test_summary": all_metrics_summary,
                "eval_results": eval_results,
                "config": {
                    "model_name": args.model_name,
                    "variant": args.variant,
                    "num_requests": args.num_requests,
                    "concurrencies": args.concurrencies,
                    "device": args.device,
                    "calibrated_slos": current_slos,
                },
            },
            f,
            indent=2,
        )

    # Optional Excel export
    try:
        import pandas as pd
        excel_file = os.path.join(args.output_dir, "performance_summary.xlsx")

        df_load = pd.DataFrame(all_metrics_summary)

        eval_rows = []
        if eval_results:
            for dataset_type, res in eval_results.items():
                if isinstance(res, dict):
                    row = dict(res)
                    row["dataset"] = dataset_type
                    eval_rows.append(row)
        df_eval = pd.DataFrame(eval_rows)
        df_details = pd.DataFrame(detailed_predictions)

        with pd.ExcelWriter(excel_file) as writer:
            df_load.to_excel(writer, sheet_name="Load Test Metrics", index=False)
            if not df_eval.empty:
                df_eval.to_excel(writer, sheet_name="Accuracy Metrics", index=False)
            if not df_details.empty:
                df_details.to_excel(writer, sheet_name="Model Outputs", index=False)

        logger.info(f"Saved Excel summary to {excel_file}")
    except ImportError:
        logger.warning("pandas or openpyxl not installed, skipping Excel export")
    except Exception as e:
        logger.error(f"Failed to save Excel report: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end baseline MED-only server evaluation")

    # Data configuration
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--data_dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--data_subset", type=int, default=0, help="Use subset of data (0=all, >0=limit to N)")

    # Model configuration
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B", help="HuggingFace model name")
    parser.add_argument("--device", default="cuda", help="Device: cuda/cpu")
    parser.add_argument("--dtype", default="auto", help="Data type: auto/float16/bfloat16")
    parser.add_argument("--variant", default="med", help="Server variant: base/med/cheap")

    # Load test configuration
    parser.add_argument("--num_requests", type=int, default=5000, help="Number of requests per concurrency level")
    parser.add_argument("--concurrencies", nargs='+', type=int, default=[1, 4, 8], help="List of concurrency levels")

    # SLO calibration configuration
    parser.add_argument("--disable_slo_calibration", action="store_true", help="Disable SLO calibration")
    parser.add_argument("--prompt_mode", choices=["accuracy","slo"], default="accuracy", help="Prompt mode: accuracy=better correctness (brief reasoning for GSM8K), slo=short answer-only outputs")
    parser.add_argument("--calibration_requests", type=int, default=200, help="#requests for calibration run")
    parser.add_argument("--calibration_concurrency", type=int, default=1, help="concurrency for calibration run")
    parser.add_argument("--slo_percentile", type=float, default=95.0, help="percentile for calibration (e.g., 95)")
    parser.add_argument("--slo_slack", type=float, default=1.10, help="slack multiplier on calibrated SLOs")

    # Evaluation verbosity
    parser.add_argument("--verbose_eval", action="store_true", help="Print a few prompt/outputs during eval")
    parser.add_argument("--max_verbose_eval", type=int, default=5, help="How many examples to print if verbose_eval")

    # Output configuration
    parser.add_argument("--output_dir", default="results/baseline_med", help="Output directory")

    args = parser.parse_args()

    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Requests per test: {args.num_requests}")
    logger.info(f"  Concurrency levels: {args.concurrencies}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  SLO calibration disabled: {args.disable_slo_calibration}")

    main(args)