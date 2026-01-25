# run_baseline_evaluation.py
"""
End-to-end baseline evaluation script.

Supports:
- prompt_mode: accuracy | slo
- optional SLO calibration
- optional load tests at different concurrencies
- accuracy evaluation (GSM8K + MMLU)
- saving outputs: eval_results.json, summary.json, performance_summary.xlsx

Example:
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt_mode accuracy \
  --disable_slo_calibration \
  --skip_load_test \
  --concurrencies 1 \
  --num_requests 5 \
  --data_subset 200 \
  --output_dir "/kaggle/working/med_cuda_v0"
"""

from __future__ import annotations

import argparse
import json
import os
import logging
from typing import List, Dict, Any, Optional

import pandas as pd

from server import SingleVariantServer
from evaluation import Evaluator
from load_generator import ClosedLoopLoadGenerator
from metrics import MetricsCalculator, calibrate_slos
from prompt_templates import build_llama_formatted_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("__main__")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_excel(path: str, accuracy_results: Dict[str, Any], load_test_rows: List[Dict[str, Any]], model_rows: List[Dict[str, Any]]) -> None:
    # Accuracy metrics table
    acc_rows = []
    for k, v in accuracy_results.items():
        acc_rows.append({
            "Dataset": k,
            "N": v.get("n", 0),
            "Correct": v.get("correct", 0),
            "Accuracy": v.get("accuracy", 0.0),
            "Format OK": v.get("format_ok", 0.0),
        })
    acc_df = pd.DataFrame(acc_rows)

    load_df = pd.DataFrame(load_test_rows) if load_test_rows else pd.DataFrame(
        columns=["concurrency", "throughput_tokens_per_sec", "ttft_p99_ms", "tpot_p95_ms", "e2e_p99_ms", "slo_compliance"]
    )

    model_df = pd.DataFrame(model_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        load_df.to_excel(writer, index=False, sheet_name="Load Test Metrics")
        acc_df.to_excel(writer, index=False, sheet_name="Accuracy Metrics")
        model_df.to_excel(writer, index=False, sheet_name="Model Outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="(optional) run preprocessing step (if available)")
    parser.add_argument("--data_dir", default="data/raw", help="Raw data dir (if preprocessing is run)")
    parser.add_argument("--processed_dir", default="data/processed", help="Processed data dir containing jsonl files")
    parser.add_argument("--data_subset", type=int, default=200, help="Subset size for val/test evaluation pools")

    # Model args
    parser.add_argument("--model", dest="model_name", default="meta-llama/Llama-3.1-8B-Instruct", help="HF model name")
    parser.add_argument("--model_name", dest="model_name_alt", default=None, help="(alias) HF model name")
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    parser.add_argument("--variant", default="med", help="server variant (med)")

    # Run config
    parser.add_argument("--prompt_mode", default="slo", choices=["slo", "accuracy"], help="Prompting mode")
    parser.add_argument("--num_requests", type=int, default=10, help="Requests per load test")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1], help="Concurrency levels to test")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")

    # SLO flags
    parser.add_argument("--disable_slo_calibration", action="store_true", help="Skip SLO calibration step")
    parser.add_argument("--skip_load_test", action="store_true", help="Skip load tests (accuracy-only runs)")

    # Calibration settings
    parser.add_argument("--calibration_requests", type=int, default=200, help="Requests for calibration run")
    parser.add_argument("--slo_percentile", type=float, default=95.0, help="Percentile for SLO calibration")
    parser.add_argument("--slo_slack", type=float, default=1.1, help="Slack multiplier applied to calibrated SLOs")

    args = parser.parse_args()

    model_name = args.model_name_alt or args.model_name
    prompt_mode = args.prompt_mode.lower()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("CONFIGURATION:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Variant: {args.variant}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Prompt mode: {prompt_mode}")
    logger.info(f"  Requests per test: {args.num_requests}")
    logger.info(f"  Concurrency levels: {args.concurrencies}")
    logger.info(f"  Data subset: {args.data_subset}")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  SLO calibration disabled: {args.disable_slo_calibration}")
    logger.info(f"  Skip load test: {args.skip_load_test}")
    logger.info("=" * 80)

    # ---------------------------------------------------------------------
    # STEP 1: Load processed data
    # ---------------------------------------------------------------------
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

    # Subset
    val_subset = val_data[: args.data_subset]
    test_subset = test_data[: args.data_subset]
    logger.info(f"Using subset: val={len(val_subset)}, test={len(test_subset)}")

    # ---------------------------------------------------------------------
    # STEP 2: Initialize server
    # ---------------------------------------------------------------------
    logger.info("\n[STEP 2] INITIALIZING SERVER")
    logger.info("-" * 80)

    server = SingleVariantServer(
        model_name=model_name,
        variant=args.variant,
        device=args.device,
        dtype=args.dtype,
    )

    # ---------------------------------------------------------------------
    # STEP 3: (optional) Load tests + SLO calibration
    # ---------------------------------------------------------------------
    slo_thresholds = None
    slo_path = os.path.join(args.output_dir, "slo_thresholds.json")

    load_test_rows: List[Dict[str, Any]] = []

    if args.skip_load_test:
        logger.info("\n[STEP 3] RUNNING LOAD TESTS")
        logger.info("-" * 80)
        logger.info("Skipping load tests (--skip_load_test).")
    else:
        logger.info("\n[STEP 3.5] CALIBRATING SLOs (measurement-only)")
        logger.info("-" * 80)

        if not args.disable_slo_calibration:
            # Calibration run (concurrency=1)
            def infer_one(example: Dict[str, Any]):
                ds = (example.get("dataset") or example.get("dataset_type") or "").lower()
                diff = (example.get("difficulty") or "medium").lower()
                messages, formatted_prompt, max_tokens = build_llama_formatted_prompt(
                    example,
                    dataset_type=ds,
                    prompt_mode="slo",  # always calibrate using SLO mode behavior
                )
                return server.generate(
                    messages=messages,
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    difficulty=diff,
                    dataset_type=ds,
                    prompt_mode="slo",
                    do_sample=False,
                )

            logger.info(f"Calibration run: {args.calibration_requests} requests @ concurrency=1")
            calib_gen = ClosedLoopLoadGenerator(
                inference_func=infer_one,
                max_concurrency=1,
                num_requests=args.calibration_requests,
                data_loader=test_subset,
                prompt_mode="slo",
            )
            calib_metrics = calib_gen.run()

            # Calibrate pXX
            slo_thresholds = calibrate_slos(calib_metrics, percentile=args.slo_percentile)
            logger.info(f"Calibrated SLOs at p{args.slo_percentile}: {slo_thresholds}")
            save_json(slo_path, slo_thresholds)
            logger.info(f"Saved calibrated SLOs to {slo_path}")

            # Apply slack
            slo_thresholds = {
                k: {
                    "ttft_ms": float(v["ttft_ms"]) * float(args.slo_slack),
                    "tpot_ms": float(v["tpot_ms"]) * float(args.slo_slack),
                }
                for k, v in slo_thresholds.items()
            }
            logger.info(f"Calibrated SLOs (p{args.slo_percentile} with slack x{args.slo_slack}): {slo_thresholds}")
        else:
            logger.info("SLO calibration disabled. Will use default SLOs.")
            slo_thresholds = None

        logger.info("\n[STEP 3] RUNNING LOAD TESTS")
        logger.info("-" * 80)

        # Define inference function for load tests
        def infer_one(example: Dict[str, Any]):
            ds = (example.get("dataset") or example.get("dataset_type") or "").lower()
            diff = (example.get("difficulty") or "medium").lower()
            # Use SLO prompt_mode for load tests by default (measuring SLO path)
            messages, formatted_prompt, max_tokens = build_llama_formatted_prompt(
                example,
                dataset_type=ds,
                prompt_mode="slo",
            )
            return server.generate(
                messages=messages,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                difficulty=diff,
                dataset_type=ds,
                prompt_mode="slo",
                do_sample=False,
            )

        for c in args.concurrencies:
            logger.info(f"\n>>> Testing with concurrency={c}")
            gen = ClosedLoopLoadGenerator(
                inference_func=infer_one,
                max_concurrency=c,
                num_requests=args.num_requests,
                data_loader=test_subset,
                prompt_mode="slo",
            )
            req_metrics = gen.run()

            mc = MetricsCalculator(req_metrics, slo_thresholds)
            metrics = mc.compute_all_metrics()
            mc.print_summary(concurrency=c)

            # Save raw metrics
            metrics_path = os.path.join(args.output_dir, f"metrics_concurrency_{c}.json")
            requests_path = os.path.join(args.output_dir, f"requests_concurrency_{c}.jsonl")
            mc.save_metrics(metrics_path)
            gen.save_requests_jsonl(requests_path)

            load_test_rows.append({
                "concurrency": c,
                "throughput_tokens_per_sec": metrics.get("throughput", {}).get("overall_tokens_per_sec", 0.0),
                "ttft_p99_ms": metrics.get("ttft_ms", {}).get("p99", 0.0),
                "tpot_p95_ms": metrics.get("tpot_ms", {}).get("p95", 0.0),
                "e2e_p99_ms": metrics.get("e2e_latency_ms", {}).get("p99", 0.0),
                "slo_compliance": metrics.get("slo_compliance", 0.0),
            })

    # ---------------------------------------------------------------------
    # STEP 4: Evaluate accuracy (uses args.prompt_mode)
    # ---------------------------------------------------------------------
    logger.info("\n[STEP 4] EVALUATING ACCURACY")
    logger.info("-" * 80)

    evaluator = Evaluator(server, prompt_mode=prompt_mode)
    accuracy_results, detailed_rows = evaluator.evaluate(val_subset)

    # Print a compact report
    gsm = accuracy_results.get("GSM8K", {})
    mmlu = accuracy_results.get("MMLU", {})
    overall = accuracy_results.get("OVERALL", {})

    logger.info("\n" + "=" * 70)
    logger.info("GSM8K Results")
    logger.info(f"  Accuracy: {gsm.get('accuracy', 0.0) * 100:.2f}% ({gsm.get('correct', 0)}/{gsm.get('n', 0)})")
    logger.info(f"  Format OK: {gsm.get('format_ok', 0.0) * 100:.2f}% ({int(gsm.get('format_ok',0.0)*gsm.get('n',0))}/{gsm.get('n',0)})")
    logger.info("MMLU Results")
    logger.info(f"  Accuracy: {mmlu.get('accuracy', 0.0) * 100:.2f}% ({mmlu.get('correct', 0)}/{mmlu.get('n', 0)})")
    logger.info(f"  Format OK: {mmlu.get('format_ok', 0.0) * 100:.2f}% ({int(mmlu.get('format_ok',0.0)*mmlu.get('n',0))}/{mmlu.get('n',0)})")
    logger.info("=" * 70)
    logger.info("OVERALL RESULTS")
    logger.info(f"  Accuracy: {overall.get('accuracy', 0.0) * 100:.2f}% ({overall.get('correct', 0)}/{overall.get('n', 0)})")
    logger.info(f"  Format OK: {overall.get('format_ok', 0.0) * 100:.2f}%")
    logger.info("=" * 70)

    # ---------------------------------------------------------------------
    # STEP 5: Save outputs
    # ---------------------------------------------------------------------
    logger.info("\n[STEP 5] SAVING OUTPUTS")
    logger.info("-" * 80)

    eval_path = os.path.join(args.output_dir, "eval_results.json")
    summary_path = os.path.join(args.output_dir, "summary.json")
    excel_path = os.path.join(args.output_dir, "performance_summary.xlsx")

    save_json(eval_path, accuracy_results)
    logger.info(f"Saved evaluation results to {eval_path}")

    summary = {
        "config": {
            "model": model_name,
            "variant": args.variant,
            "device": args.device,
            "dtype": args.dtype,
            "prompt_mode": prompt_mode,
            "requests_per_test": args.num_requests,
            "concurrencies": args.concurrencies,
            "data_subset": args.data_subset,
            "output_dir": args.output_dir,
            "disable_slo_calibration": args.disable_slo_calibration,
            "skip_load_test": args.skip_load_test,
        },
        "accuracy_results": accuracy_results,
        "load_test_results": load_test_rows,
        "slo_thresholds_path": slo_path if os.path.exists(slo_path) else None,
        "slo_thresholds": slo_thresholds,
    }
    save_json(summary_path, summary)
    logger.info(f"Saved overall summary to {summary_path}")

    save_excel(excel_path, accuracy_results, load_test_rows, detailed_rows)
    logger.info(f"Saved Excel summary to {excel_path}")

    logger.info("\n" + "=" * 80)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
