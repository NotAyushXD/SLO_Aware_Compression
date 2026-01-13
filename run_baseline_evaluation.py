# run_baseline_evaluation.py
"""
End-to-end baseline evaluation orchestration
Runs complete pipeline: preprocessing → server → load tests → evaluation
"""

import json
import argparse
import os
from pathlib import Path
import logging
import time

from preprocessing import DataPreprocessor
from server import SingleVariantServer
from load_generator import ClosedLoopLoadGenerator
from metrics import MetricsCalculator
from evaluation import HeldOutEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_dir: str = "data/processed"):
    """Load preprocessed datasets from JSONL files"""
    train_data = []
    val_data = []
    test_data = []
    
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


def main(args):
    """Run end-to-end baseline evaluation pipeline"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("END-TO-END BASELINE EVALUATION: MED-ONLY SERVER (8-BIT QUANTIZATION)")
    logger.info("="*80)
    
    # Step 1: Data preprocessing (if needed)
    if args.preprocess:
        logger.info("\n[STEP 0] PREPROCESSING DATASETS")
        logger.info("-"*80)
        
        preprocessor = DataPreprocessor(
            data_dir=args.data_dir,
            output_dir=args.processed_dir
        )
        train, val, test = preprocessor.run_pipeline()
    
    # Step 2: Load preprocessed data
    logger.info("\n[STEP 1] LOADING DATA")
    logger.info("-"*80)
    
    train_data, val_data, test_data = load_data(args.processed_dir)
    
    if not val_data or not test_data:
        logger.error("No validation or test data found!")
        return
    
    # Use subset for faster iteration (optional)
    if args.data_subset > 0:
        val_data = val_data[:args.data_subset]
        test_data = test_data[:args.data_subset]
        logger.info(f"Using subset: val={len(val_data)}, test={len(test_data)}")
    
    # Step 3: Initialize server
    logger.info("\n[STEP 2] INITIALIZING SERVER")
    logger.info("-"*80)
    
    try:
        server = SingleVariantServer(
            model_name=args.model_name,
            variant="med",
            device=args.device,
            dtype=args.dtype
        )
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        return
    
    # Step 4: Run load tests at multiple concurrency levels
    logger.info("\n[STEP 3] RUNNING LOAD TESTS")
    logger.info("-"*80)
    
    load_test_results = {}
    all_metrics_summary = []
    
    for concurrency in args.concurrencies:
        logger.info(f"\n>>> Testing with concurrency={concurrency}")
        
        # Create load generator
        load_gen = ClosedLoopLoadGenerator(
            inference_func=server.generate,
            max_concurrency=concurrency,
            num_requests=args.num_requests,
            data_loader=val_data
        )
        
        # Run load test
        start_time = time.time()
        metrics = load_gen.run()
        load_duration = time.time() - start_time
        
        # Calculate metrics
        calc = MetricsCalculator(metrics)
        test_metrics = calc.compute_all_metrics()
        load_test_results[concurrency] = test_metrics
        
        # Print report
        calc.print_report(
            title=f"LOAD TEST RESULTS (Concurrency {concurrency})"
        )
        
        # Save detailed metrics to JSON
        metrics_file = os.path.join(
            args.output_dir,
            f"metrics_concurrency_{concurrency}.json"
        )
        calc.save_metrics(metrics_file)
        
        # Save individual request logs
        requests_file = os.path.join(
            args.output_dir,
            f"requests_concurrency_{concurrency}.jsonl"
        )
        load_gen.save_metrics(requests_file)
        
        # Summary entry
        summary_entry = {
            "concurrency": concurrency,
            "num_requests": args.num_requests,
            "duration_sec": load_duration,
            "success_rate": test_metrics["summary"]["success_rate"],
            "throughput_tokens_per_sec": test_metrics["summary"]["throughput_tokens_per_sec"],
            "ttft_p99_ms": test_metrics["ttft"]["p99"],
            "tpot_p95_ms": test_metrics["tpot"]["p95"],
            "e2e_p99_ms": test_metrics["e2e_latency"]["p99"],
            "slo_compliance": test_metrics["summary"]["slo_compliance"],
            "slo_violations": test_metrics["summary"]["slo_violations"]
        }
        all_metrics_summary.append(summary_entry)
    
    # Step 5: Evaluate accuracy on held-out test set
    logger.info("\n[STEP 4] EVALUATING ACCURACY")
    logger.info("-"*80)
    
    try:
        evaluator = HeldOutEvaluator(
            model=server,
            data_loader=test_data,
            batch_size=32
        )
        eval_results = evaluator.evaluate()
        
        # Save eval results
        eval_file = os.path.join(args.output_dir, "eval_results.json")
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {eval_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        eval_results = {}
    
    # Step 6: Summary report
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY REPORT")
    logger.info("="*80)
    
    print("\nLoad Test Results by Concurrency:")
    print(f"{'Concurrency':<12} {'Throughput':<18} {'TTFT P99':<12} {'TPOT P95':<12} {'E2E P99':<12} {'SLO Compl':<10}")
    print("-" * 80)
    
    for summary in all_metrics_summary:
        print(f"{summary['concurrency']:<12} "
              f"{summary['throughput_tokens_per_sec']:<18.1f} "
              f"{summary['ttft_p99_ms']:<12.1f} "
              f"{summary['tpot_p95_ms']:<12.1f} "
              f"{summary['e2e_p99_ms']:<12.1f} "
              f"{summary['slo_compliance']*100:<10.1f}%")
    
    print("\nAccuracy Results:")
    if eval_results:
        for dataset_type in sorted(eval_results.keys()):
            if dataset_type != "overall":
                result = eval_results[dataset_type]
                print(f"  {dataset_type.upper():<10s}: {result['em']*100:6.2f}% "
                      f"({result['correct_count']}/{result['total_count']})")
        
        overall = eval_results.get("overall", {})
        print(f"  {'OVERALL':<10s}: {overall.get('em', 0)*100:6.2f}% "
              f"({overall.get('correct_count', 0)}/{overall.get('total_count', 0)})")
    else:
        print("  No evaluation results available")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "load_test_summary": all_metrics_summary,
            "eval_results": eval_results,
            "config": {
                "model_name": args.model_name,
                "variant": "med",
                "num_requests": args.num_requests,
                "concurrencies": args.concurrencies,
                "device": args.device
            }
        }, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("BASELINE EVALUATION COMPLETE")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)
    logger.info("\nFiles generated:")
    logger.info(f"  - summary.json (overall summary)")
    logger.info(f"  - eval_results.json (accuracy metrics)")
    for concurrency in args.concurrencies:
        logger.info(f"  - metrics_concurrency_{concurrency}.json")
        logger.info(f"  - requests_concurrency_{concurrency}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end baseline MED-only server evaluation"
    )
    
    # Data configuration
    parser.add_argument("--preprocess", action="store_true",
                       help="Run preprocessing (download and process datasets)")
    parser.add_argument("--data_dir", default="data/raw",
                       help="Raw data directory")
    parser.add_argument("--processed_dir", default="data/processed",
                       help="Processed data directory")
    parser.add_argument("--data_subset", type=int, default=0,
                       help="Use subset of data (0=all, >0=limit to N examples)")
    
    # Model configuration
    parser.add_argument("--model_name", 
                       default="meta-llama/Llama-2-7b-chat-hf",
                       help="HuggingFace model name")
    parser.add_argument("--device", default="cuda",
                       help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--dtype", default="auto",
                       help="Data type: 'auto', 'float16', 'bfloat16'")
    
    # Load test configuration
    parser.add_argument("--num_requests", type=int, default=5000,
                       help="Number of requests per concurrency level")
    parser.add_argument("--concurrencies", type=int, nargs="+",
                       default=[1, 2, 4, 8, 16, 32],
                       help="Concurrency levels to test")
    
    # Output configuration
    parser.add_argument("--output_dir", default="results/baseline_med",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Requests per test: {args.num_requests}")
    logger.info(f"  Concurrency levels: {args.concurrencies}")
    logger.info(f"  Output dir: {args.output_dir}")
    
    main(args)
