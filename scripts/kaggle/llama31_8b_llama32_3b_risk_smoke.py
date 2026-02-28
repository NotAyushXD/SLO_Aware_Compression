"""Kaggle-friendly end-to-end smoke for the RiskRouter pipeline.

This script is intentionally small/fast and is designed for Kaggle T4 GPUs.

It runs:
  1) Trace collection (train_learned_router.py --collect_only)
  2) RiskRouter training (train_risk_router.py)
  3) A tiny multi-variant evaluation (run_baseline_evaluation.py)

Model mix (paper-facing "portfolio"):
  - base, med: meta-llama/Llama-3.1-8B-Instruct (different quantization)
  - cheap:    meta-llama/Llama-3.2-3B (or -Instruct)

Example:
  python scripts/kaggle/llama31_8b_llama32_3b_risk_smoke.py \
    --outroot /kaggle/working/router_models/kaggle_smoke

Notes:
  - You must have HF auth configured for gated Llama models.
  - If you OOM on a T4, set --max_loaded_variants 1 or 2.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HF model id for BASE+MED (default: Llama-3.1-8B-Instruct)",
    )
    p.add_argument(
        "--cheap_model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="HF model id for CHEAP (default: Llama-3.2-3B)",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="auto")

    p.add_argument("--base_quantization", type=str, default="fp16", choices=["fp16", "bf16", "int8", "int4", "none"])
    p.add_argument("--med_quantization", type=str, default="int8", choices=["fp16", "bf16", "int8", "int4", "none"])
    p.add_argument("--cheap_quantization", type=str, default="fp16", choices=["fp16", "bf16", "int8", "int4", "none"])

    p.add_argument("--outroot", type=str, required=True)
    p.add_argument("--processed_dir", type=str, default="data/processed")

    p.add_argument("--max_examples", type=int, default=60)
    p.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2])
    p.add_argument("--multi_variants", nargs="+", type=str, default=["cheap", "base"])

    p.add_argument("--max_batch_size", type=int, default=4)
    p.add_argument("--batch_wait_ms", type=int, default=8)

    # Residency / swap knobs (important on T4)
    p.add_argument("--load_strategy", type=str, default="auto")
    p.add_argument("--max_loaded_variants", type=int, default=None)
    p.add_argument("--preload_variants", nargs="*", default=None)
    p.add_argument("--warmup", action="store_true")

    # Risk-router knobs
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.20)
    p.add_argument("--min_rows_per_variant", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    outroot = Path(args.outroot)
    trace_root = outroot / "trace"
    bundle_dir = outroot / "risk_router_bundle"
    eval_dir = outroot / "multi_risk"
    trace_root.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect traces
    run(
        [
            "python",
            "scripts/train_learned_router.py",
            "--model",
            args.base_model,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--cheap_model",
            args.cheap_model,
            "--base_quantization",
            args.base_quantization,
            "--med_quantization",
            args.med_quantization,
            "--cheap_quantization",
            args.cheap_quantization,
            "--processed_dir",
            args.processed_dir,
            "--prompt_mode",
            "slo",
            "--output_root",
            str(trace_root),
            "--variants",
            *args.multi_variants,
            "--concurrencies",
            *[str(c) for c in args.concurrencies],
            "--max_examples",
            str(args.max_examples),
            "--max_batch_size",
            str(args.max_batch_size),
            "--batch_wait_ms",
            str(args.batch_wait_ms),
            "--collect_only",
            "--load_strategy",
            str(args.load_strategy),
        ]
        + (["--max_loaded_variants", str(args.max_loaded_variants)] if args.max_loaded_variants is not None else [])
        + (["--preload_variants", *args.preload_variants] if args.preload_variants else [])
        + (["--warmup"] if args.warmup else [])
    )

    trace_jsonl = trace_root / "trainval_traces.jsonl"
    if not trace_jsonl.exists():
        raise RuntimeError(f"Trace JSONL missing: {trace_jsonl}")

    # 2) Train risk router
    run(
        [
            "python",
            "scripts/train_risk_router.py",
            "--trace_jsonl",
            str(trace_jsonl),
            "--output_dir",
            str(bundle_dir),
            "--min_rows_per_variant",
            str(args.min_rows_per_variant),
        ]
    )

    # 3) Small multi-variant eval
    run(
        [
            "python",
            "run_baseline_evaluation.py",
            "--backend",
            "hf",
            "--service",
            "multi",
            "--router_mode",
            "risk",
            "--risk_router_dir",
            str(bundle_dir),
            "--risk_latency_delta",
            str(args.delta),
            "--risk_quality_epsilon",
            str(args.epsilon),
            "--dispatcher_policy",
            "edf",
            "--model",
            args.base_model,
            "--cheap_model",
            args.cheap_model,
            "--base_quantization",
            args.base_quantization,
            "--med_quantization",
            args.med_quantization,
            "--cheap_quantization",
            args.cheap_quantization,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--multi_variants",
            *args.multi_variants,
            "--prompt_mode",
            "slo",
            "--skip_accuracy_eval",
            "--num_requests",
            "10",
            "--concurrencies",
            *[str(c) for c in args.concurrencies],
            "--max_batch_size",
            str(args.max_batch_size),
            "--batch_wait_ms",
            str(args.batch_wait_ms),
            "--output_dir",
            str(eval_dir),
            "--load_strategy",
            str(args.load_strategy),
        ]
        + (["--max_loaded_variants", str(args.max_loaded_variants)] if args.max_loaded_variants is not None else [])
        + (["--preload_variants", *args.preload_variants] if args.preload_variants else [])
        + (["--warmup"] if args.warmup else [])
    )

    print("\n[OK] Risk-router smoke complete.")
    print("Trace root:", trace_root)
    print("Bundle dir:", bundle_dir)
    print("Eval dir:", eval_dir)


if __name__ == "__main__":
    main()
