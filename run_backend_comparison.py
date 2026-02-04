#!/usr/bin/env python3
# run_backend_comparison.py
"""Run HF vs vLLM comparisons for paper tables.

This script is a convenience wrapper around run_baseline_evaluation.py that:
- Runs the same workload under:
    1) HF (no micro-batching)
    2) HF (micro-batching enabled)
    3) vLLM (continuous batching; HF micro-batching flags ignored)
- Aggregates the resulting metrics_concurrency_*.json into a single CSV/JSON summary.

Example (load test, SLO prompts):
python run_backend_comparison.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant base \
  --prompt_mode slo \
  --num_requests 500 \
  --concurrencies 1 4 \
  --max_batch_size 8 \
  --batch_wait_ms 8 \
  --seed 42 \
  --output_dir ./runs/compare_hf_vs_vllm_fp16
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


@dataclass
class RunSpec:
    name: str
    backend: str
    batching_mode: str  # no_batch / micro_batch / vllm_continuous
    extra_args: List[str]


def _run_one(spec: RunSpec, base_args: List[str], out_dir: str, dry_run: bool) -> str:
    run_dir = os.path.join(out_dir, spec.name)
    cmd = ["python", "run_baseline_evaluation.py"] + base_args + spec.extra_args + ["--output_dir", run_dir]
    print("\n" + "=" * 80)
    print(f"RUN: {spec.name}")
    print("CMD:", " ".join(cmd))
    print("=" * 80)
    if not dry_run:
        subprocess.check_call(cmd)
    return run_dir


def _collect_metrics(run_dir: str, backend: str, batching_mode: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Find all metrics_concurrency_*.json
    for fn in sorted(os.listdir(run_dir)):
        if not fn.startswith("metrics_concurrency_") or not fn.endswith(".json"):
            continue
        conc = fn.replace("metrics_concurrency_", "").replace(".json", "")
        path = os.path.join(run_dir, fn)
        m = _read_json(path)

        row: Dict[str, Any] = {
            "backend": backend,
            "batching_mode": batching_mode,
            "concurrency": int(conc),
            "run_dir": run_dir,
        }

        # Percentile blocks produced by MetricsCalculator
        for metric_key in ["ttft", "tpot", "e2e_latency", "queue_wait"]:
            blk = m.get(metric_key)
            if not isinstance(blk, dict):
                continue
            for k in ["p50", "p90", "p95", "p99", "mean"]:
                if k in blk:
                    row[f"{metric_key}_{k}"] = blk[k]

        # Summary (throughput + SLO compliance)
        summ = m.get("summary", {}) if isinstance(m, dict) else {}
        if isinstance(summ, dict):
            for k in ["success_rate", "throughput_tokens_per_sec", "slo_compliance", "escalation_rate"]:
                if k in summ:
                    row[k] = summ[k]

        rows.append(row)

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--variant", default="base", choices=["base", "med", "cheap"])
    p.add_argument("--prompt_mode", default="slo", choices=["slo", "accuracy"])
    p.add_argument("--num_requests", type=int, default=200)
    p.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4])
    p.add_argument("--max_batch_size", type=int, default=8)
    p.add_argument("--batch_wait_ms", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dry_run", action="store_true")

    # vLLM optional knobs
    p.add_argument("--vllm_model_override", type=str, default=None)
    p.add_argument("--vllm_quantization", type=str, default=None)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--vllm_max_model_len", type=int, default=4096)
    p.add_argument("--vllm_max_num_seqs", type=int, default=128)
    p.add_argument("--vllm_enforce_eager", action="store_true")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    base_args = [
        "--model", args.model,
        "--variant", args.variant,
        "--prompt_mode", args.prompt_mode,
        "--skip_accuracy_eval",
        "--num_requests", str(args.num_requests),
        "--concurrencies", *[str(c) for c in args.concurrencies],
        "--seed", str(args.seed),
    ]

    # HF (no micro-batching)
    hf_no = RunSpec(
        name="hf_no_batch",
        backend="hf",
        batching_mode="no_batch",
        extra_args=["--backend", "hf", "--disable_batching"],
    )

    # HF (micro-batching)
    hf_mb = RunSpec(
        name="hf_micro_batch",
        backend="hf",
        batching_mode="micro_batch",
        extra_args=[
            "--backend", "hf",
            "--enable_batching",
            "--max_batch_size", str(args.max_batch_size),
            "--batch_wait_ms", str(args.batch_wait_ms),
        ],
    )

    # vLLM (continuous batching)
    vllm_args = ["--backend", "vllm"]
    if args.vllm_model_override:
        vllm_args += ["--vllm_model_override", args.vllm_model_override]
    if args.vllm_quantization:
        vllm_args += ["--vllm_quantization", args.vllm_quantization]
    vllm_args += [
        "--vllm_gpu_memory_utilization", str(args.vllm_gpu_memory_utilization),
        "--vllm_max_model_len", str(args.vllm_max_model_len),
        "--vllm_max_num_seqs", str(args.vllm_max_num_seqs),
    ]
    if args.vllm_enforce_eager:
        vllm_args.append("--vllm_enforce_eager")

    vllm = RunSpec(
        name="vllm_continuous",
        backend="vllm",
        batching_mode="continuous",
        extra_args=vllm_args,
    )

    specs = [hf_no, hf_mb, vllm]

    run_dirs = []
    for spec in specs:
        run_dirs.append(_run_one(spec, base_args, args.output_dir, dry_run=args.dry_run))

    # Collect + write summary
    all_rows: List[Dict[str, Any]] = []
    for spec, rd in zip(specs, run_dirs):
        if os.path.isdir(rd):
            all_rows.extend(_collect_metrics(rd, backend=spec.backend, batching_mode=spec.batching_mode))

    _write_json(os.path.join(args.output_dir, "backend_comparison_summary.json"), all_rows)
    _write_csv(os.path.join(args.output_dir, "backend_comparison_summary.csv"), all_rows)

    print("\nWrote:")
    print(" -", os.path.join(args.output_dir, "backend_comparison_summary.json"))
    print(" -", os.path.join(args.output_dir, "backend_comparison_summary.csv"))


if __name__ == "__main__":
    main()
