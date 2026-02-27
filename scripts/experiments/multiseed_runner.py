#!/usr/bin/env python
"""Generic multi-seed runner for run_baseline_evaluation.py.

This is a thin wrapper that:
- runs the same config for multiple seeds
- collects the resulting metrics JSON files
- emits mean/CI summary

Example:
  python scripts/experiments/multiseed_runner.py \
    --config configs/always_base.json \
    --seeds 0 1 2 3 4 \
    --out_root outputs/always_base_multiseed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

from scripts.experiments.utils import (
    extract_frontier_point,
    find_metrics_file,
    load_experiment_config,
    load_metrics,
    run_baseline_eval,
)


def mean_ci(xs: List[float], alpha: float = 0.05) -> Dict[str, float]:
    xs = [float(x) for x in xs]
    n = len(xs)
    if n == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n": 0}
    m = float(np.mean(xs))
    if n == 1:
        return {"mean": m, "low": m, "high": m, "n": 1}
    se = float(np.std(xs, ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    return {"mean": m, "low": m - tcrit * se, "high": m + tcrit * se, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument(
        "--extra_args",
        type=str,
        nargs="*",
        default=None,
        help="Extra args appended to the run_baseline_evaluation invocation.",
    )
    args = ap.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]
    cfg = load_experiment_config(args.config)
    base_cmd = list(cfg["command"])

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    per_seed: List[Dict[str, float]] = []

    for seed in args.seeds:
        run_dir = out_root / f"seed_{seed}"
        run_baseline_eval(
            repo_dir=repo_dir,
            base_command=base_cmd,
            out_dir=run_dir,
            seed=int(seed),
            extra_args=list(args.extra_args) if args.extra_args else None,
        )
        metrics = load_metrics(find_metrics_file(run_dir))
        per_seed.append(extract_frontier_point(metrics))

    summary = {
        "config": str(args.config),
        "seeds": [int(s) for s in args.seeds],
        "extra_args": list(args.extra_args) if args.extra_args else None,
        "accuracy": mean_ci([r["accuracy"] for r in per_seed]),
        "cost_per_request": mean_ci([r["cost_per_request"] for r in per_seed]),
        "violation_rate": mean_ci([r["violation_rate"] for r in per_seed]),
        "per_seed": per_seed,
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
