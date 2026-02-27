#!/usr/bin/env python
"""E1: delta sweep for the primal-dual bandit.

Runs multiple deltas, emits per-delta JSON summaries and a cost/quality frontier plot.

Usage:
  python scripts/experiments/e1_delta_sweep.py \
    --config configs/bandit_base.json \
    --deltas 0.01 0.02 0.05 0.1 \
    --seeds 0 1 2 3 4 \
    --out_root outputs/e1_delta_sweep

The config file is a JSON with a "command" list of args for run_baseline_evaluation.py.
The script overrides --bandit_delta and --seed/--output_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
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
    ap.add_argument("--deltas", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_root", type=str, required=True)
    args = ap.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]
    cfg = load_experiment_config(args.config)
    base_cmd = list(cfg["command"])

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for delta in args.deltas:
        per_seed = []
        for seed in args.seeds:
            run_dir = out_root / f"delta_{delta:.4f}" / f"seed_{seed}"
            run_baseline_eval(
                repo_dir=repo_dir,
                base_command=base_cmd,
                out_dir=run_dir,
                seed=int(seed),
                extra_args=["--bandit_delta", str(float(delta))],
            )
            mpath = find_metrics_file(run_dir)
            metrics = load_metrics(mpath)
            per_seed.append(extract_frontier_point(metrics))

        agg = {
            "delta": float(delta),
            "accuracy": mean_ci([r["accuracy"] for r in per_seed]),
            "cost_per_request": mean_ci([r["cost_per_request"] for r in per_seed]),
            "violation_rate": mean_ci([r["violation_rate"] for r in per_seed]),
            "slo_compliance": mean_ci([r["slo_compliance"] for r in per_seed]),
            "per_seed": per_seed,
        }
        results.append(agg)
        (out_root / f"summary_delta_{delta:.4f}.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Frontier plot (mean)
    deltas = [r["delta"] for r in results]
    xs = [r["cost_per_request"]["mean"] for r in results]
    ys = [r["accuracy"]["mean"] for r in results]

    plt.figure()
    plt.scatter(xs, ys)
    for d, x, y in zip(deltas, xs, ys):
        plt.annotate(f"{d:g}", (x, y))
    plt.xlabel("cost per request (token-equivalent units)")
    plt.ylabel("accuracy")
    plt.title("E1: delta sweep frontier")
    plt.tight_layout()
    plt.savefig(out_root / "frontier_delta_sweep.png", dpi=200)
    plt.close()

    (out_root / "summary_all.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
