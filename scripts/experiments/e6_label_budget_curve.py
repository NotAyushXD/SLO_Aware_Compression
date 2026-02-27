#!/usr/bin/env python
"""E6: label budget curve for contextual bandits.

Sweep bandit_label_budget_p and plot quality vs label rate while monitoring risk.

Usage:
  python scripts/experiments/e6_label_budget_curve.py \
    --config configs/bandit_base.json \
    --label_ps 0.01 0.02 0.05 0.1 1.0 \
    --seeds 0 1 2 3 4 \
    --out_root outputs/e6_label_budget
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
    ap.add_argument("--label_ps", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_root", type=str, required=True)
    args = ap.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]
    cfg = load_experiment_config(args.config)
    base_cmd = list(cfg["command"])

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for p in args.label_ps:
        per_seed = []
        per_seed_bandit_label_used = []
        for seed in args.seeds:
            run_dir = out_root / f"label_p_{p:.4f}" / f"seed_{seed}"
            run_baseline_eval(
                repo_dir=repo_dir,
                base_command=base_cmd,
                out_dir=run_dir,
                seed=int(seed),
                extra_args=["--bandit_label_budget_p", str(float(p))],
            )
            mpath = find_metrics_file(run_dir)
            metrics = load_metrics(mpath)
            per_seed.append(extract_frontier_point(metrics))
            # pull observed label-used rate if present
            try:
                per_seed_bandit_label_used.append(float((metrics.get("summary") or {}).get("bandit_quality_label_used_rate", 0.0) or 0.0))
            except Exception:
                per_seed_bandit_label_used.append(0.0)

        agg = {
            "label_budget_p": float(p),
            "accuracy": mean_ci([r["accuracy"] for r in per_seed]),
            "cost_per_request": mean_ci([r["cost_per_request"] for r in per_seed]),
            "violation_rate": mean_ci([r["violation_rate"] for r in per_seed]),
            "observed_label_used_rate": mean_ci(per_seed_bandit_label_used),
            "per_seed": per_seed,
        }
        results.append(agg)
        (out_root / f"summary_label_p_{p:.4f}.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Plot: accuracy vs observed label rate (mean)
    xs = [r["observed_label_used_rate"]["mean"] for r in results]
    ys = [r["accuracy"]["mean"] for r in results]

    plt.figure()
    plt.scatter(xs, ys)
    for r, x, y in zip(results, xs, ys):
        plt.annotate(f"p={r['label_budget_p']:g}", (x, y))
    plt.xlabel("observed label-used rate")
    plt.ylabel("accuracy")
    plt.title("E6: quality vs label rate")
    plt.tight_layout()
    plt.savefig(out_root / "label_budget_curve.png", dpi=200)
    plt.close()

    (out_root / "summary_all.json").write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
