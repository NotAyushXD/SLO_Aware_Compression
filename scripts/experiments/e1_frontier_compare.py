#!/usr/bin/env python
"""E1 (extended): Pareto frontier comparison across multiple routers.

The original e1_delta_sweep.py is a single-config δ sweep.
For a paper figure, we typically want a *single plot* that overlays:
  - our δ-sweep curve, and
  - baseline routers as points (or curves if they respond to δ).

This script runs multiple configs across a shared δ grid and aggregates
mean + 95% CI over seeds.

Output format is compatible with the legacy analysis/make_all.py expectation:
  {
    "labelA": {"points": [{"delta":...,"cost":...,"quality":...,"violation":...}, ...]},
    "labelB": {"points": [...]},
    ...
  }
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.experiments.utils import load_config_command, load_metrics, run_baseline_eval


def _mean_ci(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = float(sum(xs) / float(len(xs)))
    if len(xs) <= 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": len(xs)}
    var = float(sum((x - mean) ** 2 for x in xs) / float(len(xs) - 1))
    se = math.sqrt(var / float(len(xs)))
    z = 1.96
    return {"mean": mean, "ci_low": mean - z * se, "ci_high": mean + z * se, "n": len(xs)}


def _extract(metrics: Dict[str, Any]) -> Dict[str, float]:
    s = metrics.get("summary", {}) or {}

    def _f(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    succ = max(_f(s.get("successful_requests")), 1.0)
    total_cost = _f(s.get("total_cost_units"))
    cost_per_req = _f(s.get("cost_per_request")) if "cost_per_request" in s else total_cost / succ

    slo = _f(s.get("slo_compliance"))
    viol = max(0.0, min(1.0, 1.0 - slo))

    return {
        "cost": float(cost_per_req),
        "quality": _f(s.get("accuracy_success")),
        "violation": float(viol),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="One or more entries of the form 'label:path/to/config.json'.",
    )
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num_requests", type=int, default=400)
    ap.add_argument("--out_root", type=str, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    parsed: List[Tuple[str, str]] = []
    for item in args.configs:
        if ":" not in item:
            raise ValueError(f"Bad --configs entry (expected label:path): {item}")
        label, path = item.split(":", 1)
        parsed.append((label, path))

    out: Dict[str, Any] = {}

    for label, cfg_path in parsed:
        cmd = load_config_command(cfg_path)
        label_dir = out_root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        pts: List[Dict[str, Any]] = []

        for delta in args.deltas:
            per_seed: List[Dict[str, float]] = []
            for seed in args.seeds:
                run_dir = label_dir / f"delta_{delta:g}" / f"seed_{int(seed)}"
                run_dir.mkdir(parents=True, exist_ok=True)

                extra_args = [
                    "--seed",
                    str(int(seed)),
                    "--concurrencies",
                    str(int(args.concurrency)),
                    "--num_requests",
                    str(int(args.num_requests)),
                    # Keep a single δ sweep consistent across routers.
                    # - BanditRouter reads --bandit_delta
                    # - RiskRouter reads --risk_latency_delta
                    "--bandit_delta",
                    str(float(delta)),
                    "--risk_latency_delta",
                    str(float(delta)),
                ]

                print(f"\n=== [E1] {label} delta={delta:g} seed={seed} ===")
                run_baseline_eval(cmd, str(run_dir), extra_args=extra_args)

                mpath = run_dir / f"metrics_concurrency_{int(args.concurrency)}.json"
                if not mpath.exists():
                    raise FileNotFoundError(f"Missing metrics: {mpath}")
                metrics = load_metrics(mpath)
                per_seed.append(_extract(metrics))

            # Aggregate this δ
            cost_ci = _mean_ci([p["cost"] for p in per_seed])
            qual_ci = _mean_ci([p["quality"] for p in per_seed])
            viol_ci = _mean_ci([p["violation"] for p in per_seed])

            agg = {
                "delta": float(delta),
                "cost": float(cost_ci["mean"]),
                "cost_ci_low": float(cost_ci["ci_low"]),
                "cost_ci_high": float(cost_ci["ci_high"]),
                "quality": float(qual_ci["mean"]),
                "quality_ci_low": float(qual_ci["ci_low"]),
                "quality_ci_high": float(qual_ci["ci_high"]),
                "violation": float(viol_ci["mean"]),
                "violation_ci_low": float(viol_ci["ci_low"]),
                "violation_ci_high": float(viol_ci["ci_high"]),
                "n_seeds": int(cost_ci["n"]),
            }
            pts.append(agg)

        out[label] = {"points": pts, "config": str(cfg_path)}

    (out_root / "summary_all.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
