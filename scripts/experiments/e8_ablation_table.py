#!/usr/bin/env python
"""E8: Ablation table runner for BanditRouter.

This is not strictly required by the patch bundle, but it is a *paper-grade*
experiment harness for the blueprint's Table-2 style ablations.

It runs a base config + multiple ablation flags across multiple seeds,
aggregates mean + 95% CI, and writes JSON + CSV.

Example:
  python scripts/experiments/e8_ablation_table.py \
    --config configs/bandit_base.json \
    --concurrency 4 --num_requests 400 \
    --seeds 0 1 2 3 4 \
    --out_root outputs/e8_ablations
"""

from __future__ import annotations

import argparse
import csv
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

    ttft = metrics.get("ttft", {}) or {}
    e2e = metrics.get("e2e_latency", {}) or {}

    return {
        "accuracy": _f(s.get("accuracy_success")),
        "violation_rate": 1.0 - _f(s.get("slo_compliance")),
        "cost_per_request": _f(s.get("cost_per_request"))
        if "cost_per_request" in s
        else (_f(s.get("total_cost_units")) / max(_f(s.get("successful_requests")), 1.0)),
        "cost_per_goodput_request": _f(s.get("cost_per_goodput_request")),
        "cost_per_qa_goodput_request": _f(s.get("cost_per_quality_adjusted_goodput_request")),
        "p99_ttft_ms": _f(ttft.get("p99")),
        "p99_e2e_ms": _f(e2e.get("p99")),
        "false_accept_rate_cheap": _f(s.get("false_accept_rate_cheap")),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num_requests", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Optional override for --bandit_delta (appended to the command).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_cmd = load_config_command(args.config)

    conditions: List[Tuple[str, List[str]]] = [
        ("ours", []),
        ("no_sys_state", ["--bandit_disable_system_features"]),
        ("no_adapter_state", ["--bandit_disable_adapter_features"]),
        ("no_fallback", ["--bandit_disable_conservative_fallback"]),
        ("no_overhead_cost", ["--bandit_disable_overhead_cost"]),
        ("no_primal_dual", ["--bandit_disable_primal_dual"]),
        ("frozen_policy", ["--bandit_force_freeze"]),
    ]

    rows: List[Dict[str, Any]] = []
    for label, extra in conditions:
        pts: List[Dict[str, float]] = []
        for seed in args.seeds:
            run_dir = out_root / label / f"seed_{int(seed)}"
            run_dir.mkdir(parents=True, exist_ok=True)

            extra_args = [
                "--concurrencies",
                str(int(args.concurrency)),
                "--num_requests",
                str(int(args.num_requests)),
                "--seed",
                str(int(seed)),
            ] + list(extra)

            if args.delta is not None:
                extra_args += ["--bandit_delta", str(float(args.delta))]

            print(f"\n=== [E8] {label} seed={seed} ===")
            run_baseline_eval(base_cmd, str(run_dir), extra_args=extra_args)

            mpath = run_dir / f"metrics_concurrency_{int(args.concurrency)}.json"
            if not mpath.exists():
                raise FileNotFoundError(f"Missing metrics: {mpath}")
            metrics = load_metrics(mpath)
            pts.append(_extract(metrics))

        # Aggregate across seeds
        agg: Dict[str, Any] = {
            "condition": label,
            "n_seeds": len(pts),
            "concurrency": int(args.concurrency),
            "num_requests": int(args.num_requests),
        }
        for k in pts[0].keys() if pts else []:
            agg[k] = _mean_ci([float(p.get(k, 0.0)) for p in pts])
        rows.append(agg)

        # Save per-condition
        (out_root / label / "summary_ci.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Save combined
    out_json = out_root / "ablation_table.json"
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")

    out_csv = out_root / "ablation_table.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = [
            "condition",
            "accuracy_mean",
            "accuracy_ci_low",
            "accuracy_ci_high",
            "violation_mean",
            "violation_ci_low",
            "violation_ci_high",
            "cost_per_request_mean",
            "cost_per_request_ci_low",
            "cost_per_request_ci_high",
            "p99_e2e_ms_mean",
            "p99_e2e_ms_ci_low",
            "p99_e2e_ms_ci_high",
            "false_accept_rate_cheap_mean",
        ]
        writer.writerow(header)
        for r in rows:
            acc = r.get("accuracy", {}) or {}
            viol = r.get("violation_rate", {}) or {}
            cost = r.get("cost_per_request", {}) or {}
            p99 = r.get("p99_e2e_ms", {}) or {}
            far = r.get("false_accept_rate_cheap", {}) or {}
            writer.writerow(
                [
                    r.get("condition"),
                    acc.get("mean"),
                    acc.get("ci_low"),
                    acc.get("ci_high"),
                    viol.get("mean"),
                    viol.get("ci_low"),
                    viol.get("ci_high"),
                    cost.get("mean"),
                    cost.get("ci_low"),
                    cost.get("ci_high"),
                    p99.get("mean"),
                    p99.get("ci_low"),
                    p99.get("ci_high"),
                    far.get("mean"),
                ]
            )


if __name__ == "__main__":
    main()
