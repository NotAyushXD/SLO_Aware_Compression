#!/usr/bin/env python
"""E3: Offered-load sweep (concurrency sweep) + multi-seed aggregation.

Blueprint / execution-plan requirement:
  - Vary offered load (concurrency) under a fixed workload
  - Plot p99 latency and violation rate vs load, plus cost vs load
  - Run multiple seeds and report mean + 95% CI

This script is a thin orchestration wrapper around run_baseline_evaluation.py.

Example (Kaggle / T4 smoke):

  python scripts/experiments/e3_offered_load_sweep.py \
    --configs ours=configs/bandit_base.json \
    --concurrencies 1 2 4 8 \
    --seeds 0 1 2 3 4 \
    --num_requests 200 \
    --out_root outputs/e3_offered_load

To add the required ablation (no system-state features):

  python scripts/experiments/e3_offered_load_sweep.py \
    --configs ours=configs/bandit_base.json \
    --add_no_system_state_ablation \
    ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from scripts.experiments.utils import extract_frontier_point, load_config_command, run_baseline_eval


def _mean_ci(xs: List[float]) -> Tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    mean = sum(xs) / float(len(xs))
    if len(xs) <= 1:
        return mean, mean, mean
    var = sum((x - mean) ** 2 for x in xs) / float(len(xs) - 1)
    se = math.sqrt(var / float(len(xs)))
    z = 1.96
    return mean, mean - z * se, mean + z * se


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_point(metrics: Dict[str, Any], conc: int) -> Dict[str, float]:
    pt = extract_frontier_point(metrics)
    pt["concurrency"] = float(conc)
    return pt


def _parse_configs(items: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for it in items:
        if "=" in it:
            name, path = it.split("=", 1)
            out.append((name.strip(), path.strip()))
        else:
            p = Path(it)
            out.append((p.stem, str(p)))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="One or more configs: label=path/to/config.json (or a bare path).",
    )
    ap.add_argument("--concurrencies", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--num_requests", type=int, default=200)
    ap.add_argument("--out_root", type=str, required=True)

    # Convenience flag for the paper-required ablation: remove system-state features.
    ap.add_argument(
        "--add_no_system_state_ablation",
        action="store_true",
        help="If set, also runs an ablation condition per config with --bandit_disable_system_features.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = _parse_configs(list(args.configs))
    if args.add_no_system_state_ablation:
        ablated: List[Tuple[str, str]] = []
        for label, cfg in configs:
            ablated.append((label, cfg))
            ablated.append((f"{label}__no_sys", cfg))
        configs = ablated

    results: Dict[str, Any] = {}

    for label, cfg_path in configs:
        cmd = load_config_command(cfg_path)
        label_root = out_root / label
        label_root.mkdir(parents=True, exist_ok=True)

        agg_by_conc: Dict[int, Dict[str, List[float]]] = {}

        for conc in args.concurrencies:
            conc = int(conc)
            for seed in args.seeds:
                run_dir = label_root / f"concurrency_{conc}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                extra_args = [
                    "--concurrencies",
                    str(conc),
                    "--num_requests",
                    str(int(args.num_requests)),
                    "--seed",
                    str(int(seed)),
                ]
                if args.add_no_system_state_ablation and label.endswith("__no_sys"):
                    extra_args += ["--bandit_disable_system_features"]

                print(f"\n=== [E3] {label} conc={conc} seed={seed} ===")
                run_baseline_eval(cmd, str(run_dir), extra_args=extra_args)

                metrics_path = run_dir / f"metrics_concurrency_{conc}.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(f"Missing metrics: {metrics_path}")
                m = _read_json(metrics_path)
                pt = _extract_point(m, conc)

                agg_by_conc.setdefault(conc, {}).setdefault("p99_e2e_ms", []).append(pt["p99_e2e_ms"])
                agg_by_conc.setdefault(conc, {}).setdefault("violation_rate", []).append(pt["violation_rate"])
                agg_by_conc.setdefault(conc, {}).setdefault("cost_per_request", []).append(pt["cost_per_request"])
                agg_by_conc.setdefault(conc, {}).setdefault("accuracy", []).append(pt["accuracy"])

        # Aggregate across seeds -> mean/CI per concurrency.
        summary: Dict[str, Any] = {"label": label, "config": cfg_path, "num_requests": int(args.num_requests)}
        rows: List[Dict[str, Any]] = []
        for conc in sorted(agg_by_conc.keys()):
            row: Dict[str, Any] = {"concurrency": int(conc)}
            for k, xs in agg_by_conc[conc].items():
                mu, lo, hi = _mean_ci([float(x) for x in xs])
                row[k] = {"mean": mu, "ci_low": lo, "ci_high": hi, "n": len(xs)}
            rows.append(row)
        summary["by_concurrency"] = rows
        results[label] = summary

        with (label_root / "summary_by_concurrency.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Write combined summary
    with (out_root / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot: one figure per metric (line per label)
    def _plot_metric(metric: str, ylabel: str, fname: str) -> None:
        plt.figure()
        for label, summ in results.items():
            xs: List[float] = []
            ys: List[float] = []
            ylo: List[float] = []
            yhi: List[float] = []
            for r in summ.get("by_concurrency", []):
                xs.append(float(r["concurrency"]))
                m = r.get(metric, {})
                ys.append(float((m.get("mean") or 0.0)))
                ylo.append(float((m.get("ci_low") or 0.0)))
                yhi.append(float((m.get("ci_high") or 0.0)))
            if xs:
                plt.plot(xs, ys, marker="o", label=label)
                # CI band (lightweight)
                try:
                    plt.fill_between(xs, ylo, yhi, alpha=0.15)
                except Exception:
                    pass
        plt.xlabel("concurrency")
        plt.ylabel(ylabel)
        plt.title(f"E3 offered-load sweep: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_root / fname, dpi=200)
        plt.close()

    _plot_metric("p99_e2e_ms", "p99 E2E latency (ms)", "e3_p99_e2e_vs_load.png")
    _plot_metric("violation_rate", "violation rate", "e3_violation_vs_load.png")
    _plot_metric("cost_per_request", "cost per request (cost_units)", "e3_cost_vs_load.png")


if __name__ == "__main__":
    main()
