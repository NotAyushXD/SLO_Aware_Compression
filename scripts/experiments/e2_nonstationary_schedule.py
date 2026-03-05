#!/usr/bin/env python
"""E2: Nonstationary offered-load schedule.

Execution-plan requirement:
  - Run a *single* evaluation where offered load changes over time.
  - Produce time-series plots of violation rate / latency / cost.

This script drives run_baseline_evaluation.py via its --concurrency_schedule
support and (optionally) aggregates multiple seeds.

Example:
  python scripts/experiments/e2_nonstationary_schedule.py \
    --config configs/bandit_base.json \
    --schedule "1:200,8:200,2:200" \
    --seeds 0 1 2 3 4 \
    --out_root outputs/e2_nonstationary
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scripts.experiments.utils import extract_frontier_point, load_config_command, run_baseline_eval


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument(
        "--schedule",
        type=str,
        required=True,
        help="Concurrency schedule '<conc>:<nreq>,<conc>:<nreq>,...' (commas or semicolons).",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument(
        "--primary_seed",
        type=int,
        default=0,
        help="Seed whose schedule logs are copied to the out_root for plotting.",
    )
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument(
        "--plot",
        action="store_true",
        help="If set, generates time-series plots in <out_root>/analysis using scripts/analysis/plot_timeseries.py.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cmd = load_config_command(args.config)

    per_seed_pts: List[Dict[str, float]] = []

    for seed in args.seeds:
        run_dir = out_root / f"seed_{int(seed)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        extra_args = [
            "--seed",
            str(int(seed)),
            "--concurrency_schedule",
            str(args.schedule),
        ]

        print(f"\n=== [E2] nonstationary schedule seed={seed} ===")
        run_baseline_eval(cmd, str(run_dir), extra_args=extra_args)

        mpath = run_dir / "metrics_schedule.json"
        if not mpath.exists():
            raise FileNotFoundError(f"Missing schedule metrics: {mpath}")
        metrics = _read_json(mpath)
        per_seed_pts.append(extract_frontier_point(metrics))

    # Aggregate overall metrics across seeds (not time-series)
    agg = {
        "config": str(args.config),
        "schedule": str(args.schedule),
        "seeds": [int(s) for s in args.seeds],
        "primary_seed": int(args.primary_seed),
        "accuracy": _mean_ci([p.get("accuracy", 0.0) for p in per_seed_pts]),
        "violation_rate": _mean_ci([p.get("violation_rate", 0.0) for p in per_seed_pts]),
        "cost_per_request": _mean_ci([p.get("cost_per_request", 0.0) for p in per_seed_pts]),
        "p99_e2e_ms": _mean_ci([p.get("p99_e2e_ms", 0.0) for p in per_seed_pts]),
        "p99_ttft_ms": _mean_ci([p.get("p99_ttft_ms", 0.0) for p in per_seed_pts]),
        "per_seed": per_seed_pts,
    }
    (out_root / "summary_ci.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    # Copy the primary seed logs to out_root root so analysis/make_all.py can find them.
    primary_dir = out_root / f"seed_{int(args.primary_seed)}"
    if primary_dir.exists():
        for fname_src, fname_dst in [
            ("requests_schedule.jsonl", "requests_schedule.jsonl"),
            ("metrics_schedule.json", "summary_schedule.json"),
            ("metrics_schedule.json", "metrics_schedule.json"),
        ]:
            src = primary_dir / fname_src
            dst = out_root / fname_dst
            if src.exists():
                shutil.copy2(src, dst)

    # Optional: generate time-series plots in the out_root itself.
    if args.plot:
        reqs = out_root / "requests_schedule.jsonl"
        if reqs.exists():
            analysis_dir = out_root / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            repo_dir = Path(__file__).resolve().parents[2]
            plot_script = repo_dir / "scripts" / "analysis" / "plot_timeseries.py"
            cmdline = [
                "python",
                str(plot_script),
                "--requests_jsonl",
                str(reqs),
                "--out_dir",
                str(analysis_dir),
            ]
            subprocess.run(cmdline, check=False)


if __name__ == "__main__":
    main()
