#!/usr/bin/env python
"""E5: Domain/length shift mid-run (nonstationary request mix).

Execution-plan requirement:
  - Change the request distribution mid-run (domain shift or length shift)
  - Plot quality/risk over time and adaptation curves
  - Compare: ours (online updates) vs frozen policy (no online updates)

This script uses the new --data_schedule support in run_baseline_evaluation.py.

Examples
--------

Dataset/domain shift (gsm8k -> mmlu), with a warmup learning phase:

  python scripts/experiments/e5_domain_length_shift.py \
    --config configs/bandit_base.json \
    --shift_mode dataset --phase1_value gsm8k --phase2_value mmlu \
    --phase_requests 200 200 --concurrency 4 \
    --warmup_requests 200 --warmup_concurrency 4 \
    --seeds 0 1 2 3 4 \
    --out_root outputs/e5_shift

Length shift (short -> long):

  python scripts/experiments/e5_domain_length_shift.py \
    --config configs/bandit_base.json \
    --shift_mode length --phase_requests 200 200 --concurrency 4 \
    --seeds 0 1 2 3 4 \
    --out_root outputs/e5_shift
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to a baseline config JSON.")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--shift_mode",
        type=str,
        default="dataset",
        choices=["dataset", "length"],
        help="Shift type: dataset domain shift or prompt length shift.",
    )
    ap.add_argument("--phase_requests", type=int, nargs=2, default=[200, 200], help="Requests per phase.")
    ap.add_argument("--phase1_value", type=str, default="gsm8k", help="dataset name for phase1 (dataset shift)")
    ap.add_argument("--phase2_value", type=str, default="mmlu", help="dataset name for phase2 (dataset shift)")

    ap.add_argument(
        "--warmup_requests",
        type=int,
        default=0,
        help="If >0, run a bandit learning phase of this many requests before the shift schedule.",
    )
    ap.add_argument(
        "--warmup_concurrency",
        type=int,
        default=None,
        help="Concurrency for warmup learning phase (defaults to --concurrency).",
    )
    ap.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling window size for time-series plots (requests).",
    )
    return ap.parse_args()


def _make_data_schedule(args: argparse.Namespace) -> str:
    n1, n2 = int(args.phase_requests[0]), int(args.phase_requests[1])
    if args.shift_mode == "dataset":
        v1 = str(args.phase1_value).strip().lower()
        v2 = str(args.phase2_value).strip().lower()
        return f"dataset={v1}:{n1},dataset={v2}:{n2}"
    if args.shift_mode == "length":
        return f"length=short:{n1},length=long:{n2}"
    raise ValueError(f"Unknown shift_mode: {args.shift_mode}")


def _run_plot_timeseries(req_jsonl: Path, out_dir: Path, window: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        "scripts/analysis/plot_timeseries.py",
        "--requests_jsonl",
        str(req_jsonl),
        "--out_dir",
        str(out_dir),
        "--window",
        str(int(window)),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base_cmd = load_config_command(args.config)
    data_schedule = _make_data_schedule(args)

    conc = int(args.concurrency)
    warmup_n = int(args.warmup_requests)
    warmup_conc = int(args.warmup_concurrency or conc)

    # Two conditions:
    #   - online : allow learning during the shift schedule
    #   - frozen : learn during warmup (optional), then freeze during schedule
    conditions = ["online", "frozen"]

    per_cond_phase: Dict[str, Dict[int, Dict[str, List[float]]]] = {c: {} for c in conditions}

    for cond in conditions:
        for seed in args.seeds:
            run_dir = out_root / cond / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            extra_args: List[str] = [
                "--seed",
                str(int(seed)),
                "--data_schedule",
                data_schedule,
                "--data_schedule_concurrency",
                str(conc),
                "--concurrencies",
                str(conc),
                "--num_requests",
                str(int(sum(args.phase_requests))),
            ]

            if warmup_n > 0:
                extra_args += [
                    "--bandit_learn_requests",
                    str(int(warmup_n)),
                    "--bandit_learn_concurrency",
                    str(int(warmup_conc)),
                ]

            if cond == "online":
                # Ensure bandit keeps updating after warmup.
                extra_args += ["--bandit_keep_learning_during_eval"]
            elif cond == "frozen":
                # Freeze after warmup; if warmup_n==0, force-freeze from start.
                if warmup_n <= 0:
                    extra_args += ["--bandit_force_freeze"]

            print(f"\n=== [E5] {cond} seed={seed} schedule={data_schedule} ===")
            run_baseline_eval(base_cmd, str(run_dir), extra_args=extra_args)

            # Generate required adaptation plots
            req_path = run_dir / "requests_schedule.jsonl"
            if req_path.exists():
                _run_plot_timeseries(req_path, run_dir / "analysis", window=int(args.window))

            # Collect per-phase summary metrics for CI tables
            # (metrics files are written as metrics_phase_{i}_...json)
            phase_files = sorted(run_dir.glob("metrics_phase_*.json"))
            for pf in phase_files:
                try:
                    m = _read_json(pf)
                except Exception:
                    continue
                phase_idx = int(m.get("phase", 0) or 0)
                if phase_idx <= 0:
                    continue
                pt = extract_frontier_point(m)
                viol = float(pt.get("violation_rate", 0.0) or 0.0)
                acc = float(pt.get("accuracy", 0.0) or 0.0)
                cost = float(pt.get("cost_per_request", 0.0) or 0.0)

                per_cond_phase[cond].setdefault(phase_idx, {}).setdefault("violation_rate", []).append(viol)
                per_cond_phase[cond].setdefault(phase_idx, {}).setdefault("accuracy", []).append(acc)
                per_cond_phase[cond].setdefault(phase_idx, {}).setdefault("cost_per_request", []).append(cost)

    # Aggregate CI summary across seeds.
    summary: Dict[str, Any] = {
        "shift_mode": args.shift_mode,
        "data_schedule": data_schedule,
        "concurrency": conc,
        "warmup_requests": warmup_n,
        "warmup_concurrency": warmup_conc,
        "seeds": list(args.seeds),
    }

    for cond in conditions:
        cond_rows: List[Dict[str, Any]] = []
        for phase_idx in sorted(per_cond_phase[cond].keys()):
            row: Dict[str, Any] = {"phase": int(phase_idx)}
            for k, xs in per_cond_phase[cond][phase_idx].items():
                mu, lo, hi = _mean_ci([float(x) for x in xs])
                row[k] = {"mean": mu, "ci_low": lo, "ci_high": hi, "n": len(xs)}
            cond_rows.append(row)
        summary[cond] = {"by_phase": cond_rows}

    with (out_root / "summary_ci.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved E5 summary to: {out_root / 'summary_ci.json'}")


if __name__ == "__main__":
    main()
