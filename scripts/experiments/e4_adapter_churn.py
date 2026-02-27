#!/usr/bin/env python
"""E4: adapter churn + cache size sweep.

This experiment increases adapter switching frequency and sweeps max_loaded_adapters.
It reports:
- adapter cache hit rate
- adapter/swap overhead per request (token-equivalent units)

The script works even without real adapter artifacts if you run with:
  --enable_adapters --adapter_allow_missing
and set synthetic overhead:
  --adapter_synthetic_load_ms, --adapter_synthetic_switch_ms

Usage:
  python scripts/experiments/e4_adapter_churn.py \
    --config configs/bandit_with_adapters.json \
    --base_requests_jsonl data/processed/gsm8k_val.jsonl \
    --churn_rates 0.0 0.25 0.5 1.0 \
    --cache_sizes 1 2 4 8 \
    --adapter_ids a0 a1 a2 a3 a4 a5 a6 a7 \
    --seeds 0 1 2 \
    --out_root outputs/e4_adapter_churn

Notes:
- churn_rate is approximated via block sizes: block ~= round(1/max(churn_rate,eps)).
  churn_rate=1.0 => alternating adapter every request.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from scripts.experiments.utils import (
    find_metrics_file,
    load_experiment_config,
    load_metrics,
    run_baseline_eval,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _augment_with_adapter_ids(rows: List[Dict[str, Any]], adapter_ids: List[str], block: int) -> List[Dict[str, Any]]:
    out = []
    if not adapter_ids:
        adapter_ids = ["a0", "a1"]
    block = max(1, int(block))
    for i, r in enumerate(rows):
        rr = dict(r)
        rr["adapter_id"] = str(adapter_ids[(i // block) % len(adapter_ids)])
        out.append(rr)
    return out


def _extract(metrics: Dict[str, Any]) -> Dict[str, float]:
    summ = metrics.get("summary") or {}
    succ = float(summ.get("successful_requests", 0) or 0.0)

    cache_hit = float(summ.get("adapter_cache_hit_rate", 0.0) or 0.0)
    adap_ov = float(summ.get("total_adapter_overhead_units", 0.0) or 0.0)
    swap_ov = float(summ.get("total_swap_overhead_units", 0.0) or 0.0)
    ov_per_req = (adap_ov + swap_ov) / succ if succ > 0 else 0.0

    return {
        "adapter_cache_hit_rate": cache_hit,
        "overhead_units_per_request": float(ov_per_req),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--base_requests_jsonl", type=str, required=True)
    ap.add_argument("--churn_rates", type=float, nargs="+", required=True)
    ap.add_argument("--cache_sizes", type=int, nargs="+", required=True)
    ap.add_argument("--adapter_ids", type=str, nargs="+", default=["a0", "a1", "a2", "a3"]) 
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--eps", type=float, default=1e-6)
    args = ap.parse_args()

    repo_dir = Path(__file__).resolve().parents[2]
    cfg = load_experiment_config(args.config)
    base_cmd = list(cfg["command"])

    base_rows = _read_jsonl(Path(args.base_requests_jsonl))
    if not base_rows:
        raise RuntimeError(f"Empty base_requests_jsonl: {args.base_requests_jsonl}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Aggregate results: cache_size -> churn_rate -> list[seed_result]
    grid: Dict[int, Dict[float, List[Dict[str, float]]]] = {}

    for cache in args.cache_sizes:
        grid[int(cache)] = {}
        for churn in args.churn_rates:
            # approximate churn via block size
            if float(churn) <= 0.0:
                block = 10**9
            else:
                block = int(max(1, round(1.0 / max(float(churn), float(args.eps)))))

            grid[int(cache)][float(churn)] = []

            for seed in args.seeds:
                run_dir = out_root / f"cache_{cache}" / f"churn_{churn:.4f}" / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                aug_path = run_dir / "load_pool_with_adapters.jsonl"
                aug_rows = _augment_with_adapter_ids(base_rows, list(args.adapter_ids), block=block)
                _write_jsonl(aug_path, aug_rows)

                extra = [
                    "--max_loaded_adapters",
                    str(int(cache)),
                    "--load_test_jsonl",
                    str(aug_path),
                ]

                run_baseline_eval(
                    repo_dir=repo_dir,
                    base_command=base_cmd,
                    out_dir=run_dir,
                    seed=int(seed),
                    extra_args=extra,
                )

                metrics = load_metrics(find_metrics_file(run_dir))
                grid[int(cache)][float(churn)].append(_extract(metrics))

    # Save raw grid
    raw_out = {str(k): {str(c): v for c, v in vv.items()} for k, vv in grid.items()}
    (out_root / "summary_raw.json").write_text(json.dumps(raw_out, indent=2), encoding="utf-8")

    # Plot cache hit rate vs churn (one line per cache size)
    plt.figure()
    for cache in sorted(grid.keys()):
        xs = []
        ys = []
        for churn in sorted(grid[cache].keys()):
            vals = [r["adapter_cache_hit_rate"] for r in grid[cache][churn]]
            xs.append(churn)
            ys.append(float(np.mean(vals)) if vals else 0.0)
        plt.plot(xs, ys, marker="o", label=f"cache={cache}")
    plt.xlabel("churn rate (approx)")
    plt.ylabel("adapter cache hit rate")
    plt.title("E4: cache hit rate vs churn")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "cache_hit_vs_churn.png", dpi=200)
    plt.close()

    # Plot overhead per request vs churn
    plt.figure()
    for cache in sorted(grid.keys()):
        xs = []
        ys = []
        for churn in sorted(grid[cache].keys()):
            vals = [r["overhead_units_per_request"] for r in grid[cache][churn]]
            xs.append(churn)
            ys.append(float(np.mean(vals)) if vals else 0.0)
        plt.plot(xs, ys, marker="o", label=f"cache={cache}")
    plt.xlabel("churn rate (approx)")
    plt.ylabel("adapter+swap overhead (cost units / request)")
    plt.title("E4: overhead vs churn")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / "overhead_vs_churn.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
