"""Paper-ready plotting for the "adapter hotness / setup-aware" polish step.

This script is intentionally lightweight: it reads the standard `metrics_concurrency_*.json`
files produced by `run_baseline_evaluation.py` and plots:

  (1) SLO violation rate (lower is better)
  (2) Quality-adjusted goodput (higher is better)

Typical usage (offered-load sweep / concurrency as x-axis):

  python scripts/plot_polish_adapter_features.py \
    --x_axis concurrency \
    --run "risk+adapter_features=/kaggle/working/exp_risk_adapterfeat" \
    --run "risk_no_adapter_features=/kaggle/working/exp_risk_no_adapterfeat" \
    --run "risk_adapterfeat_setupaware_off=/kaggle/working/exp_risk_adapterfeat_no_setupaware" \
    --run "heuristic=/kaggle/working/exp_heuristic" \
    --out_dir /kaggle/working/paper_plots

Alternate usage (x-axis is a knob like synthetic setup cost or #adapters):

  - Run multiple experiments with different knob values.
  - Pass each run with the SAME label (the script will combine them into a curve)
    and select a single concurrency via `--select_concurrency`.

Example:

  python scripts/plot_polish_adapter_features.py \
    --x_axis setup_cost_ms --select_concurrency 4 \
    --run "risk+adapter_features=/kaggle/working/exp_setup0" \
    --run "risk+adapter_features=/kaggle/working/exp_setup20" \
    --run "risk+adapter_features=/kaggle/working/exp_setup50" \
    --out_dir /kaggle/working/paper_plots
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_metrics_files(run_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(run_dir, "metrics_concurrency_*.json")))


def _parse_concurrency_from_filename(path: str) -> Optional[int]:
    m = re.search(r"metrics_concurrency_(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _extract_summary_metrics(metrics_json: Dict[str, Any]) -> Tuple[float, float]:
    """Return (slo_violation_rate, qa_goodput_tokens_per_sec)."""
    summary = metrics_json.get("summary", {})

    # SLO violation rate
    if "slo_compliance" in summary:
        slo_violation = float(1.0 - float(summary["slo_compliance"]))
    elif "slo_violations" in summary and "total_requests" in summary:
        total = max(1.0, float(summary["total_requests"]))
        slo_violation = float(summary["slo_violations"]) / total
    else:
        slo_violation = float("nan")

    # QA goodput
    qa_goodput = None
    for k in (
        "quality_adjusted_goodput_tokens_per_sec",
        "quality_adjusted_goodput_tokens_per_sec_parseable",
        "qa_goodput_tokens_per_sec",
        "goodput_tokens_per_sec",
    ):
        if k in summary:
            qa_goodput = float(summary[k])
            break
    if qa_goodput is None:
        qa_goodput = float("nan")

    return slo_violation, qa_goodput


def _get_x_from_config(config_path: str, x_axis: str) -> Optional[float]:
    if not os.path.exists(config_path):
        return None
    cfg = _read_json(config_path)

    if x_axis == "setup_cost_ms":
        # Convention used by adapter experiments.
        load_ms = float(cfg.get("adapter_synthetic_load_ms", 0.0) or 0.0)
        switch_ms = float(cfg.get("adapter_synthetic_switch_ms", 0.0) or 0.0)
        return load_ms + switch_ms

    if x_axis == "num_adapters":
        # Convention used by adapter experiments.
        # If not present, return None and let caller derive from logs.
        v = cfg.get("adapter_pool_size", None)
        return float(v) if v is not None else None

    return None


def _count_unique_adapters_from_requests(run_dir: str, select_concurrency: int) -> Optional[int]:
    path = os.path.join(run_dir, f"requests_concurrency_{select_concurrency}.jsonl")
    if not os.path.exists(path):
        return None
    uniq = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            adapter_id = obj.get("adapter_id", None)
            if adapter_id is not None and str(adapter_id) != "":
                uniq.add(str(adapter_id))
    return len(uniq)


@dataclass
class Point:
    x: float
    slo_violation: float
    qa_goodput: float


def load_points_for_run(
    run_dir: str,
    x_axis: str,
    select_concurrency: Optional[int],
) -> List[Point]:
    """Load points from one run directory."""

    metrics_files = _find_metrics_files(run_dir)
    if not metrics_files:
        raise FileNotFoundError(f"No metrics_concurrency_*.json files found in: {run_dir}")

    points: List[Point] = []

    if x_axis == "concurrency":
        for mf in metrics_files:
            c = _parse_concurrency_from_filename(mf)
            if c is None:
                continue
            mjs = _read_json(mf)
            slo_v, qa_gp = _extract_summary_metrics(mjs)
            points.append(Point(x=float(c), slo_violation=slo_v, qa_goodput=qa_gp))
        return sorted(points, key=lambda p: p.x)

    # Knob-based x-axis: each run_dir becomes a single point (per select_concurrency).
    if select_concurrency is None:
        raise ValueError("--select_concurrency is required when --x_axis != concurrency")

    mf = os.path.join(run_dir, f"metrics_concurrency_{select_concurrency}.json")
    if not os.path.exists(mf):
        raise FileNotFoundError(f"Missing metrics file for concurrency={select_concurrency}: {mf}")

    mjs = _read_json(mf)
    slo_v, qa_gp = _extract_summary_metrics(mjs)

    # Prefer config-derived x; fall back to requests-derived for num_adapters.
    cfg_path = os.path.join(run_dir, "config.json")
    x = _get_x_from_config(cfg_path, x_axis)
    if x is None and x_axis == "num_adapters":
        n = _count_unique_adapters_from_requests(run_dir, select_concurrency)
        x = float(n) if n is not None else None

    if x is None:
        raise ValueError(
            f"Could not derive x-axis value ({x_axis}) for run_dir={run_dir}. "
            "Ensure config.json includes the relevant fields, or provide request logs for fallback."
        )

    return [Point(x=float(x), slo_violation=slo_v, qa_goodput=qa_gp)]


def parse_runs(run_args: List[str]) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    for spec in run_args:
        if "=" not in spec:
            raise ValueError(f"Invalid --run spec (expected label=path): {spec}")
        label, path = spec.split("=", 1)
        runs.append((label.strip(), path.strip()))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as label=path (repeatable).",
    )
    ap.add_argument(
        "--x_axis",
        choices=["concurrency", "setup_cost_ms", "num_adapters"],
        default="concurrency",
        help="What to use for the x-axis.",
    )
    ap.add_argument(
        "--select_concurrency",
        type=int,
        default=None,
        help="Required when x_axis != concurrency. Which concurrency's metrics to plot.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for plots.",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="Adapter hotness + setup-aware scheduling (paper polish)",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=200,
    )
    args = ap.parse_args()

    if not args.run:
        raise ValueError("Provide at least one --run label=path")

    runs = parse_runs(args.run)

    # Build curves: multiple --run entries can share a label.
    curves: Dict[str, List[Point]] = {}
    for label, path in runs:
        pts = load_points_for_run(path, args.x_axis, args.select_concurrency)
        curves.setdefault(label, []).extend(pts)

    # De-duplicate / sort by x.
    for label, pts in list(curves.items()):
        # If multiple points share x, keep the last one.
        by_x: Dict[float, Point] = {p.x: p for p in pts}
        curves[label] = sorted(by_x.values(), key=lambda p: p.x)

    os.makedirs(args.out_dir, exist_ok=True)

    # Plot
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7.0, 6.0))
    ax_gp, ax_slo = axes[0], axes[1]

    for label, pts in curves.items():
        xs = [p.x for p in pts]
        qa = [p.qa_goodput for p in pts]
        slo = [p.slo_violation for p in pts]
        ax_gp.plot(xs, qa, marker="o", label=label)
        ax_slo.plot(xs, slo, marker="o", label=label)

    ax_gp.set_title(args.title)
    ax_gp.set_ylabel("QA goodput (tokens/sec)")
    ax_gp.grid(True, alpha=0.3)

    ax_slo.set_ylabel("SLO violation rate")
    ax_slo.set_xlabel(
        {
            "concurrency": "Offered load (concurrency)",
            "setup_cost_ms": "Synthetic setup cost (ms)",
            "num_adapters": "# unique adapters",
        }[args.x_axis]
    )
    ax_slo.grid(True, alpha=0.3)

    # One shared legend.
    handles, labels = ax_gp.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_pdf = os.path.join(args.out_dir, "polish_adapter_features.pdf")
    out_png = os.path.join(args.out_dir, "polish_adapter_features.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=args.dpi)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
