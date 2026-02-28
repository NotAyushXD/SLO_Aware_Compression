#!/usr/bin/env python
"""Time-series plots for nonstationary / online bandit experiments.

Produces:
- Rolling SLO violation rate over time
- Bandit queue Q_t over time (if present)
- Action mix over time (variants)

Input is a requests JSONL file emitted by load_generator.save_results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _rolling_mean(xs: List[float], w: int) -> List[float]:
    if w <= 1:
        return xs
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for x in xs:
        q.append(float(x))
        s += float(x)
        if len(q) > w:
            s -= q.pop(0)
        out.append(s / float(len(q)))
    return out


def _rolling_ratio(num: List[float], den: List[float], w: int) -> List[float]:
    """Rolling ratio (sum(num)/sum(den)) over a sliding window."""
    if w <= 1:
        out: List[float] = []
        for a, b in zip(num, den):
            out.append(float(a) / float(b) if float(b) > 0 else 0.0)
        return out
    out: List[float] = []
    n_sum = 0.0
    d_sum = 0.0
    buf: List[Tuple[float, float]] = []
    for a, b in zip(num, den):
        a = float(a)
        b = float(b)
        buf.append((a, b))
        n_sum += a
        d_sum += b
        if len(buf) > w:
            a0, b0 = buf.pop(0)
            n_sum -= a0
            d_sum -= b0
        out.append(n_sum / d_sum if d_sum > 0 else 0.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--window", type=int, default=50, help="Rolling window size (in requests).")
    args = ap.parse_args()

    req_path = Path(args.requests_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(req_path)
    if not rows:
        raise RuntimeError(f"Empty requests log: {req_path}")

    # Sort by end_time if present.
    try:
        rows = sorted(rows, key=lambda r: float(r.get("end_time", 0.0) or 0.0))
    except Exception:
        pass

    t = list(range(len(rows)))

    violations: List[float] = []
    correct: List[float] = []
    success: List[float] = []
    Qs: List[float] = []
    Q_present = False
    variants: List[str] = []
    datasets: List[str] = []
    phases: List[int] = []

    for r in rows:
        inf = r.get("inference_metrics", {}) or {}
        v = inf.get("risk_violation", inf.get("slo_violation", 0))
        try:
            violations.append(1.0 if int(v) != 0 else 0.0)
        except Exception:
            violations.append(0.0)

        # Bandit queue
        bu = inf.get("bandit_update", {}) or {}
        q_after = bu.get("Q_after", None)
        if q_after is None:
            # fallback to router_meta snapshot
            rm = inf.get("router_meta", {}) or {}
            q_after = ((rm.get("bandit") or {}).get("Q", None))
        if q_after is not None:
            Q_present = True
            try:
                Qs.append(float(q_after))
            except Exception:
                Qs.append(0.0)
        else:
            Qs.append(0.0)

        variants.append(str(inf.get("variant", "")) or "")

        try:
            success.append(1.0 if int(inf.get("success", 0) or 0) != 0 else 0.0)
        except Exception:
            success.append(0.0)
        try:
            correct.append(1.0 if int(inf.get("correct", 0) or 0) != 0 else 0.0)
        except Exception:
            correct.append(0.0)

        datasets.append(str(r.get("dataset_type", r.get("dataset", "")) or ""))
        try:
            phases.append(int(r.get("phase", 0) or 0))
        except Exception:
            phases.append(0)

    viol_roll = _rolling_mean(violations, int(args.window))
    acc_roll = _rolling_ratio(correct, success, int(args.window))

    # 1) Violation rate
    plt.figure()
    plt.plot(t, viol_roll)
    plt.xlabel("request index")
    plt.ylabel(f"rolling violation rate (window={args.window})")
    plt.title("SLO/Risk violation rate over time")
    plt.tight_layout()
    plt.savefig(out_dir / "violation_rate_timeseries.png", dpi=200)
    plt.close()

    # 1b) Rolling accuracy
    plt.figure()
    plt.plot(t, acc_roll)
    plt.xlabel("request index")
    plt.ylabel(f"rolling accuracy among successes (window={args.window})")
    plt.title("Quality over time (accuracy)")
    # Phase boundary markers (if present)
    try:
        last = phases[0]
        for i, p in enumerate(phases):
            if p != last:
                plt.axvline(i, linestyle="--", linewidth=1, alpha=0.6)
                last = p
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_timeseries.png", dpi=200)
    plt.close()

    # 2) Q_t
    if Q_present:
        plt.figure()
        plt.plot(t, Qs)
        plt.xlabel("request index")
        plt.ylabel("Q")
        plt.title("Bandit queue Q over time")
        plt.tight_layout()
        plt.savefig(out_dir / "bandit_Q_timeseries.png", dpi=200)
        plt.close()

    # 3) Action mix (variants)
    uniq = sorted(list({v for v in variants if v}))
    if uniq:
        # rolling proportions per variant
        w = int(args.window)
        series = {u: [] for u in uniq}
        buf: List[str] = []
        for v in variants:
            buf.append(v)
            if len(buf) > w:
                buf.pop(0)
            denom = float(len(buf))
            for u in uniq:
                series[u].append(sum(1 for x in buf if x == u) / denom)

        plt.figure()
        for u in uniq:
            plt.plot(t, series[u], label=u)
        plt.xlabel("request index")
        plt.ylabel(f"rolling action proportion (window={args.window})")
        plt.title("Action mix over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "action_mix_timeseries.png", dpi=200)
        plt.close()

    # 4) Dataset mix (if multiple domains)
    uniq_ds = sorted(list({d for d in datasets if d}))
    if len(uniq_ds) >= 2:
        w = int(args.window)
        series = {u: [] for u in uniq_ds}
        buf: List[str] = []
        for d in datasets:
            buf.append(d)
            if len(buf) > w:
                buf.pop(0)
            denom = float(len(buf))
            for u in uniq_ds:
                series[u].append(sum(1 for x in buf if x == u) / denom)
        plt.figure()
        for u in uniq_ds:
            plt.plot(t, series[u], label=u)
        plt.xlabel("request index")
        plt.ylabel(f"rolling dataset proportion (window={args.window})")
        plt.title("Request mix over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "dataset_mix_timeseries.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
