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
    Qs: List[float] = []
    Q_present = False
    variants: List[str] = []

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

    viol_roll = _rolling_mean(violations, int(args.window))

    # 1) Violation rate
    plt.figure()
    plt.plot(t, viol_roll)
    plt.xlabel("request index")
    plt.ylabel(f"rolling violation rate (window={args.window})")
    plt.title("SLO/Risk violation rate over time")
    plt.tight_layout()
    plt.savefig(out_dir / "violation_rate_timeseries.png", dpi=200)
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


if __name__ == "__main__":
    main()
