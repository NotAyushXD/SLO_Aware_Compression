#!/usr/bin/env python
"""One-command figure regeneration from saved experiment artifacts.

This is the reproducibility entrypoint referenced by the execution plan:
"Provide raw logs and a one-command script to regenerate all figures." 

It **does not rerun LLM inference**. It reads the JSON/JSONL artifacts under an
`outputs/` directory and regenerates figures/tables under `figures/` and `tables/`.

Supported experiments (best-effort):
  - E1: delta sweep (frontier)
  - E2: nonstationary load schedule (time-series)
  - E3: offered-load sweep
  - E4: adapter churn
  - E5: domain/length shift
  - E6: label budget curve
  - E7: calibration

If an experiment folder is missing, it is skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.calibration_utils import (
    apply_isotonic,
    apply_temperature,
    expected_calibration_error,
    fit_isotonic,
    fit_temperature,
    reliability_bins,
)


def _rolling_mean(xs: List[float], w: int) -> List[float]:
    if w <= 1:
        return list(xs)
    out: List[float] = []
    s = 0.0
    buf: List[float] = []
    for x in xs:
        x = float(x)
        buf.append(x)
        s += x
        if len(buf) > w:
            s -= buf.pop(0)
        out.append(s / float(len(buf)))
    return out


def _rolling_ratio(num: List[float], den: List[float], w: int) -> List[float]:
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


def _plot_timeseries(req_jsonl: Path, fig_dir: Path, prefix: str, window: int = 50) -> None:
    rows = _read_jsonl(req_jsonl)
    if not rows:
        return

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
        # Violation: prefer explicit slo_violation / risk_violation.
        v = inf.get("slo_violation", None)
        if v is None:
            v = inf.get("risk_violation", 0)
        try:
            violations.append(1.0 if int(v) != 0 else 0.0)
        except Exception:
            violations.append(0.0)

        # Variant
        variants.append(str(inf.get("variant", "")) or "")

        # Correct/success
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

        # Bandit Q_t
        q = None
        try:
            rm = inf.get("router_meta", {}) or {}
            if isinstance(rm, dict):
                bu = rm.get("bandit_update", {}) or {}
                if isinstance(bu, dict) and "Q_after" in bu:
                    q = bu.get("Q_after")
                else:
                    b = rm.get("bandit", {}) or {}
                    if isinstance(b, dict) and "Q" in b:
                        q = b.get("Q")
        except Exception:
            q = None
        if q is None:
            Qs.append(0.0)
        else:
            try:
                Qs.append(float(q))
                Q_present = True
            except Exception:
                Qs.append(0.0)

    t = list(range(len(rows)))
    viol_roll = _rolling_mean(violations, int(window))
    acc_roll = _rolling_ratio(correct, success, int(window))

    # Violation
    plt.figure()
    plt.plot(t, viol_roll)
    plt.xlabel("request index")
    plt.ylabel(f"rolling violation rate (window={window})")
    plt.title(f"{prefix}: violation rate over time")
    try:
        last = phases[0]
        for i, p in enumerate(phases):
            if p != last:
                plt.axvline(i, linestyle="--", linewidth=1, alpha=0.6)
                last = p
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(fig_dir / f"{prefix}_violation_rate_timeseries.png", dpi=200)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(t, acc_roll)
    plt.xlabel("request index")
    plt.ylabel(f"rolling accuracy among successes (window={window})")
    plt.title(f"{prefix}: accuracy over time")
    try:
        last = phases[0]
        for i, p in enumerate(phases):
            if p != last:
                plt.axvline(i, linestyle="--", linewidth=1, alpha=0.6)
                last = p
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(fig_dir / f"{prefix}_accuracy_timeseries.png", dpi=200)
    plt.close()

    # Bandit Q
    if Q_present:
        plt.figure()
        plt.plot(t, Qs)
        plt.xlabel("request index")
        plt.ylabel("Q_t")
        plt.title(f"{prefix}: Bandit Q over time")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_bandit_Q_timeseries.png", dpi=200)
        plt.close()

    # Action mix
    uniq = sorted([u for u in set(variants) if u])
    if len(uniq) >= 2:
        counts = {u: [] for u in uniq}
        w = int(window)
        buf: List[str] = []
        for v in variants:
            buf.append(v)
            if len(buf) > w:
                buf.pop(0)
            denom = float(len(buf))
            for u in uniq:
                counts[u].append(sum(1 for x in buf if x == u) / denom)
        plt.figure()
        for u in uniq:
            plt.plot(t, counts[u], label=u)
        plt.xlabel("request index")
        plt.ylabel(f"rolling action proportion (window={window})")
        plt.title(f"{prefix}: action mix")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_action_mix_timeseries.png", dpi=200)
        plt.close()

    # Dataset mix
    uniq_ds = sorted([u for u in set(datasets) if u])
    if len(uniq_ds) >= 2:
        w = int(window)
        series = {u: [] for u in uniq_ds}
        buf2: List[str] = []
        for d in datasets:
            buf2.append(d)
            if len(buf2) > w:
                buf2.pop(0)
            denom = float(len(buf2))
            for u in uniq_ds:
                series[u].append(sum(1 for x in buf2 if x == u) / denom)
        plt.figure()
        for u in uniq_ds:
            plt.plot(t, series[u], label=u)
        plt.xlabel("request index")
        plt.ylabel(f"rolling dataset proportion (window={window})")
        plt.title(f"{prefix}: request mix")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}_dataset_mix_timeseries.png", dpi=200)
        plt.close()


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _plot_e1_frontier(e1_dir: Path, fig_dir: Path) -> None:
    summ = e1_dir / "summary_all.json"
    if not summ.exists():
        return
    data = _read_json(summ)

    plt.figure()
    for label, d in data.items():
        pts = d.get("points", []) or []
        xs = [float(p.get("cost", 0.0) or 0.0) for p in pts]
        ys = [float(p.get("quality", 0.0) or 0.0) for p in pts]
        if xs and ys:
            plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("cost per request (cost_units)")
    plt.ylabel("quality (accuracy)")
    plt.title("E1 Pareto frontier (δ sweep)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "e1_frontier.png", dpi=200)
    plt.close()


def _plot_e3_offered_load(e3_dir: Path, fig_dir: Path) -> None:
    summ = e3_dir / "summary_all.json"
    if not summ.exists():
        return
    data = _read_json(summ)

    def plot(metric: str, ylabel: str, fname: str) -> None:
        plt.figure()
        for label, d in data.items():
            xs = []
            ys = []
            lo = []
            hi = []
            for r in d.get("by_concurrency", []):
                xs.append(float(r.get("concurrency", 0)))
                m = r.get(metric, {}) or {}
                ys.append(float(m.get("mean", 0.0) or 0.0))
                lo.append(float(m.get("ci_low", 0.0) or 0.0))
                hi.append(float(m.get("ci_high", 0.0) or 0.0))
            if xs:
                plt.plot(xs, ys, marker="o", label=label)
                try:
                    plt.fill_between(xs, lo, hi, alpha=0.15)
                except Exception:
                    pass
        plt.xlabel("concurrency")
        plt.ylabel(ylabel)
        plt.title(f"E3 offered-load sweep: {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=200)
        plt.close()

    plot("p99_e2e_ms", "p99 E2E latency (ms)", "e3_p99_e2e_vs_load.png")
    plot("violation_rate", "violation rate", "e3_violation_vs_load.png")
    plot("cost_per_request", "cost per request (cost_units)", "e3_cost_vs_load.png")


def _plot_e2_nonstationary(e2_dir: Path, fig_dir: Path) -> None:
    # Expect requests_schedule.jsonl at root (from run_baseline_evaluation --concurrency_schedule).
    cand = e2_dir / "requests_schedule.jsonl"
    if not cand.exists():
        # fall back: first match
        matches = sorted(e2_dir.glob("**/requests_schedule.jsonl"))
        if not matches:
            return
        cand = matches[0]
    _plot_timeseries(cand, fig_dir, prefix="e2", window=50)


def _plot_e4_churn(e4_dir: Path, fig_dir: Path) -> None:
    # Regenerate from summary_raw.json if present.
    summ = e4_dir / "summary_raw.json"
    if not summ.exists():
        return
    data = _read_json(summ)
    rows = data.get("rows", []) or []
    if not rows:
        return

    churn = [float(r.get("churn_rate", 0.0) or 0.0) for r in rows]
    hit = [float(r.get("cache_hit_rate", 0.0) or 0.0) for r in rows]
    overhead_ms = [float(r.get("avg_overhead_ms", 0.0) or 0.0) for r in rows]

    plt.figure()
    plt.plot(churn, hit, marker="o")
    plt.xlabel("adapter churn rate")
    plt.ylabel("cache hit rate")
    plt.title("E4 adapter churn: cache hit vs churn")
    plt.tight_layout()
    plt.savefig(fig_dir / "e4_cache_hit_vs_churn.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(churn, overhead_ms, marker="o")
    plt.xlabel("adapter churn rate")
    plt.ylabel("avg overhead (ms)")
    plt.title("E4 adapter churn: overhead vs churn")
    plt.tight_layout()
    plt.savefig(fig_dir / "e4_overhead_vs_churn.png", dpi=200)
    plt.close()


def _plot_e6_label_budget(e6_dir: Path, fig_dir: Path) -> None:
    summ = e6_dir / "summary_all.json"
    if not summ.exists():
        return
    data = _read_json(summ)
    results = (data.get("results") or {})
    if not results:
        return

    # Plot accuracy vs label_budget_p (mean)
    ps = sorted([float(p) for p in results.keys()])
    acc = []
    viol = []
    for p in ps:
        d = results.get(str(p), results.get(p, {})) or {}
        acc.append(float(d.get("accuracy", {}).get("mean", 0.0) or 0.0))
        viol.append(float(d.get("violation_rate", {}).get("mean", 0.0) or 0.0))

    plt.figure()
    plt.plot(ps, acc, marker="o")
    plt.xlabel("label budget p")
    plt.ylabel("accuracy")
    plt.title("E6 label budget: accuracy vs label rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "e6_accuracy_vs_label_budget.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(ps, viol, marker="o")
    plt.xlabel("label budget p")
    plt.ylabel("violation rate")
    plt.title("E6 label budget: violation rate vs label rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "e6_violation_vs_label_budget.png", dpi=200)
    plt.close()


def _plot_e5_shift(e5_dir: Path, fig_dir: Path) -> None:
    summ = e5_dir / "summary_ci.json"
    if not summ.exists():
        return
    data = _read_json(summ)

    # Simple bar plot: phase1 vs phase2 accuracy for online vs frozen.
    online = {r["phase"]: r.get("accuracy", {}) for r in data.get("online", {}).get("by_phase", [])}
    frozen = {r["phase"]: r.get("accuracy", {}) for r in data.get("frozen", {}).get("by_phase", [])}

    phases = sorted(set(list(online.keys()) + list(frozen.keys())))
    if not phases:
        return
    xs = np.arange(len(phases))

    def get(m, key):
        try:
            return float((m.get(key) or 0.0))
        except Exception:
            return 0.0

    online_mean = [get(online.get(p, {}), "mean") for p in phases]
    frozen_mean = [get(frozen.get(p, {}), "mean") for p in phases]

    width = 0.35
    plt.figure()
    plt.bar(xs - width / 2, online_mean, width=width, label="online")
    plt.bar(xs + width / 2, frozen_mean, width=width, label="frozen")
    plt.xticks(xs, [f"phase {p}" for p in phases])
    plt.ylabel("accuracy")
    plt.title(f"E5 shift ({data.get('shift_mode', '')}): phase accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "e5_phase_accuracy.png", dpi=200)
    plt.close()

    # Also copy per-run time-series plots if present (best-effort)
    for cond in ["online", "frozen"]:
        d = e5_dir / cond
        if not d.exists():
            continue
        # pick the first seed folder
        seed_dirs = sorted([p for p in d.iterdir() if p.is_dir()])
        if not seed_dirs:
            continue
        ts_dir = seed_dirs[0] / "analysis"
        if ts_dir.exists():
            for png in [
                "violation_rate_timeseries.png",
                "bandit_Q_timeseries.png",
                "action_mix_timeseries.png",
                "accuracy_timeseries.png",
                "dataset_mix_timeseries.png",
            ]:
                src = ts_dir / png
                if src.exists():
                    shutil.copyfile(src, fig_dir / f"e5_{cond}_{png}")


def _extract_pred_prob(row: Dict[str, Any]) -> Optional[float]:
    inf = row.get("inference_metrics", {}) or {}
    rm = inf.get("router_meta", {}) or {}
    # Bandit
    b = rm.get("bandit", None)
    if isinstance(b, dict):
        cc = b.get("chosen_components", {}) or {}
        if "q_mean" in cc:
            try:
                return float(cc.get("q_mean"))
            except Exception:
                pass
    # Risk / learned
    if "predicted_quality" in rm:
        try:
            return float(rm.get("predicted_quality"))
        except Exception:
            pass
    return None


def _extract_label(row: Dict[str, Any]) -> Optional[int]:
    inf = row.get("inference_metrics", {}) or {}
    if "correct" in inf:
        try:
            return int(inf.get("correct") or 0)
        except Exception:
            return None
    return None


def _plot_e7_calibration(e7_dir: Path, fig_dir: Path, n_bins: int = 15) -> None:
    # Expect structure: e7_dir/<label>/seed_*/requests_concurrency_*.jsonl
    if not e7_dir.exists():
        return
    for label_dir in sorted([p for p in e7_dir.iterdir() if p.is_dir()]):
        label = label_dir.name
        probs: List[float] = []
        labels: List[int] = []
        req_files = sorted(label_dir.glob("seed_*/requests_concurrency_*.jsonl"))
        for rf in req_files:
            rows = _read_jsonl(rf)
            for r in rows:
                p = _extract_pred_prob(r)
                y = _extract_label(r)
                if p is None or y is None:
                    continue
                if 0.0 <= float(p) <= 1.0:
                    probs.append(float(p))
                    labels.append(int(y))
        if not probs:
            continue
        y = np.asarray(labels, dtype=float)
        p_raw = np.asarray(probs, dtype=float)

        # Simple split: first half calibrate
        n = len(y)
        n_cal = max(1, min(n - 1, n // 2))
        y_cal, y_ev = y[:n_cal], y[n_cal:]
        p_cal, p_ev = p_raw[:n_cal], p_raw[n_cal:]

        T = fit_temperature(y_cal, p_cal)
        p_temp = apply_temperature(p_ev, T)
        iso = fit_isotonic(y_cal, p_cal)
        p_iso = apply_isotonic(iso, p_ev)

        # Reliability plot
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="ideal")
        for name, p in [("raw", p_ev), (f"temp(T={T:.2f})", p_temp), ("isotonic", p_iso)]:
            bins = reliability_bins(y_ev, p, n_bins=n_bins)
            conf = bins["bin_confidence"].astype(float)
            acc = bins["bin_accuracy"].astype(float)
            cnt = bins["bin_count"].astype(int)
            m = cnt > 0
            ax.plot(conf[m], acc[m], marker="o", label=name)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(f"E7 reliability ({label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"e7_reliability_{label}.png", dpi=200)
        plt.close(fig)

        # ECE table
        ece = {
            "raw": expected_calibration_error(y_ev, p_ev, n_bins=n_bins),
            "temp": expected_calibration_error(y_ev, p_temp, n_bins=n_bins),
            "isotonic": expected_calibration_error(y_ev, p_iso, n_bins=n_bins),
        }
        with (fig_dir / f"e7_ece_{label}.json").open("w", encoding="utf-8") as f:
            json.dump({"label": label, "ece": ece, "n": int(len(y_ev))}, f, indent=2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", type=str, default="outputs")
    ap.add_argument("--fig_dir", type=str, default="figures")
    ap.add_argument("--tables_dir", type=str, default="tables")
    ap.add_argument("--n_bins", type=int, default=15)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root)
    fig_dir = Path(args.fig_dir)
    tables_dir = Path(args.tables_dir)
    _ensure_dir(fig_dir)
    _ensure_dir(tables_dir)

    # Common folder conventions used by our scripts.
    candidates = {
        "e1": outputs_root / "e1_delta_sweep",
        "e2": outputs_root / "e2_nonstationary",
        "e3": outputs_root / "e3_offered_load",
        "e4": outputs_root / "e4_adapter_churn",
        "e5": outputs_root / "e5_shift",
        "e6": outputs_root / "e6_label_budget",
        "e7": outputs_root / "e7_calibration",
    }

    # Fallback: accept any folder that starts with e1/e3/e5/e7.
    for key in ["e1", "e3", "e5", "e7"]:
        if not candidates[key].exists() and outputs_root.exists():
            matches = sorted([p for p in outputs_root.glob(f"{key}*") if p.is_dir()])
            if matches:
                candidates[key] = matches[0]

    # E1
    if candidates["e1"].exists():
        _plot_e1_frontier(candidates["e1"], fig_dir)

    # E2
    if candidates["e2"].exists():
        _plot_e2_nonstationary(candidates["e2"], fig_dir)

    # E3
    if candidates["e3"].exists():
        _plot_e3_offered_load(candidates["e3"], fig_dir)

    # E4
    if candidates["e4"].exists():
        _plot_e4_churn(candidates["e4"], fig_dir)

    # E5
    if candidates["e5"].exists():
        _plot_e5_shift(candidates["e5"], fig_dir)

    # E6
    if candidates["e6"].exists():
        _plot_e6_label_budget(candidates["e6"], fig_dir)

    # E7
    if candidates["e7"].exists():
        _plot_e7_calibration(candidates["e7"], fig_dir, n_bins=int(args.n_bins))

    print(f"Done. Figures -> {fig_dir} | Tables -> {tables_dir}")


if __name__ == "__main__":
    main()
