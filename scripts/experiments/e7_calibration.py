#!/usr/bin/env python
"""E7: Calibration & threshold sensitivity.

Execution-plan requirement:
  - Reliability diagrams + ECE table
  - Compare our raw predictor vs alternative calibrations (e.g., temperature scaling, isotonic)

This script runs one or more configs (multi-seed), extracts per-request predicted quality
probabilities from the request logs, and produces:
  - reliability_<label>.png
  - threshold_sensitivity_<label>.png
  - ece_table_<label>.json
  - summary_all.json

Notes
-----
We try to auto-detect predicted quality probability from logs:
  - bandit: inference_metrics.router_meta.bandit.chosen_components.q_mean
  - risk/learned: inference_metrics.router_meta.predicted_quality

If a run does not log a usable predicted probability, it will be skipped.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from scripts.analysis.calibration_utils import (
    apply_isotonic,
    apply_temperature,
    brier_score,
    expected_calibration_error,
    fit_isotonic,
    fit_temperature,
    nll,
    reliability_bins,
)
from scripts.experiments.utils import load_config_command, run_baseline_eval


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


def _extract_pred_prob(row: Dict[str, Any]) -> Optional[float]:
    inf = row.get("inference_metrics", {}) or {}
    rm = inf.get("router_meta", {}) or {}

    # Bandit path
    try:
        b = rm.get("bandit", None)
        if isinstance(b, dict):
            cc = b.get("chosen_components", {}) or {}
            if "q_mean" in cc:
                return float(cc.get("q_mean"))
    except Exception:
        pass

    # Risk/Learned routers
    for k in ["predicted_quality", "p_correct", "quality_prob"]:
        if k in rm:
            try:
                return float(rm.get(k))
            except Exception:
                pass
        if k in inf:
            try:
                return float(inf.get(k))
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        type=str,
        nargs="+",
        required=True,
        help="One or more configs: label=path/to/config.json (or a bare path).",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num_requests", type=int, default=300)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--n_bins", type=int, default=15)
    ap.add_argument(
        "--calib_frac",
        type=float,
        default=0.5,
        help="Fraction of requests used as calibration set (rest used for evaluation).",
    )
    return ap.parse_args()


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


def _plot_reliability(ax, bins: Dict[str, np.ndarray], label: str) -> None:
    conf = bins["bin_confidence"].astype(float)
    acc = bins["bin_accuracy"].astype(float)
    cnt = bins["bin_count"].astype(int)
    m = cnt > 0
    ax.plot(conf[m], acc[m], marker="o", label=label)


def _threshold_sensitivity_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (taus, coverage, selective_accuracy)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    taus = np.linspace(0.0, 1.0, 51)
    cov = []
    acc = []
    for t in taus:
        m = y_prob >= float(t)
        cov.append(float(np.mean(m)) if y_prob.size > 0 else 0.0)
        if np.sum(m) > 0:
            acc.append(float(np.mean(y_true[m])))
        else:
            acc.append(float("nan"))
    return taus, np.asarray(cov, dtype=float), np.asarray(acc, dtype=float)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    configs = _parse_configs(list(args.configs))
    conc = int(args.concurrency)
    nreq = int(args.num_requests)

    summary_all: Dict[str, Any] = {}

    for label, cfg_path in configs:
        cmd = load_config_command(cfg_path)
        label_root = out_root / label
        label_root.mkdir(parents=True, exist_ok=True)

        all_probs: List[float] = []
        all_labels: List[int] = []

        for seed in args.seeds:
            run_dir = label_root / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            extra_args = [
                "--seed",
                str(int(seed)),
                "--concurrencies",
                str(conc),
                "--num_requests",
                str(nreq),
            ]

            print(f"\n=== [E7] {label} seed={seed} ===")
            run_baseline_eval(cmd, str(run_dir), extra_args=extra_args)

            req_path = run_dir / f"requests_concurrency_{conc}.jsonl"
            if not req_path.exists():
                raise FileNotFoundError(f"Missing request log: {req_path}")
            rows = _read_jsonl(req_path)

            for r in rows:
                p = _extract_pred_prob(r)
                y = _extract_label(r)
                if p is None or y is None:
                    continue
                if not (0.0 <= float(p) <= 1.0):
                    continue
                all_probs.append(float(p))
                all_labels.append(int(y))

        if not all_probs:
            print(f"[E7] WARNING: no usable predicted probabilities found for label={label}. Skipping.")
            continue

        y = np.asarray(all_labels, dtype=float)
        p_raw = np.asarray(all_probs, dtype=float)

        # Deterministic split (first frac = calibration)
        n = int(len(y))
        n_cal = int(max(1, min(n - 1, int(float(args.calib_frac) * n))))
        y_cal, y_ev = y[:n_cal], y[n_cal:]
        p_cal, p_ev = p_raw[:n_cal], p_raw[n_cal:]

        # Alternative calibrations
        T = fit_temperature(y_cal, p_cal)
        p_temp = apply_temperature(p_ev, T)
        iso = fit_isotonic(y_cal, p_cal)
        p_iso = apply_isotonic(iso, p_ev)

        # Metrics
        ece_raw = expected_calibration_error(y_ev, p_ev, n_bins=int(args.n_bins))
        ece_temp = expected_calibration_error(y_ev, p_temp, n_bins=int(args.n_bins))
        ece_iso = expected_calibration_error(y_ev, p_iso, n_bins=int(args.n_bins))

        out = {
            "label": label,
            "config": cfg_path,
            "n_total": int(n),
            "n_cal": int(n_cal),
            "n_eval": int(n - n_cal),
            "temperature_T": float(T),
            "ece": {
                "raw": float(ece_raw),
                "temperature": float(ece_temp),
                "isotonic": float(ece_iso),
            },
            "brier": {
                "raw": float(brier_score(y_ev, p_ev)),
                "temperature": float(brier_score(y_ev, p_temp)),
                "isotonic": float(brier_score(y_ev, p_iso)),
            },
            "nll": {
                "raw": float(nll(y_ev, p_ev)),
                "temperature": float(nll(y_ev, p_temp)),
                "isotonic": float(nll(y_ev, p_iso)),
            },
        }

        with (label_root / f"ece_table_{label}.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        summary_all[label] = out

        # Reliability plot
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="ideal")
        _plot_reliability(ax, reliability_bins(y_ev, p_ev, n_bins=int(args.n_bins)), "raw")
        _plot_reliability(ax, reliability_bins(y_ev, p_temp, n_bins=int(args.n_bins)), f"temp (T={T:.2f})")
        _plot_reliability(ax, reliability_bins(y_ev, p_iso, n_bins=int(args.n_bins)), "isotonic")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("empirical accuracy")
        ax.set_title(f"Reliability diagram ({label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(label_root / f"reliability_{label}.png", dpi=200)
        plt.close(fig)

        # Threshold sensitivity (selective accuracy / coverage)
        taus, cov_raw, acc_raw = _threshold_sensitivity_curve(y_ev, p_ev)
        taus, cov_temp, acc_temp = _threshold_sensitivity_curve(y_ev, p_temp)
        taus, cov_iso, acc_iso = _threshold_sensitivity_curve(y_ev, p_iso)

        fig, ax = plt.subplots()
        ax.plot(cov_raw, acc_raw, marker="o", label="raw")
        ax.plot(cov_temp, acc_temp, marker="o", label="temp")
        ax.plot(cov_iso, acc_iso, marker="o", label="isotonic")
        ax.set_xlabel("coverage (fraction kept)")
        ax.set_ylabel("accuracy among kept")
        ax.set_title(f"Threshold sensitivity ({label})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(label_root / f"threshold_sensitivity_{label}.png", dpi=200)
        plt.close(fig)

    with (out_root / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(summary_all, f, indent=2)

    print(f"\nSaved E7 artifacts to: {out_root}")


if __name__ == "__main__":
    main()
