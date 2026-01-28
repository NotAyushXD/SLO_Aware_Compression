"""
metrics.py

Metrics and SLO logic for the load test harness.

Paper-friendly changes:
- Summary now includes prompt/output token statistics.
- SLO compliance includes separate TTFT/TPOT violation counts (useful for analysis).
- SLO calibration supports an optional safety margin (multiplicative) so calibrated SLOs
  are a bit more robust to run-to-run noise.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SLOProfile:
    ttft_ms: float
    tpot_ms: float


@dataclass
class InferenceMetrics:
    request_id: int
    success: bool
    error: Optional[str]
    submit_time: float
    start_time: float
    end_time: float

    prompt_tokens: int
    output_tokens: int

    # From server (may be None if unavailable)
    ttft_ms: Optional[float]
    tpot_ms: Optional[float]
    queue_wait_ms: Optional[float]

    dataset_type: str
    difficulty: str
    prompt_mode: str
    variant: str = "unknown"

    def e2e_latency_ms(self) -> float:
        return max(0.0, (self.end_time - self.submit_time) * 1000.0)


class MetricsCalculator:
    def __init__(self, metrics: List[InferenceMetrics], slo_profiles: Optional[Dict[str, SLOProfile]] = None):
        self.metrics = metrics
        self.slo_profiles = slo_profiles or {}
        self._cached_summary: Optional[Dict[str, Any]] = None

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return float("nan")
        if p <= 0:
            return min(values)
        if p >= 100:
            return max(values)
        k = (len(values) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted(values)[int(k)]
        d0 = sorted(values)[int(f)] * (c - k)
        d1 = sorted(values)[int(c)] * (k - f)
        return d0 + d1

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(statistics.mean(values)) if values else float("nan")

    @staticmethod
    def _safe_stdev(values: List[float]) -> float:
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    # -----------------------------
    # Summary
    # -----------------------------
    def summary(self) -> Dict[str, Any]:
        if self._cached_summary is not None:
            return self._cached_summary

        total = len(self.metrics)
        successful = sum(1 for m in self.metrics if m.success)
        failed = total - successful

        e2e = [m.e2e_latency_ms() for m in self.metrics if m.success]
        ttft = [m.ttft_ms for m in self.metrics if m.success and m.ttft_ms is not None]
        tpot = [m.tpot_ms for m in self.metrics if m.success and m.tpot_ms is not None]
        qwait = [m.queue_wait_ms for m in self.metrics if m.success and m.queue_wait_ms is not None]

        output_tokens = [m.output_tokens for m in self.metrics if m.success]
        prompt_tokens = [m.prompt_tokens for m in self.metrics if m.success]

        # Throughput: output tokens / wall time of the whole test (using end-start across all requests)
        if self.metrics:
            test_start = min(m.submit_time for m in self.metrics)
            test_end = max(m.end_time for m in self.metrics)
            duration_s = max(1e-6, test_end - test_start)
        else:
            duration_s = 0.0

        throughput_toks = (sum(output_tokens) / duration_s) if duration_s > 0 else 0.0

        # SLO compliance
        slo_ok_count = 0
        slo_ttft_viol = 0
        slo_tpot_viol = 0
        slo_total_checked = 0

        for m in self.metrics:
            if not m.success:
                continue
            prof = self.slo_profiles.get(m.difficulty)
            if not prof:
                continue

            slo_total_checked += 1
            ok = True
            if m.ttft_ms is not None and m.ttft_ms > prof.ttft_ms:
                ok = False
                slo_ttft_viol += 1
            if m.tpot_ms is not None and m.tpot_ms > prof.tpot_ms:
                ok = False
                slo_tpot_viol += 1
            if ok:
                slo_ok_count += 1

        slo_compliance = (slo_ok_count / slo_total_checked) if slo_total_checked > 0 else float("nan")

        out: Dict[str, Any] = {
            "total_requests": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total) if total > 0 else float("nan"),
            "total_duration_sec": duration_s,
            "throughput_tokens_per_sec": throughput_toks,
            "slo_checked": slo_total_checked,
            "slo_compliance": slo_compliance,
            "slo_violations_total": (slo_total_checked - slo_ok_count) if slo_total_checked > 0 else 0,
            "slo_violations_ttft": slo_ttft_viol,
            "slo_violations_tpot": slo_tpot_viol,
            "latency_ms": {
                "e2e": self._latency_block(e2e),
                "ttft": self._latency_block(ttft),
                "tpot": self._latency_block(tpot),
                "queue_wait": self._latency_block(qwait),
            },
            "tokens": {
                "prompt_total": int(sum(prompt_tokens)),
                "output_total": int(sum(output_tokens)),
                "prompt_mean": self._safe_mean([float(x) for x in prompt_tokens]),
                "output_mean": self._safe_mean([float(x) for x in output_tokens]),
                "prompt_p95": self._percentile([float(x) for x in prompt_tokens], 95),
                "output_p95": self._percentile([float(x) for x in output_tokens], 95),
            },
        }

        self._cached_summary = out
        return out

    def _latency_block(self, values: List[float]) -> Dict[str, Any]:
        if not values:
            return {
                "p50": float("nan"),
                "p75": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
                "p99": float("nan"),
                "mean": float("nan"),
                "stdev": float("nan"),
            }
        return {
            "p50": self._percentile(values, 50),
            "p75": self._percentile(values, 75),
            "p90": self._percentile(values, 90),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
            "mean": self._safe_mean(values),
            "stdev": self._safe_stdev(values),
        }

    def to_json(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())


def calibrate_slos(
    metrics: List[InferenceMetrics],
    percentile: float = 95.0,
    safety_margin: float = 1.00,
    defaults: Optional[Dict[str, SLOProfile]] = None,
) -> Dict[str, SLOProfile]:
    """
    Calibrate SLO thresholds from a set of metrics.

    safety_margin: multiplicative factor (>1.0 loosens, <1.0 tightens)
    """
    defaults = defaults or {
        "easy": SLOProfile(ttft_ms=300, tpot_ms=150),
        "medium": SLOProfile(ttft_ms=500, tpot_ms=250),
        "hard": SLOProfile(ttft_ms=800, tpot_ms=400),
    }

    # Group by difficulty
    by_diff: Dict[str, List[InferenceMetrics]] = {}
    for m in metrics:
        if not m.success:
            continue
        by_diff.setdefault(m.difficulty, []).append(m)

    out: Dict[str, SLOProfile] = {}
    for diff, default_prof in defaults.items():
        group = by_diff.get(diff, [])
        ttft_vals = [m.ttft_ms for m in group if m.ttft_ms is not None]
        tpot_vals = [m.tpot_ms for m in group if m.tpot_ms is not None]

        if ttft_vals and tpot_vals:
            ttft_p = MetricsCalculator._percentile([float(x) for x in ttft_vals], percentile) * float(safety_margin)
            tpot_p = MetricsCalculator._percentile([float(x) for x in tpot_vals], percentile) * float(safety_margin)
            out[diff] = SLOProfile(ttft_ms=float(ttft_p), tpot_ms=float(tpot_p))
        else:
            out[diff] = default_prof

    return out
