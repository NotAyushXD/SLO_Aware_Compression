# metrics.py
"""Metrics calculation + SLO compliance.

Patch highlights (paper reliability):
- SLO calibration profiles for p90/p95/p99 (sensitivity analysis).
- Robust handling of server-side TTFT definitions (Option A already encoded in server.py).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calibrate_slos(request_metrics: List, percentile: float = 95.0) -> Dict:
    """Calibrate per-difficulty SLO thresholds at a given percentile.

    Args:
        request_metrics: List of RequestMetrics objects
        percentile: Percentile to use (e.g., 95.0)

    Returns:
        Dict[str, Dict[str, float]] keyed by difficulty with ttft_ms/tpot_ms.
    """

    if not request_metrics:
        return {k: v.copy() for k, v in MetricsCalculator.DEFAULT_SLOS.items()}

    by_diff: Dict[str, Dict[str, List[float]]] = {}
    for m in request_metrics:
        if not getattr(m, "success", False):
            continue
        diff = getattr(m, "difficulty", "medium")
        by_diff.setdefault(diff, {"ttft": [], "tpot": []})
        by_diff[diff]["ttft"].append(float(getattr(m, "ttft_ms", 0.0) or 0.0))
        by_diff[diff]["tpot"].append(float(getattr(m, "tpot_ms", 0.0) or 0.0))

    calibrated: Dict[str, Dict[str, float]] = {}
    for diff, values in by_diff.items():
        ttft_vals = [v for v in values["ttft"] if v > 0]
        tpot_vals = [v for v in values["tpot"] if v > 0]

        ttft_p = float(np.percentile(ttft_vals, percentile)) if ttft_vals else 0.0
        tpot_p = float(np.percentile(tpot_vals, percentile)) if tpot_vals else 0.0

        # Round up for cleaner, slightly-conservative SLOs.
        calibrated[diff] = {
            "ttft_ms": float(np.ceil(ttft_p)),
            "tpot_ms": float(np.ceil(tpot_p)),
        }

    # Fill missing difficulties with defaults.
    for diff in MetricsCalculator.DEFAULT_SLOS:
        calibrated.setdefault(diff, {k: float(v) for k, v in MetricsCalculator.DEFAULT_SLOS[diff].items()})

    logger.info(f"Calibrated SLOs at p{percentile:.1f}: {calibrated}")
    return calibrated


def calibrate_slo_profiles(
    request_metrics: List,
    percentiles: Sequence[float] = (90.0, 95.0, 99.0),
) -> Dict[str, Dict]:
    """Calibrate multiple percentile profiles (for paper sensitivity analysis)."""

    profiles: Dict[str, Dict] = {}
    for p in percentiles:
        key = f"p{int(p)}"
        profiles[key] = calibrate_slos(request_metrics, percentile=float(p))
    return profiles


@dataclass
class PercentileMetrics:
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    mean: float
    std: float


class MetricsCalculator:
    """Compute summary statistics and SLO compliance."""

    # Default SLOs used only if calibration is disabled/missing.
    DEFAULT_SLOS = {
        "easy": {"ttft_ms": 150, "tpot_ms": 1000},
        "medium": {"ttft_ms": 250, "tpot_ms": 1000},
        "hard": {"ttft_ms": 400, "tpot_ms": 1500},
    }

    def __init__(self, request_metrics: List, slo_dict: Optional[Dict] = None):
        self.metrics = request_metrics
        self.slo_dict = slo_dict or {k: v.copy() for k, v in self.DEFAULT_SLOS.items()}
        logger.info(f"Initialized MetricsCalculator with {len(request_metrics)} metrics")

    def calculate_percentiles(self, values: List[float]) -> PercentileMetrics:
        if not values:
            return PercentileMetrics(0, 0, 0, 0, 0, 0, 0)
        values = sorted(values)
        return PercentileMetrics(
            p50=float(np.percentile(values, 50)),
            p75=float(np.percentile(values, 75)),
            p90=float(np.percentile(values, 90)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            mean=float(np.mean(values)),
            std=float(np.std(values)),
        )

    def compute_all_metrics(self) -> Dict:
        total_requests = len(self.metrics)
        successful = sum(1 for m in self.metrics if getattr(m, "success", False))
        success_rate = successful / max(total_requests, 1)

        ttft_values = [float(m.ttft_ms) for m in self.metrics if float(getattr(m, "ttft_ms", 0.0) or 0.0) > 0]
        tpot_values = [float(m.tpot_ms) for m in self.metrics if float(getattr(m, "tpot_ms", 0.0) or 0.0) > 0]
        e2e_values = [float(getattr(m, "e2e_latency_ms", 0.0) or 0.0) for m in self.metrics]
        queue_values = [float(getattr(m, "queue_wait_time_ms", 0.0) or 0.0) for m in self.metrics]

        ttft_p = self.calculate_percentiles(ttft_values)
        tpot_p = self.calculate_percentiles(tpot_values)
        e2e_p = self.calculate_percentiles(e2e_values)
        queue_p = self.calculate_percentiles(queue_values)

        # ------------------------------------------------------------------
        # Throughput: total output tokens / wall time from first submit to last completion.
        # ------------------------------------------------------------------
        if self.metrics:
            first_submit = min(float(getattr(m, "submit_time", time.time())) for m in self.metrics)
            last_complete = max(float(getattr(m, "end_time", time.time())) for m in self.metrics)
            total_duration = max(last_complete - first_submit, 1e-3)

            total_out_tokens = sum(
                int(getattr(m, "inference_metrics", {}).get("output_length", 0) or 0)
                for m in self.metrics
                if getattr(m, "success", False)
            )
            throughput = float(total_out_tokens) / total_duration
        else:
            total_duration = 0.0
            throughput = 0.0

        
        # Escalation rate (for MultiVariantService): fraction of successful requests that escalated.
        escalations = sum(
            1
            for m in self.metrics
            if getattr(m, 'success', False)
            and bool(getattr(m, 'inference_metrics', {}).get('router_escalated', False))
        )
        escalation_rate = float(escalations) / max(successful, 1)

        # ------------------------------------------------------------------
        # SLO compliance + Goodput + Quality + Cost
        # ------------------------------------------------------------------
        # Helpers for backward compatibility with older logs.
        def _inf(m) -> Dict:
            return getattr(m, "inference_metrics", {}) or {}

        def _variant_cost_multiplier(inf: Dict) -> float:
            # Prefer server-reported multiplier if present.
            try:
                if "cost_multiplier" in inf and inf["cost_multiplier"] is not None:
                    return float(inf["cost_multiplier"])
            except Exception:
                pass

            v = (inf.get("variant_effective") or inf.get("variant") or "base")
            v = str(v).lower().strip()
            # Use the learned router's cost table if available.
            try:
                from learned_router import LearnedRouter

                return float(LearnedRouter.VARIANT_COSTS.get(v, 1.0))
            except Exception:
                # Fallback: keep in sync with learned_router.py
                return {"base": 1.0, "med": 0.6, "cheap": 0.3}.get(v, 1.0)

        def _cost_units(inf: Dict) -> float:
            # Prefer server-reported cost units if present.
            try:
                if "cost_units" in inf and inf["cost_units"] is not None:
                    return float(inf["cost_units"])
            except Exception:
                pass

            cm = _variant_cost_multiplier(inf)
            total_tokens = inf.get("total_tokens")
            if total_tokens is None:
                pt = int(inf.get("prompt_tokens", 0) or 0)
                ot = int(inf.get("output_length", inf.get("output_tokens", 0)) or 0)
                total_tokens = pt + ot
                if total_tokens <= 0:
                    total_tokens = ot
            try:
                return float(cm) * float(total_tokens)
            except Exception:
                return 0.0

        # Quality (strict primary + parseable sensitivity)
        correct_all = 0
        correct_parseable_all = 0
        format_ok_all = 0
        format_ok_parseable_all = 0

        correct_success = 0
        correct_parseable_success = 0
        format_ok_success = 0
        format_ok_parseable_success = 0

        # Goodput (SLO-compliant throughput)
        goodput_out_tokens = 0
        goodput_req = 0
        qa_goodput_out_tokens = 0
        qa_goodput_req = 0
        qa_goodput_out_tokens_parseable = 0

        # Cost
        total_cost_units = 0.0
        total_cost_units_slo_ok = 0.0
        total_cost_units_qa_ok = 0.0
        cost_mult_sum = 0.0
        cost_mult_count = 0
        total_tokens_sum = 0

        # SLO compliance
        slo_ok = 0
        slo_viol = 0
        viol_details = []

        for m in self.metrics:
            inf = _inf(m)

            # Quality signals (if present)
            try:
                c = int(inf.get("correct", 0) or 0)
                cp = int(inf.get("correct_parseable", 0) or 0)
                f = int(inf.get("format_ok", 0) or 0)
                fp = int(inf.get("format_ok_parseable", 0) or 0)
            except Exception:
                c, cp, f, fp = 0, 0, 0, 0

            correct_all += int(c)
            correct_parseable_all += int(cp)
            format_ok_all += int(f)
            format_ok_parseable_all += int(fp)

            is_success = bool(getattr(m, "success", False))
            if is_success:
                correct_success += int(c)
                correct_parseable_success += int(cp)
                format_ok_success += int(f)
                format_ok_parseable_success += int(fp)

                # Cost accounting (success-only)
                cm = _variant_cost_multiplier(inf)
                cu = _cost_units(inf)
                total_cost_units += float(cu)
                cost_mult_sum += float(cm)
                cost_mult_count += 1
                try:
                    total_tokens_sum += int(inf.get("total_tokens", 0) or 0)
                except Exception:
                    pass

            # SLO compliance + goodput
            if not is_success:
                slo_viol += 1
                continue

            diff = getattr(m, "difficulty", "medium")
            slo = self.slo_dict.get(diff, self.slo_dict.get("medium", self.DEFAULT_SLOS["medium"]))

            ttft = float(getattr(m, "ttft_ms", 0.0) or 0.0)
            tpot = float(getattr(m, "tpot_ms", 0.0) or 0.0)

            ttft_ok = ttft <= float(slo.get("ttft_ms", 0.0) or 0.0)
            tpot_ok = tpot <= float(slo.get("tpot_ms", 0.0) or 0.0)
            slo_pass = bool(ttft_ok and tpot_ok)

            out_toks = int(inf.get("output_length", inf.get("output_tokens", 0)) or 0)
            if slo_pass:
                slo_ok += 1
                goodput_req += 1
                goodput_out_tokens += max(0, int(out_toks))

                # Cost restricted to SLO-compliant requests (success-only).
                total_cost_units_slo_ok += float(_cost_units(inf))

                # Quality-adjusted goodput (strict primary)
                if int(c) == 1:
                    qa_goodput_req += 1
                    qa_goodput_out_tokens += max(0, int(out_toks))
                    total_cost_units_qa_ok += float(_cost_units(inf))

                # Quality-adjusted goodput (parseable sensitivity)
                if int(cp) == 1:
                    qa_goodput_out_tokens_parseable += max(0, int(out_toks))

            else:
                slo_viol += 1
                viol_details.append(
                    {
                        "request_id": getattr(m, "request_id", None),
                        "difficulty": diff,
                        "ttft_ms": ttft,
                        "ttft_slo": float(slo.get("ttft_ms", 0.0) or 0.0),
                        "ttft_ok": ttft_ok,
                        "tpot_ms": tpot,
                        "tpot_slo": float(slo.get("tpot_ms", 0.0) or 0.0),
                        "tpot_ok": tpot_ok,
                    }
                )

        slo_compliance = slo_ok / max(successful, 1)

        # Goodput + quality-adjusted goodput
        goodput_tokens_per_sec = float(goodput_out_tokens) / max(float(total_duration), 1e-9)
        goodput_requests_per_sec = float(goodput_req) / max(float(total_duration), 1e-9)
        qa_goodput_tokens_per_sec = float(qa_goodput_out_tokens) / max(float(total_duration), 1e-9)
        qa_goodput_requests_per_sec = float(qa_goodput_req) / max(float(total_duration), 1e-9)
        qa_goodput_tokens_per_sec_parseable = float(qa_goodput_out_tokens_parseable) / max(float(total_duration), 1e-9)

        # Accuracy under load
        accuracy_all = float(correct_all) / max(total_requests, 1)
        accuracy_success = float(correct_success) / max(successful, 1)
        accuracy_parseable_all = float(correct_parseable_all) / max(total_requests, 1)
        accuracy_parseable_success = float(correct_parseable_success) / max(successful, 1)

        # Accuracy conditional on SLO compliance (success-only)
        accuracy_slo_success = float(qa_goodput_req) / max(goodput_req, 1)

        # Cost summaries
        avg_cost_multiplier = float(cost_mult_sum) / max(cost_mult_count, 1)
        token_weighted_cost_multiplier = (
            float(total_cost_units) / max(float(total_tokens_sum), 1.0) if total_tokens_sum > 0 else float(avg_cost_multiplier)
        )
        cost_units_per_sec = float(total_cost_units) / max(float(total_duration), 1e-9)
        cost_per_goodput_token = float(total_cost_units) / max(float(goodput_out_tokens), 1.0)
        cost_per_qa_goodput_token = float(total_cost_units) / max(float(qa_goodput_out_tokens), 1.0)

        return {
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful,
                "failed_requests": total_requests - successful,
                "success_rate": success_rate,
                "slo_compliant": slo_ok,
                "slo_violations": slo_viol,
                "slo_compliance": slo_compliance,
                "escalation_rate": float(escalation_rate),
                "total_duration_sec": float(total_duration),
                "throughput_tokens_per_sec": float(throughput),
                # Goodput (SLO-compliant throughput)
                "goodput_tokens_per_sec": float(goodput_tokens_per_sec),
                "goodput_requests_per_sec": float(goodput_requests_per_sec),
                # Quality-adjusted goodput (strict primary)
                "quality_adjusted_goodput_tokens_per_sec": float(qa_goodput_tokens_per_sec),
                "quality_adjusted_goodput_requests_per_sec": float(qa_goodput_requests_per_sec),
                # Quality-adjusted goodput (parseable sensitivity)
                "quality_adjusted_goodput_tokens_per_sec_parseable": float(qa_goodput_tokens_per_sec_parseable),
                # Accuracy under load (strict primary)
                "accuracy_all": float(accuracy_all),
                "accuracy_success": float(accuracy_success),
                "accuracy_slo_compliant_success": float(accuracy_slo_success),
                # Accuracy under load (parseable sensitivity)
                "accuracy_parseable_all": float(accuracy_parseable_all),
                "accuracy_parseable_success": float(accuracy_parseable_success),
                # Cost accounting
                "avg_cost_multiplier": float(avg_cost_multiplier),
                "token_weighted_cost_multiplier": float(token_weighted_cost_multiplier),
                "total_cost_units": float(total_cost_units),
                "cost_units_per_sec": float(cost_units_per_sec),
                "total_cost_units_slo_compliant": float(total_cost_units_slo_ok),
                "total_cost_units_quality_adjusted": float(total_cost_units_qa_ok),
                "cost_per_goodput_token": float(cost_per_goodput_token),
                "cost_per_quality_adjusted_goodput_token": float(cost_per_qa_goodput_token),
            },
            "ttft": {
                "p50": ttft_p.p50,
                "p75": ttft_p.p75,
                "p90": ttft_p.p90,
                "p95": ttft_p.p95,
                "p99": ttft_p.p99,
                "mean": ttft_p.mean,
                "std": ttft_p.std,
            },
            "tpot": {
                "p50": tpot_p.p50,
                "p75": tpot_p.p75,
                "p90": tpot_p.p90,
                "p95": tpot_p.p95,
                "p99": tpot_p.p99,
                "mean": tpot_p.mean,
                "std": tpot_p.std,
            },
            "e2e_latency": {
                "p50": e2e_p.p50,
                "p75": e2e_p.p75,
                "p90": e2e_p.p90,
                "p95": e2e_p.p95,
                "p99": e2e_p.p99,
                "mean": e2e_p.mean,
                "std": e2e_p.std,
            },
            "queue_wait": {
                "p50": queue_p.p50,
                "p75": queue_p.p75,
                "p90": queue_p.p90,
                "p95": queue_p.p95,
                "p99": queue_p.p99,
                "mean": queue_p.mean,
                "std": queue_p.std,
            },
            "slo_violations": viol_details[:10],
        }

    def print_report(self, title: str = "BASELINE SERVER METRICS REPORT") -> Dict:
        metrics = self.compute_all_metrics()

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        s = metrics["summary"]
        print("\nSUMMARY:")
        print(f"  Total Requests:        {s['total_requests']:6d}")
        print(f"  Successful:            {s['successful_requests']:6d}")
        print(f"  Failed:                {s['failed_requests']:6d}")
        print(f"  Success Rate:          {s['success_rate']*100:6.2f}%")
        print(f"  Total Duration:        {s['total_duration_sec']:6.2f} seconds")
        print(f"  Throughput:            {s['throughput_tokens_per_sec']:6.1f} tokens/sec")
        print(f"  Goodput (SLO OK):       {s.get('goodput_tokens_per_sec', 0.0):6.1f} tokens/sec")
        print(f"  QA Goodput (SLO+Acc):   {s.get('quality_adjusted_goodput_tokens_per_sec', 0.0):6.1f} tokens/sec")
        print(f"  SLO Compliance:        {s['slo_compliance']*100:6.2f}%")
        print(f"  SLO Violations:        {s['slo_violations']:6d}")
        print(f"  Escalation Rate:       {s['escalation_rate']*100:6.2f}%")

        # Quality under load (if computed by load_generator)
        if 'accuracy_success' in s:
            print(f"  Accuracy (success):    {s.get('accuracy_success', 0.0)*100:6.2f}%")
            print(f"  Accuracy (SLO OK):     {s.get('accuracy_slo_compliant_success', 0.0)*100:6.2f}%")
            print(f"  Acc (parseable, succ): {s.get('accuracy_parseable_success', 0.0)*100:6.2f}%")

        # Cost accounting (unitless multipliers)
        if 'avg_cost_multiplier' in s:
            print(f"  Avg Cost Multiplier:   {s.get('avg_cost_multiplier', 0.0):6.3f}  (req-avg)")
            print(f"  Token-Wt Cost Mult:    {s.get('token_weighted_cost_multiplier', 0.0):6.3f}  (token-avg)")
            print(f"  Cost Units / sec:      {s.get('cost_units_per_sec', 0.0):6.1f}")
            print(f"  Cost / Goodput token:  {s.get('cost_per_goodput_token', 0.0):6.3f}")
            print(f"  Cost / QA token:       {s.get('cost_per_quality_adjusted_goodput_token', 0.0):6.3f}")

        print("\nTTFT (Time-to-First-Token) in milliseconds:")
        for k in ("p50", "p75", "p90", "p95", "p99"):
            print(f"  {k.upper()}:  {metrics['ttft'][k]:7.2f} ms")
        print(f"  Mean: {metrics['ttft']['mean']:7.2f} ms (±{metrics['ttft']['std']:.2f})")

        print("\nTPOT (Time-Per-Output-Token) in milliseconds:")
        for k in ("p50", "p75", "p90", "p95", "p99"):
            print(f"  {k.upper()}:  {metrics['tpot'][k]:7.2f} ms")
        print(f"  Mean: {metrics['tpot']['mean']:7.2f} ms (±{metrics['tpot']['std']:.2f})")

        print("\nE2E Latency (End-to-End) in milliseconds:")
        for k in ("p50", "p75", "p90", "p95", "p99"):
            print(f"  {k.upper()}:  {metrics['e2e_latency'][k]:7.2f} ms")
        print(f"  Mean: {metrics['e2e_latency']['mean']:7.2f} ms (±{metrics['e2e_latency']['std']:.2f})")

        print("\nQueue Wait Time in milliseconds:")
        for k in ("p50", "p95", "p99"):
            print(f"  {k.upper()}:  {metrics['queue_wait'][k]:7.2f} ms")
        print(f"  Mean: {metrics['queue_wait']['mean']:7.2f} ms")

        if metrics.get("slo_violations"):
            print("\nSample SLO Violations (first 10):")
            for i, v in enumerate(metrics["slo_violations"][:10], 1):
                parts = []
                if not v.get("ttft_ok", True):
                    parts.append(f"TTFT: {v['ttft_ms']:.1f}ms > {v['ttft_slo']}ms SLO")
                if not v.get("tpot_ok", True):
                    parts.append(f"TPOT: {v['tpot_ms']:.1f}ms > {v['tpot_slo']}ms SLO")
                if not parts:
                    parts.append(f"TTFT: {v['ttft_ms']:.1f}ms > {v['ttft_slo']}ms SLO")
                    parts.append(f"TPOT: {v['tpot_ms']:.1f}ms > {v['tpot_slo']}ms SLO")
                print(f"  {i}. Request {v.get('request_id')} ({v.get('difficulty')}): " + " | ".join(parts))

        print("\n" + "=" * 80)
        return metrics

    def save_metrics(self, output_file: str) -> None:
        metrics = self.compute_all_metrics()
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_file}")
