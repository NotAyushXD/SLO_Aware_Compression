#!/usr/bin/env python
"""Unit-ish test for metrics.py

Creates a tiny synthetic RequestMetrics list and verifies MetricsCalculator
runs end-to-end, producing the expected summary keys.

This test is intentionally lightweight (no model inference).
"""

import os
import sys
import time

# Ensure repo root is on sys.path when invoked as a file path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from load_generator import RequestMetrics  # noqa: E402
from metrics import MetricsCalculator  # noqa: E402


def main() -> None:
    now = time.time()

    metrics = []
    for i in range(5):
        submit = now + i * 0.01
        start = submit + 0.001
        end = start + 0.050
        inf = {
            "success": True,
            "ttft_ms": float(30 + i),
            "tpot_ms": float(5 + 0.1 * i),
            # Provide an explicit E2E to exercise the new v_t definition path.
            "total_latency_ms": float(80 + i),
            "output_length": 16,
            "prompt_tokens": 32,
            "variant_effective": "cheap" if i % 2 == 0 else "base",
            # Quality labels
            "correct": 1 if i % 2 == 0 else 0,
            "correct_parseable": 1 if i % 2 == 0 else 0,
            "format_ok": 1,
            "format_ok_parseable": 1,
            # Router flags
            "router_escalated": False,
        }
        metrics.append(
            RequestMetrics(
                request_id=i,
                dataset_type="gsm8k",
                submit_time=submit,
                start_time=start,
                end_time=end,
                difficulty="easy",
                inference_metrics=inf,
            )
        )

    # Provide only ttft + tpot SLOs; MetricsCalculator will derive a total_ms budget.
    slo = {
        "easy": {"ttft_ms": 100.0, "tpot_ms": 50.0},
        "medium": {"ttft_ms": 150.0, "tpot_ms": 60.0},
        "hard": {"ttft_ms": 200.0, "tpot_ms": 80.0},
    }

    mc = MetricsCalculator(metrics, slo_dict=slo)
    out = mc.compute_all_metrics()

    assert isinstance(out, dict), "metrics output must be a dict"
    assert "summary" in out and isinstance(out["summary"], dict), "missing summary block"

    s = out["summary"]
    for k in [
        "success_rate",
        "throughput_tokens_per_sec",
        "slo_compliance",
        "goodput_tokens_per_sec",
        "accuracy_success",
        "total_cost_units",
    ]:
        assert k in s, f"Missing summary key: {k}"

    print("[OK] metrics.compute_all_metrics ran successfully")
    print("Summary keys (sample):", sorted(list(s.keys()))[:15], "...")


if __name__ == "__main__":
    main()
