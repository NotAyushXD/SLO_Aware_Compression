#!/usr/bin/env python
"""Unit-ish test for metrics.py

Creates a tiny synthetic RequestMetrics list and verifies MetricsCalculator
runs end-to-end.
"""

import time

from load_generator import RequestMetrics
from metrics import MetricsCalculator


def main() -> None:
    now = time.time()
    ms = lambda x: float(x)

    metrics = []
    for i in range(5):
        submit = now + i * 0.01
        start = submit + 0.001
        end = start + 0.050
        inf = {
            'success': True,
            'ttft_ms': ms(30 + i),
            'tpot_ms': ms(5 + 0.1 * i),
            'output_length': 16,
            'prompt_tokens': 32,
            'variant_effective': 'cheap' if i % 2 == 0 else 'base',
            'correct': 1 if i % 2 == 0 else 0,
            'correct_parseable': 1 if i % 2 == 0 else 0,
            'format_ok': 1,
            'format_ok_parseable': 1,
            'router_escalated': False,
            'slo_ttft_ok': True,
            'slo_tpot_ok': True,
            'slo_total_ok': True,
            'slo_ok': True,
        }
        metrics.append(
            RequestMetrics(
                request_id=i,
                dataset_type='gsm8k',
                submit_time=submit,
                start_time=start,
                end_time=end,
                difficulty='easy',
                inference_metrics=inf,
            )
        )

    slo = {'easy': {'ttft_ms': 100.0, 'tpot_ms': 50.0}, 'medium': {'ttft_ms': 150.0, 'tpot_ms': 60.0}, 'hard': {'ttft_ms': 200.0, 'tpot_ms': 80.0}}
    mc = MetricsCalculator(metrics, slo_dict=slo)
    out = mc.compute_all_metrics()

    # Basic schema checks
    for k in ['success_rate', 'throughput_toks_per_s', 'slo_compliance_rate', 'goodput_toks_per_s']:
        assert k in out, f"Missing key: {k}"

    print('[OK] metrics.compute_all_metrics ran successfully')
    print('Keys:', sorted(list(out.keys()))[:20], '...')


if __name__ == '__main__':
    main()
