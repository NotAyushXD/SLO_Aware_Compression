import unittest

from metrics import MetricsCalculator
from load_generator import RequestMetrics


class TestMetricsFalseAccept(unittest.TestCase):
    def test_false_accept_rate_cheap(self):
        # One cheap accepted + wrong => FAR(cheap)=1.0
        m1 = RequestMetrics(
            request_id=1,
            dataset_type="gsm8k",
            submit_time=0.0,
            start_time=0.0,
            end_time=1.0,
            difficulty="easy",
            inference_metrics={
                "success": True,
                "correct": 0,
                "total_cost_units": 1.0,
                "router_escalated": False,
                "variant_effective": "cheap",
                "router_attempts": [{"variant": "cheap"}],
            },
        )

        # Cheap first, escalated to base => not counted as cheap accepted
        m2 = RequestMetrics(
            request_id=2,
            dataset_type="gsm8k",
            submit_time=0.0,
            start_time=0.0,
            end_time=1.0,
            difficulty="easy",
            inference_metrics={
                "success": True,
                "correct": 0,
                "total_cost_units": 2.0,
                "router_escalated": True,
                "variant_effective": "base",
                "router_attempts": [{"variant": "cheap"}, {"variant": "base"}],
            },
        )

        mc = MetricsCalculator([m1, m2], slo_dict={"default": {"ttft_ms": 10_000, "e2e_ms": 10_000}})
        report = mc.compute_all_metrics()
        s = report.get("summary", {})
        self.assertAlmostEqual(float(s.get("false_accept_rate_cheap", 0.0)), 1.0, places=6)
        self.assertEqual(int((s.get("accept_no_escalation_counts", {}) or {}).get("cheap", 0)), 1)


if __name__ == "__main__":
    unittest.main()
