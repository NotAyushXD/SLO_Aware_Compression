# metrics.py
"""
Metrics calculation and SLO compliance checking
Computes percentiles, throughput, and SLO attainment
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PercentileMetrics:
    """Container for percentile-based metrics"""
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    mean: float
    std: float


class MetricsCalculator:
    """Calculate comprehensive metrics from request logs"""
    
    # Default SLO targets by difficulty level
    DEFAULT_SLOS = {
        "easy": {"ttft_ms": 200, "tpot_ms": 20},
        "medium": {"ttft_ms": 300, "tpot_ms": 25},
        "hard": {"ttft_ms": 500, "tpot_ms": 40}
    }
    
    def __init__(self, request_metrics: List, slo_dict: Optional[Dict] = None):
        """
        Args:
            request_metrics: List of RequestMetrics objects
            slo_dict: SLO thresholds by difficulty; uses defaults if None
        """
        self.metrics = request_metrics
        self.slo_dict = slo_dict or self.DEFAULT_SLOS
        
        logger.info(f"Initialized MetricsCalculator with {len(request_metrics)} metrics")
    
    def calculate_percentiles(self, values: List[float]) -> PercentileMetrics:
        """Calculate percentile metrics from values list"""
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
            std=float(np.std(values))
        )
    
    def compute_all_metrics(self) -> Dict:
        """
        Compute comprehensive metrics across all requests
        
        Returns:
            Dict with summary, latency, throughput, and SLO metrics
        """
        # Basic statistics
        total_requests = len(self.metrics)
        successful = sum(1 for m in self.metrics if m.success)
        success_rate = successful / max(total_requests, 1)
        
        # Extract latency values
        ttft_values = [m.ttft_ms for m in self.metrics if m.ttft_ms > 0]
        tpot_values = [m.tpot_ms for m in self.metrics if m.tpot_ms > 0]
        e2e_latency_values = [m.e2e_latency_ms for m in self.metrics]
        queue_wait_values = [m.queue_wait_time_ms for m in self.metrics]
        
        # Calculate percentiles
        ttft_percentiles = self.calculate_percentiles(ttft_values)
        tpot_percentiles = self.calculate_percentiles(tpot_values)
        e2e_percentiles = self.calculate_percentiles(e2e_latency_values)
        queue_percentiles = self.calculate_percentiles(queue_wait_values)
        
        # Throughput calculation
        if self.metrics:
            first_submit = min(m.submit_time for m in self.metrics)
            last_complete = max(m.end_time for m in self.metrics)
            total_duration = max(last_complete - first_submit, 0.001)
            
            total_output_tokens = sum(m.inference_metrics.get("output_length", 0)
                                      for m in self.metrics if m.success)
            throughput = total_output_tokens / total_duration
        else:
            throughput = 0
        
        # SLO compliance check
        slo_compliant = 0
        slo_violations = 0
        slo_violation_details = []
        
        for m in self.metrics:
            if not m.success:
                slo_violations += 1
                continue
            
            difficulty = m.difficulty
            slo = self.slo_dict.get(difficulty, self.slo_dict["medium"])
            
            ttft = m.ttft_ms
            tpot = m.tpot_ms
            
            # Check SLO compliance
            ttft_ok = ttft <= slo["ttft_ms"]
            tpot_ok = tpot <= slo["tpot_ms"]
            
            if ttft_ok and tpot_ok:
                slo_compliant += 1
            else:
                slo_violations += 1
                slo_violation_details.append({
                    "request_id": m.request_id,
                    "difficulty": difficulty,
                    "ttft_ms": ttft,
                    "ttft_slo": slo["ttft_ms"],
                    "tpot_ms": tpot,
                    "tpot_slo": slo["tpot_ms"]
                })
        
        slo_compliance = slo_compliant / max(successful, 1)
        
        # Escalation analysis (requests routed to expensive variants)
        # For single-variant baseline, this is 0
        escalation_rate = 0.0
        
        return {
            "summary": {
                "total_requests": total_requests,
                "successful_requests": successful,
                "failed_requests": total_requests - successful,
                "success_rate": success_rate,
                "slo_compliant": slo_compliant,
                "slo_violations": slo_violations,
                "slo_compliance": slo_compliance,
                "escalation_rate": escalation_rate,
                "total_duration_sec": total_duration if self.metrics else 0,
                "throughput_tokens_per_sec": throughput
            },
            "ttft": {
                "p50": ttft_percentiles.p50,
                "p75": ttft_percentiles.p75,
                "p90": ttft_percentiles.p90,
                "p95": ttft_percentiles.p95,
                "p99": ttft_percentiles.p99,
                "mean": ttft_percentiles.mean,
                "std": ttft_percentiles.std
            },
            "tpot": {
                "p50": tpot_percentiles.p50,
                "p75": tpot_percentiles.p75,
                "p90": tpot_percentiles.p90,
                "p95": tpot_percentiles.p95,
                "p99": tpot_percentiles.p99,
                "mean": tpot_percentiles.mean,
                "std": tpot_percentiles.std
            },
            "e2e_latency": {
                "p50": e2e_percentiles.p50,
                "p75": e2e_percentiles.p75,
                "p90": e2e_percentiles.p90,
                "p95": e2e_percentiles.p95,
                "p99": e2e_percentiles.p99,
                "mean": e2e_percentiles.mean,
                "std": e2e_percentiles.std
            },
            "queue_wait": {
                "p50": queue_percentiles.p50,
                "p75": queue_percentiles.p75,
                "p90": queue_percentiles.p90,
                "p95": queue_percentiles.p95,
                "p99": queue_percentiles.p99,
                "mean": queue_percentiles.mean,
                "std": queue_percentiles.std
            },
            "slo_violations": slo_violation_details[:10]  # First 10 violations
        }
    
    def print_report(self, title: str = "BASELINE SERVER METRICS REPORT"):
        """Print formatted metrics report"""
        metrics = self.compute_all_metrics()
        
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        # Summary section
        print("\nSUMMARY:")
        print(f"  Total Requests:        {metrics['summary']['total_requests']:6d}")
        print(f"  Successful:            {metrics['summary']['successful_requests']:6d}")
        print(f"  Failed:                {metrics['summary']['failed_requests']:6d}")
        print(f"  Success Rate:          {metrics['summary']['success_rate']*100:6.2f}%")
        print(f"  Total Duration:        {metrics['summary']['total_duration_sec']:6.2f} seconds")
        print(f"  Throughput:            {metrics['summary']['throughput_tokens_per_sec']:6.1f} tokens/sec")
        print(f"  SLO Compliance:        {metrics['summary']['slo_compliance']*100:6.2f}%")
        print(f"  SLO Violations:        {metrics['summary']['slo_violations']:6d}")
        print(f"  Escalation Rate:       {metrics['summary']['escalation_rate']*100:6.2f}%")
        
        # TTFT section
        print("\nTTFT (Time-to-First-Token) in milliseconds:")
        print(f"  P50:  {metrics['ttft']['p50']:7.2f} ms")
        print(f"  P75:  {metrics['ttft']['p75']:7.2f} ms")
        print(f"  P90:  {metrics['ttft']['p90']:7.2f} ms")
        print(f"  P95:  {metrics['ttft']['p95']:7.2f} ms")
        print(f"  P99:  {metrics['ttft']['p99']:7.2f} ms")
        print(f"  Mean: {metrics['ttft']['mean']:7.2f} ms (±{metrics['ttft']['std']:.2f})")
        
        # TPOT section
        print("\nTPOT (Time-Per-Output-Token) in milliseconds:")
        print(f"  P50:  {metrics['tpot']['p50']:7.2f} ms")
        print(f"  P75:  {metrics['tpot']['p75']:7.2f} ms")
        print(f"  P90:  {metrics['tpot']['p90']:7.2f} ms")
        print(f"  P95:  {metrics['tpot']['p95']:7.2f} ms")
        print(f"  P99:  {metrics['tpot']['p99']:7.2f} ms")
        print(f"  Mean: {metrics['tpot']['mean']:7.2f} ms (±{metrics['tpot']['std']:.2f})")
        
        # E2E Latency section
        print("\nE2E Latency (End-to-End) in milliseconds:")
        print(f"  P50:  {metrics['e2e_latency']['p50']:7.2f} ms")
        print(f"  P75:  {metrics['e2e_latency']['p75']:7.2f} ms")
        print(f"  P90:  {metrics['e2e_latency']['p90']:7.2f} ms")
        print(f"  P95:  {metrics['e2e_latency']['p95']:7.2f} ms")
        print(f"  P99:  {metrics['e2e_latency']['p99']:7.2f} ms")
        print(f"  Mean: {metrics['e2e_latency']['mean']:7.2f} ms (±{metrics['e2e_latency']['std']:.2f})")
        
        # Queue wait section
        print("\nQueue Wait Time in milliseconds:")
        print(f"  P50:  {metrics['queue_wait']['p50']:7.2f} ms")
        print(f"  P95:  {metrics['queue_wait']['p95']:7.2f} ms")
        print(f"  P99:  {metrics['queue_wait']['p99']:7.2f} ms")
        print(f"  Mean: {metrics['queue_wait']['mean']:7.2f} ms")
        
        # SLO violations
        if metrics['slo_violations']:
            print("\nSample SLO Violations (first 10):")
            for i, violation in enumerate(metrics['slo_violations'][:10], 1):
                print(f"  {i}. Request {violation['request_id']} ({violation['difficulty']})")
                print(f"     TTFT: {violation['ttft_ms']:.1f}ms > {violation['ttft_slo']}ms SLO")
                print(f"     TPOT: {violation['tpot_ms']:.1f}ms > {violation['tpot_slo']}ms SLO")
        
        print("\n" + "="*80)
        
        return metrics
    
    def save_metrics(self, output_file: str):
        """Save metrics to JSON file"""
        metrics = self.compute_all_metrics()
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    # Test with mock metrics
    from load_generator import RequestMetrics
    
    # Create mock metrics
    mock_metrics = [
        RequestMetrics(
            request_id=i,
            dataset_type="mmlu",
            submit_time=time.time() - 100 + i * 0.1,
            start_time=time.time() - 100 + i * 0.1,
            end_time=time.time() - 100 + i * 0.1 + 0.5,
            difficulty="easy" if i % 3 == 0 else "medium",
            inference_metrics={
                "ttft_ms": 50 + i % 50,
                "tpot_ms": 15 + i % 10,
                "output_length": 100,
                "success": True
            }
        )
        for i in range(100)
    ]
    
    calc = MetricsCalculator(mock_metrics)
    calc.print_report("TEST METRICS REPORT")
