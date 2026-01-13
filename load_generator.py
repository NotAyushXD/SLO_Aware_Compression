# load_generator.py
"""
Closed-loop load generator: maintains constant concurrency
Submits N concurrent requests continuously and collects per-request metrics
"""

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Per-request metrics collected during load test"""
    request_id: int
    dataset_type: str
    submit_time: float
    start_time: float
    end_time: float
    difficulty: str
    inference_metrics: Dict[str, Any]
    
    @property
    def e2e_latency_ms(self) -> float:
        """End-to-end latency in milliseconds"""
        return (self.end_time - self.submit_time) * 1000
    
    @property
    def queue_wait_time_ms(self) -> float:
        """Time spent waiting in queue"""
        return (self.start_time - self.submit_time) * 1000
    
    @property
    def inference_time_ms(self) -> float:
        """Time spent in actual inference"""
        return (self.end_time - self.start_time) * 1000
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token from inference metrics"""
        return self.inference_metrics.get("ttft_ms", 0)
    
    @property
    def tpot_ms(self) -> float:
        """Time per output token from inference metrics"""
        return self.inference_metrics.get("tpot_ms", 0)
    
    @property
    def success(self) -> bool:
        """Whether request succeeded"""
        return self.inference_metrics.get("success", False)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d["e2e_latency_ms"] = self.e2e_latency_ms
        d["queue_wait_time_ms"] = self.queue_wait_time_ms
        d["inference_time_ms"] = self.inference_time_ms
        return d


class ClosedLoopLoadGenerator:
    """
    Closed-loop load generator with fixed concurrency
    Maintains constant number of concurrent requests throughout test
    """
    
    def __init__(self,
                 inference_func: Callable,
                 max_concurrency: int,
                 num_requests: int,
                 data_loader: List[Dict]):
        """
        Args:
            inference_func: Function(prompt, max_tokens) -> (text, metrics_dict)
            max_concurrency: Number of concurrent requests to maintain
            num_requests: Total number of requests to send
            data_loader: List of {prompt, answer, difficulty, dataset, ...} dicts
        """
        self.inference_func = inference_func
        self.max_concurrency = max_concurrency
        self.num_requests = num_requests
        self.data_loader = data_loader
        
        self.request_metrics: List[RequestMetrics] = []
        self.lock = threading.Lock()
        self.completed_count = 0
        
        logger.info(f"Initialized ClosedLoopLoadGenerator")
        logger.info(f"  Concurrency: {max_concurrency}")
        logger.info(f"  Total requests: {num_requests}")
        logger.info(f"  Data pool size: {len(data_loader)}")
    
    def run_request(self, request_id: int, example: Dict) -> RequestMetrics:
        """
        Execute single request with timing
        
        Returns:
            RequestMetrics object with all measurements
        """
        metrics = RequestMetrics(
            request_id=request_id,
            dataset_type=example.get("dataset", "unknown"),
            submit_time=time.time(),
            start_time=None,
            end_time=None,
            difficulty=example.get("difficulty", "medium"),
            inference_metrics={}
        )
        
        # Wait to be processed (simulating queue)
        metrics.start_time = time.time()
        
        try:
            # Call inference function
            generated_text, inference_metrics = self.inference_func(
                prompt=example["prompt"],
                max_tokens=example.get("output_length", 128)
            )
            
            # Store inference metrics
            metrics.inference_metrics = inference_metrics
            
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
            metrics.inference_metrics["success"] = False
            metrics.inference_metrics["error"] = str(e)
        
        metrics.end_time = time.time()
        
        # Store metrics (thread-safe)
        with self.lock:
            self.request_metrics.append(metrics)
            self.completed_count += 1
        
        return metrics
    
    def run(self) -> List[RequestMetrics]:
        """
        Execute load test with fixed concurrency
        
        Returns:
            List of RequestMetrics for all completed requests
        """
        logger.info("="*70)
        logger.info(f"STARTING LOAD TEST: {self.num_requests} requests @ {self.max_concurrency} concurrency")
        logger.info("="*70)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = []
            
            # Submit initial batch of requests
            for i in range(min(self.num_requests, self.max_concurrency)):
                example = self.data_loader[i % len(self.data_loader)]
                future = executor.submit(self.run_request, i, example)
                futures.append(future)
            
            # Keep track of submitted requests
            submitted = self.max_concurrency
            
            # Process results as they complete and submit new requests
            for future in as_completed(futures):
                _ = future.result()
                
                # Submit next request if available
                if submitted < self.num_requests:
                    example = self.data_loader[submitted % len(self.data_loader)]
                    new_future = executor.submit(self.run_request, submitted, example)
                    futures.append(new_future)
                    submitted += 1
                
                # Progress logging
                if self.completed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = self.completed_count / max(elapsed, 0.001)
                    eta = (self.num_requests - self.completed_count) / max(rate, 0.001)
                    logger.info(f"Progress: {self.completed_count}/{self.num_requests} "
                               f"({rate:.1f} req/sec, ETA: {eta:.0f}s)")
            
            # Wait for all remaining requests
            for future in as_completed(futures):
                _ = future.result()
        
        elapsed = time.time() - start_time
        logger.info(f"Load test complete in {elapsed:.1f}s")
        
        return self.request_metrics
    
    def save_metrics(self, output_file: str):
        """Save all request metrics to JSON file"""
        metrics_list = [m.to_dict() for m in self.request_metrics]
        with open(output_file, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        logger.info(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    # Test load generator with mock inference function
    def mock_inference(prompt: str, max_tokens: int = 128) -> tuple:
        """Mock inference function for testing"""
        time.sleep(0.1)  # Simulate inference latency
        return "mock response", {
            "ttft_ms": 45,
            "tpot_ms": 15,
            "output_length": 32,
            "success": True
        }
    
    # Create test data
    test_data = [
        {
            "prompt": f"Question {i}?",
            "answer": "Answer",
            "difficulty": "easy" if i % 3 == 0 else "medium" if i % 3 == 1 else "hard",
            "dataset": "mmlu",
            "output_length": 128
        }
        for i in range(100)
    ]
    
    # Run load test
    gen = ClosedLoopLoadGenerator(
        inference_func=mock_inference,
        max_concurrency=4,
        num_requests=50,
        data_loader=test_data
    )
    
    metrics = gen.run()
    logger.info(f"\nCollected {len(metrics)} request metrics")
    logger.info(f"Success rate: {sum(1 for m in metrics if m.success) / len(metrics) * 100:.1f}%")
