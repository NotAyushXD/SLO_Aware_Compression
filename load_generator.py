# load_generator.py
"""
Closed-loop load generator: maintains constant concurrency and realistic queueing.

Key fixes vs previous version:
1) Correctly executes *all* num_requests while maintaining max_concurrency in-flight
   (fixes the as_completed(...) + append bug).
2) Correct queue wait time: submit_time is captured when the request is submitted,
   not when the worker thread starts executing.
3) Uses the SAME formatted prompts / token budgets as evaluation by calling
   build_llama_formatted_prompt(...). This aligns load testing with benchmark prompts.

This is still *parallel* (many requests in-flight). On a single GPU, actual model
execution will be serialized unless you implement batching; we intentionally keep
client-side concurrency so you can measure queueing + tail latency under load.
"""

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import List, Dict, Callable, Any
from dataclasses import dataclass, asdict
import logging

from prompt_templates import build_llama_formatted_prompt

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
        return (self.end_time - self.submit_time) * 1000

    @property
    def queue_wait_time_ms(self) -> float:
        return (self.start_time - self.submit_time) * 1000

    @property
    def inference_time_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def ttft_ms(self) -> float:
        return float(self.inference_metrics.get("ttft_ms", 0.0) or 0.0)

    @property
    def tpot_ms(self) -> float:
        return float(self.inference_metrics.get("tpot_ms", 0.0) or 0.0)

    @property
    def success(self) -> bool:
        return bool(self.inference_metrics.get("success", False))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["e2e_latency_ms"] = self.e2e_latency_ms
        d["queue_wait_time_ms"] = self.queue_wait_time_ms
        d["inference_time_ms"] = self.inference_time_ms
        return d


class ClosedLoopLoadGenerator:
    """
    Closed-loop load generator with fixed concurrency.
    Maintains constant number of concurrent requests throughout the test.
    """

    def __init__(
        self,
        inference_func: Callable[..., Any],
        max_concurrency: int,
        num_requests: int,
        data_loader: List[Dict[str, Any]],
        seed: int = 0,
    ):
        """
        Args:
            inference_func: Function that supports:
                generate(prompt=..., max_tokens=..., difficulty=...) -> (text, metrics_dict)
            max_concurrency: Number of concurrent requests to maintain
            num_requests: Total number of requests to send
            data_loader: List of {prompt, answer, difficulty, dataset, ...} dicts
        """
        self.inference_func = inference_func
        self.max_concurrency = int(max_concurrency)
        self.num_requests = int(num_requests)
        self.data_loader = list(data_loader)

        self.request_metrics: List[RequestMetrics] = []
        self.lock = threading.Lock()
        self.completed_count = 0

        logger.info("Initialized ClosedLoopLoadGenerator")
        logger.info(f"  Concurrency: {self.max_concurrency}")
        logger.info(f"  Total requests: {self.num_requests}")
        logger.info(f"  Data pool size: {len(self.data_loader)}")

        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if not self.data_loader:
            raise ValueError("data_loader is empty")

    def run_request(self, request_id: int, example: Dict[str, Any], submit_time: float) -> RequestMetrics:
        """Execute a single request with timing."""
        metrics = RequestMetrics(
            request_id=request_id,
            dataset_type=example.get("dataset", "unknown"),
            submit_time=submit_time,
            start_time=0.0,
            end_time=0.0,
            difficulty=example.get("difficulty", "medium"),
            inference_metrics={}
        )

        # Worker starts now (queue wait = start_time - submit_time)
        metrics.start_time = time.time()

        try:
            dataset_type = example.get("dataset", "mmlu")
            difficulty = example.get("difficulty", "medium")

            # IMPORTANT: ensure load testing uses the same formatted prompts as evaluation
            formatted_prompt, max_tokens, _stops = build_llama_formatted_prompt(example, dataset_type)

            try:
                generated_text, inference_metrics = self.inference_func(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    difficulty=difficulty,
                    dataset_type=dataset_type,
                )
            except TypeError:
                # Backwards compatibility: older server.generate() without dataset_type
                generated_text, inference_metrics = self.inference_func(
                    prompt=formatted_prompt,
                    max_tokens=max_tokens,
                    difficulty=difficulty,
                )

            metrics.inference_metrics = dict(inference_metrics or {})
            # If server provides more accurate wall times (e.g., after acquiring GPU lock),
            # use them so queue_wait includes time waiting for GPU access.
            try:
                if "server_infer_start_time_wall" in metrics.inference_metrics:
                    metrics.start_time = float(metrics.inference_metrics["server_infer_start_time_wall"])
                if "server_infer_end_time_wall" in metrics.inference_metrics:
                    # We'll set end_time later, but keep this to override after inference returns.
                    pass
            except Exception:
                pass

            # Store text only if you want (can bloat logs); omit by default
            metrics.inference_metrics.setdefault("success", True)

        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
            metrics.inference_metrics = {"success": False, "error": str(e)}

        metrics.end_time = time.time()
        try:
            if "server_infer_end_time_wall" in metrics.inference_metrics:
                metrics.end_time = float(metrics.inference_metrics["server_infer_end_time_wall"])
        except Exception:
            pass


        with self.lock:
            self.request_metrics.append(metrics)
            self.completed_count += 1

        return metrics

    def run(self) -> List[RequestMetrics]:
        """Execute load test with fixed concurrency."""
        logger.info("=" * 70)
        logger.info(f"STARTING LOAD TEST: {self.num_requests} requests @ {self.max_concurrency} concurrency")
        logger.info("=" * 70)

        t0 = time.time()

        def submit_one(executor: ThreadPoolExecutor, req_id: int):
            example = self.data_loader[req_id % len(self.data_loader)]
            submit_time = time.time()  # capture as close to submission as possible
            return executor.submit(self.run_request, req_id, example, submit_time)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            in_flight = set()
            next_id = 0

            # initial wave
            while next_id < self.num_requests and len(in_flight) < self.max_concurrency:
                in_flight.add(submit_one(executor, next_id))
                next_id += 1

            # closed-loop: keep concurrency constant
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

                for fut in done:
                    try:
                        _ = fut.result()
                    except Exception as e:
                        logger.error(f"Request future failed: {e}")

                    if next_id < self.num_requests:
                        in_flight.add(submit_one(executor, next_id))
                        next_id += 1

                # progress logging
                if self.completed_count > 0 and self.completed_count % 100 == 0:
                    elapsed = time.time() - t0
                    rate = self.completed_count / max(elapsed, 1e-6)
                    eta = (self.num_requests - self.completed_count) / max(rate, 1e-6)
                    logger.info(
                        f"Progress: {self.completed_count}/{self.num_requests} "
                        f"({rate:.2f} req/sec, ETA: {eta:.0f}s)"
                    )

        elapsed = time.time() - t0
        logger.info(f"Load test complete in {elapsed:.1f}s")
        return self.request_metrics

    def save_metrics(self, output_file: str):
        """Save all request metrics to JSONL file"""
        # output_file is JSONL in the rest of your pipeline
        with open(output_file, 'w') as f:
            for m in self.request_metrics:
                f.write(json.dumps(m.to_dict()) + "\n")
        logger.info(f"Saved metrics to {output_file}")


if __name__ == "__main__":
    # Simple smoke test with mock inference
    def mock_inference(prompt: str, max_tokens: int = 16, difficulty: str = "medium"):
        time.sleep(0.05)
        return "A", {"ttft_ms": 10.0, "tpot_ms": 5.0, "output_length": 1, "success": True}

    dummy_data = [
        {"dataset": "mmlu", "prompt": "What is 2+2?\nA) 3\nB) 4\nC) 5\nD) 6", "answer": "B", "difficulty": "easy"}
    ] * 10

    lg = ClosedLoopLoadGenerator(mock_inference, max_concurrency=2, num_requests=5, data_loader=dummy_data)
    _ = lg.run()
    lg.save_metrics("/tmp/requests.jsonl")