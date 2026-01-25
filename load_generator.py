# load_generator.py
"""
Closed-loop load generator (fixed concurrency) that actually maintains the
target concurrency until num_requests is reached.

This fixes the classic bug where as_completed() only iterates over the initial
futures, causing you to run ~2x concurrency requests instead of the full count.

Used by run_baseline_evaluation.py for load tests + SLO measurement.
"""

from __future__ import annotations

import json
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

logger = logging.getLogger("load_generator")
logger.setLevel(logging.INFO)


@dataclass
class RequestMetrics:
    request_id: int
    dataset: str = ""
    difficulty: str = ""
    prompt_mode: str = ""
    prompt: str = ""
    submit_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    queue_wait_ms: float = 0.0
    inference_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "dataset": self.dataset,
            "difficulty": self.difficulty,
            "prompt_mode": self.prompt_mode,
            "prompt": self.prompt,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.end_time - self.start_time,
            "queue_wait_ms": self.queue_wait_ms,
            "inference_metrics": self.inference_metrics,
        }


class ClosedLoopLoadGenerator:
    """
    Executes a closed-loop load test:
    - maintain up to `max_concurrency` in-flight requests
    - submit next request immediately when one finishes
    - stop when `num_requests` completed
    """

    def __init__(
        self,
        inference_func: Callable[[Dict[str, Any]], tuple],
        max_concurrency: int,
        num_requests: int,
        data_loader: List[Dict[str, Any]],
        prompt_mode: str = "slo",
    ):
        self.inference_func = inference_func
        self.max_concurrency = int(max_concurrency)
        self.num_requests = int(num_requests)
        self.data_loader = list(data_loader)
        self.prompt_mode = (prompt_mode or "slo").lower()

        self.request_metrics: List[RequestMetrics] = []
        self.lock = threading.Lock()
        self.completed_count = 0

        logger.info("Initialized ClosedLoopLoadGenerator")
        logger.info(f"  Concurrency: {self.max_concurrency}")
        logger.info(f"  Total requests: {self.num_requests}")
        logger.info(f"  Data pool size: {len(self.data_loader)}")

    def _run_one(self, request_id: int, example: Dict[str, Any], submitted_time: float) -> RequestMetrics:
        m = RequestMetrics(
            request_id=request_id,
            dataset=(example.get("dataset") or example.get("dataset_type") or "").lower(),
            difficulty=(example.get("difficulty") or "medium").lower(),
            prompt_mode=self.prompt_mode,
            prompt=(example.get("prompt") or ""),
        )
        m.submit_time = submitted_time
        m.start_time = time.time()
        m.queue_wait_ms = (m.start_time - submitted_time) * 1000.0

        try:
            _, inf = self.inference_func(example)
            m.inference_metrics = inf or {}
            m.inference_metrics.setdefault("success", True)
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            m.inference_metrics = {"success": False, "error": str(e)}

        m.end_time = time.time()

        with self.lock:
            self.request_metrics.append(m)
            self.completed_count += 1

        return m

    def run(self) -> List[RequestMetrics]:
        logger.info("=" * 70)
        logger.info(f"STARTING LOAD TEST: {self.num_requests} requests @ {self.max_concurrency} concurrency")
        logger.info("=" * 70)

        start = time.time()
        next_id = 0
        in_flight = set()

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            # Prime the queue
            while next_id < self.num_requests and len(in_flight) < self.max_concurrency:
                example = self.data_loader[next_id % len(self.data_loader)]
                submitted_time = time.time()
                fut = ex.submit(self._run_one, next_id, example, submitted_time)
                in_flight.add(fut)
                next_id += 1

            # Maintain concurrency
            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)

                # Submit new tasks for each completed future
                for _ in done:
                    if next_id < self.num_requests:
                        example = self.data_loader[next_id % len(self.data_loader)]
                        submitted_time = time.time()
                        fut = ex.submit(self._run_one, next_id, example, submitted_time)
                        in_flight.add(fut)
                        next_id += 1

                # Progress logging
                if self.completed_count and self.completed_count % 100 == 0:
                    elapsed = time.time() - start
                    rate = self.completed_count / max(elapsed, 1e-6)
                    eta = (self.num_requests - self.completed_count) / max(rate, 1e-6)
                    logger.info(f"Progress: {self.completed_count}/{self.num_requests} "
                                f"({rate:.2f} req/sec, ETA: {eta:.0f}s)")

        elapsed = time.time() - start
        logger.info(f"Load test complete in {elapsed:.1f}s")
        return self.request_metrics

    def save_metrics(self, output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump([m.to_dict() for m in self.request_metrics], f, indent=2)
        logger.info(f"Saved metrics to {output_file}")

    def save_requests_jsonl(self, output_file: str) -> None:
        with open(output_file, "w") as f:
            for m in self.request_metrics:
                f.write(json.dumps(m.to_dict()) + "\n")
        logger.info(f"Saved metrics to {output_file}")