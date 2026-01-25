# load_generator.py
"""
Closed-loop load generator (fixed concurrency).

Key fixes vs earlier versions:
- True closed-loop behavior: keep ~N in-flight requests until total is reached.
- Correct queue_wait_time_ms: submit_time recorded at submission, start_time when worker starts.
- Adds prompt_mode plumbing (accuracy vs slo) without breaking parallel execution.

This tool is mainly for your SLO work. For pure accuracy debugging you can also
skip load tests via --skip_load_test in run_baseline_evaluation.py (added in v7).
"""

from __future__ import annotations

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass, asdict
import logging

from prompt_templates import build_llama_formatted_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    request_id: int
    dataset_type: str
    submit_time: float
    start_time: float
    end_time: float
    difficulty: str
    inference_metrics: Dict[str, Any]

    @property
    def e2e_latency_ms(self) -> float:
        return (self.end_time - self.submit_time) * 1000.0

    @property
    def queue_wait_time_ms(self) -> float:
        return (self.start_time - self.submit_time) * 1000.0

    @property
    def inference_time_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000.0

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
    def __init__(
        self,
        inference_func: Callable[..., Any],
        max_concurrency: int,
        num_requests: int,
        data_loader: List[Dict[str, Any]],
        prompt_mode: str = "slo",
        seed: int = 0,
    ):
        self.inference_func = inference_func
        self.max_concurrency = int(max_concurrency)
        self.num_requests = int(num_requests)
        self.data_loader = data_loader
        self.prompt_mode = prompt_mode
        self.seed = seed

        self.request_metrics: List[RequestMetrics] = []
        self._lock = threading.Lock()

        logger.info("Initialized ClosedLoopLoadGenerator")
        logger.info(f"  Concurrency: {self.max_concurrency}")
        logger.info(f"  Total requests: {self.num_requests}")
        logger.info(f"  Data pool size: {len(self.data_loader)}")
        logger.info(f"  Prompt mode: {self.prompt_mode}")

    def _select_example(self, request_id: int) -> Dict[str, Any]:
        # deterministic cycling is fine for load testing
        if not self.data_loader:
            return {"dataset": "unknown", "prompt": "", "answer": "", "difficulty": "medium"}
        return self.data_loader[request_id % len(self.data_loader)]

    def run_request(self, request_id: int, submit_time: float) -> RequestMetrics:
        ex = self._select_example(request_id)
        dataset_type = ex.get("dataset", "unknown")
        difficulty = ex.get("difficulty", "medium")

        start_time = time.time()
        inference_metrics: Dict[str, Any] = {}

        try:
            prompt, max_tokens, _stops = build_llama_formatted_prompt(ex, dataset_type, prompt_mode=self.prompt_mode)

            # Call server.generate; keep backward-compat fallbacks
            try:
                _text, inf = self.inference_func(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    difficulty=difficulty,
                    dataset_type=dataset_type,
                    prompt_mode=self.prompt_mode,
                )
            except TypeError:
                # Older server.generate signatures
                _text, inf = self.inference_func(prompt=prompt, max_tokens=max_tokens, difficulty=difficulty)

            inference_metrics = inf or {}
        except Exception as e:
            inference_metrics = {"success": False, "error": str(e)}

        end_time = time.time()

        rm = RequestMetrics(
            request_id=request_id,
            dataset_type=dataset_type,
            submit_time=submit_time,
            start_time=start_time,
            end_time=end_time,
            difficulty=difficulty,
            inference_metrics=inference_metrics,
        )
        return rm

    def run(self) -> List[RequestMetrics]:
        logger.info("=" * 70)
        logger.info(f"STARTING LOAD TEST: {self.num_requests} requests @ {self.max_concurrency} concurrency")
        logger.info("=" * 70)

        t0 = time.time()

        in_flight = {}
        next_id = 0

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            # Initial fill
            while next_id < self.num_requests and len(in_flight) < self.max_concurrency:
                submit_time = time.time()
                fut = pool.submit(self.run_request, next_id, submit_time)
                in_flight[fut] = next_id
                next_id += 1

            completed = 0
            last_log = t0

            while in_flight:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)

                for fut in done:
                    req_id = in_flight.pop(fut)
                    try:
                        rm = fut.result()
                    except Exception as e:
                        rm = RequestMetrics(
                            request_id=req_id,
                            dataset_type="unknown",
                            submit_time=t0,
                            start_time=t0,
                            end_time=time.time(),
                            difficulty="medium",
                            inference_metrics={"success": False, "error": str(e)},
                        )

                    with self._lock:
                        self.request_metrics.append(rm)

                    completed += 1

                    # Submit next to maintain closed-loop
                    if next_id < self.num_requests:
                        submit_time = time.time()
                        nfut = pool.submit(self.run_request, next_id, submit_time)
                        in_flight[nfut] = next_id
                        next_id += 1

                now = time.time()
                if now - last_log > 30 and completed > 0:
                    rate = completed / max(now - t0, 1e-6)
                    eta = (self.num_requests - completed) / max(rate, 1e-6)
                    logger.info(f"Progress: {completed}/{self.num_requests} ({rate:.2f} req/sec, ETA: {eta:.0f}s)")
                    last_log = now

        total = time.time() - t0
        logger.info(f"Load test complete in {total:.1f}s")
        return self.request_metrics

    def save_metrics(self, path: str) -> None:
        with open(path, "w") as f:
            for m in self.request_metrics:
                f.write(json.dumps(m.to_dict()) + "\n")
        logger.info(f"Saved metrics to {path}")
