# load_generator.py
"""Closed-loop load generator (fixed concurrency).

Notes for paper-grade measurements:
- submit_time is recorded at submission.
- start_time is when the worker thread begins execution (client-side queueing).
- end_time is when the response is received.

Server-side metrics (ttft_ms, tpot_ms, queue_wait_ms, ...) are passed through
from server.generate().
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from prompt_templates import build_llama_formatted_prompt
from evaluation import EvaluationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        # If False, do NOT send gold/quality labels to the server.
        # This enables realistic "delayed label" experiments where labels arrive later.
        send_labels_to_server: bool = True,
    ):
        self.inference_func = inference_func
        self.max_concurrency = int(max_concurrency)
        self.num_requests = int(num_requests)
        self.data_loader = data_loader
        self.prompt_mode = (prompt_mode or "slo").lower().strip()
        self.seed = int(seed)
        self.send_labels_to_server = bool(send_labels_to_server)

        self.request_metrics: List[RequestMetrics] = []
        self._lock = threading.Lock()

        logger.info("Initialized ClosedLoopLoadGenerator")
        logger.info(f"  Concurrency: {self.max_concurrency}")
        logger.info(f"  Total requests: {self.num_requests}")
        logger.info(f"  Data pool size: {len(self.data_loader)}")
        logger.info(f"  Prompt mode: {self.prompt_mode}")
        logger.info(f"  Seed: {self.seed}")
        logger.info(f"  Send labels to server: {self.send_labels_to_server}")

    def _select_example(self, request_id: int) -> Dict[str, Any]:
        # Deterministic selection that changes with seed.
        if not self.data_loader:
            return {"dataset": "unknown", "prompt": "", "answer": "", "difficulty": "medium"}
        idx = (request_id + self.seed) % len(self.data_loader)
        return self.data_loader[idx]

    def run_request(self, request_id: int, submit_time: float) -> RequestMetrics:
        ex = self._select_example(request_id)
        dataset_type = ex.get("dataset", "unknown")
        difficulty = ex.get("difficulty", "medium")

        worker_start_time = time.time()
        inference_metrics: Dict[str, Any] = {}

        try:
            prompt, max_tokens, _stops = build_llama_formatted_prompt(ex, dataset_type, prompt_mode=self.prompt_mode)

            # Call server.generate; keep backward-compatible fallbacks.
            try:
                call_kwargs = dict(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    dataset_type=dataset_type,
                    difficulty=difficulty,
                    prompt_mode=self.prompt_mode,
                    concurrency=self.max_concurrency,
                    request_id=int(request_id),
                )

                # Optional adapter overrides (used by adapter-churn experiments).
                if ex.get("adapter_id") is not None:
                    call_kwargs["adapter_id"] = ex.get("adapter_id")
                if ex.get("adapter_rank") is not None:
                    call_kwargs["adapter_rank"] = ex.get("adapter_rank")

                # Gold labels: only send when requested.
                if self.send_labels_to_server:
                    call_kwargs["gold_answer"] = str(ex.get("answer", ""))
                    call_kwargs["label_source"] = "gold"

                _pred_text, inference_metrics = self.inference_func(
                    **call_kwargs,
                )
            except TypeError:
                _pred_text, inference_metrics = self.inference_func(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    dataset_type=dataset_type,
                    difficulty=difficulty,
                    request_id=int(request_id),
                    gold_answer=(str(ex.get("answer", "")) if self.send_labels_to_server else None),
                    label_source=("gold" if self.send_labels_to_server else None),
                )

            # ----------------------------
            # Online quality signals (paper-facing): correctness + format adherence
            # ----------------------------
            # This enables joint latency/quality analysis under load, and supports
            # metrics like "quality-adjusted goodput".
            try:
                truth = ex.get("answer", "")
                ok, extracted, fmt_ok = EvaluationMetrics.is_correct(_pred_text, str(truth), dataset_type)
                ok_p, extracted_p, fmt_ok_p = EvaluationMetrics.is_correct_parseable(_pred_text, str(truth), dataset_type)
                # Preserve server-side correctness if already present (e.g., bandit labels).
                if "correct" not in inference_metrics:
                    inference_metrics["correct"] = int(bool(ok))
                if "format_ok" not in inference_metrics:
                    inference_metrics["format_ok"] = int(bool(fmt_ok))
                if "extracted_answer" not in inference_metrics:
                    inference_metrics["extracted_answer"] = extracted

                # Always record parseable metrics (sensitivity analysis)
                inference_metrics.update(
                    {
                        "correct_parseable": int(bool(ok_p)),
                        "format_ok_parseable": int(bool(fmt_ok_p)),
                        "extracted_answer_parseable": extracted_p,
                    }
                )
            except Exception:
                # Never fail the load generator due to evaluation quirks.
                pass

        except Exception as e:
            inference_metrics = {
                "success": False,
                "error": str(e),
                "ttft_ms": 0.0,
                "tpot_ms": 0.0,
                "output_length": 0,
                "throughput_tokens_per_sec": 0.0,
                "queue_wait_ms": 0.0,
                "total_latency_ms": 0.0,
                "correct": 0,
                "format_ok": 0,
                "correct_parseable": 0,
                "format_ok_parseable": 0,
            }

        worker_end_time = time.time()

        return RequestMetrics(
            request_id=request_id,
            dataset_type=dataset_type,
            submit_time=submit_time,
            start_time=worker_start_time,
            end_time=worker_end_time,
            difficulty=difficulty,
            inference_metrics=inference_metrics,
        )

    def run_load_test(self) -> List[RequestMetrics]:
        logger.info("=" * 70)
        logger.info(f"STARTING LOAD TEST: {self.num_requests} requests @ {self.max_concurrency} concurrency")
        logger.info("=" * 70)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            in_flight = set()
            next_request_id = 0

            # Prime pipeline.
            while next_request_id < min(self.max_concurrency, self.num_requests):
                submit_time = time.time()
                fut = executor.submit(self.run_request, next_request_id, submit_time)
                in_flight.add(fut)
                next_request_id += 1

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    rm = fut.result()
                    with self._lock:
                        self.request_metrics.append(rm)

                # Submit more.
                while next_request_id < self.num_requests and len(in_flight) < self.max_concurrency:
                    submit_time = time.time()
                    fut = executor.submit(self.run_request, next_request_id, submit_time)
                    in_flight.add(fut)
                    next_request_id += 1

                if len(self.request_metrics) % 5 == 0 and self.request_metrics:
                    elapsed = time.time() - start_time
                    rate = len(self.request_metrics) / max(elapsed, 1e-9)
                    remaining = self.num_requests - len(self.request_metrics)
                    eta = remaining / max(rate, 1e-9)
                    logger.info(
                        f"Progress: {len(self.request_metrics)}/{self.num_requests} ({rate:.2f} req/sec, ETA: {eta:.0f}s)"
                    )

        total_time = time.time() - start_time
        logger.info(f"Load test complete in {total_time:.1f}s")

        return self.request_metrics

    def save_results(self, output_file: str):
        with open(output_file, "w") as f:
            for m in self.request_metrics:
                f.write(json.dumps(m.to_dict()) + "\n")
        logger.info(f"Saved metrics to {output_file}")
