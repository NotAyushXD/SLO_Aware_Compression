"""load_generator.py

Closed-loop load generator for the SLO-aware compression harness.

Design goals:
- Simple and dependency-light (threads, not asyncio)
- Works with `SingleVariantServer.generate()` (server.py)
- Captures latency metrics at the request level so `metrics.py` can compute
  TTFT/TPOT/E2E/queue percentiles and SLO compliance.

The generator is "closed loop": each worker sends the next request only after
it finishes the previous one. This approximates many real serving setups where
clients wait for a response before sending the next request.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from metrics import InferenceMetrics
from prompt_templates import build_prompt


class ClosedLoopLoadGenerator:
    def __init__(
        self,
        server: Any,
        concurrency: int,
        total_requests: int,
        data_pool: List[Dict[str, Any]],
        prompt_mode: str,
        max_new_tokens: int,
        seed: int = 123,
    ) -> None:
        self.server = server
        self.concurrency = int(concurrency)
        self.total_requests = int(total_requests)
        self.data_pool = data_pool
        self.prompt_mode = prompt_mode
        self.max_new_tokens = int(max_new_tokens)
        self.rng = random.Random(seed)

        self._lock = threading.Lock()
        self._next_request_id = 0

    def _sample_example(self) -> Dict[str, Any]:
        return self.rng.choice(self.data_pool)

    def _run_one(self, request_id: int) -> InferenceMetrics:
        ex = self._sample_example()
        dataset_type = ex.get("dataset_type") or ex.get("dataset") or "gsm8k"
        difficulty = ex.get("difficulty", "easy")
        question = ex.get("question")

        if question is None:
            # Fallback for older processed format (stores `prompt` only)
            question = str(ex.get("prompt", ""))

        choices = ex.get("choices")

        prompt = build_prompt(
            dataset_type=dataset_type,
            question=question,
            choices=choices,
            prompt_mode=self.prompt_mode,
            difficulty=difficulty,
        )

        submit_time = time.time()
        start_time = time.time()
        outputs, per_req_metrics = self.server.generate(
            [prompt],
            dataset_type=dataset_type,
            difficulty=difficulty,
            prompt_mode=self.prompt_mode,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        end_time = time.time()

        metrics_dict = per_req_metrics[0] if per_req_metrics else {}

        return InferenceMetrics(
            request_id=request_id,
            variant=str(getattr(self.server, "variant", "unknown")),
            dataset_type=str(dataset_type),
            difficulty=str(difficulty),
            prompt_mode=str(self.prompt_mode),
            success=True,
            submit_time=submit_time,
            start_time=start_time,
            end_time=end_time,
            queue_wait_ms=float(metrics_dict.get("queue_wait_ms", 0.0) or 0.0),
            ttft_ms=float(metrics_dict.get("ttft_ms", 0.0) or 0.0),
            tpot_ms=float(metrics_dict.get("tpot_ms", 0.0) or 0.0),
            prompt_tokens=int(metrics_dict.get("prompt_tokens", 0) or 0),
            output_tokens=int(metrics_dict.get("output_tokens", metrics_dict.get("output_length", 0)) or 0),
        )

    def _worker_loop(self) -> List[InferenceMetrics]:
        results: List[InferenceMetrics] = []
        while True:
            with self._lock:
                if self._next_request_id >= self.total_requests:
                    break
                request_id = self._next_request_id
                self._next_request_id += 1

            try:
                results.append(self._run_one(request_id))
            except Exception:
                # Record failure with timestamps so metrics can still compute duration.
                now = time.time()
                results.append(
                    InferenceMetrics(
                        request_id=request_id,
                        variant=str(getattr(self.server, "variant", "unknown")),
                        dataset_type="unknown",
                        difficulty="unknown",
                        prompt_mode=str(self.prompt_mode),
                        success=False,
                        submit_time=now,
                        start_time=now,
                        end_time=now,
                        queue_wait_ms=0.0,
                        ttft_ms=0.0,
                        tpot_ms=0.0,
                        prompt_tokens=0,
                        output_tokens=0,
                    )
                )
        return results

    def run(self, log_every: int = 5) -> List[InferenceMetrics]:
        """Run the load test and return per-request metrics."""
        if not self.data_pool:
            raise ValueError("data_pool is empty; cannot run load test")

        start = time.time()
        all_results: List[InferenceMetrics] = []

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = [ex.submit(self._worker_loop) for _ in range(self.concurrency)]
            done_count = 0
            for fut in as_completed(futures):
                worker_results = fut.result()
                all_results.extend(worker_results)
                done_count += len(worker_results)
                if log_every and done_count % log_every == 0:
                    elapsed = time.time() - start
                    rps = done_count / elapsed if elapsed > 0 else 0.0
                    print(f"Progress: {done_count}/{self.total_requests} ({rps:.2f} req/sec)")

        # Ensure stable order by request_id
        all_results.sort(key=lambda m: m.request_id)
        total_time = time.time() - start
        print(f"Load test complete in {total_time:.1f}s")
        return all_results

    @staticmethod
    def save_requests_jsonl(metrics_list: List[InferenceMetrics], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for m in metrics_list:
                f.write(json.dumps(asdict(m)) + "\n")
