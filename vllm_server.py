# vllm_server.py
"""Optional vLLM backend (async streaming) for the same evaluation stack.

Design goals:
- Keep vLLM as an *optional* dependency (imported only when used).
- Preserve the SingleVariantServer.generate() contract:
    generate(prompt, max_tokens, difficulty, dataset_type, prompt_mode) -> (text, metrics_dict)
- Use queue-inclusive TTFT ("Option A") measured from request submission to first token.
- Apply the same GSM8K server-side postprocessor (append FINAL_ANSWER line when safe).

Notes:
- vLLM always performs some form of continuous batching internally. The HF "micro-batching"
  scheduler is not used here (batching flags are ignored for vLLM in run_baseline_evaluation.py).
- For fairness in HF fp16 vs vLLM fp16 comparisons, we default to greedy decoding.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from transformers import AutoTokenizer  # light dependency; already required for HF backend

from answer_utils import enforce_strict_gsm8k_final_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vllm_server")


@dataclass
class VLLMConfig:
    model: str
    dtype: str = "float16"
    quantization: Optional[str] = None
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    max_num_seqs: int = 128
    trust_remote_code: bool = True
    enforce_eager: bool = False


class VLLMVariantServer:
    """vLLM-backed server with async streaming TTFT measurement."""

    def __init__(self, model_name: str, variant: str, config: VLLMConfig):
        self.model_name = model_name
        self.variant = variant
        self.backend = "vllm"
        self.config = config

        # Tokenizer is used for output_length accounting (tokens) in metrics.
        # For fairness, use the same tokenizer family as HF.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop_worker, daemon=True)
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None

        self._engine = None
        self._SamplingParams = None  # vLLM SamplingParams class (resolved lazily)

        self._thread.start()
        self._ready.wait()
        if self._init_error is not None:
            raise RuntimeError(f"vLLM init failed: {self._init_error}") from self._init_error

        logger.info("vLLM server ready.")

    # -----------------------------
    # Loop / engine init
    # -----------------------------

    def _loop_worker(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._engine, self._SamplingParams = self._create_engine()
        except BaseException as e:
            self._init_error = e
        finally:
            self._ready.set()

        # Keep loop alive for async generate calls.
        self._loop.run_forever()

    def _create_engine(self):
        """Create an async vLLM engine. Supports multiple vLLM API versions."""
        try:
            # Newer vLLM (some installs)
            from vllm import AsyncLLM, SamplingParams  # type: ignore

            logger.info("Using vLLM AsyncLLM API")
            llm = AsyncLLM(
                model=self.config.model,
                dtype=self.config.dtype,
                quantization=self.config.quantization,
                tensor_parallel_size=self.config.tensor_parallel_size,
                max_model_len=self.config.max_model_len,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=self.config.trust_remote_code,
                max_num_seqs=self.config.max_num_seqs,
                enforce_eager=self.config.enforce_eager,
            )
            return llm, SamplingParams
        except Exception:
            pass

        # Fallback to AsyncLLMEngine API (older versions)
        from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore
        from vllm.sampling_params import SamplingParams  # type: ignore

        logger.info("Using vLLM AsyncLLMEngine API")
        engine_args = AsyncEngineArgs(
            model=self.config.model,
            dtype=self.config.dtype,
            quantization=self.config.quantization,
            tensor_parallel_size=self.config.tensor_parallel_size,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=self.config.trust_remote_code,
            max_num_seqs=self.config.max_num_seqs,
            enforce_eager=self.config.enforce_eager,
        )
        llm = AsyncLLMEngine.from_engine_args(engine_args)
        return llm, SamplingParams

    # -----------------------------
    # Public API
    # -----------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        difficulty: str = "medium",
        dataset_type: str = "mmlu",
        prompt_mode: str = "slo",
        **_: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        """Synchronous wrapper around async streaming generation."""
        fut = asyncio.run_coroutine_threadsafe(
            self._generate_async(prompt=prompt, max_tokens=max_tokens),
            self._loop,
        )
        raw_text, ttft_ms, total_latency_ms = fut.result()

        # Server-side postprocess for GSM8K formatting robustness.
        text = raw_text.strip()
        postprocessed = False
        postprocess_candidate = None
        if (dataset_type or "").lower().strip() == "gsm8k":
            fixed, cand, did = enforce_strict_gsm8k_final_answer(text)
            if did:
                text = fixed.strip()
                postprocessed = True
                postprocess_candidate = cand

        # Token counts (for performance accounting). Use raw_text to reflect model output.
        out_len = len(self.tokenizer.encode(raw_text, add_special_tokens=False))
        if out_len <= 1:
            tpot_ms = 0.0
        else:
            # Option A: queue-inclusive TTFT is already part of total latency; TPOT uses remaining time.
            tpot_ms = max(0.0, (total_latency_ms - ttft_ms)) / float(out_len - 1)

        throughput = (float(out_len) / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        metrics: Dict[str, Any] = {
            "success": True,
            "backend": "vllm",
            "variant": self.variant,
            "raw_text": raw_text,
            "postprocessed": postprocessed,
            "postprocess_candidate": postprocess_candidate,
            # Latency split (best-effort; vLLM doesn't expose model/prefill splits here)
            "ttft_ms": float(ttft_ms),
            "ttft_infer_ms": float(ttft_ms),
            "ttft_model_ms": None,
            "tpot_ms": float(tpot_ms),
            "scheduler_wait_ms": 0.0,
            "queue_wait_ms": 0.0,
            "tokenize_ms": None,
            "total_latency_ms": float(total_latency_ms),
            "output_length": int(out_len),
            "throughput_tokens_per_sec": float(throughput),
        }
        return text, metrics

    async def _generate_async(self, prompt: str, max_tokens: int) -> Tuple[str, float, float]:
        """Async generation; returns (text, ttft_ms, total_latency_ms)."""
        if self._engine is None or self._SamplingParams is None:
            raise RuntimeError("vLLM engine is not initialized")

        # Greedy decoding for HF fp16 vs vLLM fp16 comparability.
        sampling_params = self._SamplingParams(
            max_tokens=int(max_tokens),
            temperature=0.0,
            top_p=1.0,
        )
        request_id = str(uuid.uuid4())

        t0 = time.perf_counter()
        gen = self._engine.generate(prompt, sampling_params, request_id)  # type: ignore
        first_t: Optional[float] = None

        # Accumulate text in a version-agnostic way (delta vs cumulative)
        acc = ""
        last = ""

        async for out in gen:
            if first_t is None:
                first_t = time.perf_counter()

            # vLLM returns a RequestOutput; try to fetch partial text from the first sequence.
            try:
                piece = out.outputs[0].text  # type: ignore
            except Exception:
                piece = ""

            if piece:
                if piece.startswith(last):
                    # cumulative
                    last = piece
                    acc = piece
                else:
                    # delta
                    acc += piece
                    last = acc

        t1 = time.perf_counter()
        if first_t is None:
            first_t = t1  # degenerate case

        ttft_ms = max(0.0, (first_t - t0) * 1000.0)
        total_ms = max(0.0, (t1 - t0) * 1000.0)
        return acc, ttft_ms, total_ms

    def shutdown(self) -> None:
        try:
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
