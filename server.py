"""
server.py

Single-variant model server used by the evaluation + load test harness.

Key features (paper-friendly):
- True 3-variant loading:
    * base  = fp16 weights (best quality; may require more VRAM)
    * med   = 8-bit (bnb int8)
    * cheap = 4-bit (bnb nf4)
- Correct TTFT for batched generation (prompt push no longer pollutes TTFT).
- Token accounting avoids counting padding tokens in batched generation.
- Greedy-mode generation defaults are forced to avoid Transformers warnings about unused sampling params.

This module intentionally keeps the API stable:
    server.generate(prompts, dataset_type, difficulty, prompt_mode, max_new_tokens, temperature, top_p)
returns: (outputs: List[str], metrics: List[Dict[str, Any]])
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

logger = logging.getLogger("server")


# -----------------------------
# Optional stopping criteria
# -----------------------------
class StopOnTokens(StoppingCriteria):
    """Stop generation once any token in `stop_token_ids` is generated."""

    def __init__(self, stop_token_ids: List[int]):
        super().__init__()
        self.stop_token_ids = set(int(t) for t in stop_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids is None or input_ids.numel() == 0:
            return False
        last_token_id = int(input_ids[0, -1].item())
        return last_token_id in self.stop_token_ids


class TimingStreamer(TextIteratorStreamer):
    """
    Captures the wall-clock time when the FIRST *generated* token is emitted.

    HF streamers typically receive an initial "prompt push" (batch, prompt_len).
    The first generated token is usually (batch, 1). We detect the prompt push
    using the last dimension > 1.
    """

    def __init__(self, tokenizer):
        # skip_prompt=True prevents prompt text from being queued, but the underlying
        # streamer still receives the prompt tensor once. We detect and ignore it.
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.first_token_time: Optional[float] = None
        self._saw_first_put = False

    def put(self, value):
        now = time.time()

        if not self._saw_first_put:
            self._saw_first_put = True

            # Detect prompt push: tensor with seq_len > 1
            if torch.is_tensor(value) and value.ndim >= 2 and int(value.shape[-1]) > 1:
                # prompt push; do NOT record TTFT yet
                pass
            else:
                # likely the first generated token
                self.first_token_time = now
        else:
            if self.first_token_time is None:
                self.first_token_time = now

        return super().put(value)


# -----------------------------
# Micro-batching
# -----------------------------
@dataclass
class BatchRequest:
    request_id: int
    prompt: str
    dataset_type: str
    difficulty: str
    prompt_mode: str
    max_new_tokens: int
    temperature: float
    top_p: float
    submit_time: float
    callback: Callable[[str, Dict[str, Any]], None]


class MicroBatchScheduler:
    """
    Simple micro-batching scheduler: accumulates requests for up to `batch_wait_ms`
    or until `max_batch_size`, then runs a single batched model.generate.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        batch_wait_ms: int = 8,
        process_batch_fn: Callable[[List[BatchRequest]], None] | None = None,
    ):
        self.max_batch_size = max(1, int(max_batch_size))
        self.batch_wait_ms = max(0, int(batch_wait_ms))
        self.process_batch_fn = process_batch_fn

        self._queue: "queue.Queue[BatchRequest]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._stop_event.set()
        try:
            self._queue.put_nowait(
                BatchRequest(
                    request_id=-1,
                    prompt="",
                    dataset_type="",
                    difficulty="",
                    prompt_mode="",
                    max_new_tokens=0,
                    temperature=0.0,
                    top_p=1.0,
                    submit_time=time.time(),
                    callback=lambda _o, _m: None,
                )
            )
        except Exception:
            pass
        if self._worker:
            self._worker.join(timeout=2)

    def submit(self, request: BatchRequest):
        self._queue.put(request)

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                first_req = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if first_req.request_id == -1:
                continue

            batch: List[BatchRequest] = [first_req]
            batch_start_time = time.time()

            while len(batch) < self.max_batch_size:
                elapsed_ms = (time.time() - batch_start_time) * 1000
                remaining_ms = self.batch_wait_ms - elapsed_ms
                if remaining_ms <= 0:
                    break

                try:
                    req = self._queue.get(timeout=remaining_ms / 1000)
                    if req.request_id == -1:
                        continue
                    batch.append(req)
                except queue.Empty:
                    break

            if self.process_batch_fn:
                try:
                    self.process_batch_fn(batch)
                except Exception as e:
                    logger.exception(f"Error processing batch: {e}")
                    for r in batch:
                        r.callback("", {"error": str(e), "request_id": r.request_id})


# -----------------------------
# Server
# -----------------------------
class SingleVariantServer:
    def __init__(
        self,
        model_name: str,
        variant: str = "med",
        device: str = "auto",
        dtype: str = "float16",
        batching_enabled: bool = False,
        max_batch_size: int = 8,
        batch_wait_ms: int = 8,
    ):
        self.model_name = model_name
        self.variant = str(variant).lower().strip()
        self.device = device

        logger.info("=" * 69)
        logger.info(f"Initializing {self.variant.upper()} server")
        logger.info("=" * 69)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Variant: {self.variant}")
        logger.info(f"  Device: {device}")

        model_dtype = self._resolve_dtype(dtype)
        logger.info(f"  Dtype: {model_dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            # For many decoder-only LMs, using EOS as PAD is standard.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_cfg = self._quantization_config(self.variant, model_dtype)

        # For best reproducibility, keep `torch_dtype` explicit even if quantized.
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": model_dtype,
            "low_cpu_mem_usage": True,
        }

        # Device mapping
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            model_kwargs["device_map"] = {"": 0}
        elif device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        else:
            # allow passing custom accelerate device_map strings
            model_kwargs["device_map"] = device

        if quant_cfg is not None:
            model_kwargs["quantization_config"] = quant_cfg

        # Prefer SDPA where available; safe no-op if unsupported.
        model_kwargs["attn_implementation"] = "sdpa"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()

        # Make greedy generation clean (prevents warnings about temperature/top_p when do_sample=False).
        self._force_greedy_generation_defaults()

        self._eos_token_ids = self._build_eos_token_ids()

        # Useful for MMLU: constrain to A/B/C/D token ids
        self.mmlu_allowed_token_ids = self._get_mmlu_allowed_token_ids()
        logger.info(f"MMLU allowed token ids: {len(self.mmlu_allowed_token_ids)}")
        logger.info(f"Tokenizer loaded: {type(self.tokenizer).__name__} (chat_template={self.tokenizer.chat_template is not None})")

        self._lock = threading.Lock()

        # Batching
        self.batching_enabled = bool(batching_enabled)
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_ms = int(batch_wait_ms)

        self._scheduler: Optional[MicroBatchScheduler] = None
        if self.batching_enabled:
            self._scheduler = MicroBatchScheduler(
                max_batch_size=self.max_batch_size,
                batch_wait_ms=self.batch_wait_ms,
                process_batch_fn=self._process_batch,
            )
            self._scheduler.start()
            logger.info(f"Batching enabled: max_batch_size={self.max_batch_size}, batch_wait_ms={self.batch_wait_ms}")

        self._warmup(iterations=3)

    # -----------------------------
    # Model / tokenizer helpers
    # -----------------------------
    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        dtype = str(dtype).lower()
        if dtype in {"float16", "fp16"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        if dtype in {"float32", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype}")

    @staticmethod
    def _quantization_config(variant: str, compute_dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
        v = str(variant).lower().strip()
        if v == "base":
            return None
        if v == "med":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        if v == "cheap":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        raise ValueError(f"Unknown variant '{variant}'. Choose from: base, med, cheap.")

    def _force_greedy_generation_defaults(self) -> None:
        try:
            gen_cfg = self.model.generation_config
            gen_cfg.do_sample = False
            gen_cfg.temperature = 1.0
            gen_cfg.top_p = 1.0
        except Exception:
            # If generation_config is missing or immutable, ignore.
            pass

    def _build_eos_token_ids(self) -> List[int]:
        eos_ids: List[int] = []
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(int(self.tokenizer.eos_token_id))
        # Llama 3.x uses <|eot_id|> as an end-of-turn token
        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and int(eot_id) != int(self.tokenizer.unk_token_id or -1):
            eos_ids.append(int(eot_id))
        # de-dup while preserving order
        seen = set()
        out = []
        for x in eos_ids:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _get_mmlu_allowed_token_ids(self) -> List[int]:
        # Constrain first generated token to one of: A, B, C, D (plus common variants with preceding space)
        options = ["A", "B", "C", "D", " A", " B", " C", " D"]
        ids = []
        for s in options:
            tok = self.tokenizer.encode(s, add_special_tokens=False)
            if len(tok) == 1:
                ids.append(int(tok[0]))
        # de-dup
        return sorted(set(ids))

    def _warmup(self, iterations: int = 3):
        logger.info(f"Warming up server ({iterations} iterations, deterministic)...")
        try:
            dummy = "Hello"
            for _ in range(max(1, int(iterations))):
                self.generate(
                    prompts=[dummy],
                    dataset_type="mmlu",
                    difficulty="easy",
                    prompt_mode="slo",
                    max_new_tokens=4,
                    temperature=0.0,
                    top_p=1.0,
                )
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        logger.info("Warmup complete")

    # -----------------------------
    # Public API
    # -----------------------------
    def shutdown(self):
        if self._scheduler:
            self._scheduler.stop()

    def generate(
        self,
        prompts: List[str],
        dataset_type: str = "gsm8k",
        difficulty: str = "easy",
        prompt_mode: str = "accuracy",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Generate outputs for a list of prompts. If micro-batching is enabled, this
        may group prompts across concurrent calls.
        """
        if not prompts:
            return [], []

        if self.batching_enabled and self._scheduler is not None:
            results: List[Tuple[str, Dict[str, Any]]] = [("", {}) for _ in range(len(prompts))]
            done = threading.Event()
            remaining = len(prompts)

            def _make_cb(idx: int):
                def _cb(out_text: str, metrics: Dict[str, Any]):
                    nonlocal remaining
                    results[idx] = (out_text, metrics)
                    remaining -= 1
                    if remaining <= 0:
                        done.set()

                return _cb

            submit_time = time.time()
            for i, p in enumerate(prompts):
                req = BatchRequest(
                    request_id=i,
                    prompt=p,
                    dataset_type=dataset_type,
                    difficulty=difficulty,
                    prompt_mode=prompt_mode,
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    submit_time=submit_time,
                    callback=_make_cb(i),
                )
                self._scheduler.submit(req)

            done.wait()
            outputs = [r[0] for r in results]
            metrics = [r[1] for r in results]
            return outputs, metrics

        # Non-batched: serialize using a lock
        lock_start = time.time()
        with self._lock:
            queue_wait_ms = int((time.time() - lock_start) * 1000)
            return self._generate_hf_batch(
                prompts=prompts,
                dataset_type=dataset_type,
                difficulty=difficulty,
                prompt_mode=prompt_mode,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                queue_wait_ms_list=[queue_wait_ms] * len(prompts),
            )

    # -----------------------------
    # Batch processing
    # -----------------------------
    def _process_batch(self, batch: List[BatchRequest]):
        if not batch:
            return

        prompts = [r.prompt for r in batch]
        dataset_type = batch[0].dataset_type
        difficulty = batch[0].difficulty
        prompt_mode = batch[0].prompt_mode
        max_new_tokens = batch[0].max_new_tokens
        temperature = batch[0].temperature
        top_p = batch[0].top_p

        # Queue wait per request
        now = time.time()
        queue_wait_ms_list = [int((now - r.submit_time) * 1000) for r in batch]

        outputs, metrics = self._generate_hf_batch(
            prompts=prompts,
            dataset_type=dataset_type,
            difficulty=difficulty,
            prompt_mode=prompt_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            queue_wait_ms_list=queue_wait_ms_list,
        )

        for r, out_text, m in zip(batch, outputs, metrics):
            r.callback(out_text, m)

    # -----------------------------
    # Core generation
    # -----------------------------
    def _trim_generated_token_ids(self, token_ids: torch.Tensor) -> List[int]:
        ids = [int(x) for x in token_ids.tolist()]

        # Trim trailing PAD
        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        while pad_id is not None and ids and ids[-1] == pad_id:
            ids.pop()

        # Trim from first EOS/EOT (exclude the EOS token itself)
        eos_set = set(int(x) for x in self._eos_token_ids)
        for j, tid in enumerate(ids):
            if tid in eos_set:
                ids = ids[:j]
                break

        return ids

    def _generate_hf_batch(
        self,
        prompts: List[str],
        dataset_type: str,
        difficulty: str,
        prompt_mode: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        queue_wait_ms_list: List[int],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        start_time = time.time()

        # Tokenize (padding to batch max)
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attn_mask = enc.get("attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.model.device)

        prompt_token_counts = [int(x) for x in enc["attention_mask"].sum(dim=1).tolist()] if attn_mask is not None else [int(input_ids.shape[1])] * len(prompts)

        # MMLU: constrain to A/B/C/D token ids
        logits_processor = LogitsProcessorList()
        if dataset_type == "mmlu" and self.mmlu_allowed_token_ids:
            def _prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
                # Constrain only the FIRST generated token
                generated_len = int(input_ids.shape[-1]) - int(enc["input_ids"].shape[-1])
                if generated_len <= 0:
                    return self.mmlu_allowed_token_ids
                return list(range(int(self.tokenizer.vocab_size)))

            logits_processor.append(PrefixConstrainedLogitsProcessor(_prefix_allowed_tokens_fn, self.model.device))

        stopping_criteria = StoppingCriteriaList()
        # Stop early on EOS/EOT; generate() already handles eos_token_id, but StopOnTokens
        # also plays nicely when eos_token_id is a list.
        if self._eos_token_ids:
            stopping_criteria.append(StopOnTokens(self._eos_token_ids))

        # Timing streamer (TTFT)
        streamer = TimingStreamer(self.tokenizer)

        # Sampling / decoding config
        do_sample = bool(prompt_mode != "slo" and temperature and float(temperature) > 0.0)
        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "streamer": streamer,
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            "eos_token_id": self._eos_token_ids if self._eos_token_ids else self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            outputs = self.model.generate(**gen_kwargs)

        end_time = time.time()

        # TTFT in ms
        ttft_ms: Optional[float] = None
        if streamer.first_token_time is not None:
            ttft_ms = max(0.0, (streamer.first_token_time - start_time) * 1000)

        # Decode + token accounting (avoid counting padding)
        batch_size = len(prompts)
        outputs_text: List[str] = []
        output_token_counts: List[int] = []

        input_len = int(enc["input_ids"].shape[1])
        for i in range(batch_size):
            gen_slice = outputs[i, input_len:]
            trimmed_ids = self._trim_generated_token_ids(gen_slice)
            output_token_counts.append(int(len(trimmed_ids)))
            text = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)
            outputs_text.append(text.strip())

        # TPOT: use max generated tokens across batch (proxy for decoding steps)
        max_out = max(output_token_counts) if output_token_counts else 0
        tpot_ms: Optional[float] = None
        if ttft_ms is not None and max_out > 1:
            total_ms = max(0.0, (end_time - start_time) * 1000)
            gen_only_ms = max(0.0, total_ms - ttft_ms)
            tpot_ms = gen_only_ms / float(max_out - 1)

        metrics: List[Dict[str, Any]] = []
        for i in range(batch_size):
            metrics.append(
                {
                    "variant": self.variant,
                    "batched": bool(self.batching_enabled),
                    "batch_size": int(batch_size),
                    "prompt_mode": prompt_mode,
                    "dataset_type": dataset_type,
                    "difficulty": difficulty,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "queue_wait_ms": int(queue_wait_ms_list[i]) if i < len(queue_wait_ms_list) else 0,
                    "prompt_tokens": int(prompt_token_counts[i]),
                    "output_length": int(output_token_counts[i]),
                }
            )

        return outputs_text, metrics
