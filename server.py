import gc
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from prompt_templates import get_max_tokens
from answer_utils import (
    enforce_strict_gsm8k_final_answer,
    extract_gsm8k_parseable,
    extract_gsm8k_strict,
    extract_mmlu_answer,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")


def _resolve_dtype(requested: str, device: str) -> str:
    """Resolve "auto" into an explicit dtype where it materially helps."""

    if requested != "auto":
        return requested
    if device != "cuda":
        return requested

    # Prefer bf16 on GPUs that support it; otherwise float16.
    try:
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8 and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            return "bfloat16"
    except Exception:
        pass
    return "float16"


# -------------------------------
# Utilities
# -------------------------------


def split_system_user(prompt: str) -> Tuple[str, str]:
    """Split the legacy 'SYSTEM\n\nUSER' prompt format into (system, user)."""

    parts = prompt.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", prompt.strip()


class GPUMonitor:
    @staticmethod
    def get_gpu_info() -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"available": 0, "total": 0, "allocated": 0, "free": 0}
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free = total - allocated
        return {"available": 1, "total": total, "allocated": allocated, "free": free}

    @staticmethod
    def log_gpu_status(prefix: str = "") -> None:
        info = GPUMonitor.get_gpu_info()
        if not info.get("available"):
            logger.info(f"{prefix}No CUDA GPU available")
            return
        name = torch.cuda.get_device_name(0)
        logger.info(f"{prefix}GPU: {name}")
        logger.info(
            f"{prefix}  Memory: {info['allocated']:.2f}GB allocated, "
            f"{info['free']:.2f}GB free / {info['total']:.2f}GB total"
        )


# -------------------------------
# Timing + stopping criteria
# -------------------------------


class TimingStreamer:
    """Record time-to-first-generated-token (TTFT).

    In transformers generation, streamer.put() is called once with the *prompt*
    token IDs before any generation steps. We ignore that prompt push and timestamp
    the first generated token push.

    NOTE: prior versions tried to detect prompt-push by `numel()>1`, which breaks
    for batch_size>1 because the first generated step also has `numel()==batch_size`.
    """

    def __init__(self, sync_fn):
        self._sync_fn = sync_fn
        self.first_token_time: Optional[float] = None
        self._ignored_first_put = False

    def put(self, _value):
        # Ignore the first put() call (prompt echo).
        if not self._ignored_first_put:
            self._ignored_first_put = True
            return

        if self.first_token_time is None:
            self._sync_fn()
            self.first_token_time = time.perf_counter()

    def end(self):
        return


class StopOnFinalAnswer(StoppingCriteria):
    """Stop once FINAL_ANSWER is present.

    For batch generation, set require_all=True to stop only after *all* rows
    contain a FINAL_ANSWER (safe for batched GSM8K).

    We keep decoding lightweight by only decoding a tail window of tokens.
    """

    _PATTERN = re.compile(r"FINAL_ANSWER\s*[:=\s]*[-+]?\d[\d,]*(?:\.\d+)?")

    def __init__(self, tokenizer, prompt_len: int, require_all: bool = False, tail_tokens: int = 256):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.require_all = bool(require_all)
        self.tail_tokens = int(max(32, tail_tokens))
        self._done: Optional[List[bool]] = None

    def _check_one(self, seq: torch.Tensor) -> bool:
        # seq: [seq_len]
        seq_len = int(seq.shape[0])
        start = max(self.prompt_len, seq_len - self.tail_tokens)
        text = self.tokenizer.decode(seq[start:], skip_special_tokens=True)
        m = self._PATTERN.search(text)
        return bool(m and m.end() < len(text))

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.ndim == 1:
            return self._check_one(input_ids)

        bsz = int(input_ids.shape[0])
        if self._done is None or len(self._done) != bsz:
            self._done = [False] * bsz

        for i in range(bsz):
            if self._done[i]:
                continue
            if self._check_one(input_ids[i]):
                self._done[i] = True

        return all(self._done) if self.require_all else any(self._done)


# -------------------------------
# Request batching
# -------------------------------


@dataclass
class _PendingRequest:
    prompt: str
    dataset_type: str
    difficulty: str
    max_tokens: int
    prompt_mode: str
    temperature: float
    top_p: float
    enqueue_time: float
    queue_depth_at_submit: int = 0
    event: threading.Event
    result_text: Optional[str] = None
    result_metrics: Optional[Dict] = None
    error: Optional[str] = None

    def batch_key(self) -> Tuple[str, str, int, bool]:
        # In SLO mode we always use greedy decoding, so sampling params are irrelevant.
        do_sample = bool(self.prompt_mode != "slo" and self.temperature and self.temperature > 0.0)
        return (self.dataset_type, self.prompt_mode, int(self.max_tokens), do_sample)


class _BatchingScheduler:
    def __init__(self, server: "SingleVariantServer", max_batch_size: int = 4, batch_wait_ms: int = 8):
        self.server = server
        self.max_batch_size = int(max(1, max_batch_size))
        self.batch_wait_s = max(0.0, float(batch_wait_ms) / 1000.0)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._pending: List[_PendingRequest] = []
        self._stop = False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        # Head-of-line blocking can make TTFT explode under load when long-form GSM8K
        # requests sit ahead of short MMLU requests in FIFO order. We mitigate this
        # with a simple "short-job-first" selection of the *next* batch key, while
        # still providing an aging-based escape hatch to avoid starvation.
        self._starvation_s = 0.25  # seconds

        # Adaptive batching window:
        # For long generations (e.g., GSM8K), a slightly larger batching window
        # can dramatically reduce lock-wait-driven TTFT tails by allowing more
        # requests to join the same micro-batch.
        # For short generations (e.g., single-token MMLU), keep the window tight.
        self._long_job_min_wait_s = 0.05  # 50ms
        self._long_job_token_threshold = 64

    def submit(self, req: _PendingRequest) -> None:
        with self._cv:
            self._pending.append(req)
            self._cv.notify()

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=1.0)

    def get_queue_depth(self) -> int:
        """Return number of pending requests waiting to be batched."""
        with self._cv:
            return int(len(self._pending))

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if self._stop and not self._pending:
                    return

                # --------------------------------------------------------------
                # Choose which request to serve next.
                # - Default: prioritize smaller max_tokens (short-job-first)
                # - If anything has waited too long, serve the oldest (anti-starvation)
                # --------------------------------------------------------------
                now = time.perf_counter()
                oldest_idx = min(range(len(self._pending)), key=lambda i: self._pending[i].enqueue_time)
                if (now - self._pending[oldest_idx].enqueue_time) > self._starvation_s:
                    chosen_idx = oldest_idx
                else:
                    chosen_idx = min(
                        range(len(self._pending)),
                        key=lambda i: (int(self._pending[i].max_tokens), self._pending[i].enqueue_time),
                    )

                first = self._pending.pop(chosen_idx)
                key = first.batch_key()
                batch: List[_PendingRequest] = [first]

                # Use a slightly longer wait window for long jobs to improve
                # batching and reduce TTFT tails caused by lock contention.
                if first.max_tokens >= self._long_job_token_threshold:
                    wait_s = max(self.batch_wait_s, self._long_job_min_wait_s)
                else:
                    wait_s = self.batch_wait_s

                deadline = time.perf_counter() + wait_s
                while len(batch) < self.max_batch_size:
                    idx = next((i for i, r in enumerate(self._pending) if r.batch_key() == key), None)
                    if idx is not None:
                        batch.append(self._pending.pop(idx))
                        continue

                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)

            self._process_batch(batch)

    def _process_batch(self, batch: List[_PendingRequest]) -> None:
        dequeue_t = time.perf_counter()

        try:
            prompts = [r.prompt for r in batch]
            dataset_type = batch[0].dataset_type
            prompt_mode = batch[0].prompt_mode
            max_tokens = batch[0].max_tokens
            temperature = batch[0].temperature
            top_p = batch[0].top_p

            texts, metrics_list, lock_wait_ms = self.server._generate_hf_batch(
                prompts=prompts,
                dataset_type=dataset_type,
                max_tokens=max_tokens,
                prompt_mode=prompt_mode,
                temperature=temperature,
                top_p=top_p,
                require_all_final_answers=(dataset_type == "gsm8k"),
            )

            for r, text, m in zip(batch, texts, metrics_list):
                scheduler_wait_ms = (dequeue_t - r.enqueue_time) * 1000.0
                scheduler_wait_ms = float(max(0.0, scheduler_wait_ms))

                m["scheduler_wait_ms"] = scheduler_wait_ms
                m["lock_wait_ms"] = float(max(0.0, lock_wait_ms))

                # queue_wait_ms is used by load_generator to compute total queueing.
                m["queue_wait_ms"] = float(max(0.0, scheduler_wait_ms + float(lock_wait_ms)))
                m["queue_depth_at_submit"] = int(getattr(req, "queue_depth_at_submit", 0))

                # ------------------------------------------------------------------
                # Option A (paper definition): queue-inclusive TTFT
                #   TTFT_A = scheduler_wait + (tokenize + lock_wait + model_prefill + first_decode)
                # ------------------------------------------------------------------
                ttft_infer_ms = float(m.get("ttft_infer_ms", m.get("ttft_ms", 0.0)) or 0.0)
                m["ttft_ms"] = float(max(0.0, scheduler_wait_ms + ttft_infer_ms))

                # Make total latency consistent with that same service-level definition.
                m["total_latency_ms"] = float(max(0.0, float(m.get("total_latency_ms", 0.0) or 0.0) + scheduler_wait_ms))

                r.result_text = text
                r.result_metrics = m
                r.event.set()

        except Exception as e:
            for r in batch:
                scheduler_wait_ms = (dequeue_t - r.enqueue_time) * 1000.0
                r.error = str(e)
                r.result_text = ""
                r.result_metrics = {
                    "success": False,
                    "backend": "hf",
                    "error": str(e),
                    "scheduler_wait_ms": float(max(0.0, scheduler_wait_ms)),
                    "queue_wait_ms": float(max(0.0, scheduler_wait_ms)),
                    "ttft_ms": 0.0,
                    "tpot_ms": 0.0,
                    "output_length": 0,
                    "throughput_tokens_per_sec": 0.0,
                    "total_latency_ms": float(max(0.0, scheduler_wait_ms)),
                    "variant": self.server.variant,
                    "model": self.server.model_name,
                    "device": self.server.device,
                }
                r.event.set()


# -------------------------------
# Main server
# -------------------------------


class SingleVariantServer:
    """A single-process, single-GPU server with optional micro-batching."""

    MMLU_ALLOWED_CHARS = [" A", " B", " C", " D", "A", "B", "C", "D"]

    def __init__(
        self,
        model_name: str,
        variant: str = "med",
        device: str = "auto",
        dtype: str = "auto",
        enable_batching: bool = False,
        max_batch_size: int = 4,
        batch_wait_ms: int = 8,
    ):
        self.model_name = model_name
        self.variant_requested = variant
        self.variant = variant
        self.variant_effective = variant
        self.quantization = "unknown"
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.dtype = _resolve_dtype(dtype, self.device)

        logger.info("=" * 69)
        logger.info(f"Initializing {variant.upper()} server")
        logger.info("=" * 69)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {self.dtype}")
        if self.device == "cuda":
            GPUMonitor.log_gpu_status(prefix="  ")

        self._generation_lock = threading.Lock()

        # Tokenizer (cheap)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Precompute MMLU allowed ids
        self.mmlu_allowed_token_ids = self._compute_mmlu_allowed_token_ids()
        logger.info(f"MMLU allowed token ids: {len(self.mmlu_allowed_token_ids)}")
        logger.info(
            f"Tokenizer loaded: {type(self.tokenizer).__name__} (chat_template={bool(getattr(self.tokenizer, 'chat_template', None))})"
        )

        # Model dtype
        model_dtype = "auto"
        if self.dtype == "float16":
            model_dtype = torch.float16
        elif self.dtype == "bfloat16":
            model_dtype = torch.bfloat16

        # Quantization by variant (with safety fallbacks)
        quant_config = None
        load_kwargs: Dict[str, Any] = {
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # Detect compute capability for guardrails (e.g., P100 + bnb int8 can be unstable).
        cc_major: Optional[int] = None
        if self.device == "cuda":
            try:
                cc_major, _cc_minor = torch.cuda.get_device_capability(0)
            except Exception:
                cc_major = None

        if self.variant == "base":
            self.quantization = f"{self.dtype}"
            quant_config = None
        elif self.variant == "med":
            # bitsandbytes int8 is CUDA-only and can be flaky on older GPUs.
            if self.device != "cuda":
                logger.warning("MED (int8) requested on non-CUDA device; falling back to fp16.")
                self.variant_effective = "base"
                self.quantization = f"{self.dtype}"
                quant_config = None
            elif cc_major is not None and cc_major < 7:
                logger.warning(
                    f"MED (bnb int8) is disabled on this GPU (compute_capability={cc_major}.x); falling back to fp16."
                )
                self.variant_effective = "base"
                self.quantization = f"{self.dtype}"
                quant_config = None
            else:
                self.quantization = "bnb_int8"
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.variant == "cheap":
            if self.device != "cuda":
                logger.warning("CHEAP (int4) requested on non-CUDA device; falling back to fp16.")
                self.variant_effective = "base"
                self.quantization = f"{self.dtype}"
                quant_config = None
            elif cc_major is not None and cc_major < 7:
                logger.warning(f"CHEAP (int4) requested on compute capability {cc_major}.x (<7.0); falling back to fp16.")
                self.variant_effective = "base"
                self.quantization = f"{self.dtype}"
                quant_config = None
            else:
                self.quantization = "bnb_nf4_int4"
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=model_dtype if model_dtype != "auto" else torch.float16,
                )
        else:
            logger.warning(f"Unknown variant '{self.variant}', defaulting to base (fp16)")
            self.variant_effective = "base"
            self.quantization = f"{self.dtype}"
            quant_config = None

        if self.variant_effective != self.variant:
            logger.warning(
                f"Variant '{self.variant}' is not supported/unstable on this device; using '{self.variant_effective}' weights (quantization={self.quantization})."
            )

        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        # transformers>=4.57 deprecates `torch_dtype` in favor of `dtype`.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype, **load_kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, **load_kwargs)

        self.model.eval()

        # Silence generate() warnings for deterministic decoding (do_sample=False).
        try:
            gcfg = getattr(self.model, "generation_config", None)
            if gcfg is not None:
                gcfg.temperature = 1.0
                gcfg.top_p = 1.0
        except Exception:
            pass

        logger.info("Model loaded successfully")
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model size: {num_params/1e9:.2f}B parameters")
        except Exception:
            pass

        if self.device == "cuda":
            logger.info("Post-load GPU status:")
            GPUMonitor.log_gpu_status(prefix="  ")

        # Optional micro-batching scheduler
        self._scheduler: Optional[_BatchingScheduler] = None
        if enable_batching:
            logger.info(f"Batching enabled: max_batch_size={max_batch_size}, batch_wait_ms={batch_wait_ms}")
            self._scheduler = _BatchingScheduler(
                server=self, max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms
            )

        self._warmup()

    def _compute_mmlu_allowed_token_ids(self) -> List[int]:
        ids: List[int] = []
        for s in self.MMLU_ALLOWED_CHARS:
            t = self.tokenizer.encode(s, add_special_tokens=False)
            if t:
                ids.append(int(t[0]))
        seen = set()
        out: List[int] = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _synchronize_device(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def _build_input_text(self, prompt: str) -> str:
        system, user = split_system_user(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if system:
            return f"SYSTEM: {system}\nUSER: {user}\nASSISTANT:"
        return f"USER: {user}\nASSISTANT:"

    def _warmup(self) -> None:
        logger.info("Warming up server (3 iterations, deterministic)...")
        warm_prompt = "You are a helpful assistant. Reply with a single letter: A."
        failures = 0
        last_err = None
        for _ in range(3):
            try:
                _ = self.generate(
                    prompt=warm_prompt,
                    dataset_type="mmlu",
                    difficulty="easy",
                    max_tokens=1,
                    prompt_mode="slo",
                )
            except Exception as e:
                failures += 1
                last_err = e
                logger.error(f"Warmup error: {e}")
        if failures:
            raise RuntimeError(f"Warmup failed ({failures}/3). Last error: {last_err}")
        logger.info("Warmup complete")

    # -------------------------------
    # Core HF generation (batch)
    # -------------------------------

    def _generate_hf_batch(
        self,
        prompts: List[str],
        dataset_type: str,
        max_tokens: int,
        prompt_mode: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        require_all_final_answers: bool = False,
    ) -> Tuple[List[str], List[Dict], float]:
        """Run a single batched HF generate call.

        Returns:
          texts: list[str]
          metrics_list: list[dict]  (one per sample; excludes scheduler_wait_ms)
          lock_wait_ms: float       (time waiting on the GPU generation lock)
        """

        t0_total = time.perf_counter()

        dataset_type = (dataset_type or "").lower().strip()
        prompt_mode = (prompt_mode or "").lower().strip()
        bsz = len(prompts)

        # Build input texts
        input_texts = [self._build_input_text(p) for p in prompts]

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        t_after_inputs = time.perf_counter()
        prompt_len = int(inputs["input_ids"].shape[1])

        # Decoding strategy
        do_sample = bool(prompt_mode != "slo" and temperature and float(temperature) > 0.0)

        # Stopping criteria
        stopping_criteria = None
        if dataset_type == "gsm8k":
            stopper = StopOnFinalAnswer(
                tokenizer=self.tokenizer,
                prompt_len=prompt_len,
                require_all=bool(require_all_final_answers and bsz > 1),
            )
            stopping_criteria = StoppingCriteriaList([stopper])

        # MMLU restriction
        prefix_allowed_tokens_fn = None
        if dataset_type == "mmlu":
            max_tokens = 1
            allowed_ids = self.mmlu_allowed_token_ids

            def prefix_allowed_tokens_fn(_batch_id: int, _input_ids):
                return allowed_ids

        # Generation kwargs
        gen_kwargs = {
            "max_new_tokens": int(max_tokens),
            "do_sample": do_sample,
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})
        if stopping_criteria is not None:
            gen_kwargs["stopping_criteria"] = stopping_criteria
        if prefix_allowed_tokens_fn is not None:
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        streamer = TimingStreamer(self._synchronize_device)

        t_lock_req = time.perf_counter()
        with self._generation_lock:
            t_lock_acq = time.perf_counter()
            lock_wait_ms = (t_lock_acq - t_lock_req) * 1000.0

            self._synchronize_device()
            t0_gen = time.perf_counter()

            with torch.inference_mode():
                sequences = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    streamer=streamer,
                    **gen_kwargs,
                )

            self._synchronize_device()
            t1_gen = time.perf_counter()

        total_gen_time = max(0.0, t1_gen - t0_gen)

        first_tok_t = streamer.first_token_time
        if first_tok_t is None:
            # If streaming didn't fire, fall back to total time (rare).
            first_tok_t = t1_gen

        ttft_model_s = max(0.0, first_tok_t - t0_gen)
        ttft_infer_s = max(0.0, first_tok_t - t0_total)

        # Decode outputs (generated tail only)
        texts: List[str] = []
        raw_texts: List[str] = []
        postprocessed_flags: List[bool] = []
        postprocess_candidates: List[Optional[str]] = []
        out_lens: List[int] = []
        for i in range(bsz):
            gen_ids = sequences[i, prompt_len:]
            out_lens.append(int(gen_ids.numel()))
            raw_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            text = raw_text
            postprocessed = False
            postprocess_candidate = None
            if dataset_type == "gsm8k":
                fixed, cand, did = enforce_strict_gsm8k_final_answer(raw_text)
                if did:
                    text = fixed.strip()
                    postprocessed = True
                    postprocess_candidate = cand
            texts.append(text)
            raw_texts.append(raw_text)
            postprocessed_flags.append(bool(postprocessed))
            postprocess_candidates.append(postprocess_candidate)


        # Compute TPOT from generation time excluding the model TTFT phase.
        decode_s = max(0.0, total_gen_time - ttft_model_s)

        t1_total = time.perf_counter()
        total_latency_s = max(0.0, t1_total - t0_total)

        tokenize_ms = float(max(0.0, (t_after_inputs - t0_total) * 1000.0))

        metrics_list: List[Dict] = []
        for out_len, raw_text, postprocessed, postprocess_candidate in zip(out_lens, raw_texts, postprocessed_flags, postprocess_candidates):
            if out_len <= 1:
                tpot_ms = 0.0
            else:
                tpot_ms = (decode_s * 1000.0) / float(out_len - 1)

            throughput = (float(out_len) / total_gen_time) if total_gen_time > 0 else 0.0

            metrics_list.append(
                {
                    "success": True,
                    "backend": "hf",
                    "raw_text": raw_text,
                    "postprocessed": bool(postprocessed),
                    "postprocess_candidate": postprocess_candidate,
                    # ------------------------------------------------------------------
                    # Paper-facing TTFT definition (Option A) is applied in the scheduler.
                    # Here we report component timings.
                    # ------------------------------------------------------------------
                    "ttft_model_ms": float(ttft_model_s * 1000.0),
                    "ttft_infer_ms": float(ttft_infer_s * 1000.0),
                    # For non-batched calls, ttft_ms == ttft_infer_ms (no scheduler_wait).
                    "ttft_ms": float(ttft_infer_s * 1000.0),
                    "tpot_ms": float(tpot_ms),
                    "output_length": int(out_len),
                    "throughput_tokens_per_sec": float(throughput),
                    "total_latency_ms": float(total_latency_s * 1000.0),
                    "tokenize_ms": tokenize_ms,
                    # queue_wait_ms / scheduler_wait_ms filled by caller
                    "variant": self.variant,
                    "variant_effective": getattr(self, "variant_effective", self.variant),
                    "quantization": getattr(self, "quantization", None),
                    "dtype": self.dtype,
                    "model": self.model_name,
                    "device": self.device,
                }
            )

        return texts, metrics_list, float(max(0.0, lock_wait_ms))

    # -------------------------------
    # Public API
    # -------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        dataset_type: str = "gsm8k",
        difficulty: str = "easy",
        prompt_mode: str = "accuracy",
        use_batching: Optional[bool] = None,
    ) -> Tuple[str, Dict]:
        """Generate completion text plus metrics."""

        dataset_type = (dataset_type or "").lower().strip()
        prompt_mode = (prompt_mode or "").lower().strip()
        difficulty = (difficulty or "").lower().strip()

        if max_tokens is None:
            max_tokens = get_max_tokens(difficulty, dataset_type, prompt_mode)
        if dataset_type == "mmlu":
            max_tokens = 1

        batching_enabled = self._scheduler is not None
        if use_batching is None:
            # Default: only batch SLO mode.
            use_batching = bool(batching_enabled and prompt_mode == "slo")

        if use_batching:
            req = _PendingRequest(
                prompt=prompt,
                dataset_type=dataset_type,
                difficulty=difficulty,
                max_tokens=int(max_tokens),
                prompt_mode=prompt_mode,
                temperature=float(temperature),
                top_p=float(top_p),
                enqueue_time=time.perf_counter(),
                event=threading.Event(),
            )
            assert self._scheduler is not None
            # Capture queue depth just before enqueuing (best-effort; used for learned routing features)
            try:
                req.queue_depth_at_submit = int(self._scheduler.get_queue_depth())
            except Exception:
                req.queue_depth_at_submit = 0
            self._scheduler.submit(req)
            req.event.wait()

            if req.error:
                return "", {
                    "success": False,
                    "backend": "hf",
                    "error": req.error,
                    "scheduler_wait_ms": 0.0,
                    "queue_wait_ms": 0.0,
                    "ttft_ms": 0.0,
                    "tpot_ms": 0.0,
                    "output_length": 0,
                    "throughput_tokens_per_sec": 0.0,
                    "total_latency_ms": 0.0,
                    "variant": self.variant,
                    "variant_effective": getattr(self, "variant_effective", self.variant),
                    "quantization": getattr(self, "quantization", None),
                    "dtype": self.dtype,
                    "model": self.model_name,
                    "device": self.device,
                }

            return req.result_text or "", req.result_metrics or {}

        # Direct (non-batched) generation.
        texts, metrics_list, lock_wait_ms = self._generate_hf_batch(
            prompts=[prompt],
            dataset_type=dataset_type,
            max_tokens=int(max_tokens),
            prompt_mode=prompt_mode,
            temperature=float(temperature),
            top_p=float(top_p),
            require_all_final_answers=False,
        )

        metrics = metrics_list[0]
        metrics["scheduler_wait_ms"] = 0.0
        metrics["lock_wait_ms"] = float(max(0.0, lock_wait_ms))
        metrics["queue_wait_ms"] = float(max(0.0, lock_wait_ms))
        # Option A: ttft_ms already equals ttft_infer_ms for direct calls.
        return texts[0], metrics

    def cleanup(self) -> None:
        """Free GPU memory."""
        try:
            if self._scheduler is not None:
                self._scheduler.shutdown()
        except Exception:
            pass

        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -------------------------------
# Multi-variant service (router wrapper)
# -------------------------------


_VARIANT_ORDER: Tuple[str, ...] = ("cheap", "med", "base")
_VARIANT_RANK = {v: i for i, v in enumerate(_VARIANT_ORDER)}


def _normalize_variant(v: str) -> str:
    v = (v or "").lower().strip()
    if v not in _VARIANT_RANK:
        raise ValueError(f"Unknown variant '{v}'. Expected one of: {list(_VARIANT_ORDER)}")
    return v


def _stronger_variants(start: str, allowed: List[str]) -> List[str]:
    """Return variants >= start (stronger / more accurate), within allowed."""
    start = _normalize_variant(start)
    out = [v for v in allowed if _VARIANT_RANK[v] >= _VARIANT_RANK[start]]
    return sorted(out, key=lambda x: _VARIANT_RANK[x])


def _weaker_variants(start: str, allowed: List[str]) -> List[str]:
    """Return variants <= start (cheaper / faster), within allowed."""
    start = _normalize_variant(start)
    out = [v for v in allowed if _VARIANT_RANK[v] <= _VARIANT_RANK[start]]
    return sorted(out, key=lambda x: _VARIANT_RANK[x])


@dataclass
class _VariantStats:
    ema_ttft_ms: Optional[float] = None
    ema_tpot_ms: Optional[float] = None
    ema_total_ms: Optional[float] = None
    n: int = 0

    def update(self, ttft_ms: float, tpot_ms: float, total_ms: float, alpha: float) -> None:
        def _ema(prev: Optional[float], x: float) -> float:
            if prev is None:
                return float(x)
            return float(alpha) * float(x) + (1.0 - float(alpha)) * float(prev)

        self.ema_ttft_ms = _ema(self.ema_ttft_ms, float(ttft_ms))
        self.ema_tpot_ms = _ema(self.ema_tpot_ms, float(tpot_ms))
        self.ema_total_ms = _ema(self.ema_total_ms, float(total_ms))
        self.n += 1


def _is_cuda_oom(err: BaseException) -> bool:
    """Best-effort check for CUDA OOM across PyTorch/transformers/bitsandbytes."""

    msg = str(err).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or ("cublas" in msg and "alloc" in msg)
        or ("cuda error" in msg and "out of memory" in msg)
    )


def _cuda_mem_gb() -> Tuple[float, float]:
    """Return (free_gb, total_gb) for the current CUDA device."""

    if not torch.cuda.is_available():
        return 0.0, 0.0

    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return float(free_b) / (1024**3), float(total_b) / (1024**3)
    except Exception:
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        free = max(0.0, float(total - allocated))
        return float(free), float(total)



@dataclass
class _MVRequest:
    prompt: str
    dataset_type: str
    difficulty: str
    prompt_mode: str
    max_tokens: int
    temperature: float
    top_p: float
    enqueue_t: float
    path: List[str]
    event: threading.Event

    # Routing context
    router_queue_depths: Dict[str, int] = field(default_factory=dict)
    router_meta: Dict[str, Any] = field(default_factory=dict)

    attempt_idx: int = 0
    attempts: List[Dict[str, Any]] = field(default_factory=list)

    # Result
    output_text: Optional[str] = None
    output_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def current_variant(self) -> str:
        return self.path[self.attempt_idx]


class MultiVariantService:
    """A concurrency-safe multi-variant router + dispatcher.

    Key idea (paper-ready):
      - Router runs in parallel per request (cheap CPU-side logic)
      - Requests are placed into one of 3 per-variant queues
      - A single dispatcher thread drains queues in batches, pins the active variant,
        and only then loads/evicts/swaps variants.

    This removes the classic race in "swap-mode" where one thread would evict a
    variant while another thread was still about to use it.
    """

    VARIANT_ORDER = ["cheap", "med", "base"]

    def __init__(
        self,
        model_name: str,
        variants: Optional[List[str]] = None,
        device: str = "cuda",
        dtype: str = "float16",
        router_mode: str = "difficulty",
        router_fixed_variant: Optional[str] = None,
        fixed_variant: Optional[str] = None,
        learned_router_dir: Optional[str] = None,
        enable_batching: bool = True,
        max_batch_size: int = 8,
        batch_wait_ms: int = 8,
        batch_timeout_s: Optional[float] = None,
        ema_alpha: float = 0.2,
        lazy_load_base: int = 0,
        router_max_retries: Optional[int] = None,
        max_retries: int = 1,
        router_calibration_mode: Optional[str] = None,
        calibration_mode: str = "base",
        # Swap / residency
        load_strategy: str = "auto",
        max_loaded_variants: Optional[int] = None,
        preload_variants: Optional[List[str]] = None,
        warmup: bool = False,
        # Router policy
        allow_quality_downgrade_for_slo: bool = False,
        # Dispatcher policy
        dispatcher_batch_wait_s: float = 0.002,
        dispatcher_max_sticky_batches: int = 4,
        dispatcher_starvation_ms: float = 50.0,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        self.router_mode = (router_mode or "difficulty").lower()

        # Alias support
        if fixed_variant is None and router_fixed_variant is not None:
            fixed_variant = router_fixed_variant
        if router_max_retries is not None:
            max_retries = router_max_retries
        if router_calibration_mode is not None:
            calibration_mode = router_calibration_mode
        if batch_timeout_s is not None and (batch_wait_ms is None or batch_wait_ms == 8):
            try:
                batch_wait_ms = int(max(0.0, float(batch_timeout_s)) * 1000)
            except Exception:
                pass

        self.fixed_variant = _normalize_variant(fixed_variant) if fixed_variant else None

        self.enable_batching = enable_batching
        self.max_batch_size = int(max(1, max_batch_size))
        self.batch_wait_ms = int(max(0, batch_wait_ms or 0))

        self.allow_quality_downgrade_for_slo = bool(allow_quality_downgrade_for_slo)
        self.max_retries = int(max(0, max_retries))
        self.calibration_mode = (calibration_mode or "base").lower()

        self.ema_alpha = float(max(0.0, min(1.0, ema_alpha)))
        self.lazy_load_base = bool(lazy_load_base)

        self.learned_router_dir = learned_router_dir
        self._learned_router = None
        if self.router_mode in {"learned_ttft", "learned_total"}:
            if not learned_router_dir:
                raise ValueError("--learned_router_dir is required when router_mode is learned_*.")
            from learned_router import LearnedRouter
            artifacts_dir = self._resolve_learned_router_path(learned_router_dir, self.router_mode)
            self._learned_router = LearnedRouter.load(artifacts_dir)

        self.dispatcher_batch_wait_s = float(max(0.0, dispatcher_batch_wait_s))
        self.dispatcher_max_sticky_batches = int(max(1, dispatcher_max_sticky_batches))
        self.dispatcher_starvation_ms = float(max(0.0, dispatcher_starvation_ms))

        requested_variants = variants or ["cheap", "med", "base"]
        requested_variants = [_normalize_variant(v) for v in requested_variants]

        supported = [v for v in requested_variants if self._is_variant_supported(v)]
        if "base" not in supported:
            supported.append("base")
        self.variants = [v for v in self.VARIANT_ORDER if v in supported]

        if self.fixed_variant and self.fixed_variant not in self.variants:
            logger.warning(
                f"Fixed variant '{self.fixed_variant}' is not supported on this device; falling back to 'base'."
            )
            self.fixed_variant = "base"

        self._stats = {v: _VariantStats() for v in self.variants}
        self.slo_dict = {}

        self.load_strategy = (load_strategy or "auto").lower()
        self.max_loaded_variants = max_loaded_variants
        self.preload_variants = [_normalize_variant(v) for v in (preload_variants or [])]

        self._servers = {}
        self._lru = []
        self._pins = {v: 0 for v in self.variants}
        self._load_events = {}

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._queues = {v: deque() for v in self.variants}
        self._shutdown = False
        self._active_variant = None
        self._active_batches_run = 0

        self._configure_capacity_and_preload()

        if warmup:
            for v in list(self._servers.keys()):
                try:
                    self._warmup_variant(v)
                except Exception:
                    pass

        self._dispatcher = threading.Thread(target=self._dispatcher_loop, name="mv-dispatcher", daemon=True)
        self._dispatcher.start()

        logger.info("MultiVariantService initialized")
        logger.info(f"  model_name={self.model_name}")
        logger.info(f"  variants={self.variants}")
        logger.info(f"  router_mode={self.router_mode}")
        logger.info(f"  load_strategy={self.load_strategy} (max_loaded_variants={self.max_loaded_variants})")

    # -------------------------
    # Variant support / startup
    # -------------------------

    def _is_variant_supported(self, variant: str) -> bool:
        variant = _normalize_variant(variant)
        if variant == "base":
            return True
        if self.device != "cuda":
            return False
        if variant == "cheap":
            return True
        if variant == "med":
            # Guardrail: bnb int8 is often unstable on older GPUs (e.g., Pascal/P100).
            try:
                cc_major, _ = torch.cuda.get_device_capability(0)
                return cc_major >= 7
            except Exception:
                return False
        return False

    def _configure_capacity_and_preload(self) -> None:
        """Choose how many variants to keep resident (1/2/3) and preload them."""
        if self.device != "cuda":
            self.max_loaded_variants = 1
            self.preload_variants = ["base"]
            self._ensure_loaded("base")
            return

        if self.load_strategy != "auto":
            # Respect explicit configuration.
            if self.max_loaded_variants is None:
                # Reasonable default: try keep 2 loaded (cheap+base) if using swap.
                self.max_loaded_variants = 2 if self.load_strategy in {"resident", "hybrid"} else 1
            if not self.preload_variants:
                self.preload_variants = ["base"]
            for v in self.preload_variants:
                if v in self.variants:
                    self._ensure_loaded(v)
            return

        # AUTO: probe what fits.
        # Prefer keeping base resident + as many additional variants as fit with some headroom.
        candidates: List[List[str]] = []
        # Try full residency
        if all(v in self.variants for v in ["cheap", "med", "base"]):
            candidates.append(["cheap", "med", "base"])
        if all(v in self.variants for v in ["cheap", "base"]):
            candidates.append(["cheap", "base"])
        if all(v in self.variants for v in ["med", "base"]):
            candidates.append(["med", "base"])
        candidates.append(["base"])

        picked: Optional[List[str]] = None
        for plan in candidates:
            if self._try_preload_plan(plan, safety_free_gb=1.0):
                picked = plan
                break

        if not picked:
            picked = ["base"]

        self.max_loaded_variants = len(picked)
        self.preload_variants = picked

        # Optionally avoid preloading base to reduce startup VRAM pressure.
        if self.lazy_load_base and len(picked) > 1 and "base" in picked:
            plan = [v for v in picked if v != "base"]
            if plan:
                self.preload_variants = plan
                self.max_loaded_variants = len(plan)
                # Re-load only the reduced preload plan. This is safe because it uses <= selected plan.
                self._cleanup_servers()
                for v in plan:
                    self._ensure_loaded(v)


    def _try_preload_plan(self, plan: List[str], safety_free_gb: float = 1.0) -> bool:
        """Try to load a set of variants; if OOM or too little free VRAM, undo."""
        # Clear any existing servers (startup only)
        self._cleanup_servers()

        self.max_loaded_variants = len(plan)

        try:
            for v in plan:
                self._ensure_loaded(v)
            # Optional headroom check
            info = GPUMonitor.get_gpu_info()
            free_gb = info.get("free")
            if free_gb is not None and free_gb < safety_free_gb:
                logger.warning(
                    f"AUTO strategy: plan {plan} leaves low free VRAM ({free_gb:.2f}GB < {safety_free_gb}GB)."
                )
                self._cleanup_servers()
                return False
            return True
        except Exception as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg and "memory" in msg:
                logger.warning(f"AUTO strategy: plan {plan} failed to load (OOM).")
            else:
                logger.warning(f"AUTO strategy: plan {plan} failed to load: {e}")
            self._cleanup_servers()
            return False

    # -------------------------
    # Public API
    # -------------------------

    def set_slo_dict(self, slo_dict: Dict[str, Dict[str, float]]) -> None:
        self.slo_dict = slo_dict or {}

    def get_queue_depth(self, variant: Optional[str] = None) -> int:
        with self._lock:
            if variant is None:
                return sum(len(q) for q in self._queues.values())
            v = _normalize_variant(variant)
            return len(self._queues.get(v, deque()))

    def get_variant_server(self, variant: str) -> SingleVariantServer:
        """Direct access to a variant server (used by calibration code).

        NOTE: If you call generate() on the returned server concurrently with the
        MultiVariantService dispatcher, you can re-introduce eviction races.
        For normal serving, always use MultiVariantService.generate().
        """
        return self._ensure_loaded(_normalize_variant(variant))

    def _resolve_learned_router_path(self, learned_router_dir: str, router_mode: str) -> str:
        """Resolve learned router artifacts path.

        Supports BOTH patterns:
          1) root folder: <dir>/learned_ttft and <dir>/learned_total
          2) mode folder: <dir> itself is one of {learned_ttft, learned_total}
        """
        import os
        base = os.path.abspath(os.path.expanduser(learned_router_dir))
        # If this folder already looks like a mode folder, use it.
        sentinels = [
            'weights.json',
            'quality_models.pkl',
            'latency_models.pkl',
            'ttft_models.pkl',
            'tpot_models.pkl',
        ]
        if any(os.path.isfile(os.path.join(base, s)) for s in sentinels):
            return base

        # Otherwise, treat as root and look for a subdirectory.
        cand = os.path.join(base, router_mode)
        if os.path.isdir(cand):
            return cand

        # If user passed a mode folder name but it's missing files, return it for clearer error later.
        if os.path.basename(base) == router_mode:
            return base

        raise FileNotFoundError(
            f"Could not resolve learned router artifacts for mode '{router_mode}'. "
            f"Expected either a mode folder with weights/models, or a root folder containing '{router_mode}/'. "
            f"Got: {learned_router_dir}"
        )

    def _warmup_variant(self, variant: str) -> None:
        """Warmup a variant once to catch obvious load issues early."""
        try:
            srv = self.get_variant_server(variant)
            _ = srv.generate(
                prompt='Warmup.',
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                dataset_type='mmlu',
                difficulty='easy',
                prompt_mode='slo',
                use_batching=True,
            )
        except Exception:
            return

    def cleanup(self) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        try:
            self._dispatcher.join(timeout=2.0)
        except Exception:
            pass
        self._cleanup_servers()

    # -------------------------
    # Internal: load / evict (pin-safe)
    # -------------------------

    def _touch_lru_locked(self, variant: str) -> None:
        if variant in self._lru:
            self._lru.remove(variant)
        self._lru.append(variant)

    def _cleanup_servers(self) -> None:
        # Remove + cleanup outside lock (startup)
        servers = list(self._servers.values())
        self._servers.clear()
        self._lru.clear()
        self._load_events.clear()
        for s in servers:
            try:
                s.cleanup()
            except Exception:
                pass
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _evict_one_locked(self) -> Optional[SingleVariantServer]:
        # Choose the least recently used unpinned server.
        for v in list(self._lru):
            if self._pins.get(v, 0) > 0:
                continue
            srv = self._servers.pop(v, None)
            if srv is not None:
                self._lru.remove(v)
                return srv
        return None
    def _ensure_loaded(self, variant: str) -> SingleVariantServer:
        """Ensure a SingleVariantServer for `variant` exists.

        Uses an event-based ...
        """
        variant = _normalize_variant(variant)
        if variant not in self.variants:
            raise ValueError(f"Variant '{variant}' not in enabled variants {self.variants}")

        while True:
            # Fast path / coordination
            with self._lock:
                existing = self._servers.get(variant)
                if existing is not None:
                    self._touch_lru_locked(variant)
                    return existing

                ev = self._load_events.get(variant)
                if ev is not None:
                    wait_ev = ev
                else:
                    # This thread becomes the loader for this variant
                    ev = threading.Event()
                    self._load_events[variant] = ev
                    wait_ev = None

                    # Evict to capacity (pin-safe)
                    victims = []
                    cap = int(self.max_loaded_variants or 1)
                    while len(self._servers) >= cap:
                        victim = self._evict_one_locked()
                        if victim is None:
                            # Cannot evict anything (everything pinned)
                            self._load_events.pop(variant, None)
                            ev.set()
                            raise RuntimeError(
                                f"Cannot load '{variant}': all loaded variants are pinned (loaded={list(self._servers.keys())})."
                            )
                        victims.append(victim)

            if wait_ev is not None:
                # Another loader is working; wait and retry
                wait_ev.wait()
                continue

            # Loader branch: clean up victims outside the lock
            for s in victims:
                try:
                    s.cleanup()
                except Exception:
                    pass
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Actually load variant (no global lock held)
            try:
                srv = SingleVariantServer(
                    model_name=self.model_name,
                    variant=variant,
                    device=self.device,
                    dtype=self.dtype,
                    enable_batching=self.enable_batching,
                    max_batch_size=self.max_batch_size,
                    batch_wait_ms=self.batch_wait_ms,
                )
            except Exception:
                with self._lock:
                    ev = self._load_events.pop(variant, None)
                    if ev:
                        ev.set()
                raise

            with self._lock:
                self._servers[variant] = srv
                self._touch_lru_locked(variant)
                ev = self._load_events.pop(variant, None)
                if ev:
                    ev.set()
                return srv

    # -------------------------
    # Routing

    # -------------------------

    def _queue_depth_locked(self, variant: str) -> int:
        return len(self._queues.get(variant, deque()))

    def _predict_latency_ms(self, variant: str, max_tokens: int) -> float:
        """Predict end-to-end latency (excluding MultiVariant dispatcher wait)."""
        stats = self._stats.get(variant) or _VariantStats()
        base_total = stats.ema_total_ms or 1000.0
        # Simple scaling with output length
        if max_tokens and max_tokens > 64:
            base_total *= (max_tokens / 64.0) ** 0.5

        # Queue penalty: approximate by depth * EMA
        qd = self._queue_depth_locked(variant)
        penalty = qd * (stats.ema_total_ms or base_total) * 0.35
        return base_total + penalty

    def choose_variant(
        self,
        dataset_type: str,
        difficulty: str,
        prompt_mode: str,
        max_tokens: int,
        estimated_tokens: int,
        queue_depths: Dict[str, int],
    ) -> Tuple[str, str, Dict[str, Any]]:
        # Returns (variant, reason, router_meta)
        dataset_type = (dataset_type or "gsm8k").lower()
        difficulty = (difficulty or "easy").lower()
        prompt_mode = (prompt_mode or "slo").lower()

        # Explicit fixed behaviors
        if self.router_mode == "fixed" and self.fixed_variant:
            return self.fixed_variant, "fixed", {}
        if self.router_mode == "always_cheap":
            v = "cheap" if "cheap" in self.variants else ("med" if "med" in self.variants else "base")
            return v, "always_cheap", {}
        if self.router_mode == "always_base":
            return "base", "always_base", {}

        if self.fixed_variant:
            return self.fixed_variant, "fixed", {}

        # Difficulty router: cheap for easy, med for medium, base for hard.
        if self.router_mode == "difficulty":
            if difficulty in {"hard", "difficult"}:
                v = "base" if "base" in self.variants else self.variants[-1]
                return v, "difficulty=hard", {}
            if difficulty in {"medium", "med"}:
                v = "med" if "med" in self.variants else ("cheap" if "cheap" in self.variants else "base")
                return v, "difficulty=medium", {}
            v = "cheap" if "cheap" in self.variants else ("med" if "med" in self.variants else "base")
            return v, "difficulty=easy", {}

        # Learned routers
        if self.router_mode in {"learned_ttft", "learned_total"}:
            if self._learned_router is None:
                raise RuntimeError("Learned router requested but artifacts were not loaded.")
            mode = "ttft" if self.router_mode == "learned_ttft" else "total"
            decision = self._learned_router.route(
                dataset_type=dataset_type,
                difficulty=difficulty,
                max_tokens=int(max_tokens),
                estimated_tokens=int(estimated_tokens),
                queue_depths=queue_depths,
                slo_dict=self.slo_dict,
                mode=mode,
                allowed_variants=self.variants,
            )
            meta = decision.to_dict()
            meta["router_mode_label"] = "Learned-TTFT" if self.router_mode == "learned_ttft" else "Learned-Total (Derived)"
            return decision.variant, meta["router_mode_label"], meta

        # SLO-aware: choose cheapest predicted to meet SLO.
        if self.router_mode == "slo_aware" and self.slo_dict:
            slo = self.slo_dict.get(difficulty) or self.slo_dict.get("default")
            if slo:
                ttft_slo = float(slo.get("ttft_ms", 1e9))
                tpot_slo = float(slo.get("tpot_ms", 1e9))
                for v in self.VARIANT_ORDER:
                    if v not in self.variants:
                        continue
                    st = self._stats.get(v) or _VariantStats()
                    pred_ttft = st.ema_ttft_ms or (0.6 * (st.ema_total_ms or 1000.0))
                    pred_tpot = st.ema_tpot_ms or 5.0
                    if pred_ttft <= ttft_slo and pred_tpot <= tpot_slo:
                        return v, f"slo_meet({difficulty})", {}
                return "base", f"slo_miss({difficulty})", {}

        return "base", "default", {}


    def plan_path(
        self,
        dataset_type: str,
        difficulty: str,
        prompt_mode: str,
        max_tokens: int,
        estimated_tokens: Optional[int] = None,
    ) -> Tuple[List[str], str, Dict[str, Any], Dict[str, int]]:
        # Snapshot queue depths at routing time
        with self._lock:
            queue_depths = {v: len(self._queues.get(v, deque())) for v in self.variants}
            chosen, reason, meta = self.choose_variant(
                dataset_type=dataset_type,
                difficulty=difficulty,
                prompt_mode=prompt_mode,
                max_tokens=max_tokens,
                estimated_tokens=estimated_tokens,
                queue_depths=queue_depths,
            )

        path = [chosen]
        if self.max_retries > 0 and chosen != "base" and "base" in self.variants:
            if chosen == "cheap" and "med" in self.variants and len(path) < (self.max_retries + 1):
                path.append("med")
            if len(path) < (self.max_retries + 1):
                path.append("base")

        return path[: self.max_retries + 1], reason, meta, queue_depths


    def _should_retry(self, dataset_type: str, text: str) -> bool:
        dataset_type = dataset_type.lower()
        if dataset_type == "gsm8k":
            ans = extract_gsm8k_strict(text)
            return not bool(ans)
        if dataset_type == "mmlu":
            ans = extract_mmlu_answer(text)
            return ans is None
        return False

    # -------------------------
    # Serving API (thread-safe)
    # -------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        dataset_type: str = "gsm8k",
        difficulty: str = "easy",
        prompt_mode: str = "slo",
        use_batching: bool = True,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        dataset_type = dataset_type.lower()
        difficulty = (difficulty or "easy").lower()
        prompt_mode = (prompt_mode or "slo").lower()

        # Align max_tokens with SingleVariantServer defaults
        if max_tokens is None:
            if dataset_type == "mmlu":
                max_tokens = 1
            elif prompt_mode == "slo":
                max_tokens = 128
            else:
                max_tokens = 256

        est_tokens = len(prompt.split())
        path, reason, router_meta, qdepths = self.plan_path(dataset_type, difficulty, prompt_mode, int(max_tokens), estimated_tokens=est_tokens)

        req = _MVRequest(
            prompt=prompt,
            dataset_type=dataset_type,
            difficulty=difficulty,
            prompt_mode=prompt_mode,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            enqueue_t=time.perf_counter(),
            path=path,
            event=threading.Event(),
            router_queue_depths=qdepths,
            router_meta=router_meta,
        )

        # Queue request for its first variant
        first = req.current_variant
        with self._cv:
            if first not in self._queues:
                # Should not happen, but be safe
                first = "base"
                req.path = ["base"]
                req.attempt_idx = 0
            self._queues[first].append(req)
            self._cv.notify()

        # Wait for completion
        req.event.wait()

        if req.output_metrics is None:
            # Emergency fallback
            err = req.error or "unknown_error"
            metrics = {
                "success": False,
                "backend": "hf",
                "variant": "base",
                "model": self.model_name,
                "device": self.device,
                "error": err,
                "router_mode": self.router_mode,
                "router_reason": reason,
                "router_path": path,
                "router_attempts": req.attempts,
                "router_escalated": len(req.attempts) > 1,
            }
            return "", metrics

        # Attach router metadata
        req.output_metrics["router_mode"] = self.router_mode
        req.output_metrics["router_reason"] = reason
        req.output_metrics["router_path"] = path
        req.output_metrics["router_attempts"] = req.attempts
        req.output_metrics["router_escalated"] = len(req.attempts) > 1
        req.output_metrics["router_final_variant"] = req.output_metrics.get("variant")
        req.output_metrics["router_num_attempts"] = len(req.attempts)
        req.output_metrics["router_queue_depths"] = getattr(req, "router_queue_depths", {})
        req.output_metrics["router_meta"] = getattr(req, "router_meta", {})

        return req.output_text or "", req.output_metrics

    # -------------------------
    # Dispatcher
    # -------------------------

    def _select_next_variant_locked(self) -> Optional[str]:
        non_empty = [v for v, q in self._queues.items() if len(q) > 0]
        if not non_empty:
            return None

        # Stickiness: keep serving current variant if it has work and hasn't hit max sticky batches
        if self._active_variant in non_empty:
            # Starvation guard: if another queue has been waiting too long, switch.
            now = time.perf_counter()
            oldest_other_ms = None
            for v in non_empty:
                if v == self._active_variant:
                    continue
                head = self._queues[v][0]
                w_ms = (now - head.enqueue_t) * 1000.0
                if oldest_other_ms is None or w_ms > oldest_other_ms:
                    oldest_other_ms = w_ms
            if (
                self._active_batches_run < self.dispatcher_max_sticky_batches
                and (oldest_other_ms is None or oldest_other_ms < self.dispatcher_starvation_ms)
            ):
                return self._active_variant

        # Otherwise pick the queue with the oldest head-of-line request
        now = time.perf_counter()
        best_v = None
        best_wait = -1.0
        for v in non_empty:
            head = self._queues[v][0]
            w = (now - head.enqueue_t) * 1000.0
            if w > best_wait:
                best_wait = w
                best_v = v
        return best_v

    def _pop_batch_locked(self, variant: str) -> List[_MVRequest]:
        q = self._queues[variant]
        if not q:
            return []

        first = q[0]
        key = (first.dataset_type, first.prompt_mode, first.max_tokens, first.temperature, first.top_p)

        batch: List[_MVRequest] = []
        while q and len(batch) < self.max_batch_size:
            nxt = q[0]
            k2 = (nxt.dataset_type, nxt.prompt_mode, nxt.max_tokens, nxt.temperature, nxt.top_p)
            if k2 != key:
                break
            batch.append(q.popleft())

        return batch

    def _dispatcher_loop(self) -> None:
        while True:
            with self._cv:
                while not self._shutdown:
                    v = self._select_next_variant_locked()
                    if v is not None:
                        break
                    self._cv.wait(timeout=0.05)
                if self._shutdown:
                    return

                variant = v
                batch = self._pop_batch_locked(variant)

                # Update stickiness counters
                if self._active_variant == variant:
                    self._active_batches_run += 1
                else:
                    self._active_variant = variant
                    self._active_batches_run = 1

            if not batch:
                continue

            try:
                # Pin while this variant is being served
                with self._lock:
                    self._pins[variant] = self._pins.get(variant, 0) + 1

                # Ensure loaded (may evict/load)
                srv = self._ensure_loaded(variant)

                infer_start = time.perf_counter()

                prompts = [r.prompt for r in batch]
                require_all_final = batch[0].dataset_type == "gsm8k"
                outputs, metrics_list, lock_wait_ms = srv._generate_hf_batch(
                    prompts,
                    dataset_type=batch[0].dataset_type,
                    max_tokens=batch[0].max_tokens,
                    temperature=batch[0].temperature,
                    top_p=batch[0].top_p,
                    prompt_mode=batch[0].prompt_mode,
                    require_all_final_answers=require_all_final,
                )

                for r, out, m in zip(batch, outputs, metrics_list):
                    # MultiVariant dispatch wait
                    sched_wait_ms = (infer_start - r.enqueue_t) * 1000.0
                    m["scheduler_wait_ms"] = sched_wait_ms
                    m["lock_wait_ms"] = float(lock_wait_ms)
                    m["queue_wait_ms"] = sched_wait_ms + float(lock_wait_ms)

                    # TTFT includes scheduler wait (paper Option A)
                    try:
                        m["ttft_ms"] = sched_wait_ms + float(m.get("ttft_infer_ms", 0.0))
                    except Exception:
                        pass
                    try:
                        m["total_latency_ms"] = float(m.get("total_latency_ms", 0.0)) + sched_wait_ms
                    except Exception:
                        pass

                    # Record attempt
                    r.attempts.append(
                        {
                            "variant": m.get("variant"),
                            "success": bool(m.get("success", False)),
                            "ttft_ms": m.get("ttft_ms"),
                            "tpot_ms": m.get("tpot_ms"),
                            "total_latency_ms": m.get("total_latency_ms"),
                            "output_tokens": m.get("output_tokens"),
                        }
                    )

                    # Update EMA stats using *inference* components (exclude dispatcher wait)
                    with self._lock:
                        st = self._stats.get(variant)
                        if st and m.get("success"):
                            try:
                                ttft_infer = float(m.get("ttft_infer_ms", 0.0) or 0.0)
                                tpot = float(m.get("tpot_ms", 0.0) or 0.0)
                                total = float(m.get("total_latency_ms", 0.0) or 0.0) - sched_wait_ms
                                n_out = int(m.get("output_tokens", m.get("n_output_tokens", 0)) or 0)
                                q_wait = float(m.get("queue_wait_ms", 0.0) or 0.0)
                                st.update(ttft_infer, tpot, total, n_out, q_wait, alpha=self.ema_alpha)
                            except Exception:
                                pass

                    # Retry on format failure
                    if m.get("success") and self._should_retry(r.dataset_type, out):
                        if r.attempt_idx + 1 < len(r.path):
                            r.attempt_idx += 1
                            nxt = r.current_variant
                            with self._cv:
                                if nxt not in self._queues:
                                    nxt = "base"
                                self._queues[nxt].append(r)
                                self._cv.notify()
                            continue

                    # Finalize
                    r.output_text = out
                    r.output_metrics = m
                    r.event.set()

            except Exception as e:
                # Fail the whole batch
                err = str(e)
                for r in batch:
                    r.error = err
                    r.output_text = ""
                    r.output_metrics = {
                        "success": False,
                        "backend": "hf",
                        "variant": variant,
                        "model": self.model_name,
                        "device": self.device,
                        "error": err,
                        "router_mode": self.router_mode,
                        "router_path": r.path,
                        "router_attempts": r.attempts,
                    }
                    r.event.set()

            finally:
                with self._lock:
                    self._pins[variant] = max(0, self._pins.get(variant, 0) - 1)


if __name__ == "__main__":
    # Minimal smoke test (won't run without model access).
    server = SingleVariantServer(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        variant="med",
        device="auto",
        enable_batching=True,
        max_batch_size=4,
        batch_wait_ms=8,
    )
    out, m = server.generate(
        prompt="You are a helpful assistant. Reply with one letter: A.",
        dataset_type="mmlu",
        prompt_mode="slo",
    )
    print(out, m)
