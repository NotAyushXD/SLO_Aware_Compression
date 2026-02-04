import gc
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
        self.variant = variant
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

        # Quantization by variant
        quant_config = None
        load_kwargs: Dict = {
            "device_map": "auto" if self.device == "cuda" else None,
        }

        if self.variant == "med":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif self.variant == "cheap":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=model_dtype if model_dtype != "auto" else torch.float16,
            )
        elif self.variant == "base":
            quant_config = None
        else:
            logger.warning(f"Unknown variant '{self.variant}', defaulting to med (8-bit)")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config

        # transformers>=4.57 deprecates `torch_dtype` in favor of `dtype`.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype, **load_kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=model_dtype, **load_kwargs)

        self.model.eval()

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
                logger.error(f"Warmup error: {e}")
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


class MultiVariantService:
    """Single-process wrapper that routes requests across {cheap, med, base} variants.

    Goals:
      - Keep the same public API as SingleVariantServer.generate() so it can be used
        by load_generator.py and evaluation.py without changes.
      - Maintain lightweight per-variant state (in-flight, queue depth, EMA latencies)
        to support SLO-aware routing.
      - Provide a simple, paper-friendly *escalation* mechanism via format checks:
        if the chosen (cheaper) variant produces an unparsable answer format, retry
        on a stronger variant (bounded by max_retries).

    This class is intentionally conservative: it does NOT require a learned quality
    predictor yet. You can later plug in a predictor by overriding choose_variant().
    """

    DEFAULT_MIN_VARIANT_BY_DIFFICULTY = {"easy": "cheap", "medium": "med", "hard": "base"}

    # Default SLOs (used if no calibration file is supplied).
    DEFAULT_SLOS = {
        "easy": {"ttft_ms": 150.0, "tpot_ms": 1000.0},
        "medium": {"ttft_ms": 250.0, "tpot_ms": 1000.0},
        "hard": {"ttft_ms": 400.0, "tpot_ms": 1500.0},
    }

    def __init__(
        self,
        model_name: str,
        variants: Tuple[str, ...] = ("cheap", "med", "base"),
        device: str = "auto",
        dtype: str = "auto",
        enable_batching: bool = False,
        max_batch_size: int = 4,
        batch_wait_ms: int = 8,
        # Routing knobs
        router_mode: str = "difficulty",
        fixed_variant: Optional[str] = None,
        min_variant_by_difficulty: Optional[Dict[str, str]] = None,
        allow_quality_downgrade_for_slo: bool = False,
        slo_dict: Optional[Dict[str, Dict[str, float]]] = None,
        slo_safety_factor: float = 1.05,
        # Escalation (retry) knobs
        max_retries: int = 1,
        retry_on_format_error: bool = True,
        # EMA smoothing for online latency estimates
        ema_alpha: float = 0.2,
        # Loading strategy (GPU-memory aware)
        #   auto: choose an appropriate plan based on detected GPU memory
        #   eager: load all enabled variants upfront
        #   lazy_base: preload non-base variants; load base on first use
        #   swap: keep at most N variants resident (default N=1)
        load_strategy: str = "auto",
        max_loaded_variants: Optional[int] = None,
        # Backward-compatible alias (kept for older scripts)
        lazy_load_base: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_batching = bool(enable_batching)
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_ms = int(batch_wait_ms)

        self.router_mode = (router_mode or "difficulty").lower().strip()
        self.fixed_variant = _normalize_variant(fixed_variant) if fixed_variant else None

        self.allow_quality_downgrade_for_slo = bool(allow_quality_downgrade_for_slo)

        self.min_variant_by_difficulty = dict(self.DEFAULT_MIN_VARIANT_BY_DIFFICULTY)
        if min_variant_by_difficulty:
            for k, v in min_variant_by_difficulty.items():
                self.min_variant_by_difficulty[(k or "").lower().strip()] = _normalize_variant(v)

        self.slo_dict: Dict[str, Dict[str, float]] = {k: dict(v) for k, v in (slo_dict or self.DEFAULT_SLOS).items()}
        self.slo_safety_factor = float(max(0.5, slo_safety_factor))

        self.max_retries = int(max(0, max_retries))
        self.retry_on_format_error = bool(retry_on_format_error)
        self.ema_alpha = float(min(max(ema_alpha, 0.01), 0.95))

        self._resolved_device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device

        self._variants: List[str] = sorted({_normalize_variant(v) for v in variants}, key=lambda x: _VARIANT_RANK[x])
        if not self._variants:
            raise ValueError("MultiVariantService requires at least one variant")

        # bitsandbytes 4/8-bit quantization is CUDA-only. If CUDA is unavailable,
        # fall back to a single 'base' server to keep the harness runnable.
        if self._resolved_device != "cuda":
            if "base" not in self._variants or len(self._variants) > 1:
                logger.warning(
                    "CUDA is not available (or device=cpu). For multi-variant serving, "
                    "forcing a single 'base' variant on CPU (4/8-bit quantization is CUDA-only)."
                )
            self._variants = ["base"]

        # -------------------------------
        # Load strategy selection
        # -------------------------------
        self.load_strategy = (load_strategy or "auto").lower().strip()
        if lazy_load_base and self.load_strategy == "auto":
            self.load_strategy = "lazy_base"

        if self._resolved_device == "cuda" and self.load_strategy == "auto":
            _free_gb, total_gb = _cuda_mem_gb()

            # Heuristic thresholds tuned for common Llama-8B deployments.
            # We still protect with OOM fallbacks during preload.
            if total_gb >= 48.0:
                self.load_strategy = "eager"
            elif total_gb >= 24.0 and ("base" in self._variants and len(self._variants) > 1):
                self.load_strategy = "lazy_base"
            elif len(self._variants) > 1:
                self.load_strategy = "swap"
            else:
                self.load_strategy = "eager"

        if self.load_strategy not in {"auto", "eager", "lazy_base", "swap"}:
            logger.warning(f"Unknown load_strategy='{self.load_strategy}', defaulting to auto")
            self.load_strategy = "auto"

        if max_loaded_variants is not None and int(max_loaded_variants) > 0:
            self.max_loaded_variants = int(max_loaded_variants)
        else:
            if self.load_strategy == "eager":
                self.max_loaded_variants = len(self._variants)
            elif self.load_strategy == "lazy_base":
                # Keep cheap+med resident when possible; base is loaded on demand.
                non_base = [v for v in self._variants if v != "base"]
                self.max_loaded_variants = max(1, min(len(non_base), 2))
            elif self.load_strategy == "swap":
                self.max_loaded_variants = 1
            else:
                self.max_loaded_variants = len(self._variants)

        self.max_loaded_variants = int(max(1, min(self.max_loaded_variants, len(self._variants))))

        # Internal state
        self._lock = threading.Lock()
        self._servers: Dict[str, Optional[SingleVariantServer]] = {v: None for v in self._variants}
        self._stats: Dict[str, _VariantStats] = {v: _VariantStats() for v in self._variants}
        self._in_flight: Dict[str, int] = {v: 0 for v in self._variants}

        # LRU list of currently loaded variants (oldest first).
        self._lru: List[str] = []

        # Preload variants based on strategy.
        if self.load_strategy == "eager":
            preload = list(self._variants)
        elif self.load_strategy == "lazy_base":
            preload = [v for v in self._variants if v != "base"]
            if not preload:
                preload = list(self._variants)
        elif self.load_strategy == "swap":
            preload = [self._variants[0]]  # cheapest
        else:
            preload = list(self._variants)

        logger.info("MultiVariantService load strategy")
        logger.info(f"  resolved_device={self._resolved_device}")
        logger.info(f"  load_strategy={self.load_strategy}")
        logger.info(f"  max_loaded_variants={self.max_loaded_variants}")
        logger.info(f"  preload={preload}")

        for v in preload:
            try:
                self._servers[v] = self._create_server(v)
                self._lru.append(v)
            except RuntimeError as e:
                # If we OOM while preloading, downgrade to swap (one-at-a-time) mode.
                if self._resolved_device == "cuda" and _is_cuda_oom(e):
                    logger.warning(
                        f"OOM while preloading variant '{v}'. Downgrading to swap mode (1 variant resident)."
                    )
                    self.load_strategy = "swap"
                    self.max_loaded_variants = 1

                    # Keep the cheapest already-loaded variant (if any), evict the rest.
                    keep = next((vv for vv in self._variants if self._servers.get(vv) is not None), None)
                    if keep is None:
                        keep = self._variants[0]
                    self._evict_all_except({keep})
                    break
                raise

        logger.info("MultiVariantService initialized")
        logger.info(f"  model_name={self.model_name}")
        logger.info(f"  variants={self._variants}")
        logger.info(f"  router_mode={self.router_mode}")
        if self.fixed_variant:
            logger.info(f"  fixed_variant={self.fixed_variant}")

    # -------------------------------
    # Lifecycle helpers
    # -------------------------------

    def _create_server(self, variant: str) -> SingleVariantServer:
        variant = _normalize_variant(variant)
        logger.info(f"Loading variant '{variant}'...")
        # A small empty_cache between loads helps fragmentation on some GPUs.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return SingleVariantServer(
            model_name=self.model_name,
            variant=variant,
            device=self.device,
            dtype=self.dtype,
            enable_batching=self.enable_batching,
            max_batch_size=self.max_batch_size,
            batch_wait_ms=self.batch_wait_ms,
        )

    def _num_loaded_locked(self) -> int:
        return int(sum(1 for s in self._servers.values() if s is not None))

    def _touch_lru_locked(self, variant: str) -> None:
        """Move variant to the most-recently-used position."""
        try:
            if variant in self._lru:
                self._lru.remove(variant)
            self._lru.append(variant)
        except Exception:
            # LRU is best-effort; never break serving due to bookkeeping.
            pass

    def _evict_variants(self, variants: List[str]) -> None:
        """Evict (unload) the provided variants, best-effort."""

        to_cleanup: List[SingleVariantServer] = []
        with self._lock:
            for v in variants:
                srv = self._servers.get(v)
                if srv is None:
                    continue
                if int(self._in_flight.get(v, 0)) > 0:
                    continue
                self._servers[v] = None
                try:
                    if v in self._lru:
                        self._lru.remove(v)
                except Exception:
                    pass
                to_cleanup.append(srv)

        for s in to_cleanup:
            try:
                s.cleanup()
            except Exception:
                pass

        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _evict_all_except(self, keep: set) -> None:
        keep = set(keep or set())
        victims = [v for v, s in self._servers.items() if s is not None and v not in keep]
        self._evict_variants(victims)

    def _evict_until_fit(self, *, reserve_for: str, needed_slots: int) -> None:
        """Evict LRU variants until we have room for `needed_slots` new loads."""

        if self.max_loaded_variants <= 0:
            return

        victims: List[str] = []
        with self._lock:
            loaded = self._num_loaded_locked()
            target = max(0, int(self.max_loaded_variants) - int(needed_slots))
            if loaded <= target:
                return

            # Evict oldest variants first, but never evict the one we are trying to load.
            for v in list(self._lru):
                if loaded <= target:
                    break
                if v == reserve_for:
                    continue
                if int(self._in_flight.get(v, 0)) > 0:
                    continue
                victims.append(v)
                loaded -= 1

        if victims:
            self._evict_variants(victims)

    def get_variant_server(self, variant: str) -> SingleVariantServer:
        """Return a loaded server, loading lazily if needed.

        If `max_loaded_variants` is small (e.g., swap mode), this method will
        evict older variants to make room before loading the requested one.
        """
        variant = _normalize_variant(variant)
        if variant not in self._servers:
            raise ValueError(f"Variant '{variant}' is not enabled in this service")

        with self._lock:
            srv = self._servers.get(variant)
            if srv is not None:
                self._touch_lru_locked(variant)
                return srv

        # Make room for this load.
        self._evict_until_fit(reserve_for=variant, needed_slots=1)

        # Lazy load (with a single OOM-triggered retry after evicting everything else).
        try:
            new_srv = self._create_server(variant)
        except RuntimeError as e:
            if self._resolved_device == "cuda" and _is_cuda_oom(e):
                logger.warning(
                    f"OOM while loading variant '{variant}'. Evicting all other variants and retrying once."
                )
                self._evict_all_except(set())
                new_srv = self._create_server(variant)
            else:
                raise

        with self._lock:
            self._servers[variant] = new_srv
            self._touch_lru_locked(variant)
            return new_srv

    def set_slo_dict(self, slo_dict: Dict[str, Dict[str, float]]) -> None:
        """Update SLO thresholds used by SLO-aware routing."""
        if not isinstance(slo_dict, dict) or not slo_dict:
            return
        with self._lock:
            self.slo_dict = {k: dict(v) for k, v in slo_dict.items() if isinstance(v, dict)}

    def cleanup(self) -> None:
        """Free GPU memory for all loaded variants."""
        with self._lock:
            servers = [s for s in self._servers.values() if s is not None]
        for s in servers:
            try:
                s.cleanup()
            except Exception:
                pass

    # -------------------------------
    # State / prediction helpers
    # -------------------------------

    def _queue_depth(self, variant: str) -> int:
        try:
            srv = self.get_variant_server(variant)
        except Exception:
            return 0
        try:
            return int(srv.get_queue_depth())
        except Exception:
            return 0

    def _predict_latency(self, variant: str) -> Tuple[float, float]:
        """Return (pred_ttft_ms, pred_tpot_ms) using simple EMA + queue/inflight penalty."""
        variant = _normalize_variant(variant)
        st = self._stats.get(variant) or _VariantStats()

        # Bootstraps: conservative-ish per-variant defaults (only used until EMA warms up)
        base_ttft = {"cheap": 160.0, "med": 220.0, "base": 280.0}.get(variant, 220.0)
        base_tpot = {"cheap": 25.0, "med": 30.0, "base": 35.0}.get(variant, 30.0)

        ttft = float(st.ema_ttft_ms if st.ema_ttft_ms is not None else base_ttft)
        tpot = float(st.ema_tpot_ms if st.ema_tpot_ms is not None else base_tpot)

        with self._lock:
            inflight = int(self._in_flight.get(variant, 0))
        qdepth = self._queue_depth(variant)

        # Penalty heuristic: inflate with outstanding work.
        load = float(inflight + qdepth)
        ttft *= (1.0 + 0.12 * load)
        tpot *= (1.0 + 0.06 * load)

        return float(ttft), float(tpot)

    def _format_ok(self, text: str, dataset_type: str) -> bool:
        dt = (dataset_type or "").lower().strip()
        if dt == "mmlu":
            return bool(extract_mmlu_answer(text))
        if dt == "gsm8k":
            # Strict is the primary metric; parseable is used only as a fallback signal.
            return bool(extract_gsm8k_strict(text) or extract_gsm8k_parseable(text))
        return True

    # -------------------------------
    # Routing policy
    # -------------------------------

    def choose_variant(
        self,
        *,
        dataset_type: str,
        difficulty: str,
        prompt_mode: str,
    ) -> Tuple[str, str, Dict[str, float]]:
        """Choose an initial variant.

        Returns:
          (variant, reason, debug_pred)

        You can override this method to plug in a learned router.
        """
        difficulty = (difficulty or "medium").lower().strip()
        prompt_mode = (prompt_mode or "slo").lower().strip()

        if self.router_mode in ("fixed", "single"):
            if not self.fixed_variant:
                raise ValueError("router_mode='fixed' requires fixed_variant")
            v = self.fixed_variant
            ttft, tpot = self._predict_latency(v)
            return v, "fixed", {"pred_ttft_ms": ttft, "pred_tpot_ms": tpot}

        # Accuracy mode: default to strongest available (base if enabled)
        if prompt_mode == "accuracy":
            v = "base" if "base" in self._variants else self._variants[-1]
            ttft, tpot = self._predict_latency(v)
            return v, "accuracy_mode_strongest", {"pred_ttft_ms": ttft, "pred_tpot_ms": tpot}

        # Difficulty proxy for quality
        min_v = self.min_variant_by_difficulty.get(difficulty, "med")
        if min_v not in self._variants:
            # pick the strongest enabled
            min_v = self._variants[-1]

        if self.router_mode == "difficulty":
            ttft, tpot = self._predict_latency(min_v)
            return min_v, f"difficulty_min({difficulty})", {"pred_ttft_ms": ttft, "pred_tpot_ms": tpot}

        if self.router_mode in ("always_cheap", "cheap"):
            v = "cheap" if "cheap" in self._variants else self._variants[0]
            ttft, tpot = self._predict_latency(v)
            return v, "always_cheap", {"pred_ttft_ms": ttft, "pred_tpot_ms": tpot}

        if self.router_mode in ("always_base", "base"):
            v = "base" if "base" in self._variants else self._variants[-1]
            ttft, tpot = self._predict_latency(v)
            return v, "always_base", {"pred_ttft_ms": ttft, "pred_tpot_ms": tpot}

        # SLO-aware heuristic:
        #   Start from min_v (quality proxy), but allow downshift to meet SLO if enabled.
        slo = self.slo_dict.get(difficulty, self.slo_dict.get("medium", self.DEFAULT_SLOS["medium"]))
        ttft_slo = float(slo.get("ttft_ms", 0.0) or 0.0) * self.slo_safety_factor
        tpot_slo = float(slo.get("tpot_ms", 0.0) or 0.0) * self.slo_safety_factor

        # Candidates to try (default: min_v and stronger). If allowed to downgrade for SLO, try cheaper first.
        if self.allow_quality_downgrade_for_slo:
            candidates = _weaker_variants(min_v, self._variants)
        else:
            candidates = _stronger_variants(min_v, self._variants)

        chosen = candidates[-1]  # fallback strongest in candidate set
        chosen_pred = {"pred_ttft_ms": 0.0, "pred_tpot_ms": 0.0}
        for v in candidates:
            pred_ttft, pred_tpot = self._predict_latency(v)
            if pred_ttft <= ttft_slo and pred_tpot <= tpot_slo:
                chosen = v
                chosen_pred = {"pred_ttft_ms": float(pred_ttft), "pred_tpot_ms": float(pred_tpot)}
                return chosen, f"slo_ok({difficulty})", chosen_pred

            # Keep last prediction for debug
            chosen_pred = {"pred_ttft_ms": float(pred_ttft), "pred_tpot_ms": float(pred_tpot)}

        return chosen, f"slo_fallback({difficulty})", chosen_pred

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
        **kwargs,
    ) -> Tuple[str, Dict]:
        """Route a request and run generation on the selected variant."""

        dataset_type_n = (dataset_type or "").lower().strip()
        difficulty_n = (difficulty or "medium").lower().strip()
        prompt_mode_n = (prompt_mode or "slo").lower().strip()

        # Initial route decision
        try:
            chosen, reason, pred = self.choose_variant(
                dataset_type=dataset_type_n,
                difficulty=difficulty_n,
                prompt_mode=prompt_mode_n,
            )
        except Exception as e:
            chosen, reason, pred = (self._variants[-1], f"router_error:{e}", {"pred_ttft_ms": 0.0, "pred_tpot_ms": 0.0})

        # Build escalation path (stronger variants only)
        path = _stronger_variants(chosen, self._variants)

        attempts = []
        final_text = ""
        final_metrics: Dict = {}
        escalated = False

        max_attempts = 1 + self.max_retries
        for attempt_idx, variant in enumerate(path[:max_attempts]):
            srv = self.get_variant_server(variant)

            with self._lock:
                self._in_flight[variant] = int(self._in_flight.get(variant, 0)) + 1
            try:
                text, metrics = srv.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    dataset_type=dataset_type,
                    difficulty=difficulty,
                    prompt_mode=prompt_mode,
                    use_batching=use_batching,
                    **kwargs,
                )
            finally:
                with self._lock:
                    self._in_flight[variant] = max(0, int(self._in_flight.get(variant, 0)) - 1)

            success = bool(metrics.get("success", False))
            fmt_ok = self._format_ok(text, dataset_type_n)

            attempts.append(
                {
                    "variant": variant,
                    "success": success,
                    "format_ok": fmt_ok,
                    "ttft_ms": metrics.get("ttft_ms"),
                    "tpot_ms": metrics.get("tpot_ms"),
                    "total_latency_ms": metrics.get("total_latency_ms"),
                }
            )

            final_text, final_metrics = text, metrics

            # Update online latency stats (only on success)
            if success:
                try:
                    self._stats[variant].update(
                        ttft_ms=float(metrics.get("ttft_ms", 0.0) or 0.0),
                        tpot_ms=float(metrics.get("tpot_ms", 0.0) or 0.0),
                        total_ms=float(metrics.get("total_latency_ms", 0.0) or 0.0),
                        alpha=self.ema_alpha,
                    )
                except Exception:
                    pass

            # Decide whether to stop or escalate
            if not self.retry_on_format_error:
                break

            if success and fmt_ok:
                break

            # Escalate if we have more variants to try
            if attempt_idx < (len(path[:max_attempts]) - 1):
                escalated = True
                continue
            break

        # Attach router metadata for analysis
        if isinstance(final_metrics, dict):
            final_metrics = dict(final_metrics)  # shallow copy
            final_metrics["router_mode"] = self.router_mode
            final_metrics["router_reason"] = reason
            final_metrics["router_initial_variant"] = chosen
            final_metrics["router_final_variant"] = final_metrics.get("variant", path[min(len(path)-1, max_attempts-1)])
            final_metrics["router_escalated"] = bool(escalated)
            final_metrics["router_attempts"] = attempts
            final_metrics["router_pred_ttft_ms"] = float(pred.get("pred_ttft_ms", 0.0) or 0.0)
            final_metrics["router_pred_tpot_ms"] = float(pred.get("pred_tpot_ms", 0.0) or 0.0)

        return final_text, final_metrics


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
