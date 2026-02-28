import gc
import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from collections import deque

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Logits processors (optional, for NaN-safe sampling)
try:
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
except Exception:  # pragma: no cover
    LogitsProcessor = object  # type: ignore
    LogitsProcessorList = list  # type: ignore

from prompt_templates import get_max_tokens
from answer_utils import (
    enforce_strict_gsm8k_final_answer,
    extract_gsm8k_parseable,
    extract_gsm8k_strict,
    extract_mmlu_answer,
    numbers_equal,
)

# SLO-safe contextual bandit router (paper contribution)
from bandit_router import BanditAction, BanditRouter, BanditRouterConfig

# Optional PEFT / LoRA adapter support (paper portfolio extension)
from adapter_utils import (
    AdapterManager,
    AdapterRegistry,
    choose_active_rank,
    choose_adapter_id,
    load_adapter_config,
    peft_available,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")


# -----------------------------------------------------------------------------
# Cost accounting (paper-facing)
# -----------------------------------------------------------------------------
# These are *relative* cost multipliers used for reporting and for the learned
# router's objective (cheaper variants should have lower multipliers).
#
# NOTE: This is intentionally simple and stable. If you later want a more
# realistic cost model, you can change this to: cost_per_token = (ms/token) *
# (GPU-$ / hour) or similar. For now we keep it unitless but consistent.
VARIANT_COST_MULTIPLIERS: Dict[str, float] = {
    "base": 1.0,
    "med": 0.6,
    "cheap": 0.3,
}


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


class NaNSafeLogitsProcessor(LogitsProcessor):
    """Make sampling robust to NaNs/Infs in logits.

    Some combinations of:
      - 4-bit quantization (bnb),
      - CUDA kernels,
      - and stochastic decoding
    can occasionally produce NaN/Inf logits. When `do_sample=True`, transformers
    calls `torch.multinomial`, which can throw a CUDA "device-side assert" if the
    probability tensor contains NaNs/Infs.

    This processor sanitizes logits *before* softmax/sampling:
      - NaN -> large negative
      - +Inf -> large positive (then clamped)
      - -Inf is preserved (important for constrained decoding masks)

    The goal is stability, not accuracy.
    """

    def __init__(self, clamp_min: float = -1e4, clamp_max: float = 1e4):
        super().__init__()
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        try:
            if not torch.is_floating_point(scores):
                return scores

            # Preserve -inf entries (used to ban tokens under constrained decoding).
            neginf_mask = torch.isneginf(scores)

            # Replace NaN / +inf.
            scores = torch.where(
                torch.isnan(scores),
                torch.full_like(scores, self.clamp_min),
                scores,
            )
            scores = torch.where(
                torch.isposinf(scores),
                torch.full_like(scores, self.clamp_max),
                scores,
            )

            # Clamp only finite values to avoid turning -inf into finite.
            finite = torch.isfinite(scores)
            if finite.any():
                clamped = scores.clamp(min=self.clamp_min, max=self.clamp_max)
                scores = torch.where(finite, clamped, scores)

            # Restore -inf.
            if neginf_mask.any():
                scores = scores.masked_fill(neginf_mask, float("-inf"))

            return scores
        except Exception:
            # Never crash generation due to a defensive processor.
            return scores


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


class StopOnDegenerateGibberish(StoppingCriteria):
    """Stop generation early if output collapses into obvious gibberish.

    Motivation:
    - Some 4-bit quantized stacks (bnb nf4) can enter repetitive token loops on
      long-form generations (notably GSM8K) under greedy decoding.
    - When this happens, continuing to generate to max_new_tokens wastes time and
      guarantees a format failure.

    This criterion is intentionally conservative and is only enabled for
    CHEAP+GSM8K.
    """

    def __init__(self, tokenizer, prompt_len: int, tail_tokens: int = 128):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.tail_tokens = int(max(32, tail_tokens))

    def _looks_degenerate(self, text: str) -> bool:
        if not text:
            return False
        tail = text[-256:]
        if len(tail) < 64:
            return False

        # High ratio of these chars is a strong signal of token-decode collapse.
        bad_chars = set("<>_?")
        bad_ratio = sum((c in bad_chars) for c in tail) / float(len(tail))
        if bad_ratio < 0.85:
            return False

        # If almost all characters come from a tiny alphabet, it's a loop.
        if len(set(tail)) <= 6:
            return True
        return False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if input_ids.ndim == 1:
            seq = input_ids
            start = max(self.prompt_len, int(seq.shape[0]) - self.tail_tokens)
            text = self.tokenizer.decode(seq[start:], skip_special_tokens=True)
            return self._looks_degenerate(text)

        # Batch: stop if any row looks degenerate.
        bsz = int(input_ids.shape[0])
        for i in range(bsz):
            seq = input_ids[i]
            start = max(self.prompt_len, int(seq.shape[0]) - self.tail_tokens)
            text = self.tokenizer.decode(seq[start:], skip_special_tokens=True)
            if self._looks_degenerate(text):
                return True
        return False


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
    concurrency: int = 1
    # Adapter context (optional)
    adapter_id: str = ""
    adapter_rank: Optional[int] = None
    # NOTE: In dataclasses, non-default fields must come before default fields.
    # This event is created by the scheduler for each request.
    event: threading.Event = field(default_factory=threading.Event)
    queue_depth_at_submit: int = 0
    result_text: Optional[str] = None
    result_metrics: Optional[Dict] = None
    error: Optional[str] = None

    def batch_key(self) -> Tuple[str, str, int, bool, str, int]:
        # In SLO mode we always use greedy decoding, so sampling params are irrelevant.
        do_sample = bool(self.prompt_mode != "slo" and self.temperature and self.temperature > 0.0)
        # Include adapter info so batches do not mix adapters / rank tiers.
        return (
            self.dataset_type,
            self.prompt_mode,
            int(self.max_tokens),
            do_sample,
            str(getattr(self, "adapter_id", "") or ""),
            int(getattr(self, "adapter_rank", 0) or 0),
        )


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
            adapter_id = getattr(batch[0], "adapter_id", "") or ""
            adapter_rank = getattr(batch[0], "adapter_rank", None)

            texts, metrics_list, lock_wait_ms = self.server._generate_hf_batch(
                prompts=prompts,
                dataset_type=dataset_type,
                max_tokens=max_tokens,
                prompt_mode=prompt_mode,
                temperature=temperature,
                top_p=top_p,
                adapter_id=str(adapter_id),
                adapter_rank=int(adapter_rank) if adapter_rank is not None else None,
                require_all_final_answers=(dataset_type == "gsm8k"),
            )

            for r, text, m in zip(batch, texts, metrics_list):
                scheduler_wait_ms = (dequeue_t - r.enqueue_time) * 1000.0
                scheduler_wait_ms = float(max(0.0, scheduler_wait_ms))

                m["scheduler_wait_ms"] = scheduler_wait_ms
                m["lock_wait_ms"] = float(max(0.0, lock_wait_ms))

                # queue_wait_ms is used by load_generator to compute total queueing.
                m["queue_wait_ms"] = float(max(0.0, scheduler_wait_ms + float(lock_wait_ms)))
                m["queue_depth_at_submit"] = int(getattr(r, "queue_depth_at_submit", 0))
                m["concurrency"] = int(getattr(r, "concurrency", 1))

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
        # Optional override for variant quantization.
        # Examples:
        #   - "fp16" / "none": load full-precision (no bnb quant)
        #   - "int8": bitsandbytes 8-bit
        #   - "int4": bitsandbytes 4-bit (nf4)
        # If None, the server uses the default mapping from `variant`.
        quantization_override: Optional[str] = None,
        enable_batching: bool = False,
        max_batch_size: int = 4,
        batch_wait_ms: int = 8,
        # PEFT / adapters (optional)
        enable_adapters: bool = False,
        adapter_root: Optional[str] = None,
        adapter_policy: str = "none",
        adapter_fixed: Optional[str] = None,
        adapter_rank_policy: str = "max",
        adapter_rank_tiers: str = "8,16,32",
        adapter_fixed_rank: Optional[int] = None,
        max_loaded_adapters: int = 8,
        adapter_eviction_policy: str = "lru",
        adapter_synthetic_load_ms: float = 0.0,
        adapter_synthetic_switch_ms: float = 0.0,
        adapter_allow_missing: bool = False,
        # Convert overhead milliseconds into token-equivalent cost units.
        overhead_ms_to_cost_units: float = 0.1,
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

        # Adapter configuration (selection policy is applied in generate())
        self.enable_adapters = bool(enable_adapters)
        self.adapter_root = adapter_root
        self.adapter_policy = (adapter_policy or "none").lower().strip()
        self.adapter_fixed = adapter_fixed
        self.adapter_rank_policy = (adapter_rank_policy or "max").lower().strip()
        self.adapter_rank_tiers = [
            int(x) for x in str(adapter_rank_tiers or "").replace(" ", "").split(",") if str(x).strip().isdigit()
        ]
        if not self.adapter_rank_tiers:
            self.adapter_rank_tiers = [8, 16, 32]
        self.adapter_fixed_rank = int(adapter_fixed_rank) if adapter_fixed_rank is not None else None
        self.max_loaded_adapters = int(max(1, max_loaded_adapters))
        self.adapter_eviction_policy = (adapter_eviction_policy or "lru").lower().strip()
        self.adapter_synthetic_load_ms = float(max(0.0, adapter_synthetic_load_ms))
        self.adapter_synthetic_switch_ms = float(max(0.0, adapter_synthetic_switch_ms))
        self.adapter_allow_missing = bool(adapter_allow_missing)

        # Cost model (shared across variants): how many "cost units" per 1ms of overhead.
        self.overhead_ms_to_cost_units = float(max(0.0, overhead_ms_to_cost_units))

        # Tokenizer (cheap)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Precompute MMLU allowed ids
        self.mmlu_allowed_token_ids = self._compute_mmlu_allowed_token_ids()
        logger.info(f"MMLU allowed token ids: {len(self.mmlu_allowed_token_ids)}")

        # Precompute numeric-only allowed ids (used as a GSM8K rescue path for CHEAP)
        self.numeric_token_ids = self._compute_numeric_token_ids()
        logger.info(f"Numeric-only allowed token ids: {len(self.numeric_token_ids)}")
        logger.info(
            f"Tokenizer loaded: {type(self.tokenizer).__name__} (chat_template={bool(getattr(self.tokenizer, 'chat_template', None))})"
        )

        # Model dtype
        model_dtype = "auto"
        if self.dtype == "float16":
            model_dtype = torch.float16
        elif self.dtype == "bfloat16":
            model_dtype = torch.bfloat16

        # Quantization by variant (with optional overrides + safety fallbacks)
        quant_config = None
        load_kwargs: Dict[str, Any] = {
            "device_map": "auto" if self.device == "cuda" else None,
        }

        # Normalize override string early.
        q_override = (quantization_override or "").lower().strip() if quantization_override is not None else ""

        # Detect compute capability for guardrails (e.g., P100 + bnb int8 can be unstable).
        cc_major: Optional[int] = None
        if self.device == "cuda":
            try:
                cc_major, _cc_minor = torch.cuda.get_device_capability(0)
            except Exception:
                cc_major = None

        # --------------------------------------
        # Quantization override (highest priority)
        # --------------------------------------
        override_applied = False
        if q_override:
            if q_override in ("none", "fp16", "float16", "fp32", "float32", "bf16", "bfloat16"):
                # "16-bit quantization" == just load fp16/bf16 weights.
                self.quantization = f"{self.dtype}"
                quant_config = None
                override_applied = True
            elif q_override in ("int8", "8bit", "8-bit", "bnb_int8"):
                if self.device != "cuda":
                    logger.warning("int8 requested on non-CUDA device; falling back to fp16.")
                    self.variant_effective = "base"
                    self.quantization = f"{self.dtype}"
                    quant_config = None
                elif cc_major is not None and cc_major < 7:
                    logger.warning(
                        f"bnb int8 is disabled on this GPU (compute_capability={cc_major}.x); falling back to fp16."
                    )
                    self.variant_effective = "base"
                    self.quantization = f"{self.dtype}"
                    quant_config = None
                else:
                    self.quantization = "bnb_int8"
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                override_applied = True
            elif q_override in ("int4", "4bit", "4-bit", "nf4", "bnb_nf4_int4", "bnb_int4"):
                if self.device != "cuda":
                    logger.warning("int4 requested on non-CUDA device; falling back to fp16.")
                    self.variant_effective = "base"
                    self.quantization = f"{self.dtype}"
                    quant_config = None
                elif cc_major is not None and cc_major < 7:
                    logger.warning(
                        f"int4 requested on compute capability {cc_major}.x (<7.0); falling back to fp16."
                    )
                    self.variant_effective = "base"
                    self.quantization = f"{self.dtype}"
                    quant_config = None
                else:
                    self.quantization = "bnb_nf4_int4"
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=False,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                override_applied = True
            else:
                logger.warning(f"Unknown quantization_override='{quantization_override}'. Using default mapping.")
                q_override = ""

        # --------------------------------------
        # Default mapping from `variant`
        # --------------------------------------
        if not q_override or not override_applied:
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
                    logger.warning(
                        f"CHEAP (int4) requested on compute capability {cc_major}.x (<7.0); falling back to fp16."
                    )
                    self.variant_effective = "base"
                    self.quantization = f"{self.dtype}"
                    quant_config = None
                else:
                    self.quantization = "bnb_nf4_int4"
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        # NOTE:
                        # - Double-quant can save memory, but it is also one of the main knobs that can
                        #   increase output degeneracy for some instruct models (esp. long-form GSM8K).
                        # - For "cheap" we prefer *stability/quality* over the last bit of VRAM.
                        bnb_4bit_use_double_quant=False,
                        # Use fp16 compute for stability across common Kaggle/Colab GPUs.
                        # (bf16 compute can be unstable for some bnb 4-bit stacks; fp16 is the safest.)
                        bnb_4bit_compute_dtype=torch.float16,
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

        # Optional PEFT adapter manager (wrap-on-first-use)
        self._adapter_registry: Optional[AdapterRegistry] = None
        self._adapter_manager: Optional[AdapterManager] = None
        if self.enable_adapters:
            if (not peft_available()) and (not self.adapter_allow_missing):
                raise RuntimeError(
                    "--enable_adapters was set but `peft` is not installed. "
                    "Install with: pip install peft"
                )
            self._adapter_registry = AdapterRegistry(self.adapter_root)
            self._adapter_manager = AdapterManager(
                base_model=self.model,
                adapter_registry=self._adapter_registry,
                max_loaded_adapters=self.max_loaded_adapters,
                eviction_policy=self.adapter_eviction_policy,
                synthetic_load_ms=self.adapter_synthetic_load_ms,
                synthetic_switch_ms=self.adapter_synthetic_switch_ms,
                allow_missing_adapters=self.adapter_allow_missing,
            )
            logger.info(
                f"Adapters enabled: policy={self.adapter_policy} adapter_root={self.adapter_root} "
                f"max_loaded_adapters={self.max_loaded_adapters} nested_rank_policy={self.adapter_rank_policy} tiers={self.adapter_rank_tiers}"
            )

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

    def _compute_numeric_token_ids(self) -> List[int]:
        '''Token ids whose string forms are numeric/whitespace only.

        Used for a last-resort constrained decoding path in CHEAP GSM8K when 4-bit
        generation collapses into gibberish.

        We build this once at init so per-token constraint checks are cheap.
        '''

        allowed: set[int] = set()

        # Always allow EOS if available
        try:
            if self.tokenizer.eos_token_id is not None:
                allowed.add(int(self.tokenizer.eos_token_id))
        except Exception:
            pass

        # Always allow common whitespace tokens
        for s in [" ", "\n", "\t", "\r"]:
            try:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                for _id in ids:
                    allowed.add(int(_id))
            except Exception:
                continue

        # SentencePiece boundary marker often used by Llama tokenizers
        allowed_chars = set("0123456789-+.,▁")
        allowed_chars.update({"\n", "\t", "\r", " "})

        # Iterate over vocabulary using convert_ids_to_tokens (fast)
        try:
            vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or len(self.tokenizer))
        except Exception:
            vocab_size = 0

        if vocab_size <= 0:
            return sorted(allowed)

        specials = set(getattr(self.tokenizer, "all_special_tokens", []) or [])

        for i in range(vocab_size):
            try:
                tok = self.tokenizer.convert_ids_to_tokens(i)
            except Exception:
                continue
            if not tok:
                continue
            if tok in specials:
                continue

            ok = True
            for ch in tok:
                if ch in allowed_chars:
                    continue
                ok = False
                break

            if ok:
                allowed.add(int(i))

        return sorted(allowed)

        specials = set(getattr(self.tokenizer, "all_special_tokens", []) or [])

        for i in range(vocab_size):
            try:
                tok = self.tokenizer.convert_ids_to_tokens(i)
            except Exception:
                continue
            if not tok:
                continue
            if tok in specials:
                continue

            ok = True
            for ch in tok:
                if ch in allowed_chars:
                    continue
                ok = False
                break

            if ok:
                allowed.add(int(i))

        return sorted(allowed)

    def _compute_ascii_token_ids(self) -> List[int]:
        """Token ids that decode to ASCII-only text (plus \n/\t).

        This is a lightweight stabilization mechanism for CHEAP (4-bit) long-form decoding:
        by disallowing non-ASCII tokens, we prevent the common 'multilingual gibberish'
        collapse mode while still allowing normal English + math symbols.

        Computed lazily and cached in self._ascii_token_ids.
        """
        allowed: List[int] = []
        vocab_size = int(len(self.tokenizer))

        for tid in range(vocab_size):
            try:
                s = self.tokenizer.decode([tid], skip_special_tokens=True)
            except Exception:
                continue
            if not s:
                continue

            ok = True
            for ch in s:
                o = ord(ch)
                if ch in ("\n", "\t"):
                    continue
                # printable ASCII range
                if o < 32 or o >= 127:
                    ok = False
                    break
            if ok:
                allowed.append(tid)

        return allowed

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

    # -------------------------------
    # Cheap/med quality guardrails (accuracy mode only)
    # -------------------------------

    @staticmethod
    def _stable_u32_seed(*parts: str) -> int:
        """Deterministic 32-bit seed derived from text (stable across runs)."""

        s = "|".join([p or "" for p in parts])
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    @staticmethod
    def _extract_gsm8k_question_from_prompt(formatted_prompt: str) -> str:
        """Best-effort extraction of the GSM8K question from our GSM8K prompt templates.

        We *must* handle multiple exemplars, each containing "Problem:", and we want the
        final "Now solve:" block's question. This function is used for format-only retries
        and cheap-variant compact prompting, so it should be robust.
        """

        if not formatted_prompt:
            return ""

        # First split our legacy format: "system\n\nuser".
        try:
            _sys, user = split_system_user(formatted_prompt)
        except Exception:
            user = formatted_prompt

        user = user or ""

        # Prefer the final "Now solve:" section if present.
        tail = user
        idx_now = tail.rfind("Now solve:")
        if idx_now >= 0:
            tail = tail[idx_now:]

        # Find the last occurrence of "Problem:" after the last Now solve.
        matches = list(re.finditer(r"\bProblem:\s*", tail))
        if matches:
            start = matches[-1].end()
            q_tail = tail[start:]

            # Cut off at the first blank-line + Solution, or at a standalone "Solution:".
            # (Our templates typically have "\n\nSolution:" right after the question.)
            cut_patterns = [r"\n\s*\n\s*Solution:\s*", r"\n\s*Solution:\s*"]
            cut = None
            for pat in cut_patterns:
                m = re.search(pat, q_tail)
                if m:
                    cut = m.start()
                    break
            if cut is not None:
                q = q_tail[:cut]
            else:
                q = q_tail

            q = (q or "").strip()
            # GSM8K questions are usually single-paragraph; collapse excessive whitespace.
            q = re.sub(r"\s+", " ", q).strip()
            return q

        # Fallback: if there is no "Problem:", return the last non-empty line.
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        if not lines:
            return ""
        q = lines[-1]
        q = re.sub(r"\s+", " ", q).strip()
        return q

    @classmethod
    def _build_gsm8k_answer_only_prompt(cls, formatted_prompt: str) -> str:
        """A short, format-focused GSM8K prompt used as an *accuracy-mode* retry.

        This is intentionally brief to reduce long-form degeneration (common in 4-bit)
        and to improve strict-format compliance.
        """

        q = cls._extract_gsm8k_question_from_prompt(formatted_prompt)
        system = "You are a careful math problem solver."
        user = (
            "Return ONLY the final numeric answer in the exact format:\n"
            "FINAL_ANSWER: <number>\n"
            "No other text. No units. No punctuation.\n\n"
            f"Problem: {q}\n"
        )
        return f"{system}\n\n{user}".strip()

    
    @classmethod
    def _build_gsm8k_compact_prompt(cls, formatted_prompt: str) -> str:
        """Compact GSM8K prompt for CHEAP (4-bit) stability.

        Removes few-shot exemplars and uses a single short worked example to
        strongly anchor the required output format without bloating the prompt.
        """
        q = cls._extract_gsm8k_question_from_prompt(formatted_prompt)
        system = "You are a careful math problem solver."
        exemplar = (
            "Example:\n"
            "Problem: Lisa has 6 candies and gives 2 away. How many candies does she have left?\n"
            "Solution:\n"
            "6 - 2 = 4\n"
            "FINAL_ANSWER: 4\n"
        )
        user = (
            "You will solve a grade-school math word problem.\n"
            "Show your working as short equations (not long prose).\n"
            "Your LAST line must be exactly:\n"
            "FINAL_ANSWER: <number>\n"
            "Where <number> is the final numeric answer (no units, no extra words).\n"
            "Do NOT write anything after the FINAL_ANSWER line.\n\n"
            f"{exemplar}\n"
            f"Now solve:\nProblem: {q}\n\nSolution:"
        )
        return f"{system}\n\n{user}".strip()

    @classmethod
    def _build_gsm8k_fill_blank_prompt(cls, formatted_prompt: str) -> str:
        """Ultra-short 'fill the blank' GSM8K retry prompt (format-only)."""
        q = cls._extract_gsm8k_question_from_prompt(formatted_prompt)
        system = "You are a careful math problem solver."
        user = (
            "Compute the final numeric answer.\n"
            "Respond by completing the last line with ONLY the number.\n"
            "Do not add any other text.\n\n"
            f"Problem: {q}\n"
            "FINAL_ANSWER: "
        )
        return f"{system}\n\n{user}".strip()

    @classmethod
    def _build_gsm8k_digits_only_prompt(cls, formatted_prompt: str) -> str:
        '''Last-resort GSM8K prompt for CHEAP: ask for digits only.

        We combine this with numeric-only constrained decoding so that even if the
        4-bit model cannot produce coherent text, it can still emit a numeric
        answer candidate that is parseable.
        '''
        q = cls._extract_gsm8k_question_from_prompt(formatted_prompt)
        system = "You are a careful math problem solver."
        user = (
            "Solve the problem. Output ONLY the final numeric answer.\n"
            "No words, no units, no punctuation.\n"
            "If the answer is an integer, output digits only.\n\n"
            f"Problem: {q}\n"
            "Answer:"
        )
        return f"{system}\n\n{user}".strip()

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
        constrain_numeric: bool = False,
        constrain_ascii: bool = False,
        seed_salt: int = 0,
        # Adapters
        adapter_id: str = "",
        adapter_rank: Optional[int] = None,
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

        # Per-sample prompt token counts (exclude padding). This is useful for
        # (1) paper cost accounting and (2) router feature logging.
        prompt_tokens_list: List[int] = []
        try:
            attn = inputs.get("attention_mask")
            if attn is None:
                prompt_tokens_list = [int(prompt_len)] * int(bsz)
            elif attn.ndim == 2:
                prompt_tokens_list = [int(attn[i].sum().item()) for i in range(int(bsz))]
            else:
                prompt_tokens_list = [int(prompt_len)] * int(bsz)
        except Exception:
            prompt_tokens_list = [int(prompt_len)] * int(bsz)

        # Decoding strategy
        do_sample = bool(prompt_mode != "slo" and temperature and float(temperature) > 0.0)

        # "CHEAP" historically meant 4-bit nf4 quantization. If callers override CHEAP to
        # be a small fp16 model (e.g., Llama-3B), we should *not* apply 4-bit-specific
        # guardrails (they can hurt quality / truncate generations).
        variant_eff = getattr(self, "variant_effective", self.variant)
        quant_str = str(getattr(self, "quantization", "") or "").lower()
        is_4bit_cheap = bool(
            variant_eff == "cheap" and any(t in quant_str for t in ("4bit", "int4", "nf4", "fp4"))
        )

        # Stopping criteria
        stopping_criteria = None
        if dataset_type == "gsm8k":
            stopper = StopOnFinalAnswer(
                tokenizer=self.tokenizer,
                prompt_len=prompt_len,
                require_all=bool(require_all_final_answers and bsz > 1),
            )
            criteria = [stopper]
            if is_4bit_cheap:
                criteria.append(StopOnDegenerateGibberish(tokenizer=self.tokenizer, prompt_len=prompt_len))
            stopping_criteria = StoppingCriteriaList(criteria)

        # MMLU restriction
        prefix_allowed_tokens_fn = None
        if dataset_type == "mmlu":
            max_tokens = 1
            allowed_ids = self.mmlu_allowed_token_ids

            def prefix_allowed_tokens_fn(_batch_id: int, _input_ids):
                return allowed_ids

        # Numeric-only constraint (used for CHEAP GSM8K rescue)
        if constrain_numeric and dataset_type == "gsm8k":
            allowed_ids = getattr(self, "numeric_token_ids", None)
            if allowed_ids:

                def _numeric_prefix_allowed_tokens_fn(_batch_id: int, _input_ids):
                    return allowed_ids

                prefix_allowed_tokens_fn = _numeric_prefix_allowed_tokens_fn

        # ASCII-only constraint (stabilization for CHEAP GSM8K long-form decoding)
        # This prevents the 4-bit model from drifting into non-ASCII gibberish while
        # still allowing normal English tokens (ASCII) and math symbols.
        if constrain_ascii and prefix_allowed_tokens_fn is None:
            allowed_ids = getattr(self, "_ascii_token_ids", None)
            if allowed_ids is None:
                allowed_ids = self._compute_ascii_token_ids()
                self._ascii_token_ids = allowed_ids
            if allowed_ids:

                def _ascii_prefix_allowed_tokens_fn(_batch_id: int, _input_ids):
                    return allowed_ids

                prefix_allowed_tokens_fn = _ascii_prefix_allowed_tokens_fn

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

        # ------------------------------------------------------------------
        # Variant-specific decoding guardrails
        # ------------------------------------------------------------------
        # In practice, some 4-bit (bnb nf4) stacks can fall into repetitive / token-gibberish
        # loops on longer GSM8K generations under greedy decoding. These light-weight
        # constraints are deterministic and typically improve both format adherence and
        # correctness for CHEAP without materially harming latency.
        if dataset_type == "gsm8k" and is_4bit_cheap:
            gen_kwargs.setdefault("repetition_penalty", 1.08)
            gen_kwargs.setdefault("no_repeat_ngram_size", 3)

        # NaN-safe logits sanitization for stochastic decoding.
        # This avoids rare-but-fatal "device-side assert" crashes inside torch.multinomial.
        if do_sample:
            try:
                lp = LogitsProcessorList()
                lp.append(NaNSafeLogitsProcessor())
                gen_kwargs["logits_processor"] = lp
            except Exception:
                # Best-effort; if something about the transformers version disagrees,
                # simply skip the safety processor.
                pass

        streamer = TimingStreamer(self._synchronize_device)

        # Adapter activation occurs under the generation lock to avoid races
        # between adapter switching and generation.
        adapter_metrics: Dict[str, Any] = {
            "adapter_id": str(adapter_id or ""),
            "adapter_active_rank": int(adapter_rank) if adapter_rank is not None else None,
            "adapter_cache_hit": 1,
            "adapter_load_ms": 0.0,
            "adapter_switch_ms": 0.0,
            "adapter_evicted": [],
            "adapter_num_loaded": 0,
        }

        t_lock_req = time.perf_counter()
        with self._generation_lock:
            t_lock_acq = time.perf_counter()
            lock_wait_ms = (t_lock_acq - t_lock_req) * 1000.0

            # Adapter switch / load (if enabled)
            if self._adapter_manager is not None:
                try:
                    adapter_metrics = self._adapter_manager.activate(
                        str(adapter_id or ""),
                        active_rank=int(adapter_rank) if adapter_rank is not None else None,
                    )
                    # Keep self.model pointing at the current (possibly PEFT-wrapped) model.
                    self.model = self._adapter_manager.model
                except Exception as e:
                    # Adapter failures should not crash the server; fall back to base.
                    adapter_metrics["adapter_error"] = str(e)
                    try:
                        self.model = self._adapter_manager.model
                    except Exception:
                        pass

            self._synchronize_device()
            t0_gen = time.perf_counter()

            with torch.inference_mode():
                # Deterministic sampling (used only when do_sample=True).
                # We avoid relying on Python's randomized hash; md5 is stable.
                if do_sample and bsz == 1:
                    seed = self._stable_u32_seed(
                        "sample",
                        getattr(self, "variant_effective", self.variant),
                        dataset_type,
                        prompts[0],
                        str(max_tokens),
                        str(seed_salt),
                    )
                    try:
                        torch.manual_seed(seed)
                        if self.device == "cuda":
                            torch.cuda.manual_seed_all(seed)
                    except Exception:
                        pass
                # If adapters are enabled but this request wants the base model,
                # run generate() inside the disable_adapter() context.
                if self._adapter_manager is not None and self._adapter_manager.is_peft_wrapped and not str(adapter_id or "").strip():
                    ctx = getattr(self.model, "disable_adapter", None)
                    if ctx is not None:
                        with ctx():
                            sequences = self.model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs.get("attention_mask"),
                                streamer=streamer,
                                **gen_kwargs,
                            )
                    else:
                        sequences = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            streamer=streamer,
                            **gen_kwargs,
                        )
                else:
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

        # Cost accounting (unitless multipliers). We attach both the multiplier
        # and a token-weighted "cost_units" so downstream reports can compute
        # total cost / cost-per-good-output.
        variant_eff = getattr(self, "variant_effective", self.variant)
        cost_mult = float(VARIANT_COST_MULTIPLIERS.get(str(variant_eff), 1.0))

        metrics_list: List[Dict] = []
        batch_size = int(max(1, len(out_lens)))
        # Adapter setup happens once per batch; allocate to each request.
        adapter_setup_ms_batch = float(adapter_metrics.get("adapter_setup_ms", 0.0) or 0.0)
        adapter_overhead_ms_alloc = adapter_setup_ms_batch / float(batch_size)
        adapter_overhead_units_alloc = float(self.overhead_ms_to_cost_units) * float(adapter_overhead_ms_alloc)
        for out_len, raw_text, postprocessed, postprocess_candidate, prompt_tokens in zip(
            out_lens,
            raw_texts,
            postprocessed_flags,
            postprocess_candidates,
            prompt_tokens_list,
        ):
            if out_len <= 1:
                tpot_ms = 0.0
            else:
                tpot_ms = (decode_s * 1000.0) / float(out_len - 1)

            throughput = (float(out_len) / total_gen_time) if total_gen_time > 0 else 0.0
            total_tokens = int(max(0, int(prompt_tokens))) + int(max(0, int(out_len)))
            token_cost_units = float(cost_mult) * float(total_tokens)
            total_cost_units = float(token_cost_units) + float(adapter_overhead_units_alloc)

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
                    "prompt_tokens": int(prompt_tokens),
                    "total_tokens": int(total_tokens),
                    "throughput_tokens_per_sec": float(throughput),
                    "total_latency_ms": float(total_latency_s * 1000.0),
                    "tokenize_ms": tokenize_ms,
                    "cost_multiplier": float(cost_mult),
                    # Cost breakdown (token-equivalent cost units)
                    "token_cost_units": float(token_cost_units),
                    "adapter_overhead_ms_alloc": float(adapter_overhead_ms_alloc),
                    "adapter_overhead_units": float(adapter_overhead_units_alloc),
                    # Multi-variant service may add swap_overhead_units.
                    "swap_overhead_units": 0.0,
                    "total_cost_units": float(total_cost_units),
                    # Backwards-compatible name: now total cost (tokens + overhead).
                    "cost_units": float(total_cost_units),
                    "batch_size": int(batch_size),
                    # queue_wait_ms / scheduler_wait_ms filled by caller
                    "variant": self.variant,
                    "variant_effective": getattr(self, "variant_effective", self.variant),
                    "quantization": getattr(self, "quantization", None),
                    # Adapter info (optional)
                    "adapter_id": adapter_metrics.get("adapter_id", ""),
                    "adapter_active_rank": adapter_metrics.get("adapter_active_rank", None),
                    "adapter_cache_hit": int(adapter_metrics.get("adapter_cache_hit", 1) or 0),
                    "adapter_load_ms": float(adapter_metrics.get("adapter_load_ms", 0.0) or 0.0),
                    "adapter_switch_ms": float(adapter_metrics.get("adapter_switch_ms", 0.0) or 0.0),
                    "adapter_setup_ms": float(adapter_metrics.get("adapter_setup_ms", 0.0) or 0.0),
                    "adapter_num_loaded": int(adapter_metrics.get("adapter_num_loaded", 0) or 0),
                    "adapter_evicted": adapter_metrics.get("adapter_evicted", []) or [],
                    "adapter_error": adapter_metrics.get("adapter_error"),
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
        concurrency: int = 1,
        use_batching: Optional[bool] = None,
        # Adapters (optional)
        adapter_id: Optional[str] = None,
        adapter_rank: Optional[int] = None,
        # Optional experiment / logging fields (ignored for generation)
        request_id: Optional[int] = None,
        gold_answer: Optional[str] = None,
        label: Optional[int] = None,
        label_source: Optional[str] = None,
        label_budget_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict]:
        """Generate completion text plus metrics."""

        dataset_type = (dataset_type or "").lower().strip()
        prompt_mode = (prompt_mode or "").lower().strip()
        difficulty = (difficulty or "").lower().strip()

        if max_tokens is None:
            max_tokens = get_max_tokens(difficulty, dataset_type, prompt_mode)
        if dataset_type == "mmlu":
            max_tokens = 1

        # Resolve adapter selection (if enabled)
        adapter_id_use = ""
        adapter_rank_use: Optional[int] = None
        if self.enable_adapters:
            adapter_id_use = choose_adapter_id(
                policy=self.adapter_policy,
                dataset_type=dataset_type,
                fixed_adapter=self.adapter_fixed,
                explicit_adapter=adapter_id,
            )
            try:
                qd = int(self._scheduler.get_queue_depth()) if self._scheduler is not None else 0
            except Exception:
                qd = 0
            if adapter_id_use:
                if adapter_rank is not None:
                    adapter_rank_use = int(adapter_rank)
                else:
                    adapter_rank_use = choose_active_rank(
                        policy=self.adapter_rank_policy,
                        difficulty=difficulty,
                        total_queue_depth=qd,
                        tiers=self.adapter_rank_tiers,
                        fixed_rank=self.adapter_fixed_rank,
                    )

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
                concurrency=int(concurrency),
                adapter_id=str(adapter_id_use or ""),
                adapter_rank=int(adapter_rank_use) if adapter_rank_use is not None else None,
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
        # NOTE: In accuracy mode we may run multiple internal attempts (self-consistency
        # or format retries). We measure end-to-end wall time here so `total_latency_ms`
        # matches what the client observes.
        req_t0 = time.perf_counter()
        total_cost_units: float = 0.0
        total_lock_wait_ms: float = 0.0
        num_attempts: int = 0

        def _accumulate_attempt(m: Dict[str, Any], lw_ms: float) -> None:
            nonlocal total_cost_units, total_lock_wait_ms, num_attempts
            num_attempts += 1
            try:
                total_cost_units += float(m.get("cost_units") or 0.0)
            except Exception:
                pass
            try:
                total_lock_wait_ms += float(max(0.0, lw_ms or 0.0))
            except Exception:
                pass

        # Determine whether this server's CHEAP variant is actually 4-bit. If CHEAP is
        # overridden to a small fp16/bf16 model, we disable 4-bit-specific hacks.
        variant_eff = getattr(self, "variant_effective", self.variant)
        quant_str = str(getattr(self, "quantization", "") or "").lower()
        is_4bit_cheap = bool(
            variant_eff == "cheap" and any(t in quant_str for t in ("4bit", "int4", "nf4", "fp4"))
        )

        # ------------------------------------------------------------------
        # CHEAP GSM8K primary rescue path (format-first, same variant)
        # ------------------------------------------------------------------
        # Empirically, some bnb 4-bit stacks on Kaggle can produce coherent 1-token
        # multiple-choice answers (MMLU) but collapse into gibberish on longer
        # free-form generations (GSM8K). To keep CHEAP non-trivial (format_ok > 0)
        # and to avoid wasting time generating 256 tokens of garbage, we run a
        # short, format-first ladder up-front for CHEAP+GSM8K+accuracy.
        #
        # Cheap (4-bit) GSM8K accuracy tends to collapse under long few-shot prompts + greedy decoding.
        # For CHEAP+GSM8K+accuracy, we use a compact single-example prompt and enable light,
        # deterministic sampling (seeded inside _generate_hf_batch) to prevent repetition loops.
        prompt_use = prompt
        variant_eff = getattr(self, "variant_effective", self.variant)
        if dataset_type == "gsm8k" and prompt_mode == "accuracy" and is_4bit_cheap:
            try:
                prompt_use = self._build_gsm8k_compact_prompt(prompt)
            except Exception:
                prompt_use = prompt
            if float(temperature or 0.0) <= 0.0:
                temperature = 0.20
            if float(top_p or 1.0) >= 1.0:
                top_p = 0.90

        # For CHEAP+GSM8K+accuracy: use a small, deterministic self-consistency pass.
        # 4-bit decoding can be brittle on long-form math; sampling a few short candidates and
        # taking a majority vote often improves correctness while keeping behavior reproducible.
        if dataset_type == "gsm8k" and prompt_mode == "accuracy" and is_4bit_cheap:
            samples = []
            for s in range(3):
                s_texts, s_metrics_list, s_lock_wait_ms = self._generate_hf_batch(
                    prompts=[prompt_use],
                    dataset_type=dataset_type,
                    max_tokens=int(max_tokens),
                    prompt_mode=prompt_mode,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    adapter_id=str(adapter_id_use or ""),
                    adapter_rank=int(adapter_rank_use) if adapter_rank_use is not None else None,
                    require_all_final_answers=False,
                    constrain_ascii=True,
                    seed_salt=int(s),
                )
                s_text = s_texts[0]
                try:
                    cand = extract_gsm8k_strict(s_text) or extract_gsm8k_parseable(s_text)
                except Exception:
                    cand = ""
                _accumulate_attempt(s_metrics_list[0], s_lock_wait_ms)
                samples.append((s_text, s_metrics_list[0], float(max(0.0, s_lock_wait_ms)), cand))

            # Choose candidate by majority vote over extracted numbers (fall back to the first sample).
            chosen_text, chosen_metrics, chosen_lock_wait_ms = samples[0][0], samples[0][1], samples[0][2]
            cand_list = [c for (_, _, _, c) in samples if c]
            if cand_list:
                from collections import Counter
                best = Counter(cand_list).most_common(1)[0][0]
                for (t, m, lw, c) in samples:
                    if c == best:
                        chosen_text, chosen_metrics, chosen_lock_wait_ms = t, m, lw
                        break

            texts, metrics_list, lock_wait_ms = [chosen_text], [chosen_metrics], chosen_lock_wait_ms
        else:
            texts, metrics_list, lock_wait_ms = self._generate_hf_batch(
                prompts=[prompt_use],
                dataset_type=dataset_type,
                max_tokens=int(max_tokens),
                prompt_mode=prompt_mode,
                temperature=float(temperature),
                top_p=float(top_p),
                adapter_id=str(adapter_id_use or ""),
                adapter_rank=int(adapter_rank_use) if adapter_rank_use is not None else None,
                require_all_final_answers=False,
            )

            # Single-attempt direct generation.
            _accumulate_attempt(metrics_list[0], lock_wait_ms)

        metrics = metrics_list[0]
        metrics["scheduler_wait_ms"] = 0.0
        metrics["lock_wait_ms"] = float(max(0.0, lock_wait_ms))
        metrics["queue_wait_ms"] = float(max(0.0, lock_wait_ms))

        text = texts[0]

        # ------------------------------------------------------------------
        # Accuracy-mode GSM8K retry (format-only, same variant)
        # ------------------------------------------------------------------
        # Goal: avoid "all-format-fail" runs in CHEAP (4-bit) and recover
        # occasional truncation-driven format failures in MED.
        # This is ONLY enabled for prompt_mode=accuracy and ONLY triggered on
        # strict+parseable format failure (no quality-based retry).
        variant_eff = getattr(self, "variant_effective", self.variant)
        if (
            dataset_type == "gsm8k"
            and prompt_mode == "accuracy"
            and variant_eff in ("cheap", "med")
        ):
            try:
                strict = extract_gsm8k_strict(text)
                parseable = strict or extract_gsm8k_parseable(text)
            except Exception:
                strict, parseable = "", ""

            if not parseable:
                # Format-only rescue ladder:
                #  - MED: single answer-only retry (greedy) to recover occasional format drift.
                #  - CHEAP: two short retries with deterministic sampling:
                #      (1) answer-only, (2) fill-the-blank ending with 'FINAL_ANSWER: '.
                attempts = []
                if is_4bit_cheap:
                    attempts = [
                        ("answer_only", self._build_gsm8k_answer_only_prompt, 0.20, 0.90, 24, True),
                        ("fill_blank", self._build_gsm8k_fill_blank_prompt, 0.20, 0.90, 12, True),
                    ]
                else:
                    attempts = [
                        ("answer_only", self._build_gsm8k_answer_only_prompt, 0.0, 1.0, 48, False),
                    ]

                first_attempt_output_len = int(metrics.get("output_length") or 0)
                first_attempt_tpot_ms = float(metrics.get("tpot_ms") or 0.0)
                first_attempt_total_ms = float(metrics.get("total_latency_ms") or 0.0)

                for (tag, builder, r_temp, r_top_p, r_max_tokens, r_constrain_ascii) in attempts:
                    try:
                        retry_prompt = builder(prompt)
                        r_texts, r_metrics_list, r_lock_wait_ms = self._generate_hf_batch(
                            prompts=[retry_prompt],
                            dataset_type=dataset_type,
                            max_tokens=int(r_max_tokens),
                            prompt_mode=prompt_mode,
                            temperature=float(r_temp),
                            top_p=float(r_top_p),
                            adapter_id=str(adapter_id_use or ""),
                            adapter_rank=int(adapter_rank_use) if adapter_rank_use is not None else None,
                            require_all_final_answers=False,
                            constrain_ascii=bool(r_constrain_ascii),
                        )

                        r_text = r_texts[0]
                        _accumulate_attempt(r_metrics_list[0], r_lock_wait_ms)
                        try:
                            r_ok = bool(extract_gsm8k_parseable(r_text) or extract_gsm8k_strict(r_text))
                        except Exception:
                            r_ok = False

                        if r_ok:
                            r_metrics = r_metrics_list[0]
                            r_metrics["scheduler_wait_ms"] = 0.0
                            r_metrics["lock_wait_ms"] = float(max(0.0, r_lock_wait_ms))
                            r_metrics["queue_wait_ms"] = float(max(0.0, r_lock_wait_ms))

                            r_metrics["format_retry"] = {
                                "enabled": True,
                                "reason": "gsm8k_format_failure",
                                "variant_effective": variant_eff,
                                "attempt": tag,
                                "first_attempt_output_len": first_attempt_output_len,
                                "first_attempt_tpot_ms": first_attempt_tpot_ms,
                                "first_attempt_total_ms": first_attempt_total_ms,
                            }
                            # Swap in the successful retry output, but do not return yet.
                            # We'll overwrite latency/cost below to reflect *all* attempts.
                            text = r_text
                            metrics = r_metrics
                            break
                    except Exception:
                        # If a retry fails for any reason, try the next attempt.
                        continue


        # ------------------------------------------------------------------
        # Fix latency/cost accounting for multi-attempt direct calls
        # ------------------------------------------------------------------
        # In accuracy mode we may run multiple internal attempts (self-consistency,
        # format retries). The client waits for *all* of them, so `total_latency_ms`
        # must reflect end-to-end wall time, not just the last attempt.
        req_total_ms = float(max(0.0, (time.perf_counter() - req_t0) * 1000.0))
        metrics["total_latency_ms"] = req_total_ms
        if num_attempts > 0:
            metrics["cost_units"] = float(total_cost_units)
            metrics["num_attempts"] = int(num_attempts)
            metrics["lock_wait_ms"] = float(total_lock_wait_ms)
            metrics["queue_wait_ms"] = float(total_lock_wait_ms)
        # Effective throughput for the returned output.
        try:
            out_len = int(metrics.get("output_length") or 0)
            if req_total_ms > 0.0:
                metrics["throughput_tokens_per_sec"] = float(out_len) / (req_total_ms / 1000.0)
        except Exception:
            pass

        # Attach client-side concurrency hint (used for router training features).
        metrics["concurrency"] = int(concurrency)

        # Optional logging join key (used for bandit delayed labels).
        if request_id is not None:
            try:
                metrics["request_id"] = int(request_id)
            except Exception:
                metrics["request_id"] = request_id

        # Option A: ttft_ms already equals ttft_infer_ms for direct calls.
        return text, metrics

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
    """Lightweight EWMA tracking for per-variant latency + contention signals.

    Used by the heuristic SLO-aware router and exposed for logging/debugging.
    """
    ema_ttft_ms: Optional[float] = None
    ema_tpot_ms: Optional[float] = None
    ema_total_ms: Optional[float] = None
    ema_output_tokens: Optional[float] = None
    ema_queue_wait_ms: Optional[float] = None
    ema_lock_wait_ms: Optional[float] = None
    n: int = 0

    def update(
        self,
        ttft_ms: float,
        tpot_ms: float,
        total_ms: float,
        output_tokens: int,
        queue_wait_ms: float,
        lock_wait_ms: float,
        *,
        alpha: float,
    ) -> None:
        def _ema(prev: Optional[float], x: float) -> float:
            if prev is None:
                return float(x)
            return float(alpha) * float(x) + (1.0 - float(alpha)) * float(prev)

        self.ema_ttft_ms = _ema(self.ema_ttft_ms, float(ttft_ms))
        self.ema_tpot_ms = _ema(self.ema_tpot_ms, float(tpot_ms))
        self.ema_total_ms = _ema(self.ema_total_ms, float(total_ms))
        self.ema_output_tokens = _ema(self.ema_output_tokens, float(output_tokens))
        self.ema_queue_wait_ms = _ema(self.ema_queue_wait_ms, float(queue_wait_ms))
        self.ema_lock_wait_ms = _ema(self.ema_lock_wait_ms, float(lock_wait_ms))
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

    # Adapter context (optional)
    adapter_id: str = ""
    adapter_rank: Optional[int] = None

    # Client-side offered-load hint (used for learned router features/logging).
    # Must come *after* non-default fields for dataclass correctness.
    concurrency: int = 1

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
        # Optional per-variant overrides (useful when CHEAP is a different model size).
        # Example:
        #   variant_models={"cheap": "meta-llama/Llama-3.2-3B-Instruct"}
        #   variant_quantization={"cheap": "fp16"}
        variant_models: Optional[Dict[str, str]] = None,
        variant_quantization: Optional[Dict[str, str]] = None,
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
        # Risk router (calibrated guarantees)
        risk_router_dir: Optional[str] = None,
        risk_latency_delta: float = 0.05,
        risk_quality_epsilon: float = 0.25,
        risk_quality_alpha: float = 0.05,
        # Dispatcher policy
        dispatcher_policy: str = "age",
        dispatcher_batch_wait_s: float = 0.002,
        dispatcher_max_sticky_batches: int = 4,
        dispatcher_starvation_ms: float = 50.0,
        # PEFT / adapters (optional portfolio extension)
        enable_adapters: bool = False,
        adapter_root: Optional[str] = None,
        adapter_policy: str = "none",
        adapter_fixed: Optional[str] = None,
        adapter_rank_policy: str = "max",
        adapter_rank_tiers: str = "8,16,32",
        adapter_fixed_rank: Optional[int] = None,
        max_loaded_adapters: int = 8,
        adapter_eviction_policy: str = "lru",
        adapter_synthetic_load_ms: float = 0.0,
        adapter_synthetic_switch_ms: float = 0.0,
        adapter_allow_missing: bool = False,
        dispatcher_max_sticky_adapter_batches: int = 4,
        # Convert overhead milliseconds into token-equivalent cost units.
        overhead_ms_to_cost_units: float = 0.1,
        # Deterministic seed for routing / label subsampling.
        router_seed: int = 0,
        # ------------------------------
        # Bandit router (SLO-safe contextual bandit)
        # ------------------------------
        bandit_delta: float = 0.05,
        bandit_alpha: float = 1.0,
        bandit_beta_r: float = 2.0,
        bandit_beta_q: float = 2.0,
        bandit_eps_r: float = 0.0,
        bandit_eps_q: float = 0.0,
        bandit_beta_u: float = 0.2,
        bandit_label_budget_p: float = 1.0,
        bandit_checkpoint_path: Optional[str] = None,
        bandit_checkpoint_every: int = 500,
        bandit_require_latency_safe: bool = True,
        bandit_use_conservative_fallback: bool = True,
        bandit_use_primal_dual: bool = True,
        bandit_use_overhead_cost: bool = True,
        bandit_use_system_features: bool = True,
        bandit_use_adapter_features: bool = True,
        bandit_variant_load_synthetic_ms: float = 1000.0,
        bandit_adapter_ids: Optional[str] = None,
        bandit_rank_tiers: Optional[str] = None,
        bandit_state_path: Optional[str] = None,
        bandit_update_enabled: bool = True,
    ):
        self.model_name = model_name
        # Match SingleVariantServer behavior: support device="auto" and dtype="auto".
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.dtype = _resolve_dtype(dtype, self.device)

        # Adapter portfolio configuration (optional)
        self.enable_adapters = bool(enable_adapters)
        self.adapter_root = adapter_root
        self.adapter_policy = (adapter_policy or "none").lower().strip()
        self.adapter_fixed = adapter_fixed
        self.adapter_rank_policy = (adapter_rank_policy or "max").lower().strip()
        self.adapter_rank_tiers = [
            int(x) for x in str(adapter_rank_tiers or "").replace(" ", "").split(",") if str(x).strip().isdigit()
        ]
        if not self.adapter_rank_tiers:
            self.adapter_rank_tiers = [8, 16, 32]
        self.adapter_fixed_rank = int(adapter_fixed_rank) if adapter_fixed_rank is not None else None
        self.max_loaded_adapters = int(max(1, max_loaded_adapters))
        self.adapter_eviction_policy = (adapter_eviction_policy or "lru").lower().strip()
        self.adapter_synthetic_load_ms = float(max(0.0, adapter_synthetic_load_ms))
        self.adapter_synthetic_switch_ms = float(max(0.0, adapter_synthetic_switch_ms))
        self.adapter_allow_missing = bool(adapter_allow_missing)
        self.dispatcher_max_sticky_adapter_batches = int(max(1, dispatcher_max_sticky_adapter_batches))

        # Cost model: convert overhead milliseconds into token-equivalent cost units.
        self.overhead_ms_to_cost_units = float(max(0.0, overhead_ms_to_cost_units))

        # Variant token-cost multipliers (used by bandit scoring + cost accounting).
        # Keep this in sync with VARIANT_COST_MULTIPLIERS.
        self.cost_multipliers = VARIANT_COST_MULTIPLIERS

        # Deterministic seed for routing-related randomness.
        self.router_seed = int(router_seed)

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

        # --------------------------------------------------------------
        # Resolve enabled variants *early*.
        #
        # Several downstream components (e.g., bandit cost/feature setup) need
        # `self.variants` during initialization. Keep this block before any
        # router-specific setup that references `self.variants`.
        # --------------------------------------------------------------
        requested_variants = variants or ["cheap", "med", "base"]
        requested_variants = [_normalize_variant(v) for v in requested_variants]

        supported = [v for v in requested_variants if self._is_variant_supported(v)]
        if "base" not in supported:
            supported.append("base")
        self.variants = [v for v in self.VARIANT_ORDER if v in supported]

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

        # Risk router (conformal latency bounds + calibrated quality gating)
        self.risk_router_dir = risk_router_dir
        self.risk_latency_delta = float(risk_latency_delta)
        self.risk_quality_epsilon = float(risk_quality_epsilon)
        self.risk_quality_alpha = float(risk_quality_alpha)
        self._risk_router = None
        if self.router_mode in {"risk"} or self.router_mode.startswith("bandit"):
            if not risk_router_dir:
                raise ValueError("--risk_router_dir is required when router_mode is risk or bandit.")
            from risk_router import RiskRouter

            rdir = os.path.abspath(os.path.expanduser(risk_router_dir))
            self._risk_router = RiskRouter.load_bundle(rdir)
            try:
                self._risk_router.quality_alpha = float(self.risk_quality_alpha)
            except Exception:
                pass

        # ------------------------------
        # Bandit router (SLO-safe contextual bandit)
        # ------------------------------
        self._bandit_router: Optional[BanditRouter] = None
        self._bandit_router_update_enabled = bool(bandit_update_enabled)
        self._bandit_variant_load_synth_ms = float(max(0.0, bandit_variant_load_synthetic_ms))
        # EWMA of observed variant swap/load times (ms)
        self._bandit_variant_load_ema_ms: Dict[str, float] = {
            v: float(self._bandit_variant_load_synth_ms) for v in self.variants
        }
        self._bandit_variant_load_ema_alpha = 0.1

        # Action space configuration (adapters + rank tiers)
        self._bandit_adapter_ids_cfg = bandit_adapter_ids
        self._bandit_rank_tiers_cfg = bandit_rank_tiers

        if self.router_mode.startswith("bandit"):
            # Feature dim = LearnedRouter base feature dim (43) + optional system features
            base_dim = 43
            extra_dim = 0
            if bool(bandit_use_system_features):
                # swap_mode, gpu_free_frac, loaded_indicator per variant, batch_wait_ms, max_loaded_variants, queue_depth_sum
                extra_dim = 1 + 1 + len(self.variants) + 1 + 1 + 1
            feat_dim = int(base_dim + extra_dim)

            cfg = BanditRouterConfig(
                delta=float(bandit_delta),
                alpha=float(bandit_alpha),
                beta_r=float(bandit_beta_r),
                beta_q=float(bandit_beta_q),
                eps_r=float(bandit_eps_r),
                eps_q=float(bandit_eps_q),
                beta_u=float(bandit_beta_u),
                overhead_ms_to_cost_units=float(self.overhead_ms_to_cost_units),
                require_action_latency_safe=bool(bandit_require_latency_safe),
                use_conservative_fallback=bool(bandit_use_conservative_fallback),
                use_primal_dual=bool(bandit_use_primal_dual),
                use_overhead_cost=bool(bandit_use_overhead_cost),
                use_system_features=bool(bandit_use_system_features),
                use_adapter_features=bool(bandit_use_adapter_features),
                seed=int(self.router_seed),
                label_budget_p=float(bandit_label_budget_p),
                checkpoint_path=str(bandit_checkpoint_path) if bandit_checkpoint_path else None,
                checkpoint_every=int(max(1, bandit_checkpoint_every)),
            )

            if bandit_state_path:
                loaded = BanditRouter.load(str(bandit_state_path))
                if int(getattr(loaded, "feature_dim", -1)) != int(feat_dim):
                    raise ValueError(
                        f"Bandit state feature_dim mismatch: state has {loaded.feature_dim}, expected {feat_dim}. "
                        "(Did you change system-feature flags?)"
                    )
                # Keep the learned parameters, but apply current-run config overrides.
                loaded.config = cfg
                self._bandit_router = loaded
            else:
                self._bandit_router = BanditRouter(feature_dim=feat_dim, config=cfg)

        # Validate adapter dependency early.
        if self.enable_adapters and not peft_available():
            raise RuntimeError(
                "Adapters are enabled (enable_adapters=True) but `peft` is not installed. "
                "Install with: pip install peft"
            )

        self.dispatcher_policy = (dispatcher_policy or "age").lower().strip()
        if self.dispatcher_policy not in {"age", "edf", "lstf", "setup_aware", "setup_edf", "setup_lstf"}:
            logger.warning(f"Unknown dispatcher_policy='{dispatcher_policy}'. Falling back to 'age'.")
            self.dispatcher_policy = "age"

        self.dispatcher_batch_wait_s = float(max(0.0, dispatcher_batch_wait_s))
        self.dispatcher_max_sticky_batches = int(max(1, dispatcher_max_sticky_batches))
        self.dispatcher_starvation_ms = float(max(0.0, dispatcher_starvation_ms))

        if self.fixed_variant and self.fixed_variant not in self.variants:
            logger.warning(
                f"Fixed variant '{self.fixed_variant}' is not supported on this device; falling back to 'base'."
            )
            self.fixed_variant = "base"

        self._stats = {v: _VariantStats() for v in self.variants}

        # ------------------------------------------------------------------
        # Per-variant model / quantization overrides
        # ------------------------------------------------------------------
        # This enables experiments where (for example) CHEAP is a smaller fp16 model
        # rather than a 4-bit quantized version of the same base model.
        self.model_name_by_variant: Dict[str, str] = {v: self.model_name for v in self.variants}
        for k, v in (variant_models or {}).items():
            try:
                nk = _normalize_variant(str(k))
            except Exception:
                continue
            if nk in self.model_name_by_variant and v:
                self.model_name_by_variant[nk] = str(v)

        self.quantization_override_by_variant: Dict[str, str] = {}
        for k, v in (variant_quantization or {}).items():
            try:
                nk = _normalize_variant(str(k))
            except Exception:
                continue
            if nk in self.variants and v:
                self.quantization_override_by_variant[nk] = str(v)
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
        # Adapter stickiness / setup-aware batching (per-variant)
        self._last_adapter_key_by_variant: Dict[str, Tuple[str, int]] = {}
        self._active_adapter_batches_run = 0
        # Lightweight tokenizer for routing feature extraction (CPU-side)
        try:
            from transformers import AutoTokenizer
            self._router_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            self._router_tokenizer = None


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
        logger.info(f"  dispatcher_policy={self.dispatcher_policy}")
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

    # Bandit controls (used by evaluation scripts to freeze learning)
    def set_bandit_update_enabled(self, enabled: bool) -> None:
        """Enable/disable online bandit updates.

        This does not change routing decisions, only whether the router updates
        its online models after each request.
        """
        self._bandit_router_update_enabled = bool(enabled)

    def save_bandit_state(self, base_path: str) -> bool:
        """Save the current bandit router state (if any)."""
        if self._bandit_router is None:
            return False
        self._bandit_router.save(str(base_path))
        return True

    def ingest_quality_label(self, join_key: str, quality_label: int) -> Dict[str, Any]:
        """Ingest a delayed quality label into the bandit router.

        join_key is typically the request_id (stringified) so it can be joined
        against offline judge outputs.
        """

        if self._bandit_router is None:
            return {"updated": False, "error": "bandit_router_not_initialized"}
        try:
            return self._bandit_router.ingest_quality_label(str(join_key), int(quality_label))
        except Exception as e:
            return {"updated": False, "error": str(e)}

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
        """Ensure a SingleVariantServer for `variant` exists."""

        srv, _meta = self._ensure_loaded_with_metrics(variant)
        return srv

    def _ensure_loaded_with_metrics(self, variant: str) -> Tuple[SingleVariantServer, Dict[str, Any]]:
        """Ensure a SingleVariantServer for `variant` exists and return swap metrics.

        Returns
        -------
        (srv, meta)
          meta contains:
            - swap_loaded: bool (this call performed a load)
            - swap_load_ms: float (0 if not loaded)
            - swap_evicted_variants: list[str]

        Notes
        -----
        If another thread is currently loading the same variant, this call will
        wait and then return swap_loaded=False (the wait time is captured in the
        request's scheduler_wait_ms rather than swap_load_ms).
        """
        variant = _normalize_variant(variant)
        if variant not in self.variants:
            raise ValueError(f"Variant '{variant}' not in enabled variants {self.variants}")

        while True:
            with self._lock:
                existing = self._servers.get(variant)
                if existing is not None:
                    self._touch_lru_locked(variant)
                    return existing, {
                        "swap_loaded": False,
                        "swap_load_ms": 0.0,
                        "swap_evicted_variants": [],
                    }

                ev = self._load_events.get(variant)
                if ev is not None:
                    wait_ev = ev
                else:
                    ev = threading.Event()
                    self._load_events[variant] = ev
                    wait_ev = None

                    victims: List[SingleVariantServer] = []
                    victim_names: List[str] = []
                    cap = int(self.max_loaded_variants or 1)
                    while len(self._servers) >= cap:
                        victim = self._evict_one_locked()
                        if victim is None:
                            self._load_events.pop(variant, None)
                            ev.set()
                            raise RuntimeError(
                                f"Cannot load '{variant}': all loaded variants are pinned (loaded={list(self._servers.keys())})."
                            )
                        victims.append(victim)
                        try:
                            victim_names.append(str(getattr(victim, "variant", "")))
                        except Exception:
                            victim_names.append("")

            if wait_ev is not None:
                wait_ev.wait()
                continue

            # Loader branch: measure swap overhead outside the lock.
            t0 = time.perf_counter()

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

            try:
                model_name = self.model_name_by_variant.get(variant, self.model_name)
                quant_override = self.quantization_override_by_variant.get(variant)
                srv = SingleVariantServer(
                    model_name=model_name,
                    variant=variant,
                    device=self.device,
                    dtype=self.dtype,
                    quantization_override=quant_override,
                    enable_batching=self.enable_batching,
                    max_batch_size=self.max_batch_size,
                    batch_wait_ms=self.batch_wait_ms,
                    enable_adapters=self.enable_adapters,
                    adapter_root=self.adapter_root,
                    adapter_policy=self.adapter_policy,
                    adapter_fixed=self.adapter_fixed,
                    adapter_rank_policy=self.adapter_rank_policy,
                    adapter_rank_tiers=",".join([str(x) for x in self.adapter_rank_tiers]),
                    adapter_fixed_rank=self.adapter_fixed_rank,
                    max_loaded_adapters=self.max_loaded_adapters,
                    adapter_eviction_policy=self.adapter_eviction_policy,
                    adapter_synthetic_load_ms=self.adapter_synthetic_load_ms,
                    adapter_synthetic_switch_ms=self.adapter_synthetic_switch_ms,
                    adapter_allow_missing=bool(getattr(self, "adapter_allow_missing", False)),
                    overhead_ms_to_cost_units=getattr(self, "overhead_ms_to_cost_units", 0.1),
                )
            except Exception:
                with self._lock:
                    ev = self._load_events.pop(variant, None)
                    if ev:
                        ev.set()
                raise

            swap_load_ms = float((time.perf_counter() - t0) * 1000.0)

            with self._lock:
                self._servers[variant] = srv
                self._touch_lru_locked(variant)
                ev = self._load_events.pop(variant, None)
                if ev:
                    ev.set()

                # Update EWMA for bandit cost estimation
                try:
                    prev = float(self._bandit_variant_load_ema_ms.get(variant, self._bandit_variant_load_synth_ms))
                    a = float(getattr(self, "_bandit_variant_load_ema_alpha", 0.1))
                    self._bandit_variant_load_ema_ms[variant] = (1.0 - a) * prev + a * float(swap_load_ms)
                except Exception:
                    pass

                return srv, {
                    "swap_loaded": True,
                    "swap_load_ms": float(swap_load_ms),
                    "swap_evicted_variants": victim_names,
                }

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

    def _get_latency_budgets_ms(self, difficulty: str, max_tokens: int) -> Tuple[float, float]:
        """Return (ttft_budget_ms, total_budget_ms) from the current SLO dict.

        If total_ms is not provided, we derive total_budget_ms as:
          total = ttft_ms + tpot_ms * max_tokens
        """

        diff = (difficulty or "easy").lower().strip()
        slo = None
        if isinstance(self.slo_dict, dict) and self.slo_dict:
            slo = self.slo_dict.get(diff) or self.slo_dict.get("default")
        if not isinstance(slo, dict) or not slo:
            # If SLOs are not configured, treat as effectively unconstrained.
            return 1e9, 1e9

        try:
            ttft_budget = float(slo.get("ttft_ms", 1e9))
        except Exception:
            ttft_budget = 1e9

        if "total_ms" in slo:
            try:
                total_budget = float(slo.get("total_ms", 1e9))
            except Exception:
                total_budget = 1e9
        else:
            try:
                tpot_budget = float(slo.get("tpot_ms", 1e9))
            except Exception:
                tpot_budget = 1e9
            total_budget = ttft_budget + tpot_budget * float(max_tokens)

        return float(ttft_budget), float(total_budget)

    # -------------------------
    # Adapter-aware routing features
    # -------------------------

    def _resolve_adapter_choice_locked(
        self,
        *,
        dataset_type: str,
        difficulty: str,
        queue_depths: Dict[str, int],
        explicit_adapter: Optional[str] = None,
        explicit_rank: Optional[int] = None,
    ) -> Tuple[str, Optional[int]]:
        """Resolve (adapter_id, adapter_rank) for this request.

        This is treated as part of the *variant configuration* for batching/scheduling.
        We resolve it *before* routing so the router can use adapter hotness/setup features.
        """

        if not self.enable_adapters:
            return "", None

        # Decide adapter ID.
        adapter_id_use = choose_adapter_id(
            policy=self.adapter_policy,
            dataset_type=dataset_type,
            fixed_adapter=self.adapter_fixed,
            explicit_adapter=str(explicit_adapter) if explicit_adapter is not None else None,
        )

        # Best-effort existence check: if adapter directory missing, fall back to base.
        # For synthetic adapter experiments (adapter_allow_missing=True), keep the id.
        if adapter_id_use and self.adapter_root and (not bool(getattr(self, "adapter_allow_missing", False))):
            expected = os.path.join(os.path.abspath(os.path.expanduser(self.adapter_root)), adapter_id_use)
            if not os.path.isdir(expected):
                adapter_id_use = ""

        adapter_rank_use: Optional[int] = None
        if adapter_id_use:
            if explicit_rank is not None:
                try:
                    adapter_rank_use = int(explicit_rank)
                except Exception:
                    adapter_rank_use = None
            else:
                total_q = 0
                try:
                    total_q = int(sum(int(v or 0) for v in (queue_depths or {}).values()))
                except Exception:
                    total_q = 0
                adapter_rank_use = choose_active_rank(
                    policy=self.adapter_rank_policy,
                    difficulty=difficulty,
                    total_queue_depth=total_q,
                    tiers=self.adapter_rank_tiers,
                    fixed_rank=self.adapter_fixed_rank,
                )

        return str(adapter_id_use or ""), (int(adapter_rank_use) if adapter_rank_use is not None else None)

    def _snapshot_adapter_state_locked(self, *, adapter_id: str, adapter_rank: Optional[int]) -> Dict[str, Dict[str, Any]]:
        """Best-effort per-variant adapter hotness/residency snapshot.

        Returned dict is safe to JSON-serialize and to include in router_meta.
        """

        aid = str(adapter_id or "").strip()
        try:
            rank_i = int(adapter_rank) if adapter_rank is not None else 0
        except Exception:
            rank_i = 0
        key = (aid, int(rank_i))

        out: Dict[str, Dict[str, Any]] = {}
        for v in self.variants:
            # Default: cold / unknown.
            st: Dict[str, Any] = {
                "adapter_id": aid,
                "adapter_active_rank": int(rank_i) if adapter_rank is not None else None,
                "resident": 0,
                "active": 0,
                "hot": 0,
                "num_loaded": 0,
                "capacity": int(self.max_loaded_adapters),
                "ewma_load_ms": float(self.adapter_synthetic_load_ms),
                "ewma_switch_ms": float(self.adapter_synthetic_switch_ms),
                "setup_est_ms": 0.0,
            }

            if not self.enable_adapters or not aid:
                out[v] = st
                continue

            # Hotness is based on the variant's most recently served adapter key.
            try:
                st["hot"] = int(1 if self._last_adapter_key_by_variant.get(v) == key else 0)
            except Exception:
                st["hot"] = 0

            srv = self._servers.get(v)
            mgr = getattr(srv, "_adapter_manager", None) if srv is not None else None

            if mgr is not None:
                # Use AdapterManager's snapshot (includes EWMA estimates).
                try:
                    snap = mgr.snapshot_for(aid, active_rank=(int(rank_i) if adapter_rank is not None else None))
                    if isinstance(snap, dict):
                        st.update(snap)
                except Exception:
                    pass

                # Ensure fields exist.
                try:
                    st.setdefault("resident", int(1 if mgr.is_loaded(aid) else 0))
                except Exception:
                    st.setdefault("resident", 0)
                try:
                    st.setdefault("active", int(1 if getattr(mgr, "active_adapter", None) == aid else 0))
                except Exception:
                    st.setdefault("active", 0)
                try:
                    st.setdefault("num_loaded", int(len(getattr(mgr, "loaded_adapters")() or [])))
                except Exception:
                    try:
                        st.setdefault("num_loaded", int(len(getattr(mgr, "_lru", {}) or {})))
                    except Exception:
                        st.setdefault("num_loaded", 0)

                # If hot, treat as no-setup (adapter already active with the same rank).
                if int(st.get("hot", 0) or 0) == 1:
                    st["setup_est_ms"] = 0.0
            else:
                # Variant not loaded or adapters not initialized yet.
                # Predict a miss cost using synthetic params (paper knob).
                st["resident"] = 0
                st["active"] = 0
                st["num_loaded"] = 0
                st["ewma_load_ms"] = float(self.adapter_synthetic_load_ms)
                st["ewma_switch_ms"] = float(self.adapter_synthetic_switch_ms)
                if aid:
                    if int(st.get("hot", 0) or 0) == 1:
                        st["setup_est_ms"] = 0.0
                    else:
                        st["setup_est_ms"] = float(max(0.0, st["ewma_load_ms"] + st["ewma_switch_ms"]))

            out[v] = st

        return out

    def choose_variant(
        self,
        dataset_type: str,
        difficulty: str,
        prompt_mode: str,
        max_tokens: int,
        prompt_tokens: int,
        concurrency: int,
        queue_depths: Dict[str, int],
        # Adapter-aware routing inputs (optional)
        adapter_id: str = "",
        adapter_rank: Optional[int] = None,
        adapter_state: Optional[Dict[str, Dict[str, Any]]] = None,
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
                prompt_tokens=int(prompt_tokens),
                concurrency=int(concurrency),
                queue_depths=queue_depths,
                adapter_id=str(adapter_id or ""),
                adapter_rank=int(adapter_rank) if adapter_rank is not None else None,
                adapter_state=adapter_state,
                slo_dict=self.slo_dict,
                mode=mode,
                allowed_variants=self.variants,
            )
            meta = decision.to_dict()
            meta["router_mode_label"] = "Learned-TTFT" if self.router_mode == "learned_ttft" else "Learned-Total (Derived)"
            return decision.variant, meta["router_mode_label"], meta

        # Risk router (calibrated latency upper bounds + calibrated quality gating)
        if self.router_mode == "risk":
            if self._risk_router is None:
                raise RuntimeError("Risk router requested but artifacts were not loaded.")
            decision = self._risk_router.route(
                dataset_type=dataset_type,
                difficulty=difficulty,
                max_tokens=int(max_tokens),
                prompt_tokens=int(prompt_tokens),
                concurrency=int(concurrency),
                queue_depths=queue_depths,
                adapter_id=str(adapter_id or ""),
                adapter_rank=int(adapter_rank) if adapter_rank is not None else None,
                adapter_state=adapter_state,
                slo_dict=self.slo_dict,
                latency_delta=float(self.risk_latency_delta),
                quality_epsilon=float(self.risk_quality_epsilon),
                allowed_variants=self.variants,
            )
            meta = decision.to_dict()
            meta["router_mode_label"] = f"Risk(δ={self.risk_latency_delta}, ε={self.risk_quality_epsilon})"
            meta["risk_latency_delta"] = float(self.risk_latency_delta)
            meta["risk_quality_epsilon"] = float(self.risk_quality_epsilon)
            meta["risk_quality_alpha"] = float(self.risk_quality_alpha)
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
        prompt_tokens: Optional[int] = None,
        concurrency: int = 1,
        # Optional explicit adapter override (paper: "active adapters" are part of state s)
        explicit_adapter: Optional[str] = None,
        explicit_rank: Optional[int] = None,
    ) -> Tuple[List[str], str, Dict[str, Any], Dict[str, int], str, Optional[int]]:
        """Compute the escalation path and routing metadata.

        Returns:
          path, reason, router_meta, queue_depths, adapter_id, adapter_rank
        """

        # Snapshot queue depths and adapter hotness at routing time.
        with self._lock:
            queue_depths = {v: len(self._queues.get(v, deque())) for v in self.variants}

            adapter_id_use, adapter_rank_use = self._resolve_adapter_choice_locked(
                dataset_type=dataset_type,
                difficulty=difficulty,
                queue_depths=queue_depths,
                explicit_adapter=explicit_adapter,
                explicit_rank=explicit_rank,
            )
            adapter_state = self._snapshot_adapter_state_locked(adapter_id=adapter_id_use, adapter_rank=adapter_rank_use)

            # ------------------------------
            # Bandit routing (variant + adapter + rank)
            # ------------------------------
            if self.router_mode.startswith("bandit") and self._bandit_router is not None and self._risk_router is not None:
                # Baseline action: use the risk router (strong, SLO-aware baseline) with the
                # standard adapter policy resolution.
                base_dec = self._risk_router.route(
                    dataset_type=dataset_type,
                    difficulty=difficulty,
                    max_tokens=int(max_tokens or 0),
                    prompt_tokens=int(prompt_tokens or 0),
                    concurrency=int(concurrency or 1),
                    queue_depths=queue_depths,
                    adapter_id=adapter_id_use,
                    adapter_rank=adapter_rank_use,
                    adapter_state=adapter_state,
                    allowed_variants=self.variants,
                    slo_dict=self.slo_dict or {},
                    latency_delta=float(self.risk_latency_delta),
                    quality_epsilon=float(self.risk_quality_epsilon),
                )
                # RiskRouter returns a RiskRouterDecision; unpack for backward compatibility.
                base_variant = str(getattr(base_dec, 'variant', 'base'))
                base_reason = str(getattr(base_dec, 'reason', ''))
                try:
                    base_meta = base_dec.to_dict()  # type: ignore[attr-defined]
                except Exception:
                    base_meta = {'decision': str(base_dec)}
                baseline_action = BanditAction(
                    variant=str(base_variant),
                    adapter_id=str(adapter_id_use or ""),
                    adapter_rank=(int(adapter_rank_use) if adapter_rank_use is not None else None),
                )

                # System snapshot for bandit features/cost estimation
                try:
                    free_gb, total_gb = _cuda_mem_gb()
                    gpu_free_frac = float(free_gb) / float(total_gb) if float(total_gb) > 0 else 0.0
                except Exception:
                    gpu_free_frac = 0.0
                swap_mode = 1 if (self.max_loaded_variants is not None and int(self.max_loaded_variants) < len(self.variants)) else 0
                variant_loaded = {v: (1 if v in self._servers else 0) for v in self.variants}
                system_snapshot = {
                    "swap_mode": int(swap_mode),
                    "gpu_free_frac": float(max(0.0, min(1.0, gpu_free_frac))),
                    "variant_loaded": variant_loaded,
                    "batch_wait_ms": float(getattr(self, "batch_wait_ms", 0.0) or 0.0),
                    "max_loaded_variants": int(self.max_loaded_variants or len(self.variants)),
                    "queue_depth_sum": int(sum(int(x) for x in queue_depths.values())),
                }

                # Build candidate action set.
                actions: List[BanditAction] = []
                adapter_state_map: Dict[Tuple[str, Optional[int]], Optional[Dict[str, Any]]] = {}

                # Adapter ids
                adapter_ids: List[str] = []
                if not self.enable_adapters:
                    adapter_ids = [""]
                else:
                    if explicit_adapter is not None:
                        adapter_ids = [str(explicit_adapter or "")]
                    else:
                        raw = str(self._bandit_adapter_ids_cfg or "").strip()
                        if raw:
                            adapter_ids = [a.strip() for a in raw.split(",") if a.strip()]
                        else:
                            # Default: allow "none" and the policy-selected adapter (if any).
                            adapter_ids = []
                            if adapter_id_use:
                                adapter_ids.append(str(adapter_id_use))
                        # Always include "no adapter" as a candidate.
                        if "" not in adapter_ids:
                            adapter_ids = [""] + adapter_ids

                    # Filter to adapters that exist on disk (if adapter_root is set)
                    if self.adapter_root:
                        filtered_ids: List[str] = []
                        for aid in adapter_ids:
                            if not aid:
                                filtered_ids.append("")
                                continue
                            ap = os.path.join(self.adapter_root, aid)
                            if os.path.isdir(ap):
                                filtered_ids.append(aid)
                        adapter_ids = filtered_ids or [""]

                # Rank tiers
                rank_tiers_default = list(self.adapter_rank_tiers)
                if self._bandit_rank_tiers_cfg:
                    try:
                        rank_tiers_default = [int(x.strip()) for x in str(self._bandit_rank_tiers_cfg).split(",") if x.strip()]
                    except Exception:
                        rank_tiers_default = list(self.adapter_rank_tiers)

                for aid in adapter_ids:
                    if not aid:
                        actions.extend([BanditAction(variant=v, adapter_id="", adapter_rank=None) for v in self.variants])
                        adapter_state_map[("", None)] = None
                    else:
                        if explicit_rank is not None:
                            tiers = [int(explicit_rank)]
                        else:
                            tiers = rank_tiers_default
                        for rnk in tiers:
                            akey = (aid, int(rnk))
                            if akey not in adapter_state_map:
                                adapter_state_map[akey] = self._snapshot_adapter_state_locked(adapter_id=aid, adapter_rank=int(rnk))
                            actions.extend([BanditAction(variant=v, adapter_id=aid, adapter_rank=int(rnk)) for v in self.variants])

                # Feature + cost per action
                features_by_action: Dict[str, np.ndarray] = {}
                cost_hat_by_action: Dict[str, float] = {}
                action_info: Dict[str, Dict[str, Any]] = {}

                # Latency guard budgets
                ttft_budget_ms, total_budget_ms = self._get_latency_budgets_ms(difficulty, int(max_tokens or 0))

                for a in actions + [baseline_action]:
                    k = a.key()
                    ast = adapter_state_map.get((a.adapter_id, a.adapter_rank))
                    # Ensure we have adapter state for the baseline action even if it's not
                    # part of the configured exploration adapter set.
                    if ast is None and (a.adapter_id or "") != "":
                        try:
                            ast = self._snapshot_adapter_state_locked(
                                adapter_id=str(a.adapter_id or ""),
                                adapter_rank=(int(a.adapter_rank) if a.adapter_rank is not None else None),
                            )
                            adapter_state_map[(str(a.adapter_id or ""), int(a.adapter_rank) if a.adapter_rank is not None else None)] = ast
                        except Exception:
                            ast = None

                    # Base (43D) feature vector used by the learned/risk routers
                    base_x = self._risk_router.extract_features(
                        dataset_type=dataset_type,
                        difficulty=difficulty,
                        max_tokens=int(max_tokens or 0),
                        prompt_tokens=int(prompt_tokens or 0),
                        concurrency=int(concurrency or 1),
                        queue_depths=queue_depths,
                        adapter_id=str(a.adapter_id or ""),
                        adapter_rank=(int(a.adapter_rank) if a.adapter_rank is not None else None),
                        adapter_state=ast,
                    )

                    # Optional adapter feature ablation
                    if not bool(self._bandit_router.config.use_adapter_features):
                        try:
                            base_x = np.asarray(base_x, dtype=np.float32).reshape(-1)
                            base_x = base_x.copy()
                            base_x[22:] = 0.0
                        except Exception:
                            pass

                    full_x = np.asarray(base_x, dtype=np.float32).reshape(-1)
                    # Optional system features
                    if bool(self._bandit_router.config.use_system_features):
                        extra: List[float] = []
                        extra.append(float(system_snapshot.get("swap_mode", 0)))
                        extra.append(float(system_snapshot.get("gpu_free_frac", 0.0)))
                        lv = system_snapshot.get("variant_loaded", {})
                        for v in self.variants:
                            extra.append(float(1.0 if lv.get(v, 0) else 0.0))
                        extra.append(float(system_snapshot.get("batch_wait_ms", 0.0)) / 1000.0)
                        mlv = float(system_snapshot.get("max_loaded_variants", len(self.variants)) or len(self.variants))
                        extra.append(float(mlv) / 10.0)
                        extra.append(float(system_snapshot.get("queue_depth_sum", 0.0)) / 100.0)
                        full_x = np.concatenate([full_x, np.asarray(extra, dtype=np.float32)], axis=0)

                    features_by_action[k] = full_x

                    # Cost estimate for scoring
                    try:
                        cost_mult = float(self.cost_multipliers.get(a.variant, 1.0))
                    except Exception:
                        cost_mult = 1.0
                    # Token component: use max_tokens as a conservative proxy for expected output length
                    total_tokens_est = int(max(0, int(prompt_tokens or 0))) + int(max(0, int(max_tokens or 0)))
                    token_cost_hat = float(cost_mult) * float(total_tokens_est)
                    overhead_hat_units = 0.0
                    if bool(self._bandit_router.config.use_overhead_cost):
                        # Adapter overhead (setup estimate)
                        try:
                            if ast and a.variant in ast:
                                overhead_hat_units += float(self.overhead_ms_to_cost_units) * float(ast[a.variant].get("setup_est_ms", 0.0) or 0.0)
                        except Exception:
                            pass
                        # Variant swap/load overhead if not loaded
                        try:
                            if int(system_snapshot.get("swap_mode", 0)) == 1 and int(variant_loaded.get(a.variant, 0)) == 0:
                                est_ms = float(self._bandit_variant_load_ema_ms.get(a.variant, self._bandit_variant_load_synth_ms))
                                overhead_hat_units += float(self.overhead_ms_to_cost_units) * float(est_ms)
                        except Exception:
                            pass
                    cost_hat_by_action[k] = float(token_cost_hat + overhead_hat_units)

                    # Optional latency-safe guard from risk router (conformal upper bounds)
                    latency_safe = True
                    ttft_ub = None
                    total_ub = None
                    try:
                        feat43 = np.asarray(base_x, dtype=np.float32).reshape(1, -1)
                        ttft_pred = float(self._risk_router.predict_ttft_ms(a.variant, feat43))
                        total_pred = float(self._risk_router.predict_total_ms(a.variant, feat43))
                        ttft_q = float(self._risk_router.conformal_q(a.variant, "ttft", delta=float(self.risk_latency_delta)))
                        total_q = float(self._risk_router.conformal_q(a.variant, "total", delta=float(self.risk_latency_delta)))
                        ttft_ub = float(ttft_pred + ttft_q)
                        total_ub = float(total_pred + total_q)
                        latency_safe = bool(ttft_ub <= float(ttft_budget_ms) and total_ub <= float(total_budget_ms))
                    except Exception:
                        latency_safe = True

                    action_info[k] = {
                        "latency_safe": bool(latency_safe),
                        "ttft_ub": (float(ttft_ub) if ttft_ub is not None else None),
                        "total_ub": (float(total_ub) if total_ub is not None else None),
                    }

                chosen_action, bandit_meta = self._bandit_router.route(
                    actions=actions,
                    features_by_action=features_by_action,
                    cost_hat_by_action=cost_hat_by_action,
                    baseline_action=baseline_action,
                    action_info=action_info,
                )

                chosen = str(chosen_action.variant)
                reason = "bandit"
                meta = {
                    "baseline": {"variant": str(base_variant), "reason": str(base_reason), "meta": base_meta},
                    "system_snapshot": system_snapshot,
                    "action_info": action_info.get(chosen_action.key(), {}),
                }
                # Merge in bandit meta (already JSON-serializable)
                try:
                    meta.update(bandit_meta or {})
                except Exception:
                    meta["bandit_meta_unparsed"] = str(bandit_meta)

                # Store the decision-time feature vector for the chosen action (used for online updates).
                try:
                    x_list = features_by_action.get(chosen_action.key())
                    if x_list is not None:
                        meta["bandit_x"] = [float(v) for v in np.asarray(x_list).reshape(-1).tolist()]
                except Exception:
                    pass

                # Override adapter choice with the bandit's selected action.
                adapter_id_use = str(chosen_action.adapter_id or "")
                adapter_rank_use = int(chosen_action.adapter_rank) if chosen_action.adapter_rank is not None else None
                adapter_state = self._snapshot_adapter_state_locked(adapter_id=adapter_id_use, adapter_rank=adapter_rank_use)
            else:
                chosen, reason, meta = self.choose_variant(
                    dataset_type=dataset_type,
                    difficulty=difficulty,
                    prompt_mode=prompt_mode,
                    max_tokens=max_tokens,
                    prompt_tokens=prompt_tokens,
                    concurrency=concurrency,
                    queue_depths=queue_depths,
                    adapter_id=adapter_id_use,
                    adapter_rank=adapter_rank_use,
                    adapter_state=adapter_state,
                )

            # Always include adapter context in router_meta for training/analysis.
            try:
                meta = dict(meta or {})
            except Exception:
                meta = {"router_meta_unparsed": str(meta)}
            meta.setdefault("adapter_id", adapter_id_use)
            meta.setdefault("adapter_rank", adapter_rank_use)
            meta.setdefault("adapter_state", adapter_state)

        path = [chosen]
        if self.max_retries > 0 and chosen != "base" and "base" in self.variants:
            if chosen == "cheap" and "med" in self.variants and len(path) < (self.max_retries + 1):
                path.append("med")
            if len(path) < (self.max_retries + 1):
                path.append("base")

        return path[: self.max_retries + 1], reason, meta, queue_depths, adapter_id_use, adapter_rank_use


    def _should_retry(self, dataset_type: str, text: str) -> bool:
        dataset_type = dataset_type.lower()
        if dataset_type == "gsm8k":
            ans = extract_gsm8k_strict(text)
            return not bool(ans)
        if dataset_type == "mmlu":
            ans = extract_mmlu_answer(text)
            # Format-only escalation: MMLU must be one of {A,B,C,D}.
            # Our extractor returns "" on failure, never None.
            return not bool(ans)
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

        # Offered-load hint (from the closed-loop load generator). Used only as a
        # routing feature + for logging. Keep a sane default.
        try:
            concurrency = int(kwargs.pop("concurrency", 1) or 1)
        except Exception:
            concurrency = 1
        concurrency = max(1, int(concurrency))

        # Optional experiment fields (used for bandit logging / delayed labels)
        request_id = kwargs.pop("request_id", None)
        gold_answer = kwargs.pop("gold_answer", None)
        explicit_label = kwargs.pop("label", None)
        label_source = kwargs.pop("label_source", None)

        # Align max_tokens with SingleVariantServer defaults
        if max_tokens is None:
            if dataset_type == "mmlu":
                max_tokens = 1
            elif prompt_mode == "slo":
                max_tokens = 128
            else:
                max_tokens = 256

        # Optional per-request override used by learned-router training data collection.
        # If set, we bypass routing and send this request directly to the specified variant.
        force_variant = kwargs.pop("force_variant", None) or kwargs.pop("router_fixed_variant", None)
        force_variant = _normalize_variant(force_variant) if force_variant else None

        # ------------------------------------------------------------
        # Adapter selection (optional).
        # IMPORTANT: we resolve adapter_id/adapter_rank *before routing*
        # so learned/risk routers can use adapter hotness/setup features.
        # ------------------------------------------------------------
        explicit_adapter = kwargs.pop("adapter_id", None)
        explicit_rank = kwargs.pop("adapter_rank", None)
        try:
            explicit_rank = int(explicit_rank) if explicit_rank is not None else None
        except Exception:
            explicit_rank = None

        # Prompt length (tokens) used as a feature for learned routing.
        prompt_tokens = 0
        try:
            if getattr(self, "_router_tokenizer", None) is not None:
                prompt_tokens = int(len(self._router_tokenizer(prompt, add_special_tokens=False).input_ids))
            else:
                prompt_tokens = int(len(prompt.split()))
        except Exception:
            prompt_tokens = int(len(prompt.split()))
        adapter_id_use = ""
        adapter_rank_use: Optional[int] = None

        if force_variant:
            # Forced path (trace collection) still needs a consistent adapter snapshot.
            with self._lock:
                qdepths = {v: len(self._queues.get(v, deque())) for v in self.variants}
                adapter_id_use, adapter_rank_use = self._resolve_adapter_choice_locked(
                    dataset_type=dataset_type,
                    difficulty=difficulty,
                    queue_depths=qdepths,
                    explicit_adapter=str(explicit_adapter) if explicit_adapter is not None else None,
                    explicit_rank=explicit_rank,
                )
                adapter_state = self._snapshot_adapter_state_locked(adapter_id=adapter_id_use, adapter_rank=adapter_rank_use)

            chosen = force_variant if force_variant in self.variants else "base"
            path, reason, router_meta = [chosen], f"forced:{chosen}", {
                "forced_variant": chosen,
                "adapter_id": adapter_id_use,
                "adapter_rank": adapter_rank_use,
                "adapter_state": adapter_state,
            }
        else:
            path, reason, router_meta, qdepths, adapter_id_use, adapter_rank_use = self.plan_path(
                dataset_type,
                difficulty,
                prompt_mode,
                int(max_tokens),
                prompt_tokens=prompt_tokens,
                concurrency=int(concurrency),
                explicit_adapter=str(explicit_adapter) if explicit_adapter is not None else None,
                explicit_rank=explicit_rank,
            )

        if router_meta is None:
            router_meta = {}
        try:
            router_meta = dict(router_meta)
        except Exception:
            router_meta = {"router_meta_unparsed": str(router_meta)}
        router_meta.setdefault("adapter_id", adapter_id_use)
        router_meta.setdefault("adapter_rank", adapter_rank_use)
        # Adapter state is a *router feature* (hotness/setup cost) and used in trace logs.
        if "adapter_state" not in router_meta:
            try:
                with self._lock:
                    router_meta["adapter_state"] = self._snapshot_adapter_state_locked(
                        adapter_id=str(adapter_id_use or ""),
                        adapter_rank=(int(adapter_rank_use) if adapter_rank_use is not None else None),
                    )
            except Exception:
                router_meta["adapter_state"] = {}

        req = _MVRequest(
            prompt=prompt,
            dataset_type=dataset_type,
            difficulty=difficulty,
            prompt_mode=prompt_mode,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            concurrency=int(concurrency),
            enqueue_t=time.perf_counter(),
            path=path,
            event=threading.Event(),
            router_queue_depths=qdepths,
            router_meta=router_meta,
            adapter_id=str(adapter_id_use or ""),
            adapter_rank=int(adapter_rank_use) if adapter_rank_use is not None else None,
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

        # Aggregate cost across attempts (e.g., format retries / escalation).
        try:
            attempts = list(req.attempts or [])
            total_cost = float(sum(float(a.get("cost_units", 0.0) or 0.0) for a in attempts))
            total_token_cost = float(sum(float(a.get("token_cost_units", 0.0) or 0.0) for a in attempts))
            total_adapter_ov = float(sum(float(a.get("adapter_overhead_units", 0.0) or 0.0) for a in attempts))
            total_swap_ov = float(sum(float(a.get("swap_overhead_units", 0.0) or 0.0) for a in attempts))
            any_slo_violation = bool(any(bool(a.get("slo_violation", False)) for a in attempts))
            any_risk_violation = int(any(int(a.get("risk_violation", 0) or 0) for a in attempts))
            if "cost_units" in req.output_metrics:
                req.output_metrics["cost_units_last_attempt"] = req.output_metrics.get("cost_units")
            req.output_metrics["cost_units"] = float(total_cost)
            req.output_metrics["token_cost_units"] = float(total_token_cost)
            req.output_metrics["adapter_overhead_units"] = float(total_adapter_ov)
            req.output_metrics["swap_overhead_units"] = float(total_swap_ov)
            req.output_metrics["total_cost_units"] = float(total_cost)
            # SLO / risk summary across attempts
            req.output_metrics["slo_violation_any"] = bool(any_slo_violation)
            req.output_metrics["risk_violation_any"] = int(any_risk_violation)
            # For compatibility, also surface as top-level signals.
            req.output_metrics["slo_violation"] = bool(any_slo_violation)
            req.output_metrics["risk_violation"] = int(any_risk_violation)
        except Exception:
            pass
        req.output_metrics["router_escalated"] = len(req.attempts) > 1
        req.output_metrics["router_final_variant"] = req.output_metrics.get("variant")
        req.output_metrics["router_num_attempts"] = len(req.attempts)
        req.output_metrics["router_queue_depths"] = getattr(req, "router_queue_depths", {})
        req.output_metrics["router_meta"] = getattr(req, "router_meta", {})
        req.output_metrics["concurrency"] = int(concurrency)

        # Optional join key for offline logs (used for delayed labels).
        if request_id is not None:
            try:
                req.output_metrics["request_id"] = int(request_id)
            except Exception:
                req.output_metrics["request_id"] = request_id
            try:
                req.output_metrics["router_meta"]["request_id"] = req.output_metrics["request_id"]
            except Exception:
                pass

        # Compute / attach quality label if provided (gold answer or explicit label).
        quality_label: Optional[int] = None
        label_src = None
        if explicit_label is not None:
            try:
                quality_label = 1 if int(explicit_label) != 0 else 0
                label_src = str(label_source or "explicit")
            except Exception:
                quality_label = None
        elif gold_answer is not None:
            try:
                dt = str(dataset_type or "").lower()
                pred_text = str(req.output_text or "")
                if dt == "mmlu":
                    pred = extract_mmlu_answer(pred_text)
                    gold = str(gold_answer).strip().upper()
                    if pred is None:
                        quality_label = 0
                    else:
                        quality_label = 1 if str(pred).strip().upper() == gold else 0
                    label_src = str(label_source or "gold")
                elif dt == "gsm8k":
                    pred_num = extract_gsm8k_strict(pred_text) or extract_gsm8k_parseable(pred_text)
                    gold_num = extract_gsm8k_strict(str(gold_answer)) or extract_gsm8k_parseable(str(gold_answer))
                    if pred_num is None or gold_num is None:
                        quality_label = None
                    else:
                        quality_label = 1 if bool(numbers_equal(str(pred_num), str(gold_num))) else 0
                    label_src = str(label_source or "gold")
                else:
                    quality_label = None
            except Exception:
                quality_label = None

        if quality_label is not None:
            req.output_metrics["correct"] = int(quality_label)
            req.output_metrics["label_source"] = str(label_src or "unknown")

        # Bandit online update (if enabled)
        if self.router_mode.startswith("bandit") and self._bandit_router is not None:
            req.output_metrics["bandit_update_enabled"] = bool(self._bandit_router_update_enabled)
            if self._bandit_router_update_enabled:
                try:
                    # Skip updates if this request escalated to a different variant.
                    escalated = bool(req.output_metrics.get("router_escalated", False))

                    # Action chosen by the bandit at decision time
                    act_dict = None
                    try:
                        act_dict = (req.output_metrics.get("router_meta") or {}).get("bandit", {}).get("chosen_action")
                    except Exception:
                        act_dict = None
                    if isinstance(act_dict, dict):
                        action = BanditAction.from_dict(act_dict)
                    else:
                        action = BanditAction(
                            variant=str(req.output_metrics.get("router_attempts", [{}])[0].get("variant", req.output_metrics.get("variant", "base"))),
                            adapter_id=str(getattr(req, "adapter_id", "") or ""),
                            adapter_rank=int(getattr(req, "adapter_rank", 0)) if getattr(req, "adapter_rank", None) is not None else None,
                        )

                    # Decision-time feature vector stored in router_meta
                    x_list = (req.output_metrics.get("router_meta") or {}).get("bandit_x")
                    if x_list is None:
                        raise ValueError("missing bandit_x in router_meta")
                    x = np.asarray(x_list, dtype=np.float32).reshape(-1)

                    cost = float(req.output_metrics.get("cost_units", 0.0) or 0.0)
                    risk_violation = int(req.output_metrics.get("risk_violation", 0) or 0)
                    label_key = str(request_id) if request_id is not None else action.key()

                    upd = self._bandit_router.update(
                        action=action,
                        x=x,
                        cost=cost,
                        risk_violation=risk_violation,
                        quality_label=quality_label,
                        label_key=label_key,
                        escalated=escalated,
                    )
                    req.output_metrics["bandit_update"] = upd
                except Exception as e:
                    req.output_metrics["bandit_update"] = {"updated": False, "error": str(e)}

        return req.output_text or "", req.output_metrics

    # -------------------------
    # Dispatcher
    # -------------------------

    def _select_next_variant_locked(self) -> Optional[str]:
        non_empty = [v for v, q in self._queues.items() if len(q) > 0]
        if not non_empty:
            return None

        # "setup_*" policies reuse the same base policy for *which variant*
        # to serve next, but enable adapter-aware batching within a variant.
        base_policy = self.dispatcher_policy
        if base_policy.startswith("setup_"):
            base_policy = base_policy.replace("setup_", "", 1)
            if base_policy == "aware":
                base_policy = "age"

        # Deadline-aware scheduling policies
        if base_policy in {"edf", "lstf"}:
            now = time.perf_counter()
            best_v: Optional[str] = None
            best_score: Optional[float] = None

            for v in non_empty:
                head = self._queues[v][0]
                if base_policy == "edf":
                    # Earliest absolute SLO deadline first.
                    _ttft_b, total_b = self._get_latency_budgets_ms(head.difficulty, head.max_tokens)
                    deadline_t = head.enqueue_t + (float(total_b) / 1000.0)
                    score = float(deadline_t)
                else:
                    # Least slack time first.
                    slack = None
                    try:
                        rm = getattr(head, "router_meta", {}) or {}
                        if isinstance(rm, dict):
                            slack = rm.get("slack_min_ms")
                            if slack is None and isinstance(rm.get("chosen"), dict):
                                slack = rm["chosen"].get("slack_min_ms")
                    except Exception:
                        slack = None

                    if slack is None:
                        # Fall back to an EMA-based latency estimate.
                        _ttft_b, total_b = self._get_latency_budgets_ms(head.difficulty, head.max_tokens)
                        pred = self._predict_latency_ms(v, head.max_tokens)
                        slack = float(total_b) - float(pred)
                    score = float(slack)

                if best_score is None or score < best_score:
                    best_score = score
                    best_v = v

            # Starvation guard: if something has waited too long, serve it.
            if best_v is not None:
                oldest_ms = 0.0
                oldest_v = None
                for v in non_empty:
                    w_ms = (now - self._queues[v][0].enqueue_t) * 1000.0
                    if w_ms > oldest_ms:
                        oldest_ms = w_ms
                        oldest_v = v
                if oldest_v is not None and oldest_ms >= self.dispatcher_starvation_ms:
                    return oldest_v
                return best_v
            return non_empty[0]

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

        def _k(r: _MVRequest) -> Tuple[Any, ...]:
            return (
                r.dataset_type,
                r.prompt_mode,
                int(r.max_tokens),
                float(r.temperature),
                float(r.top_p),
                str(getattr(r, "adapter_id", "") or ""),
                int(getattr(r, "adapter_rank", 0) or 0),
            )

        # Setup-aware batching: prefer the last adapter/rank used for this variant
        # (reduces adapter switch + cache-miss overhead).
        key = None
        pref = self._last_adapter_key_by_variant.get(variant)
        if (
            pref is not None
            and self.dispatcher_policy.startswith("setup")
            and self._active_variant == variant
            and int(getattr(self, "_active_adapter_batches_run", 0)) < int(getattr(self, "dispatcher_max_sticky_adapter_batches", 4))
        ):
            pref_aid, pref_rank = pref
            for r in q:
                if str(getattr(r, "adapter_id", "") or "") == str(pref_aid or "") and int(getattr(r, "adapter_rank", 0) or 0) == int(pref_rank or 0):
                    key = _k(r)
                    break

        if key is None:
            key = _k(q[0])

        # Collect a batch by scanning the queue and extracting matching items.
        # This avoids head-of-line blocking when adapters are interleaved.
        batch: List[_MVRequest] = []
        new_q: Deque[_MVRequest] = deque()
        while q:
            r = q.popleft()
            if len(batch) < self.max_batch_size and _k(r) == key:
                batch.append(r)
            else:
                new_q.append(r)
        q.extend(new_q)
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

                # Update stickiness counters (variant + adapter)
                if batch:
                    batch_adapter_key = (
                        str(getattr(batch[0], "adapter_id", "") or ""),
                        int(getattr(batch[0], "adapter_rank", 0) or 0),
                    )
                else:
                    batch_adapter_key = ("", 0)

                if self._active_variant == variant:
                    self._active_batches_run += 1
                    if self._last_adapter_key_by_variant.get(variant) == batch_adapter_key:
                        self._active_adapter_batches_run += 1
                    else:
                        self._active_adapter_batches_run = 1
                else:
                    self._active_variant = variant
                    self._active_batches_run = 1
                    self._active_adapter_batches_run = 1

                # Remember last adapter used for this variant (setup-aware scheduling).
                self._last_adapter_key_by_variant[variant] = batch_adapter_key

            if not batch:
                continue

            try:
                # Pin while this variant is being served
                with self._lock:
                    self._pins[variant] = self._pins.get(variant, 0) + 1

                # Ensure loaded (may evict/load)
                srv, load_meta = self._ensure_loaded_with_metrics(variant)

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
                    adapter_id=str(getattr(batch[0], "adapter_id", "") or ""),
                    adapter_rank=int(getattr(batch[0], "adapter_rank", 0) or 0) or None,
                    require_all_final_answers=require_all_final,
                )

                for r, out, m in zip(batch, outputs, metrics_list):
                    # Swap / load metadata (if variant swapping is enabled)
                    try:
                        swap_loaded = bool(load_meta.get("swap_loaded", False))
                        swap_load_ms = float(load_meta.get("swap_load_ms", 0.0) or 0.0)
                        swap_evicted = list(load_meta.get("swap_evicted_variants", []) or [])
                    except Exception:
                        swap_loaded, swap_load_ms, swap_evicted = False, 0.0, []

                    m["swap_loaded"] = bool(swap_loaded)
                    m["swap_load_ms"] = float(swap_load_ms)
                    m["swap_evicted_variants"] = swap_evicted

                    # Allocate swap overhead to each request in the batch.
                    bsz = int(max(1, len(batch)))
                    swap_overhead_ms_alloc = float(swap_load_ms) / float(bsz)
                    swap_overhead_units = float(self.overhead_ms_to_cost_units) * float(swap_overhead_ms_alloc)
                    m["swap_overhead_ms_alloc"] = float(swap_overhead_ms_alloc)
                    m["swap_overhead_units"] = float(swap_overhead_units)

                    # Recompute total cost units by adding swap overhead to the server-reported base cost.
                    try:
                        base_total = float(m.get("total_cost_units", m.get("cost_units", 0.0)) or 0.0)
                        # Fill breakdown fields when missing (backwards compatibility).
                        if m.get("adapter_overhead_units", None) is None:
                            m["adapter_overhead_units"] = 0.0
                        if m.get("token_cost_units", None) is None:
                            try:
                                m["token_cost_units"] = float(max(0.0, base_total - float(m.get("adapter_overhead_units", 0.0) or 0.0)))
                            except Exception:
                                m["token_cost_units"] = float(base_total)
                        total_cost_units = float(base_total) + float(swap_overhead_units)
                        m["total_cost_units"] = float(total_cost_units)
                        m["cost_units"] = float(total_cost_units)
                    except Exception:
                        pass
                    # MultiVariant dispatch wait
                    sched_wait_ms = (infer_start - r.enqueue_t) * 1000.0
                    m["scheduler_wait_ms"] = sched_wait_ms
                    m["lock_wait_ms"] = float(lock_wait_ms)
                    m["queue_wait_ms"] = sched_wait_ms + float(lock_wait_ms)
                    m["concurrency"] = int(getattr(r, "concurrency", 1))

                    # TTFT includes scheduler wait (paper Option A)
                    try:
                        m["ttft_ms"] = sched_wait_ms + float(m.get("ttft_infer_ms", 0.0))
                    except Exception:
                        pass
                    try:
                        m["total_latency_ms"] = float(m.get("total_latency_ms", 0.0)) + sched_wait_ms
                    except Exception:
                        pass

                    # SLO budgets + violation indicators
                    try:
                        ttft_budget_ms, total_budget_ms = self._get_latency_budgets_ms(r.difficulty, r.max_tokens)
                        m["slo_ttft_budget_ms"] = float(ttft_budget_ms)
                        m["slo_total_budget_ms"] = float(total_budget_ms)

                        # TPOT budget (raw SLO threshold)
                        prof = None
                        try:
                            if isinstance(self.slo_dict, dict):
                                prof = self.slo_dict.get(r.difficulty) or self.slo_dict.get("default")
                        except Exception:
                            prof = None
                        tpot_budget_ms = float((prof or {}).get("tpot_ms", 1e9) or 1e9)
                        m["slo_tpot_budget_ms"] = float(tpot_budget_ms)

                        ttft_ms = float(m.get("ttft_ms", 0.0) or 0.0)
                        total_ms = float(m.get("total_latency_ms", 0.0) or 0.0)
                        tpot_ms = float(m.get("tpot_ms", 0.0) or 0.0)
                        ttft_v = bool(ttft_ms > float(ttft_budget_ms))
                        total_v = bool(total_ms > float(total_budget_ms))
                        tpot_v = bool(tpot_ms > float(tpot_budget_ms))
                        m["slo_ttft_violation"] = bool(ttft_v)
                        m["slo_total_violation"] = bool(total_v)
                        m["slo_tpot_violation"] = bool(tpot_v)
                        # Primary SLO event (blueprint): TTFT or E2E(total) violation.
                        # NOTE: E2E is measured server-side as `total_latency_ms` (queue-inclusive Option A).
                        m["slo_violation"] = bool(ttft_v or total_v)
                        # Convenience for bandit updates (risk label)
                        m["risk_violation"] = int(1 if (ttft_v or total_v) else 0)
                    except Exception:
                        pass

                    # Record attempt
                    r.attempts.append(
                        {
                            "variant": m.get("variant"),
                            "adapter_id": m.get("adapter_id", ""),
                            "adapter_active_rank": m.get("adapter_active_rank"),
                            "success": bool(m.get("success", False)),
                            "ttft_ms": m.get("ttft_ms"),
                            "tpot_ms": m.get("tpot_ms"),
                            "total_latency_ms": m.get("total_latency_ms"),
                            "output_tokens": m.get("output_tokens"),
                            # Cost breakdown
                            "cost_units": m.get("cost_units"),
                            "token_cost_units": m.get("token_cost_units"),
                            "adapter_overhead_units": m.get("adapter_overhead_units"),
                            "swap_overhead_units": m.get("swap_overhead_units"),
                            "swap_loaded": m.get("swap_loaded"),
                            # SLO / risk signals
                            "slo_violation": m.get("slo_violation"),
                            "risk_violation": m.get("risk_violation"),
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
                                st.update(ttft_infer, tpot, total, n_out, q_wait, float(lock_wait_ms), alpha=self.ema_alpha)
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