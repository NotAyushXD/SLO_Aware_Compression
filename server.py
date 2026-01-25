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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")


# -------------------------------
# Utilities
# -------------------------------

def split_system_user(prompt: str) -> Tuple[str, str]:
    """Split the legacy 'SYSTEM\n\nUSER' prompt format into (system, user)."""
    parts = prompt.split("\n\n", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    # Fallback: treat whole prompt as user content
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
    """Records TTFT (time-to-first-generated-token).

    HuggingFace may call streamer.put() once with the full prompt token IDs before generation.
    We ignore that *prompt push* and only timestamp the first generated token.
    This class intentionally does NOT depend on transformers.BaseStreamer (not always exported).
    """

    def __init__(self, sync_fn):
        self._sync_fn = sync_fn
        self.first_token_time: Optional[float] = None
        self._ignored_prompt_push = False

    def put(self, value):
        # The initial prompt push typically contains >1 token overall (prompt_len * batch_size).
        # Generated token pushes usually contain only 1 token per sequence (batch_size tokens total).
        try:
            n = int(value.numel()) if isinstance(value, torch.Tensor) else len(value)
        except Exception:
            n = 1

        if not self._ignored_prompt_push and n > 1:
            self._ignored_prompt_push = True
            return

        if self.first_token_time is None:
            self._sync_fn()
            self.first_token_time = time.perf_counter()

    def end(self):
        return


class StopOnFinalAnswer(StoppingCriteria):
    """Stop once FINAL_ANSWER is present.

    For batch generation, you can set require_all=True to stop only after *all* rows
    contain a FINAL_ANSWER, which makes it safe for batched GSM8K.
    """

    _PATTERN = re.compile(r"FINAL_ANSWER\s*[:=\s]*[-+]?\d+(?:\.\d+)?")

    def __init__(self, tokenizer, prompt_len: int, require_all: bool = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = int(prompt_len)
        self.require_all = bool(require_all)
        self._done: Optional[List[bool]] = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # input_ids: [batch, seq] or [seq]
        if input_ids.ndim == 1:
            text = self.tokenizer.decode(input_ids[self.prompt_len :], skip_special_tokens=True)
            return bool(self._PATTERN.search(text))

        bsz = int(input_ids.shape[0])
        if self._done is None or len(self._done) != bsz:
            self._done = [False] * bsz

        for i in range(bsz):
            if self._done[i]:
                continue
            text = self.tokenizer.decode(input_ids[i, self.prompt_len :], skip_special_tokens=True)
            if self._PATTERN.search(text):
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

    def submit(self, req: _PendingRequest) -> None:
        with self._cv:
            self._pending.append(req)
            self._cv.notify()

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if self._stop and not self._pending:
                    return

                # Start a new batch with the oldest request.
                first = self._pending.pop(0)
                key = first.batch_key()
                batch: List[_PendingRequest] = [first]

                deadline = time.perf_counter() + self.batch_wait_s
                while len(batch) < self.max_batch_size:
                    # Find another request with the same key.
                    idx = next((i for i, r in enumerate(self._pending) if r.batch_key() == key), None)
                    if idx is not None:
                        batch.append(self._pending.pop(idx))
                        continue

                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)

            # Process outside the lock.
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
                queue_wait_ms = (dequeue_t - r.enqueue_time) * 1000.0 + float(lock_wait_ms)
                m["queue_wait_ms"] = float(max(0.0, queue_wait_ms))
                r.result_text = text
                r.result_metrics = m
                r.event.set()

        except Exception as e:
            for r in batch:
                r.error = str(e)
                r.result_text = ""
                r.result_metrics = {
                    "success": False,
                    "error": str(e),
                    "ttft_ms": 0.0,
                    "tpot_ms": 0.0,
                    "output_length": 0,
                    "throughput_tokens_per_sec": 0.0,
                    "total_latency_ms": 0.0,
                    "queue_wait_ms": (dequeue_t - r.enqueue_time) * 1000.0,
                    "variant": self.server.variant,
                    "model": self.server.model_name,
                    "device": self.server.device,
                }
                r.event.set()


# -------------------------------
# Main server
# -------------------------------


class SingleVariantServer:
    """A single-process, single-GPU server with optional micro-batching.

    The micro-batcher is deliberately simple:
    - It groups requests with the same (dataset_type, prompt_mode, max_tokens, do_sample)
      arriving within a small window.
    - It runs exactly one HF `generate()` at a time, preventing the concurrency=4 TPOT
      collapse seen with parallel independent generation.
    """

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
        self.dtype = dtype

        logger.info("=" * 69)
        logger.info(f"Initializing {variant.upper()} server")
        logger.info("=" * 69)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {dtype}")
        if self.device == "cuda":
            GPUMonitor.log_gpu_status(prefix="  ")

        self._generation_lock = threading.Lock()

        # Tokenizer first (cheap)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Make batching-friendly for decoder-only models
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Precompute MMLU allowed ids
        self.mmlu_allowed_token_ids = self._compute_mmlu_allowed_token_ids()
        logger.info(f"MMLU allowed token ids: {len(self.mmlu_allowed_token_ids)}")
        logger.info(
            f"Tokenizer loaded: {type(self.tokenizer).__name__} (chat_template={bool(getattr(self.tokenizer, 'chat_template', None))})"
        )

        # Load model
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=(
                torch.float16
                if dtype == "float16"
                else torch.bfloat16
                if dtype == "bfloat16"
                else "auto"
            ),
            quantization_config=quant_config,
        )
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
            logger.info(
                f"Batching enabled: max_batch_size={max_batch_size}, batch_wait_ms={batch_wait_ms}"
            )
            self._scheduler = _BatchingScheduler(
                server=self, max_batch_size=max_batch_size, batch_wait_ms=batch_wait_ms
            )

        # Warmup (deterministic) to reduce cold-start variance
        self._warmup()

    def _compute_mmlu_allowed_token_ids(self) -> List[int]:
        ids: List[int] = []
        for s in self.MMLU_ALLOWED_CHARS:
            t = self.tokenizer.encode(s, add_special_tokens=False)
            if t:
                ids.append(int(t[0]))
        # Deduplicate while preserving order
        seen = set()
        out = []
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

        # If the tokenizer has a chat template, use it; otherwise fall back.
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Very simple fallback format
        if system:
            return f"SYSTEM: {system}\nUSER: {user}\nASSISTANT:"
        return f"USER: {user}\nASSISTANT:"

    def _warmup(self) -> None:
        logger.info("Warming up server (3 iterations, deterministic)...")
        warm_prompt = "You are a helpful assistant. Reply with a single letter: A."
        for _ in range(3):
            try:
                # Use direct generation path to avoid involving the scheduler.
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
          metrics_list: list[dict]  (one per sample, without queue_wait_ms)
          lock_wait_ms: float       (time waiting on the GPU generation lock)
        """

        t0_total = time.perf_counter()

        dataset_type = dataset_type.lower()
        prompt_mode = prompt_mode.lower()
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
            # MMLU should output exactly one token: A/B/C/D.
            max_tokens = 1

            allowed_ids = self.mmlu_allowed_token_ids

            def prefix_allowed_tokens_fn(batch_id: int, input_ids):
                # Restrict the *first* generated token to {A,B,C,D} (and variants).
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

        # Timing + lock
        streamer = TimingStreamer(self._synchronize_device)

        t_lock_req = time.perf_counter()
        with self._generation_lock:
            t_lock_acq = time.perf_counter()
            lock_wait_ms = (t_lock_acq - t_lock_req) * 1000.0

            self._synchronize_device()
            t0_gen = time.perf_counter()

            # NOTE: We avoid return_dict_in_generate/output_scores here for speed.
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
            # If for some reason streaming didn't fire, fall back to total time.
            first_tok_t = t1_gen
        ttft_s = max(0.0, first_tok_t - t0_gen)

        # Decode outputs (only the generated tail)
        texts: List[str] = []
        out_lens: List[int] = []
        for i in range(bsz):
            gen_ids = sequences[i, prompt_len:]
            out_lens.append(int(gen_ids.numel()))
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            texts.append(text)

        # Compute tpot using decode time excluding TTFT.
        decode_s = max(0.0, total_gen_time - ttft_s)

        t1_total = time.perf_counter()
        total_latency_s = max(0.0, t1_total - t0_total)

        metrics_list: List[Dict] = []
        for out_len in out_lens:
            if out_len <= 1:
                tpot_ms = 0.0
            else:
                tpot_ms = (decode_s * 1000.0) / float(out_len - 1)

            throughput = (float(out_len) / total_gen_time) if total_gen_time > 0 else 0.0

            metrics_list.append(
                {
                    "success": True,
                    "ttft_ms": float(ttft_s * 1000.0),
                    "tpot_ms": float(tpot_ms),
                    "output_length": int(out_len),
                    "throughput_tokens_per_sec": float(throughput),
                    "total_latency_ms": float(total_latency_s * 1000.0),
                    # queue_wait_ms is added by the caller (scheduler/direct)
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

        dataset_type = dataset_type.lower()
        prompt_mode = prompt_mode.lower()
        difficulty = difficulty.lower()

        if max_tokens is None:
            max_tokens = get_max_tokens(dataset_type, difficulty, prompt_mode)
        if dataset_type == "mmlu":
            max_tokens = 1

        # Decide whether to route through the batcher.
        batching_enabled = self._scheduler is not None
        if use_batching is None:
            # Default: only batch SLO mode. Accuracy evaluation is usually sequential.
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
                    "error": req.error,
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
        metrics["queue_wait_ms"] = float(lock_wait_ms)
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
