# server.py
"""
Single-variant LLM server (baseline: MED-only 8-bit by default)

This file is tuned to support the research problem statement:
- We care about *tail latency* (TTFT/TPOT/E2E) under *concurrent load*.
- We care about *benchmark-style accuracy* for short-format tasks (MMLU letter, GSM8K final number).
- We need reproducible baselines: deterministic decoding by default.

Key fixes vs your current version:
1) Deterministic decoding by default (do_sample=False, temperature=0) for stable accuracy.
2) Correct TTFT measurement: ignore the initial streamer.put(prompt_tokens) call.
3) Thread-safe inference: a generate_lock serializes GPU access (realistic for single-GPU).
   Importantly, we export server_infer_start/end wall timestamps so the load generator can
   compute queue wait time including time spent waiting for GPU access.
4) Keep efficiency: torch.inference_mode(), minimal per-request cleanup.
"""

import time
import os
import gc
import threading
from typing import Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimingStreamer(BaseStreamer):
    """Capture first *generated* token time (TTFT).

    Transformers calls streamer.put(input_ids) once before generation starts,
    which contains the full prompt (seq_len > 1). If we record TTFT on that call
    we'd get ~0-1ms TTFT. We explicitly ignore that first prompt put().
    """

    def __init__(self, sync_fn):
        self.sync_fn = sync_fn
        self.first_token_time = None
        self.generated_token_count = 0
        self._ignored_prompt = False

    def put(self, value):
        # value is usually a torch.LongTensor [batch, seq_len]
        try:
            seq_len = int(getattr(value, "shape", [1, 1])[-1])
        except Exception:
            seq_len = 1

        # Ignore prompt pushes before generation begins.
        # We ignore the first push unconditionally (almost always prompt-related),
        # and also ignore any subsequent seq_len>1 pushes until we start timing.
        if not self._ignored_prompt:
            self._ignored_prompt = True
            return

        if self.first_token_time is None and seq_len > 1 and self.generated_token_count == 0:
            return

        if self.first_token_time is None:
            # Synchronize BEFORE recording time to ensure GPU ops for the first token are complete
            self.sync_fn()
            self.first_token_time = time.perf_counter()

        # Count generated tokens (typically seq_len == 1)
        self.generated_token_count += max(seq_len, 1)

    def end(self):
        pass


class FirstTokenAllowedTokens(LogitsProcessor):
    """Restrict the first generated token to a small allowed set.

    Used for MMLU-style multiple choice so the model emits exactly one of A/B/C/D.
    """

    def __init__(self, prompt_length: int, allowed_token_ids):
        super().__init__()
        self.prompt_length = int(prompt_length)
        self.allowed_token_ids = list(allowed_token_ids) if allowed_token_ids else []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.allowed_token_ids:
            return scores

        # At the first decoding step, input_ids length == prompt_length.
        try:
            cur_len = int(input_ids.shape[1])
        except Exception:
            cur_len = -1

        if cur_len == self.prompt_length:
            masked = scores.new_full(scores.shape, -float("inf"))
            masked[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
            return masked

        return scores

class GPUMonitor:
    @staticmethod
    def is_cuda_available() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            _ = torch.zeros(1, device='cuda')
            return True
        except Exception:
            return False

    @staticmethod
    def get_gpu_info() -> Dict:
        info = {"cuda_available": False}
        if not torch.cuda.is_available():
            return info
        try:
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)

            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

            info["memory_allocated_gb"] = round(memory_allocated, 2)
            info["memory_reserved_gb"] = round(memory_reserved, 2)
            info["memory_total_gb"] = round(memory_total, 2)
            info["memory_free_gb"] = round(memory_total - memory_reserved, 2)
            info["memory_utilization_pct"] = round(memory_reserved / memory_total * 100, 1)
        except Exception as e:
            info["error"] = str(e)
        return info

    @staticmethod
    def log_gpu_status(prefix: str = ""):
        info = GPUMonitor.get_gpu_info()
        if info.get("cuda_available"):
            logger.info(f"{prefix}GPU: {info.get('device_name', 'Unknown')}")
            logger.info(
                f"{prefix}  Memory: {info.get('memory_allocated_gb', 0):.2f}GB allocated, "
                f"{info.get('memory_free_gb', 0):.2f}GB free / {info.get('memory_total_gb', 0):.2f}GB total"
            )
        else:
            logger.warning(f"{prefix}CUDA not available")


class SingleVariantServer:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B",
        variant: str = "med",
        device: str = "auto",
        dtype: str = "auto",
    ):
        self.model_name = model_name
        self.variant = variant
        self.dtype = dtype

        self.device = self._detect_device(device, variant)

        # Inference serialization lock (single GPU realism + thread safety)
        self._generate_lock = threading.Lock()

        # Memory cleanup controls
        self._cleanup_per_request = False
        self._request_count = 0
        self._cleanup_interval = 50

        logger.info("=" * 70)
        logger.info(f"Initializing {variant.upper()} server")
        logger.info("=" * 70)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {dtype}")
        GPUMonitor.log_gpu_status("  ")

        self._load_tokenizer()
        self._load_model()

        logger.info("Post-load GPU status:")
        GPUMonitor.log_gpu_status("  ")

        self._warmup()

    def _detect_device(self, requested_device: str, variant: str) -> str:
        cuda_available = GPUMonitor.is_cuda_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        if requested_device == "auto":
            if cuda_available:
                selected = "cuda"
            elif mps_available and variant == "base":
                selected = "mps"
            else:
                selected = "cpu"
        else:
            selected = requested_device

        if variant in ["med", "cheap"] and selected != "cuda":
            if cuda_available:
                logger.warning(f"Quantized models require CUDA. Switching from '{selected}' to 'cuda'")
                selected = "cuda"
            else:
                raise RuntimeError("CUDA required for quantized models (bitsandbytes).")

        return selected

    def _compute_mmlu_allowed_token_ids(self):
        """Token IDs for 'A'/'B'/'C'/'D' (and space-prefixed variants) that are single tokens."""
        candidates = ["A", "B", "C", "D", " A", " B", " C", " D"]
        ids = set()
        for s in candidates:
            try:
                enc = self.tokenizer.encode(s, add_special_tokens=False)
                if isinstance(enc, (list, tuple)) and len(enc) == 1:
                    ids.add(int(enc[0]))
            except Exception:
                continue
        return sorted(ids)

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        # Precompute allowed token ids for MMLU (A/B/C/D) to avoid empty outputs
        self._mmlu_allowed_token_ids = self._compute_mmlu_allowed_token_ids()
        logger.info(f"MMLU allowed token ids: {len(self._mmlu_allowed_token_ids)}")

    def _load_model(self):
        try:
            if self.variant == "med":
                logger.info("Loading model with 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

            elif self.variant == "cheap":
                logger.info("Loading model with 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )

            else:
                logger.info("Loading model in full precision...")
                torch_dtype = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "auto": torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16,
                }.get(self.dtype, torch.float16)

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device != "cuda":
                    self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("Model loaded successfully")

            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model size: {total_params / 1e9:.2f}B parameters")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _synchronize_device(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def _maybe_cleanup_memory(self, force: bool = False):
        self._request_count += 1
        if force or (self._cleanup_per_request and self._request_count % self._cleanup_interval == 0):
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

    def _warmup(self):
        logger.info("Warming up server (3 iterations, deterministic)...")
        try:
            warmup_prompt = "Hello"
            inputs = self.tokenizer(warmup_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            for _ in range(3):
                with torch.inference_mode():
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=8,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                self._synchronize_device()

            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        difficulty: str = "medium",
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        dataset_type: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """Generate response + timing metrics.

        Defaults are tuned for benchmarking:
        - do_sample=False (deterministic)
        - temperature=0, top_p=1
        """
        if max_tokens is None:
            from prompt_templates import get_max_tokens
            max_tokens = int(get_max_tokens(difficulty))

        metrics: Dict = {}

        # Tokenize outside the GPU lock (CPU work); keep GPU work serialized.
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        input_length = int(inputs["input_ids"].shape[1])
        metrics["input_length"] = input_length

        # Move to device inside lock (GPU work)
        with self._generate_lock:
            server_infer_start_wall = time.time()
            metrics["server_infer_start_time_wall"] = server_infer_start_wall

            # Move tensors
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            self._synchronize_device()
            t0_total = time.perf_counter()

            # Streamer for TTFT
            streamer = TimingStreamer(self._synchronize_device)
            # Optional: task-aware decoding constraints (kept lightweight for speed)
            logits_processor = None
            effective_max_new_tokens = int(max_tokens)
            if (dataset_type or '').lower() == 'mmlu':
                # Force exactly one of {A,B,C,D} as the first token.
                allowed = getattr(self, '_mmlu_allowed_token_ids', [])
                if allowed:
                    logits_processor = LogitsProcessorList([FirstTokenAllowedTokens(input_length, allowed)])
                # MMLU should be a single token output
                effective_max_new_tokens = 1
                # Always deterministic for classification-style tasks
                do_sample = False


            # Build generation kwargs
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=int(effective_max_new_tokens),
                do_sample=bool(do_sample),
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
                use_cache=True,
                min_new_tokens=1,
                logits_processor=logits_processor,
            )
            if do_sample and (dataset_type or '').lower() != 'mmlu':
                gen_kwargs.update(dict(temperature=float(temperature), top_p=float(top_p)))

            try:
                self._synchronize_device()
                t0_generate = time.perf_counter()

                with torch.inference_mode():
                    outputs = self.model.generate(**gen_kwargs)

                self._synchronize_device()
                t_generate_end = time.perf_counter()

                # TTFT
                if streamer.first_token_time is not None:
                    t_ttft = streamer.first_token_time - t0_generate
                else:
                    t_ttft = t_generate_end - t0_generate
                metrics["ttft_ms"] = float(t_ttft * 1000.0)

                generated_ids = outputs.sequences[0, input_length:]
                output_length = int(generated_ids.shape[0])
                metrics["output_length"] = output_length

                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

                total_gen_time = t_generate_end - t0_generate
                t_decode = max(total_gen_time - t_ttft, 1e-6)
                metrics["total_decode_latency_ms"] = float(t_decode * 1000.0)

                if output_length > 1:
                    metrics["tpot_ms"] = float((t_decode * 1000.0) / (output_length - 1))
                else:
                    metrics["tpot_ms"] = 0.0

                metrics["throughput_tokens_per_sec"] = float(output_length / max(total_gen_time, 1e-6))

                t_total = time.perf_counter() - t0_total
                metrics["total_latency_ms"] = float(t_total * 1000.0)

                metrics["variant"] = self.variant
                metrics["model"] = self.model_name.split('/')[-1]
                metrics["device"] = self.device
                metrics["success"] = True

                # Wall clock end for load generator
                server_infer_end_wall = time.time()
                metrics["server_infer_end_time_wall"] = server_infer_end_wall

                # Cleanup
                del outputs
                self._maybe_cleanup_memory()

                return generated_text, metrics

            except Exception as e:
                metrics["success"] = False
                metrics["error"] = str(e)
                metrics["server_infer_end_time_wall"] = time.time()
                logger.error(f"Error during generation: {e}")
                return "", metrics

    def get_gpu_stats(self) -> Dict:
        return GPUMonitor.get_gpu_info()

    def force_memory_cleanup(self):
        self._maybe_cleanup_memory(force=True)
        GPUMonitor.log_gpu_status("After cleanup: ")