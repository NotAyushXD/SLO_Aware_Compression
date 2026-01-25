# server.py
"""
Single-variant LLM server.

This version (v7) is focused on **accuracy debugging** while keeping the hooks
needed for later SLO work:
- Supports prompt_mode ("accuracy" vs "slo") via kwargs.
- Uses chat templates automatically for Instruct models (when available).
- Enforces MMLU output to be a single letter (A/B/C/D) via token restriction.
- Adds an optional stopping criterion for GSM8K to stop once FINAL_ANSWER is produced.

NOTE: BitsAndBytes quantization (8-bit / 4-bit) requires CUDA.
"""

from __future__ import annotations

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from typing import Dict, Tuple, Optional, List
import logging
import gc
import os
import re

from prompt_templates import get_max_tokens, split_system_user

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TimingStreamer(BaseStreamer):
    """
    Streamer capturing first-token time for TTFT measurement.
    """
    def __init__(self, sync_fn):
        self.sync_fn = sync_fn
        self.first_token_time = None
        self.token_count = 0

    def put(self, value):
        if self.first_token_time is None:
            self.sync_fn()
            self.first_token_time = time.perf_counter()
        self.token_count += 1

    def end(self):
        pass


class StopOnFinalAnswer(StoppingCriteria):
    """
    Stop generation once we observe a FINAL_ANSWER line with a number.
    """
    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self._pat = re.compile(r"FINAL_ANSWER\s*[:=\s]*[-+]?\d+(?:\.\d+)?", re.IGNORECASE)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # input_ids shape: [batch, seq]
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return bool(self._pat.search(text))


class GPUMonitor:
    """Monitor and log GPU utilization metrics"""
    @staticmethod
    def is_cuda_available() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            t = torch.zeros(1, device="cuda")
            del t
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

            info["memory_allocated_gb"] = memory_allocated
            info["memory_reserved_gb"] = memory_reserved
            info["memory_total_gb"] = memory_total
            info["memory_free_gb"] = memory_total - memory_reserved
        except Exception as e:
            info["error"] = str(e)
        return info

    @staticmethod
    def log_gpu_status(prefix: str = ""):
        info = GPUMonitor.get_gpu_info()
        if not info.get("cuda_available"):
            logger.info(prefix + "GPU: Not available")
            return
        logger.info(prefix + f"GPU: {info.get('device_name')}")
        logger.info(prefix + f"  Memory: {info.get('memory_allocated_gb', 0):.2f}GB allocated, "
                    f"{info.get('memory_free_gb', 0):.2f}GB free / {info.get('memory_total_gb', 0):.2f}GB total")


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

    # ---------------------------------------------------------------------
    # Device / loading
    # ---------------------------------------------------------------------

    def _detect_device(self, requested_device: str, variant: str) -> str:
        req = (requested_device or "auto").lower().strip()
        cuda_ok = GPUMonitor.is_cuda_available()
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        # Quantized variants require CUDA (bitsandbytes)
        if variant in ("med", "cheap") and not cuda_ok:
            logger.warning(f"Variant '{variant}' needs CUDA for bitsandbytes; falling back to CPU/base behavior may fail.")
            # We'll still return 'cpu' so the error is explicit at load time.
            return "cpu"

        if req == "cuda":
            return "cuda" if cuda_ok else "cpu"
        if req == "mps":
            return "mps" if mps_ok else "cpu"
        if req == "cpu":
            return "cpu"

        # auto
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    def _load_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Chat template support (best for Instruct models)
            self._supports_chat = bool(getattr(self.tokenizer, "chat_template", None)) and hasattr(self.tokenizer, "apply_chat_template")

            # Precompute allowed token ids for MMLU (single-token letters)
            self._mmlu_allowed_token_ids = self._compute_mmlu_allowed_token_ids()
            if self._mmlu_allowed_token_ids:
                logger.info(f"MMLU allowed token ids: {len(self._mmlu_allowed_token_ids)}")
            else:
                logger.warning("Could not compute single-token A/B/C/D ids; MMLU restriction will be disabled.")

            logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__} (chat_template={self._supports_chat})")

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def _compute_mmlu_allowed_token_ids(self) -> List[int]:
        allowed = set()
        candidates = ["A", "B", "C", "D", " A", " B", " C", " D", "\nA", "\nB", "\nC", "\nD"]
        for s in candidates:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                allowed.add(ids[0])
        return sorted(allowed)

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

            # Parameters count (approx)
            try:
                n_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model size: {n_params/1e9:.2f}B parameters")
            except Exception:
                pass

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
        if not force and not self._cleanup_per_request:
            if self._request_count % self._cleanup_interval != 0:
                return
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def _warmup(self, iterations: int = 3):
        logger.info(f"Warming up server ({iterations} iterations, deterministic)...")
        try:
            # A short deterministic warmup to populate caches
            for _ in range(iterations):
                _ = self.generate("Warmup", max_tokens=4, temperature=0.0, top_p=1.0, dataset_type="mmlu", prompt_mode="slo")
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    def _build_input_text(self, prompt: str) -> str:
        """
        If chat_template is available (Instruct models), wrap as:
          system: <system>
          user: <user>
        Otherwise, return prompt verbatim.
        """
        if not self._supports_chat:
            return prompt

        system, user = split_system_user(prompt)
        if not user:
            user = prompt
            system = ""

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback to raw prompt if chat template fails
            return prompt

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        difficulty: str = "medium",
        dataset_type: str = "mmlu",
        prompt_mode: str = "slo",
    ) -> Tuple[str, Dict]:
        """
        Generate response and collect latency metrics.

        Returns:
            generated_text, metrics_dict
        """
        metrics: Dict = {
            "success": False,
            "ttft_ms": 0.0,
            "tpot_ms": 0.0,
            "output_length": 0,
            "throughput_tokens_per_sec": 0.0,
            "total_latency_ms": 0.0,
            "variant": self.variant,
            "model": self.model_name.split("/")[-1],
            "device": self.device,
        }

        dataset_type = (dataset_type or "mmlu").lower().strip()

        if max_tokens is None:
            max_tokens = get_max_tokens(difficulty=difficulty, dataset_type=dataset_type, prompt_mode=prompt_mode)

        # MMLU: force 1 token output (we also restrict allowed tokens)
        if dataset_type == "mmlu":
            max_tokens = 1

        t0_total = time.perf_counter()

        try:
            input_text = self._build_input_text(prompt)

            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            input_len = int(inputs["input_ids"].shape[1])

            self._synchronize_device()

            # Prepare generation args
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=int(max_tokens),
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

            # Sampling vs greedy
            do_sample = bool(temperature and temperature > 0.0)
            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)
            else:
                gen_kwargs["do_sample"] = False

            # TTFT timing streamer
            streamer = TimingStreamer(sync_fn=self._synchronize_device)
            gen_kwargs["streamer"] = streamer

            # Dataset-specific controls
            if dataset_type == "mmlu" and self._mmlu_allowed_token_ids:
                prompt_len = input_len

                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    # First generated token: must be A/B/C/D variants
                    if input_ids.shape[1] == prompt_len:
                        return self._mmlu_allowed_token_ids
                    # After first token, allow eos only (shouldn't matter because max_new_tokens=1)
                    return [self.tokenizer.eos_token_id]

                gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

            stopping = None
            if dataset_type == "gsm8k":
                stopping = StoppingCriteriaList([StopOnFinalAnswer(self.tokenizer, prompt_len=input_len)])
                gen_kwargs["stopping_criteria"] = stopping

            # Generate
            t0_gen = time.perf_counter()
            outputs = self.model.generate(**gen_kwargs)
            self._synchronize_device()
            t1_gen = time.perf_counter()

            # TTFT
            if streamer.first_token_time is not None:
                t_ttft = streamer.first_token_time - t0_gen
            else:
                t_ttft = t1_gen - t0_gen
            metrics["ttft_ms"] = t_ttft * 1000.0

            # Decode generated tokens (exclude prompt)
            generated_ids = outputs.sequences[0, input_len:]
            out_len = int(generated_ids.shape[0])
            metrics["output_length"] = out_len

            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            total_gen_time = max(t1_gen - t0_gen, 1e-6)

            # TPOT (after first token)
            t_decode = max(total_gen_time - t_ttft, 1e-6)
            if out_len > 1:
                metrics["tpot_ms"] = (t_decode * 1000.0) / (out_len - 1)
            else:
                metrics["tpot_ms"] = 0.0

            metrics["throughput_tokens_per_sec"] = out_len / total_gen_time

            # Total end-to-end latency
            self._synchronize_device()
            metrics["total_latency_ms"] = (time.perf_counter() - t0_total) * 1000.0

            metrics["success"] = True

            # Cleanup
            del outputs
            self._maybe_cleanup_memory()

            return generated_text, metrics

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            metrics["success"] = False
            metrics["error"] = str(e)
            self._maybe_cleanup_memory()
            return "", metrics

    def get_gpu_stats(self) -> Dict:
        return GPUMonitor.get_gpu_info()

    def force_memory_cleanup(self):
        self._maybe_cleanup_memory(force=True)
        GPUMonitor.log_gpu_status("After cleanup: ")


if __name__ == "__main__":
    # Simple smoke test (CPU-friendly model)
    logger.info("Testing SingleVariantServer with a small model...")
    srv = SingleVariantServer(model_name="gpt2", variant="base", device="cpu")
    out, m = srv.generate("Answer: A", max_tokens=1, dataset_type="mmlu")
    print(out, m)
