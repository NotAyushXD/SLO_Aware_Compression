# server.py
"""
Single-variant LLM server (MED-only) with:
- chat-template prompting for Llama-3.x Instruct models
- robust TTFT/TPOT metrics
- GSM8K early stopping that avoids the "trailing zeros dropped" bug
- optional MMLU single-letter constrained decoding in SLO mode

This file is meant to be used by:
- run_baseline_evaluation.py (evaluation + load tests)
- load_generator.py (concurrency tests)
"""

from __future__ import annotations

import time
import logging
import re
from typing import Dict, Any, Optional, Tuple, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.generation.streamers import BaseStreamer
from transformers import StoppingCriteria, StoppingCriteriaList

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("server")


# -----------------------------------------------------------------------------
# Streamer for TTFT measurement
# -----------------------------------------------------------------------------

class TimingStreamer(BaseStreamer):
    """
    Minimal streamer to capture the time-to-first-token (TTFT).

    HF generate() calls streamer.put() every step with newly generated tokens.
    """
    def __init__(self):
        super().__init__()
        self.first_token_time: Optional[float] = None

    def put(self, value):
        if self.first_token_time is None:
            # synchronize to make TTFT meaningful on GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.first_token_time = time.time()

    def end(self):
        return


# -----------------------------------------------------------------------------
# GSM8K early-stop criteria (permanent fix #1)
# -----------------------------------------------------------------------------

class StopOnFinalAnswer(StoppingCriteria):
    """
    Stop generation once we observe a FINAL_ANSWER line with a number AND
    at least one non-digit character after the number.

    Why this matters:
    - The earlier pattern stopped as soon as it saw "FINAL_ANSWER: 190"
      which caused "1900" -> "190" (dropping trailing zeros).
    - Requiring a non-digit char after the number ensures the number token
      is complete (e.g., space/newline/punctuation after it).
    """
    def __init__(self, tokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        # Require a non-digit after the number to avoid premature stopping.
        self._pat = re.compile(
            r"FINAL_ANSWER\s*[:=\s]*[-+]?\d+(?:\.\d+)?\D",
            re.IGNORECASE
        )

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # input_ids can be [batch, seq] or [seq] depending on transformers version
        if getattr(input_ids, "ndim", 2) == 1:
            gen_ids = input_ids[self.prompt_len:]
        else:
            gen_ids = input_ids[0, self.prompt_len:]

        if gen_ids.numel() == 0:
            return False

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return bool(self._pat.search(text))


# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------

class SingleVariantServer:
    """Single-variant server (MED variant) with 8-bit quantization by default."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        variant: str = "med",
        device: str = "auto",
        dtype: str = "auto",
        max_length: int = 4096,
    ):
        self.model_name = model_name
        self.variant = variant
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        logger.info("=" * 70)
        logger.info("Initializing MED server")
        logger.info("=" * 70)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {dtype}")

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Decide whether to use chat template
        self._has_chat_template = hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None)
        if self._has_chat_template:
            logger.info("Tokenizer has chat template; will use apply_chat_template().")
        else:
            logger.info("Tokenizer has no chat template; will use raw text prompts.")

        # Precompute MMLU allowed token ids (single-token A/B/C/D and space-prefixed variants)
        self.mmlu_allowed_token_ids = self._compute_mmlu_allowed_token_ids()
        logger.info(f"MMLU allowed token ids: {len(self.mmlu_allowed_token_ids)}")

        # Load model (8-bit quantization for MED)
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Loading model with 8-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=None if dtype == "auto" else getattr(torch, dtype),
        )
        self.model.eval()

        # GPU status (best-effort)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"  GPU: {gpu_name}")
            logger.info(f"    Memory: {allocated:.2f}GB allocated, {total-allocated:.2f}GB free / {total:.2f}GB total")

        # Warmup
        self._warmup()

    def _compute_mmlu_allowed_token_ids(self) -> List[int]:
        ids: List[int] = []
        for ch in ["A", "B", "C", "D"]:
            for s in [ch, " " + ch]:
                try:
                    enc = self.tokenizer.encode(s, add_special_tokens=False)
                    if len(enc) == 1:
                        ids.append(enc[0])
                except Exception:
                    continue
        return sorted(set(ids))

    def _warmup(self, iters: int = 3):
        """Warm-up generation to stabilize latency and catch obvious errors early."""
        logger.info(f"Warming up server ({iters} iterations, deterministic)...")
        for _ in range(iters):
            try:
                _ = self.generate(
                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                              {"role": "user", "content": "Say 'ok'."}],
                    dataset_type="mmlu",
                    prompt_mode="slo",
                    max_tokens=4,
                    do_sample=False,
                )
            except Exception as e:
                logger.error(f"Warmup error: {e}")
        logger.info("Warmup complete")

    def _encode(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode either chat messages (preferred) or a raw prompt string.
        """
        if messages is not None and self._has_chat_template:
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(input_ids, dict):
                    # Some transformers versions return a dict
                    enc = input_ids
                else:
                    enc = {"input_ids": input_ids}
            except TypeError:
                # Older signature fallback
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                enc = {"input_ids": input_ids}
        else:
            text = (prompt or "").strip()
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

        # Attention mask
        if "attention_mask" not in enc:
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])

        # Move to device
        for k in enc:
            enc[k] = enc[k].to(self.model.device)

        return enc

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 128,
        difficulty: str = "medium",
        dataset_type: str = "gsm8k",
        prompt_mode: str = "slo",
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text + metrics.

        Args:
            prompt/messages: input prompt
            max_tokens: max_new_tokens (output token budget)
            dataset_type: "gsm8k" or "mmlu"
            prompt_mode: "accuracy" or "slo"
        """
        enc = self._encode(prompt=prompt, messages=messages)
        input_ids = enc["input_ids"]
        prompt_len = input_ids.shape[-1]

        # Optional early stop for GSM8K
        stopping_criteria = None
        if dataset_type.lower() == "gsm8k":
            stopping_criteria = StoppingCriteriaList([StopOnFinalAnswer(self.tokenizer, prompt_len=prompt_len)])

        # Optional constrained decoding for MMLU in SLO mode
        prefix_allowed_tokens_fn = None
        if dataset_type.lower() == "mmlu" and (prompt_mode or "").lower() != "accuracy":
            if self.mmlu_allowed_token_ids:
                def _allow(batch_id, input_ids_local):
                    return self.mmlu_allowed_token_ids
                prefix_allowed_tokens_fn = _allow
                # In SLO mode we really just want the letter
                max_tokens = min(max_tokens, 2)

        streamer = TimingStreamer()

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.time()

        try:
            gen_kwargs = dict(
                **enc,
                max_new_tokens=int(max_tokens),
                do_sample=bool(do_sample),
                return_dict_in_generate=True,
                output_scores=False,
                streamer=streamer,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria
            if prefix_allowed_tokens_fn is not None:
                gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

            # Only pass sampling params when do_sample=True to avoid warnings.
            if do_sample:
                gen_kwargs["temperature"] = float(temperature)
                gen_kwargs["top_p"] = float(top_p)

            out = self.model.generate(**gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            seq = out.sequences[0]
            gen_ids = seq[prompt_len:]
            output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            output_len = int(gen_ids.shape[-1])
            total_gen_time = end_time - inference_start

            # TTFT
            ttft_ms = 0.0
            if streamer.first_token_time is not None:
                ttft_ms = (streamer.first_token_time - inference_start) * 1000.0

            # TPOT
            tpot_ms = (total_gen_time * 1000.0 / max(output_len, 1))

            metrics = {
                "success": True,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "output_length": output_len,
                "inference_time_s": total_gen_time,
                "throughput_tokens_per_sec": (output_len / max(total_gen_time, 1e-6)),
                "difficulty": difficulty,
                "dataset_type": dataset_type,
                "prompt_mode": prompt_mode,
            }
            return output_text, metrics

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            metrics = {
                "success": False,
                "error": str(e),
                "ttft_ms": 0.0,
                "tpot_ms": 0.0,
                "output_length": 0,
                "inference_time_s": end_time - inference_start,
                "throughput_tokens_per_sec": 0.0,
                "difficulty": difficulty,
                "dataset_type": dataset_type,
                "prompt_mode": prompt_mode,
            }
            return "", metrics
