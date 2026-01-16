# server.py
"""
Single-variant LLM server (MED-only: 8-bit quantization)
Handles inference requests and collects detailed latency metrics
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, Optional
import logging
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from transformers.generation.streamers import BaseStreamer

class TimingStreamer(BaseStreamer):
    """
    Streamer that captures the time of the first token generation
    to measure TTFT (Time To First Token) accurately during a single pass.
    """
    def __init__(self, tokenizer, server_instance):
        self.tokenizer = tokenizer
        self.server = server_instance
        self.first_token_time = None
        self.tokens = []
        
    def put(self, value):
        """
        Callback for new tokens.
        value: tensor of new tokens (shape [batch, 1])
        """
        # If this is the first token(s), record the time
        if self.first_token_time is None:
            self.server._synchronize_device()
            self.first_token_time = time.time()
            
        self.tokens.append(value)
        
    def end(self):
        pass


class SingleVariantServer:
    """
    Single-variant LLM server for inference
    Collects granular latency metrics: TTFT, TPOT, throughput
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 variant: str = "med",
                 device: str = "cuda",
                 dtype: str = "auto"):
        """
        Args:
            model_name: HuggingFace model identifier
            variant: "base" (FP16), "med" (8-bit), "cheap" (4-bit)
            device: "cuda" or "cpu"
            dtype: "auto", "float16", "bfloat16"
        """
        self.model_name = model_name
        self.variant = variant

        if torch.backends.mps.is_available():
            self.device = "mps"      # Mac GPU
        elif torch.cuda.is_available():
            self.device = "cuda"     # NVIDIA
        else:
            self.device = "cpu"      # Fallback

        self.dtype = dtype
        
        logger.info(f"Initializing {variant.upper()} server")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {dtype}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Load model with variant-specific settings
        try:
            if variant == "med":
                # 8-bit quantization
                logger.info("Loading model with 8-bit quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            elif variant == "cheap":
                # 4-bit quantization
                logger.info("Loading model with 4-bit quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_4bit=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            else:  # base
                # Full precision
                logger.info("Loading model in full precision...")
                torch_dtype = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "auto": torch.bfloat16
                }.get(dtype, torch.bfloat16)
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto"
                )
            
            self.model.eval()
            logger.info(f"Model loaded successfully")
            
            # Get model size
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model size: {total_params / 1e9:.2f}B parameters")
            
            # Warmup
            self._warmup()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _warmup(self):
        """Run dummy generations to warm up CUDA kernels"""
        logger.info("Warming up server (3 iterations)...")
        try:
            # Simple warmup prompt
            dummy_input = self.tokenizer("Hello", return_tensors="pt")
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            
            for i in range(5):
                self.model.generate(
                    **dummy_input, 
                    max_new_tokens=16, 
                    do_sample=False,
                    use_cache=True
                )
                self._synchronize_device()
            
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def _synchronize_device(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
    
    def generate(self,
                 prompt: str,
                 max_tokens: int = 128,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> Tuple[str, Dict]:
        """
        Generate response and collect detailed latency metrics
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            (generated_text, metrics_dict)
        """
        metrics = {}
        
        try:
            # Phase 1: Tokenization
            self._synchronize_device()
            t0_total = time.time()
            t0_tokenize = time.time()
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            metrics["input_length"] = input_length
            
            self._synchronize_device()
            t_tokenize = time.time() - t0_tokenize
            metrics["tokenize_ms"] = t_tokenize * 1000
            
            # Phase 2: Generation (Single Pass with TimingStreamer)
            streamer = TimingStreamer(self.tokenizer, self)
            
            self._synchronize_device()
            t0_generate = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    return_dict_in_generate=True,
                    output_scores=False,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    use_cache=True
                )
            
            self._synchronize_device()
            t_generate_end = time.time()
            
            # metrics calculation logic
            
            # TTFT (Time To First Token)
            if streamer.first_token_time:
                t_ttft = streamer.first_token_time - t0_generate
            else:
                # Fallback if no tokens generated or streamer failed
                t_ttft = t_generate_end - t0_generate
                
            metrics["ttft_ms"] = t_ttft * 1000
            
            # Extract generated text
            generated_ids = outputs.sequences[0, input_length:]
            output_length = len(generated_ids)
            metrics["output_length"] = output_length
            
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Decode Latency (Post-TTFT)
            # Total generation time = TTFT + Decode Time
            # Decode Time = Total generation time - TTFT
            total_gen_time = t_generate_end - t0_generate
            t_decode = max(total_gen_time - t_ttft, 0.0001)
            
            metrics["total_decode_latency_ms"] = t_decode * 1000
            
            # TPOT (Time Per Output Token)
            # We divide decode time by (output_length - 1) because the first token is covered by TTFT
            # If only 1 token, TPOT is 0 or undefined, but we can just use decode time
            # For robustness, we usually define TPOT for the decoding phase
            if output_length > 1:
                tpot = (t_decode * 1000) / (output_length - 1)
            else:
                tpot = 0.0 
                
            metrics["tpot_ms"] = tpot
            
            # Throughput
            metrics["throughput_tokens_per_sec"] = output_length / max(total_gen_time, 0.001)
            
            # Total end-to-end latency
            self._synchronize_device()
            t_total = time.time() - t0_total
            metrics["total_latency_ms"] = t_total * 1000
            
            # Model/variant info
            metrics["variant"] = self.variant
            metrics["model"] = self.model_name.split('/')[-1]
            
            # Success flag
            metrics["success"] = True
            
            # Clean up GPU memory
            del outputs, inputs, streamer
            torch.cuda.empty_cache()
            if self.device == "mps":
                torch.mps.empty_cache()
            
            return generated_text, metrics
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            metrics["success"] = False
            metrics["error"] = str(e)
            return "", metrics


if __name__ == "__main__":
    # Test server initialization and inference
    logger.info("Testing SingleVariantServer")
    
    try:
        # Initialize server
        server = SingleVariantServer(
            model_name="gpt2",  # Use small model for testing
            variant="med",
            device="cpu"  # Use CPU for testing
        )
        
        # Generate sample response
        test_prompt = "What is the capital of France?"
        generated_text, metrics = server.generate(
            prompt=test_prompt,
            max_tokens=32
        )
        
        logger.info("\n" + "="*70)
        logger.info("INFERENCE TEST RESULTS")
        logger.info("="*70)
        logger.info(f"Prompt: {test_prompt}")
        logger.info(f"Generated: {generated_text}")
        logger.info(f"\nMetrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key:30s}: {value:10.2f}")
            else:
                logger.info(f"  {key:30s}: {value}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
