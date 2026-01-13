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
        self.device = device
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
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
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
        
        Metrics include:
            - ttft_ms: Time to first token (milliseconds)
            - tpot_ms: Time per output token (milliseconds)
            - total_latency_ms: Total generation time
            - throughput_tokens_per_sec: Generation throughput
            - input_length: Number of input tokens
            - output_length: Number of output tokens
        """
        metrics = {}
        
        try:
            # Phase 1: Tokenization
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
            
            t_tokenize = time.time() - t0_tokenize
            metrics["tokenize_ms"] = t_tokenize * 1000
            
            # Phase 2: Time to first token (TTFT)
            # Generate only the first token to measure latency
            t0_first = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_p=top_p,
                    return_dict_in_generate=True,
                    output_scores=False,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            t_ttft = time.time() - t0_first
            metrics["ttft_ms"] = t_ttft * 1000
            
            # Phase 3: Full generation (remaining tokens)
            t0_decode = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    return_dict_in_generate=True,
                    output_scores=False,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            t_decode = time.time() - t0_decode
            
            # Extract generated text
            generated_ids = outputs.sequences[0, input_length:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # Calculate metrics
            output_length = len(generated_ids)
            metrics["output_length"] = output_length
            
            # Total decode time (in milliseconds)
            metrics["total_decode_latency_ms"] = t_decode * 1000
            
            # Time per output token (TPOT) - average of all tokens
            tpot = (t_decode * 1000) / max(output_length, 1)
            metrics["tpot_ms"] = tpot
            
            # Throughput
            metrics["throughput_tokens_per_sec"] = output_length / max(t_decode, 0.001)
            
            # Total end-to-end latency
            t_total = time.time() - t0_total
            metrics["total_latency_ms"] = t_total * 1000
            
            # Model/variant info
            metrics["variant"] = self.variant
            metrics["model"] = self.model_name.split('/')[-1]
            
            # Success flag
            metrics["success"] = True
            
            # Clean up GPU memory
            del outputs, inputs
            torch.cuda.empty_cache()
            
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
