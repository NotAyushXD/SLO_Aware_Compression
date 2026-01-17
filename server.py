# server.py
"""
Single-variant LLM server (MED-only: 8-bit quantization)
Handles inference requests and collects detailed latency metrics

Optimizations:
- Proper GPU detection with CUDA verification for quantization
- Accurate TTFT/TPOT timing with correct synchronization
- GPU utilization logging for diagnostics
- Optimized memory management (reduced per-request overhead)
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, Tuple, Optional
import logging
import gc
import os

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
    
    IMPORTANT: Synchronization happens BEFORE recording time to ensure
    GPU operations have completed when we capture the timestamp.
    """
    def __init__(self, tokenizer, sync_fn):
        self.tokenizer = tokenizer
        self.sync_fn = sync_fn  # Function to synchronize device
        self.first_token_time = None
        self.token_count = 0
        
    def put(self, value):
        """
        Callback for new tokens.
        value: tensor of new tokens (shape [batch, 1])
        """
        # Synchronize BEFORE recording time to ensure GPU ops complete
        if self.first_token_time is None:
            self.sync_fn()
            self.first_token_time = time.perf_counter()  # Use perf_counter for precision
        
        self.token_count += 1
        
    def end(self):
        pass


class GPUMonitor:
    """Monitor and log GPU utilization metrics"""
    
    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is truly available and functional"""
        if not torch.cuda.is_available():
            return False
        try:
            # Actually try to use CUDA
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_gpu_info() -> Dict:
        """Get current GPU memory and utilization info"""
        info = {"cuda_available": False}
        
        if not torch.cuda.is_available():
            return info
            
        try:
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(0)
            
            # Memory info (in GB)
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
        """Log current GPU status"""
        info = GPUMonitor.get_gpu_info()
        if info.get("cuda_available"):
            logger.info(f"{prefix}GPU: {info.get('device_name', 'Unknown')}")
            logger.info(f"{prefix}  Memory: {info.get('memory_allocated_gb', 0):.2f}GB allocated, "
                       f"{info.get('memory_free_gb', 0):.2f}GB free / {info.get('memory_total_gb', 0):.2f}GB total")
        else:
            logger.warning(f"{prefix}CUDA not available")


class SingleVariantServer:
    """
    Single-variant LLM server for inference
    Collects granular latency metrics: TTFT, TPOT, throughput
    
    Optimizations:
    - Proper CUDA detection with fallback handling
    - Consistent warmup/generation parameters
    - Accurate timing with proper synchronization
    - Memory-efficient per-request handling
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 variant: str = "med",
                 device: str = "auto",
                 dtype: str = "auto"):
        """
        Args:
            model_name: HuggingFace model identifier
            variant: "base" (FP16), "med" (8-bit), "cheap" (4-bit)
            device: "auto", "cuda", "mps", or "cpu"
            dtype: "auto", "float16", "bfloat16"
        """
        self.model_name = model_name
        self.variant = variant
        self.dtype = dtype
        
        # Determine device with proper CUDA verification
        self.device = self._detect_device(device, variant)
        
        # Track if we need per-request memory cleanup (disabled by default for speed)
        self._cleanup_per_request = False
        self._request_count = 0
        self._cleanup_interval = 50  # Cleanup every N requests
        
        logger.info("="*70)
        logger.info(f"Initializing {variant.upper()} server")
        logger.info("="*70)
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Variant: {variant}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Dtype: {dtype}")
        
        # Log GPU status
        GPUMonitor.log_gpu_status("  ")
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Load model with variant-specific settings
        self._load_model()
        
        # Log post-load GPU status
        logger.info("Post-load GPU status:")
        GPUMonitor.log_gpu_status("  ")
        
        # Warmup with consistent parameters
        self._warmup()
        
    def _detect_device(self, requested_device: str, variant: str) -> str:
        """
        Detect the best available device with proper validation.
        
        For quantized models (med/cheap), CUDA is required for bitsandbytes.
        Falls back gracefully with warnings.
        """
        # Check CUDA availability properly
        cuda_available = GPUMonitor.is_cuda_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        if requested_device == "auto":
            if cuda_available:
                selected = "cuda"
            elif mps_available and variant == "base":
                # MPS doesn't support quantization
                selected = "mps"
            else:
                selected = "cpu"
        else:
            selected = requested_device
        
        # Validate device for quantized models
        if variant in ["med", "cheap"]:
            if selected != "cuda":
                if cuda_available:
                    logger.warning(f"Quantized models require CUDA. Switching from '{selected}' to 'cuda'")
                    selected = "cuda"
                else:
                    logger.error("="*70)
                    logger.error("CRITICAL: Quantized models (8-bit/4-bit) require CUDA!")
                    logger.error("bitsandbytes does not support CPU or MPS.")
                    logger.error("Options:")
                    logger.error("  1. Use variant='base' for CPU/MPS")
                    logger.error("  2. Run on a CUDA-enabled system")
                    logger.error("="*70)
                    raise RuntimeError("CUDA required for quantized models")
        
        return selected
    
    def _load_tokenizer(self):
        """Load tokenizer with proper configuration"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded: {self.tokenizer.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self):
        """Load model with variant-specific quantization settings"""
        try:
            if self.variant == "med":
                # 8-bit quantization using BitsAndBytesConfig (recommended API)
                logger.info("Loading model with 8-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
            elif self.variant == "cheap":
                # 4-bit quantization
                logger.info("Loading model with 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                
            else:  # base - full precision
                logger.info("Loading model in full precision...")
                torch_dtype = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "auto": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                }.get(self.dtype, torch.float16)
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
            # Get model size
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model size: {total_params / 1e9:.2f}B parameters")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _warmup(self):
        """
        Run warmup generations to initialize CUDA kernels.
        
        IMPORTANT: Uses same generation parameters as actual inference
        to ensure consistent kernel compilation.
        """
        logger.info("Warming up server (5 iterations with sampling)...")
        try:
            # Use a realistic prompt for warmup
            warmup_prompt = "[INST] Hello [/INST]"
            dummy_input = self.tokenizer(warmup_prompt, return_tensors="pt")
            
            # Move to appropriate device
            if self.device == "cuda":
                dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
            elif self.device == "mps":
                dummy_input = {k: v.to("mps") for k, v in dummy_input.items()}
            
            for i in range(5):
                with torch.no_grad():
                    # Use SAME parameters as actual generation for consistency
                    self.model.generate(
                        **dummy_input, 
                        max_new_tokens=16, 
                        do_sample=True,  # Match actual generation
                        temperature=0.7,
                        top_p=0.9,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                self._synchronize_device()
            
            logger.info("Warmup complete")
            
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def _synchronize_device(self):
        """Synchronize device to ensure all GPU operations complete"""
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()
    
    def _maybe_cleanup_memory(self, force: bool = False):
        """
        Conditionally cleanup GPU memory.
        
        Only runs every N requests to reduce overhead.
        Set force=True for immediate cleanup.
        """
        self._request_count += 1
        
        if force or (self._cleanup_per_request and 
                     self._request_count % self._cleanup_interval == 0):
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
    
    def generate(self,
                 prompt: str,
                 max_tokens: int = 128,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> Tuple[str, Dict]:
        """
        Generate response and collect detailed latency metrics.
        
        Timing methodology:
        - TTFT: Time from generation start until first token is ready
        - TPOT: Average time per output token (excluding first token)
        - Throughput: Total tokens / total generation time
        
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
            # ============================================================
            # Phase 1: Tokenization
            # ============================================================
            self._synchronize_device()
            t0_total = time.perf_counter()
            t0_tokenize = time.perf_counter()
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            metrics["input_length"] = input_length
            
            self._synchronize_device()
            t_tokenize = time.perf_counter() - t0_tokenize
            metrics["tokenize_ms"] = t_tokenize * 1000
            
            # ============================================================
            # Phase 2: Generation with accurate timing
            # ============================================================
            # Create streamer with sync function reference
            streamer = TimingStreamer(self.tokenizer, self._synchronize_device)
            
            self._synchronize_device()
            t0_generate = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                    use_cache=True
                )
            
            self._synchronize_device()
            t_generate_end = time.perf_counter()
            
            # ============================================================
            # Phase 3: Calculate metrics
            # ============================================================
            
            # TTFT (Time To First Token)
            if streamer.first_token_time is not None:
                t_ttft = streamer.first_token_time - t0_generate
            else:
                # Fallback if no tokens generated
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
            
            # Total generation time
            total_gen_time = t_generate_end - t0_generate
            
            # Decode latency (time after first token)
            t_decode = max(total_gen_time - t_ttft, 0.0001)
            metrics["total_decode_latency_ms"] = t_decode * 1000
            
            # TPOT (Time Per Output Token) - for tokens after the first
            if output_length > 1:
                tpot = (t_decode * 1000) / (output_length - 1)
            else:
                tpot = 0.0
            metrics["tpot_ms"] = tpot
            
            # Throughput
            metrics["throughput_tokens_per_sec"] = output_length / max(total_gen_time, 0.001)
            
            # Total end-to-end latency (includes tokenization)
            self._synchronize_device()
            t_total = time.perf_counter() - t0_total
            metrics["total_latency_ms"] = t_total * 1000
            
            # Model/variant info
            metrics["variant"] = self.variant
            metrics["model"] = self.model_name.split('/')[-1]
            metrics["device"] = self.device
            
            # Success flag
            metrics["success"] = True
            
            # ============================================================
            # Phase 4: Memory management (optimized)
            # ============================================================
            # Only delete large objects, skip per-request cache clearing
            del outputs
            self._maybe_cleanup_memory()
            
            return generated_text, metrics
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            metrics["success"] = False
            metrics["error"] = str(e)
            return "", metrics
    
    def get_gpu_stats(self) -> Dict:
        """Get current GPU statistics for monitoring"""
        return GPUMonitor.get_gpu_info()
    
    def force_memory_cleanup(self):
        """Force GPU memory cleanup (use sparingly)"""
        self._maybe_cleanup_memory(force=True)
        GPUMonitor.log_gpu_status("After cleanup: ")


if __name__ == "__main__":
    # Test server initialization and inference
    logger.info("Testing SingleVariantServer")
    
    try:
        # Check GPU availability first
        gpu_info = GPUMonitor.get_gpu_info()
        logger.info(f"GPU Info: {gpu_info}")
        
        # Initialize server (will auto-detect device)
        server = SingleVariantServer(
            model_name="gpt2",  # Use small model for testing
            variant="base",    # Use base for CPU/MPS compatibility
            device="auto"
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
        
        # Show GPU stats
        logger.info("\nFinal GPU Status:")
        server.get_gpu_stats()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
