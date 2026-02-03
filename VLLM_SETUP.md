# vLLM Backend (Optional)

This patch adds an optional vLLM backend to improve throughput and tail latency.

## Why vLLM?
- vLLM uses *continuous batching* and optimized KV-cache management.
- Under load, this typically reduces queue buildup and improves p95/p99 TTFT compared to a vanilla HF server.

## Install (optional)
vLLM is NOT required for the HF backend.

1) Install base deps (HF backend):
```bash
pip install -r requirements_kaggle.txt
```

2) Install vLLM (optional):
```bash
pip install -r requirements_vllm.txt
```

> Note: vLLM support depends on CUDA, GPU type, and Python version.  
> If your environment is Python 3.12 and `pip install vllm` fails, use Python 3.10/3.11.

## Running with vLLM

### Base (fp16) — paper fairness baseline
```bash
python run_baseline_evaluation.py \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant base \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 200 \
  --concurrencies 1 4 \
  --seed 42 \
  --output_dir ./runs/vllm_base_fp16
```

### Med (int8) — **requires a pre-quantized checkpoint**
For `--variant med` we require you to pass a model path/name that is already quantized
(e.g., AWQ or GPTQ). This is what we called **Option 1**.

```bash
python run_baseline_evaluation.py \
  --backend vllm \
  --model <tokenizer-compatible-base-model> \
  --variant med \
  --vllm_model_override <PATH_OR_HF_ID_TO_QUANTIZED_CHECKPOINT> \
  --vllm_quantization awq \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 200 \
  --concurrencies 1 4 \
  --seed 42 \
  --output_dir ./runs/vllm_med_quant
```

### Cheap (int4)
If your vLLM build supports bitsandbytes quantization you can try:
```bash
python run_baseline_evaluation.py \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant cheap \
  --vllm_quantization bitsandbytes \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 200 \
  --concurrencies 1 4 \
  --seed 42 \
  --output_dir ./runs/vllm_cheap
```

If it fails, use AWQ/GPTQ quantized checkpoints instead.

## HF vs vLLM comparison runner
```bash
python run_backend_comparison.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --variant base \
  --prompt_mode slo \
  --num_requests 200 \
  --concurrencies 1 4 \
  --max_batch_size 8 \
  --batch_wait_ms 8 \
  --seed 42 \
  --output_dir ./runs/compare_hf_vs_vllm_fp16
```

This produces:
- `backend_comparison_summary.csv`
- `backend_comparison_summary.json`

Both are paper-friendly inputs for tables/plots.
