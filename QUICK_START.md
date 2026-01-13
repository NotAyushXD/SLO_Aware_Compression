# QUICK_START.md
# Quick Start Guide: Baseline Evaluation

Fast setup and execution guide for SLO-Aware compression baseline.

## 1. Environment Setup (5 minutes)

```bash
# Clone/download repository
cd slo-aware-compression

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (for model access)
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
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
```

## 2. Run Full Baseline (Option A: Complete)

```bash
# This will:
# 1. Download MMLU (500MB) and GSM8K (10MB)
# 2. Preprocess and split data
# 3. Initialize MED-only server (8-bit Llama-2-7B)
# 4. Run load tests at 6 concurrency levels with 5K requests each
# 5. Evaluate accuracy on 5K held-out test set

python run_baseline_evaluation.py \
  --preprocess \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_requests 5000 \
  --concurrencies 1 2 4 8 16 32 \
  --device cuda \
  --output_dir results/baseline_med

# Expected time: 4-6 hours
# GPU memory: 40GB+ (NVIDIA A100 or RTX 6000)
```

## 3. Faster Option (Option B: Skip Preprocessing)

If you've already downloaded datasets:

```bash
# Just run load tests + evaluation (skip preprocessing)
python run_baseline_evaluation.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_requests 5000 \
  --concurrencies 1 4 8 16 \
  --device cuda \
  --output_dir results/baseline_med

# Expected time: 2-3 hours
```

## 4. Quick Test (Option C: Validation)

Test everything with small datasets before full run:

```bash
# Test with 100 examples, GPT-2, CPU
python run_baseline_evaluation.py \
  --preprocess \
  --model_name gpt2 \
  --num_requests 100 \
  --concurrencies 1 2 \
  --device cpu \
  --data_subset 100 \
  --output_dir results/test_run

# Expected time: 5-10 minutes
# This validates entire pipeline without long wait times
```

## 5. Check Results

```bash
# View summary
cat results/baseline_med/summary.json | python -m json.tool

# View accuracy
cat results/baseline_med/eval_results.json | python -m json.tool

# View specific concurrency metrics
cat results/baseline_med/metrics_concurrency_16.json | python -m json.tool
```

## 6. Understand Output Files

After running `run_baseline_evaluation.py`, you'll get:

```
results/baseline_med/
├── summary.json
│   └── Complete summary: load tests across all concurrencies + eval results
│
├── eval_results.json
│   └── Accuracy breakdown:
│       - MMLU: exact match on multiple choice
│       - GSM8K: exact match on math problems
│       - OVERALL: combined accuracy
│
├── metrics_concurrency_1.json
├── metrics_concurrency_4.json
├── metrics_concurrency_8.json
├── metrics_concurrency_16.json
│   └── Each contains:
│       - summary: request counts, success rate, throughput, SLO compliance
│       - ttft: percentiles for time-to-first-token (ms)
│       - tpot: percentiles for time-per-output-token (ms)
│       - e2e_latency: percentiles for end-to-end latency (ms)
│       - queue_wait: percentiles for queue waiting time (ms)
│
├── requests_concurrency_1.jsonl
├── requests_concurrency_4.jsonl
├── requests_concurrency_8.jsonl
├── requests_concurrency_16.jsonl
    └── Per-request logs (one JSON per line):
        - request_id, submit_time, start_time, end_time
        - e2e_latency_ms, queue_wait_time_ms
        - inference_metrics (ttft, tpot, output_length)
```

## 7. Interpret Key Metrics

### Throughput (tokens/sec)
- Higher is better
- Baseline MED: ~1200-5200 tokens/sec (1-32 concurrency)

### TTFT P99 (milliseconds)
- Time to first token at 99th percentile
- Lower is better (< 200ms is excellent)
- Baseline MED: ~190ms at concurrency 1

### TPOT P95 (milliseconds)
- Time per output token at 95th percentile
- Lower is better (< 30ms is good)
- Baseline MED: ~20ms at concurrency 1

### SLO Compliance (%)
- Percentage of requests meeting SLO targets
- Target: TTFT < 200-500ms depending on difficulty
- Baseline MED: ~98% at concurrency 1

### Accuracy / EM
- Exact match on test set
- MMLU: Multiple choice (A/B/C/D)
- GSM8K: Numeric answer extraction
- Baseline MED: ~69% overall

## 8. Example Results Interpretation

```
Load Test at Concurrency 16:
  ✓ Throughput: 5,234 tokens/sec (excellent for batch)
  ✓ TTFT P99: 789ms (acceptable for complex tasks)
  ✓ TPOT P95: 89ms (slower per token due to contention)
  ✓ E2E P99: 9,876ms (queuing delays at high concurrency)
  ⚠ SLO Compliance: 87.6% (some timeout violations)

→ Conclusion: MED variant saturates at concurrency 16.
  For production, keep concurrency ≤ 8 OR add cheaper variants.
```

## 9. Modify Concurrency Test

Test only specific concurrency levels:

```bash
python run_baseline_evaluation.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_requests 5000 \
  --concurrencies 1 8 16  # Only test these three
  --output_dir results/baseline_med
```

## 10. Reduce Data for Faster Testing

```bash
# Use only first 500 examples from each split
python run_baseline_evaluation.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_requests 500  # Fewer requests
  --data_subset 500   # Smaller dataset
  --concurrencies 1 8  # Fewer concurrency levels
  --output_dir results/baseline_med
```

## 11. Troubleshooting

### Error: Out of Memory
```bash
# Use CPU instead of GPU
python run_baseline_evaluation.py \
  --device cpu \
  --num_requests 1000  # Reduce requests
  ...
```

### Error: Model not found
```bash
# Ensure HuggingFace login and model access
huggingface-cli login
# Or try a smaller model:
--model_name gpt2
--model_name distilgpt2
```

### Error: Data not found
```bash
# Run preprocessing explicitly
python preprocessing.py
# Then run baseline without --preprocess flag
```

### Slow Download
- First run downloads datasets from HuggingFace (~500MB)
- Cached in `data/raw/` for subsequent runs
- Expected: 5-15 minutes depending on internet speed

## 12. Next Steps After Baseline

1. **Analyze results**: Review summary.json and eval_results.json
2. **Optimize concurrency**: Find sweet spot between throughput and latency
3. **Reduce cost**: Test 4-bit (CHEAP) or FP16 (BASE) variants
4. **Build router**: Classify tasks by difficulty for variant selection
5. **Online adaptation**: Retrain router periodically on live logs

## 13. Key Files to Understand

| File | Purpose | Learn About |
|------|---------|-------------|
| `preprocessing.py` | Data pipeline | Download, preprocess, split MMLU/GSM8K |
| `server.py` | LLM inference | 8-bit quantization, generation, metrics |
| `load_generator.py` | Load testing | Closed-loop concurrency, request queuing |
| `metrics.py` | Metrics collection | Percentiles, SLO compliance, throughput |
| `evaluation.py` | Accuracy eval | Exact match scoring for MMLU/GSM8K |
| `run_baseline_evaluation.py` | Orchestration | End-to-end pipeline control |

## 14. Contact & Issues

- Check GPU memory: `nvidia-smi`
- Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify datasets: `ls -lah data/processed/`
- Check results: `cat results/baseline_med/summary.json`

---

**Ready to start?** Run Option C first to validate setup, then Option A for full baseline!
