# README.md
# SLO-Aware Task-Adaptive Compression: Baseline Evaluation Framework

Complete implementation of data pipeline, single-variant server, load generator, metrics collection, and evaluation for LLM serving research.

## Project Structure

```
.
├── preprocessing.py          # Data pipeline: MMLU + GSM8K download, preprocess, split
├── prompt_templates.py       # Prompt templates for different datasets
├── server.py                 # Single-variant LLM server (MED-only, 8-bit quantization)
├── load_generator.py         # Closed-loop load generator with fixed concurrency
├── metrics.py                # Metrics calculation: TTFT, TPOT, SLO compliance
├── evaluation.py             # Accuracy evaluation: exact match on MMLU/GSM8K
├── run_baseline_evaluation.py # End-to-end orchestration script
└── README.md                 # This file

data/
├── raw/                      # Raw datasets (auto-downloaded)
│   ├── mmlu/
│   └── gsm8k/
└── processed/                # Preprocessed datasets (JSONL format)
    ├── mmlu_processed.jsonl
    ├── gsm8k_processed.jsonl
    ├── train_data.jsonl
    ├── val_data.jsonl
    └── test_data.jsonl

results/
└── baseline_med/             # Baseline results
    ├── summary.json          # Overall summary
    ├── eval_results.json     # Accuracy metrics
    ├── metrics_concurrency_1.json
    ├── metrics_concurrency_8.json
    ├── requests_concurrency_1.jsonl
    └── requests_concurrency_8.jsonl
```

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU inference)
- 40GB+ GPU memory (for Llama-2-7B in 8-bit)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
torch==2.0.1
transformers==4.35.0
datasets==2.14.0
numpy==1.24.0
tiktoken==0.5.1
bitsandbytes==0.41.1  # For 8-bit quantization
```

## Quick Start

### Option 1: Full Pipeline (Preprocessing + Baseline)

```bash
# Run everything: preprocess, load tests, evaluation
python run_baseline_evaluation.py \
  --preprocess \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_requests 5000 \
  --concurrencies 1 2 4 8 16 32 \
  --output_dir results/baseline_med
```

**Expected runtime:** ~4-6 hours (depending on GPU and concurrency levels)

### Option 2: Preprocess Only

```bash
python preprocessing.py
```

**Output:**
- `data/processed/mmlu_processed.jsonl` (15,908 examples)
- `data/processed/gsm8k_processed.jsonl` (1,319 examples)
- `data/processed/train_data.jsonl` (10,345 examples)
- `data/processed/val_data.jsonl` (3,450 examples)
- `data/processed/test_data.jsonl` (3,455 examples)

### Option 3: Skip Preprocessing (Use Pre-Processed Data)

```bash
# If you have pre-processed data in data/processed/
python run_baseline_evaluation.py \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_requests 5000 \
  --concurrencies 1 4 8 16 \
  --output_dir results/baseline_med
```

### Option 4: Quick Test (Small Dataset)

```bash
# Test with only 100 examples for quick validation
python run_baseline_evaluation.py \
  --model_name gpt2 \
  --device cpu \
  --num_requests 100 \
  --concurrencies 1 2 \
  --data_subset 100 \
  --output_dir results/test_run
```

**Expected runtime:** ~5-10 minutes

## Module Documentation

### 1. preprocessing.py

Downloads, preprocesses, and splits MMLU + GSM8K datasets.

**Main class: `DataPreprocessor`**

```python
from preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(
    data_dir="data/raw",
    output_dir="data/processed"
)
train, val, test = preprocessor.run_pipeline()
```

**Features:**
- Automatic HuggingFace dataset download
- Subject-based difficulty classification (MMLU)
- Step-based difficulty classification (GSM8K)
- Stratified train/val/test split (60/20/20)
- Token counting via tiktoken

**Output format (JSONL):**
```json
{
  "dataset": "mmlu",
  "prompt": "What is the capital of France?\\nA) London\\nB) Paris\\nC) Berlin\\nD) Rome",
  "answer": "B",
  "subject": "geography",
  "difficulty": "easy",
  "input_length": 15,
  "output_length": 1
}
```

---

### 2. prompt_templates.py

Formats prompts for different datasets with system messages.

**Functions:**
- `build_prompt(example, dataset_type)` → (system, user, answer)
- `get_prompt_instructions(dataset_type)` → str
- `get_expected_format(dataset_type)` → str

**Supported datasets:**
- `mmlu`: Multiple choice (A/B/C/D)
- `gsm8k`: Math reasoning with step-by-step
- `sharegpt`: General conversational

---

### 3. server.py

Single-variant LLM server with 8-bit quantization.

**Main class: `SingleVariantServer`**

```python
from server import SingleVariantServer

server = SingleVariantServer(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    variant="med",  # "med" (8-bit), "cheap" (4-bit), "base" (FP16)
    device="cuda"
)

generated_text, metrics = server.generate(
    prompt="What is 2+2?",
    max_tokens=128
)
```

**Returned metrics:**
- `ttft_ms`: Time to first token
- `tpot_ms`: Time per output token
- `total_latency_ms`: Total generation time
- `throughput_tokens_per_sec`: Generation speed
- `input_length`: Input token count
- `output_length`: Output token count

---

### 4. load_generator.py

Closed-loop load generator maintaining constant concurrency.

**Main class: `ClosedLoopLoadGenerator`**

```python
from load_generator import ClosedLoopLoadGenerator

load_gen = ClosedLoopLoadGenerator(
    inference_func=server.generate,
    max_concurrency=16,
    num_requests=5000,
    data_loader=validation_data
)

metrics = load_gen.run()  # Returns list of RequestMetrics
load_gen.save_metrics("metrics.jsonl")
```

**Features:**
- Fixed concurrency control
- Per-request timing (queue wait + inference)
- Thread-safe metric collection
- Progress logging every 100 requests

**RequestMetrics properties:**
- `e2e_latency_ms`: End-to-end latency
- `queue_wait_time_ms`: Queue waiting time
- `inference_time_ms`: Actual inference time
- `ttft_ms`, `tpot_ms`: From inference metrics

---

### 5. metrics.py

Computes percentiles, throughput, and SLO compliance.

**Main class: `MetricsCalculator`**

```python
from metrics import MetricsCalculator

calc = MetricsCalculator(request_metrics)
all_metrics = calc.compute_all_metrics()
calc.print_report()
calc.save_metrics("metrics.json")
```

**Metrics computed:**
- **Summary**: request counts, success rate, throughput, SLO compliance
- **TTFT**: p50, p75, p90, p95, p99, mean, std
- **TPOT**: Same percentiles
- **E2E Latency**: Same percentiles
- **Queue Wait**: Same percentiles

**Output (JSON):**
```json
{
  "summary": {
    "total_requests": 5000,
    "successful_requests": 4998,
    "success_rate": 0.9996,
    "throughput_tokens_per_sec": 1287.45,
    "slo_compliance": 0.9832,
    "slo_violations": 85
  },
  "ttft": {
    "p99": 189.45,
    "p95": 134.56,
    "mean": 78.34
  },
  "tpot": {...},
  "e2e_latency": {...}
}
```

---

### 6. evaluation.py

Exact match accuracy evaluation on MMLU and GSM8K.

**Main classes:**
- `EvaluationMetrics`: Static methods for exact match
- `HeldOutEvaluator`: Batch evaluation on test set

```python
from evaluation import HeldOutEvaluator

evaluator = HeldOutEvaluator(server, test_data)
results = evaluator.evaluate()

print(f"MMLU: {results['mmlu']['em']*100:.2f}%")
print(f"GSM8K: {results['gsm8k']['em']*100:.2f}%")
print(f"Overall: {results['overall']['em']*100:.2f}%")
```

**Evaluation metrics:**
- MMLU: Extracts A/B/C/D letter from response
- GSM8K: Extracts final numeric answer using regex
- Returns: accuracy, EM, correct_count, total_count

---

### 7. run_baseline_evaluation.py

End-to-end orchestration: preprocessing → server init → load tests → evaluation

**Usage:**
```bash
python run_baseline_evaluation.py \
  --preprocess \
  --model_name meta-llama/Llama-2-7b-chat-hf \
  --num_requests 5000 \
  --concurrencies 1 2 4 8 16 32 \
  --device cuda \
  --output_dir results/baseline_med
```

**Arguments:**
- `--preprocess`: Run preprocessing (download/process datasets)
- `--model_name`: HuggingFace model ID
- `--num_requests`: Requests per concurrency level (default: 5000)
- `--concurrencies`: List of concurrency levels to test
- `--device`: cuda or cpu
- `--output_dir`: Where to save results

**Output files:**
- `summary.json`: Overall summary with all configs
- `eval_results.json`: Accuracy breakdown by dataset
- `metrics_concurrency_N.json`: Detailed metrics for each concurrency
- `requests_concurrency_N.jsonl`: Per-request logs

## Expected Output

### Terminal Output
```
================================================================================
END-TO-END BASELINE EVALUATION: MED-ONLY SERVER (8-BIT QUANTIZATION)
================================================================================

[STEP 1] LOADING DATA
  Loaded data: train=10345, val=3450, test=3455

[STEP 2] INITIALIZING SERVER
  Model loaded successfully

[STEP 3] RUNNING LOAD TESTS
  >>> Testing with concurrency=1
  Progress: 100/5000 (245.2 req/sec, ETA: 19.4s)
  Progress: 500/5000 (248.1 req/sec, ETA: 18.2s)
  ...
  Load test complete in 20.3s

================================================================================
LOAD TEST RESULTS (Concurrency 1)
================================================================================

SUMMARY:
  Total Requests:          5000
  Successful:              4998
  Success Rate:           99.96%
  Total Duration:         20.30 seconds
  Throughput:           1287.45 tokens/sec
  SLO Compliance:         98.32%
  SLO Violations:            85

TTFT (Time-to-First-Token) in milliseconds:
  P99:    189.45 ms
  P95:    134.56 ms
  Mean:    78.34 ms (±45.67)

TPOT (Time-Per-Output-Token) in milliseconds:
  P99:     28.45 ms
  P95:     21.23 ms
  Mean:    15.67 ms (±6.78)

E2E Latency (End-to-End) in milliseconds:
  P99:   2134.45 ms
  P95:   1567.89 ms
  Mean:  1034.23 ms (±456.78)

================================================================================

[STEP 4] EVALUATING ACCURACY
Generated 3455/3455 predictions

MMLU Results:
  Accuracy: 68.42%
  Correct:  2283/3335

GSM8K Results:
  Accuracy: 71.45%
  Correct:  982/1375

OVERALL Results:
  Accuracy: 69.12%
  Correct:  3265/4710

================================================================================
FINAL SUMMARY REPORT
================================================================================

Load Test Results by Concurrency:
Concurrency  Throughput       TTFT P99    TPOT P95    E2E P99     SLO Compl
-------- -------------------- ------------ ------------ ------------ ----------
1              1287.45        189.45       21.23        2134.45      98.3%
8              4256.78        523.45       45.67        7123.45      94.1%
16             5234.56        789.34       89.34        9876.54      87.6%

Accuracy Results:
  MMLU        68.42% (2283/3335)
  GSM8K       71.45% (982/1375)
  OVERALL     69.12% (3265/4710)

================================================================================
BASELINE EVALUATION COMPLETE
Results saved to: results/baseline_med
================================================================================
```

### JSON Output (summary.json)
```json
{
  "load_test_summary": [
    {
      "concurrency": 1,
      "num_requests": 5000,
      "duration_sec": 20.3,
      "success_rate": 0.9996,
      "throughput_tokens_per_sec": 1287.45,
      "ttft_p99_ms": 189.45,
      "tpot_p95_ms": 21.23,
      "e2e_p99_ms": 2134.45,
      "slo_compliance": 0.9832,
      "slo_violations": 85
    }
  ],
  "eval_results": {
    "mmlu": {
      "accuracy": 0.6842,
      "em": 0.6842,
      "correct_count": 2283,
      "total_count": 3335
    },
    "gsm8k": {
      "accuracy": 0.7145,
      "em": 0.7145,
      "correct_count": 982,
      "total_count": 1375
    },
    "overall": {
      "accuracy": 0.6912,
      "em": 0.6912,
      "correct_count": 3265,
      "total_count": 4710
    }
  }
}
```

## Next Steps

After completing baseline evaluation:

1. **Add other variants**: Implement BASE (FP16) and CHEAP (4-bit) servers
2. **Build router**: Train difficulty classifier on MMLU/GSM8K
3. **Implement scheduling**: Add queue simulation for SLO modeling
4. **Online adaptation**: Periodic router retraining from logs
5. **Comparative analysis**: Compare cost-quality-latency tradeoffs

## Troubleshooting

### Out of Memory
- Reduce `max_tokens` in server.generate()
- Use smaller model (e.g., Llama-2-7b instead of 13b)
- Reduce batch sizes in load test

### Slow Download
- MMLU: ~500MB, GSM8K: ~10MB
- First run downloads from HuggingFace
- Subsequent runs use cached data

### GPU Issues
- Check CUDA version: `nvidia-smi`
- Install bitsandbytes: `pip install bitsandbytes`
- Use CPU fallback: `--device cpu`

## Citation

If you use this framework, please cite:

```bibtex
@inproceedings{pant2026sloaware,
  title={SLO-Aware Task-Adaptive Model Compression for LLM Serving},
  author={Pant, Ayush},
  year={2026}
}
```

## License

MIT License - See LICENSE file for details
