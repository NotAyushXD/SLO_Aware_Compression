# SLO-aware evaluation patch bundle (Option A, p95 primary)

This bundle contains patched versions of the baseline scripts to produce paper-grade, reproducible results for:
- Accuracy evaluation (GSM8K + MMLU)
- SLO-mode quality guardrail (SLO prompting should not collapse correctness)
- Closed-loop load testing with micro-batching, TTFT/TPOT measurement, and dynamic SLO calibration

## What changed (high level)

### 1) Correct TTFT instrumentation (server.py)
- True TTFT measured as **prefill + first decode** using a streamer timestamp.
- Exposes timing breakdown:
  - `tokenize_ms`, `lock_wait_ms`, `ttft_model_ms`, `ttft_infer_ms`, `scheduler_wait_ms`
- Implements paper definition **Option A (queue-inclusive)**:
  - `ttft_ms = scheduler_wait_ms + ttft_infer_ms`

### 1b) Tail-latency mitigation under mixed workloads (server.py)
- Implements **short-job-first** batch-key selection with an **anti-starvation** timer.
  This reduces head-of-line blocking where long-form GSM8K requests can make
  short MMLU requests miss TTFT SLOs under load.
- Adds an **adaptive batching window** for long jobs (default 50ms) to reduce
  lock-wait-driven TTFT tails by increasing the chance that concurrent requests
  enter the same micro-batch.

### 2) SLO calibration + sensitivity (metrics.py + run_baseline_evaluation.py)
- Calibrates SLO thresholds from a baseline run (default `concurrency=1`).
- Saves percentile profiles **p90/p95/p99** to `slo_thresholds.json`.
- Uses **p95 as primary** for reported compliance; p90/p99 are saved for sensitivity analysis.

### 3) Reproducibility + split integrity (preprocessing.py)
- Preserves official test splits (no leakage into calibration).
- Deterministic stratified split by difficulty for internal train/val.

### 4) SLO prompt no longer destroys GSM8K (prompt_templates.py)
- Adds explicit formatting guidance (exact `FINAL_ANSWER:` tag) and two compact examples.
- Keeps answers compact while retaining enough reasoning budget.
- Increases SLO GSM8K max_new_tokens slightly (uniform across difficulty) to reduce truncation-caused format failures.

### 5) Reduced head-of-line blocking under load (server.py)
- Replaces strict FIFO batching selection with a simple **short-job-first** policy (by `max_tokens`) plus an **anti-starvation** fallback.
- This improves tail TTFT for short requests (e.g., MMLU) when mixed with long-form GSM8K.

## How to run

### Quick smoke suite
```bash
bash run_all.sh meta-llama/Llama-3.1-8B-Instruct /kaggle/working/outputs_paper_smoke med auto 0
```

### Or run individual commands
See `run_all.sh`.

## Methodology note
- `METHODOLOGY_SLO_CALIBRATION.md` describes the exact TTFT definition and SLO calibration protocol.

## Dependency notes (Kaggle/Colab)
- `requirements_kaggle.txt` contains a recommended pin set.
  - Pin `pyarrow<21` to avoid the `PyExtensionType` removal that can break `datasets`.
  - If you use med/cheap (bitsandbytes), pin `triton<3` to avoid `triton.ops` import errors.
