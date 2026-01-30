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

### 2) SLO calibration + sensitivity (metrics.py + run_baseline_evaluation.py)
- Calibrates SLO thresholds from a baseline run (default `concurrency=1`).
- Saves percentile profiles **p90/p95/p99** to `slo_thresholds.json`.
- Uses **p95 as primary** for reported compliance; p90/p99 are saved for sensitivity analysis.

### 3) Reproducibility + split integrity (preprocessing.py)
- Preserves official test splits (no leakage into calibration).
- Deterministic stratified split by difficulty for internal train/val.

### 4) SLO prompt no longer destroys GSM8K (prompt_templates.py)
- Removes overly restrictive "<= 6 lines" constraint.
- Keeps answers compact while retaining enough reasoning budget.

## How to run

### Quick smoke suite
```bash
bash run_all.sh meta-llama/Llama-3.1-8B-Instruct /kaggle/working/outputs_paper_smoke med auto 0
```

### Or run individual commands
See `run_all.sh`.

## Methodology note
- `METHODOLOGY_SLO_CALIBRATION.md` describes the exact TTFT definition and SLO calibration protocol.
