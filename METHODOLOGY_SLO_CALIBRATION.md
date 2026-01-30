# Paper-ready methodology: SLO calibration (Option A TTFT)

This bundle is intended to support paper-quality reporting for **SLO-aware inference** (routing / adaptive compression) by making latency measurement and SLO calibration explicit, reproducible, and aligned with common serving practice.

## 1) Latency definitions

### TTFT (Option A / service-facing)
We define **TTFT** as the time from when a request is admitted to the server until the **first output token** is available to be returned.

In this implementation the paper-facing TTFT is:

**TTFT_A (ms) = scheduler_wait_ms + ttft_infer_ms**

Where:
- **scheduler_wait_ms**: time the request spends waiting to be **dequeued by the micro-batching scheduler** (enqueue → dequeue).
- **ttft_infer_ms**: time from the start of the server inference function to first token, including:
  - tokenization + tensor preparation (+ transfer to device)
  - waiting on the model generation lock (GPU contention)
  - model prefill
  - first decode step

We also expose components to support ablations and transparency:
- **tokenize_ms**: tokenization + input preparation time.
- **lock_wait_ms**: time waiting to acquire the generation lock.
- **ttft_model_ms**: `model.generate()` start → first token (prefill + first decode), excluding tokenization and lock wait.
- **ttft_infer_ms**: inference start → first token (includes tokenization + lock wait + model time).
- **ttft_ms**: the final **Option A** TTFT used for SLOs.

### TPOT (time-per-output-token)
We define **TPOT** as the average decode time per generated token after the first token:

**TPOT (ms) = (T_generate_total − TTFT_model) / (output_tokens − 1)**

Notes:
- For single-token outputs (e.g., MMLU is forced to 1 token), TPOT may be reported as 0.
- TTFT is dominated by prefill + first decode; TPOT reflects steady-state decoding.

### Total latency
Server-side `total_latency_ms` is aligned with the same service boundary and includes `scheduler_wait_ms`.

Client-side end-to-end latency is additionally computed by the load generator (`e2e_latency_ms`), which also includes client thread scheduling effects.

## 2) Dataset splits and evaluation pools

To avoid leakage and make accuracy results defensible:
- **Official test splits are never used for calibration**.
- GSM8K: we preserve the official `test` split; non-test data comes from the official `train` split.
- MMLU: we preserve the official `test` split; non-test data comes from the official `validation` split.

From the combined non-test pool, we create an internal `train`/`val` split using a deterministic RNG (`--seed`) and stratification by difficulty bucket.

- **Load tests + SLO calibration** draw from `val` by default.
- **Accuracy evaluation** uses `test_data.jsonl` (which includes only official test items).

## 3) SLO calibration protocol (p95 primary + p90/p99 sensitivity)

We follow a standard serving methodology:

1. **Calibration run** at **concurrency = 1** (unloaded baseline).
2. Compute per-difficulty thresholds using TTFT_A (`ttft_ms`) and TPOT (`tpot_ms`) at selected percentiles.
3. Save calibrated SLOs to `slo_thresholds.json`.
4. Run additional concurrencies using the fixed thresholds to measure degradation under load.

This bundle calibrates **profiles** for sensitivity analysis:
- **p90**, **p95**, **p99**

### Primary reporting choice
For the main paper results:
- Use **p95** thresholds as the *primary* SLO profile.

For robustness / sensitivity analysis:
- Report compliance under **p90** and **p99** using the same runs (no re-calibration).

## 4) How to report results in the paper

Recommended tables/plots:
- TTFT and TPOT distribution (p50/p90/p95/p99) vs concurrency.
- SLO compliance (%) vs concurrency for p95 (primary), with p90/p99 as sensitivity.
- Breakdown plots for TTFT components (scheduler_wait_ms / lock_wait_ms / ttft_model_ms) to show where latency comes from.

When discussing queueing effects, emphasize that **Option A TTFT is service-facing** and intentionally includes queue and tokenization to match user-perceived latency.

## 5) Reproducibility checklist

- Fix `--seed` for preprocessing, load selection, and evaluation.
- Preserve official test sets (explicitly stated above).
- Save artifacts: `config.json`, `slo_thresholds.json`, per-concurrency `metrics_*.json`, request logs `requests_*.jsonl`.
- Record batching params: `max_batch_size`, `batch_wait_ms`.
- Keep prompt templates versioned (this bundle includes the exact prompts used).
