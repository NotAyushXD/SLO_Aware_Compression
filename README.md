# SLO-aware evaluation patch bundle (Option A, p95 primary)

This bundle contains patched versions of the baseline scripts to produce paper-grade, reproducible results for:
- Accuracy evaluation (GSM8K + MMLU)
- SLO-mode quality guardrail (SLO prompting should not collapse correctness)
- Closed-loop load testing with micro-batching, TTFT/**E2E(total)** measurement, and dynamic SLO calibration

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
- **Primary violation definition (paper):** TTFT **or** E2E(total) exceeds the calibrated thresholds.
  - E2E(total) is measured server-side as `total_latency_ms` (queue-inclusive).

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


### 6) Multi-variant serving wrapper (server.py)
- Adds `MultiVariantService`, a single-process wrapper that can load `cheap`/`med`/`base` variants and route per-request.
- Supports routing modes: `difficulty` (easy→cheap, medium→med, hard→base), `slo_aware`, and fixed routing.
- Includes a paper-friendly *escalation* mechanism: if the chosen (cheaper) variant returns an unparsable answer format, it retries on a stronger variant (bounded by `--router_max_retries`).
- **GPU-memory aware loading (default):** the service automatically chooses a safe loading plan based on detected GPU memory.
  - Large GPUs: eager-load all variants.
  - Mid GPUs: preload `cheap+med`, lazy-load `base` on first use.
  - Small GPUs: keep only 1 variant resident and swap variants on demand.
  - You can still override manually via `--multi_variants ...` and `--router_lazy_load_base`.

Example (multi-variant, difficulty routing):
```bash
python run_baseline_evaluation.py \
  --backend hf \
  --service multi \
  --router_mode difficulty \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt_mode slo \
  --num_requests 200 \
  --concurrencies 1 4 \
  --seed 42 \
  --output_dir ./runs/multi_variant_smoke
```

## New experimental knobs (paper add-ons)

### Nonstationary load schedules (E2)
Run a phase schedule (overrides `--concurrencies`):
```bash
python run_baseline_evaluation.py \
  --service multi --router_mode bandit \
  --risk_router_dir router_models/risk_router_bundle \
  --multi_variants cheap med base \
  --prompt_mode slo \
  --num_requests 200 \
  --concurrency_schedule "1:100,8:200,2:100" \
  --output_dir ./runs/nonstationary
```
Then plot time-series from `requests_schedule.jsonl`:
```bash
python scripts/analysis/plot_timeseries.py \
  --requests_jsonl ./runs/nonstationary/requests_schedule.jsonl \
  --out_dir ./runs/nonstationary/plots
```

### Offered-load sweep (E3)
Run a concurrency sweep (multi-seed + CI + plots):
```bash
python scripts/experiments/e3_offered_load_sweep.py \
  --configs ours=configs/bandit_base.json \
  --concurrencies 1 2 4 8 \
  --seeds 0 1 2 3 4 \
  --num_requests 200 \
  --out_root outputs/e3_offered_load
```
Add the paper ablation (remove system-state features):
```bash
python scripts/experiments/e3_offered_load_sweep.py \
  --configs ours=configs/bandit_base.json \
  --add_no_system_state_ablation \
  --concurrencies 1 2 4 8 \
  --seeds 0 1 2 3 4 \
  --num_requests 200 \
  --out_root outputs/e3_offered_load
```

### Domain/length shift mid-run (E5)
This uses the `--data_schedule` mechanism in `run_baseline_evaluation.py`.

Dataset/domain shift (gsm8k → mmlu) comparing **online** vs **frozen**:
```bash
python scripts/experiments/e5_domain_length_shift.py \
  --config configs/bandit_base.json \
  --shift_mode dataset --phase1_value gsm8k --phase2_value mmlu \
  --phase_requests 200 200 --concurrency 4 \
  --warmup_requests 200 --warmup_concurrency 4 \
  --seeds 0 1 2 3 4 \
  --out_root outputs/e5_shift
```
Length shift (short → long):
```bash
python scripts/experiments/e5_domain_length_shift.py \
  --config configs/bandit_base.json \
  --shift_mode length \
  --phase_requests 200 200 --concurrency 4 \
  --seeds 0 1 2 3 4 \
  --out_root outputs/e5_shift
```
Each seed run produces time-series plots under `.../analysis/`.

### Calibration + threshold sensitivity (E7)
Reliability diagrams + ECE tables from request logs (multi-seed):
```bash
python scripts/experiments/e7_calibration.py \
  --configs ours=configs/bandit_base.json \
  --concurrency 4 --num_requests 300 \
  --seeds 0 1 2 3 4 \
  --out_root outputs/e7_calibration
```

### One-command figure regeneration
After running E1–E7 (or any subset), regenerate figures from saved artifacts:
```bash
python analysis/make_all.py \
  --outputs_root outputs \
  --fig_dir figures \
  --tables_dir tables
```

### Delayed labels (Step 3)
Run without sending gold labels to the server (bandit stores pending join_keys):
```bash
python run_baseline_evaluation.py \
  --service multi --router_mode bandit \
  --risk_router_dir router_models/risk_router_bundle \
  --server_label_mode none \
  --output_dir ./runs/delayed_labels
```

Later, ingest a judge file (JSONL/CSV) into the saved bandit state:
```bash
python scripts/replay_delayed_labels.py \
  --bandit_state_path ./runs/delayed_labels/bandit_state \
  --judge_file ./judge_outputs.jsonl
```

### Adapter churn + cache sweeps (E4)
Enable synthetic adapters (no PEFT / no on-disk adapter artifacts required):
```bash
python run_baseline_evaluation.py \
  --enable_adapters --adapter_allow_missing \
  --adapter_synthetic_load_ms 20 --adapter_synthetic_switch_ms 5 \
  --output_dir ./runs/adapters
```
Then run sweeps via `scripts/experiments/e4_adapter_churn.py`.

## How to run

### Train router artifacts (required for `--router_mode risk` and `--router_mode bandit`)
Bandit/risk modes require a trained `--risk_router_dir` bundle.

1) Collect multi-variant traces (Train+Val):
```bash
python scripts/train_learned_router.py \
  --model gpt2 \
  --collect_only \
  --output_root router_models
```
This writes `router_models/trainval_traces.jsonl`.

2) Train RiskRouter bundle:
```bash
python scripts/train_risk_router.py \
  --trace_jsonl router_models/trainval_traces.jsonl \
  --output_dir router_models/risk_router_bundle
```

3) (Optional) Train LearnedRouter bundles (quality + latency predictors):
```bash
python scripts/train_learned_router.py \
  --model gpt2 \
  --output_root router_models
```

After this, `configs/bandit_base.json` should work out-of-the-box (it already points to `router_models/risk_router_bundle`).

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
