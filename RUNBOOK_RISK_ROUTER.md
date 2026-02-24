# Runbook: Risk-controlled routing (TTFT + Total latency) for multi-variant LLM serving

This runbook explains how to reproduce the **risk router** pipeline end-to-end:

1) preprocess datasets
2) calibrate SLO thresholds (p90/p95/p99)
3) collect resumable traces for all variants
4) train predictors + build a risk-router bundle (models + calibration arrays)
5) evaluate baselines vs the risk router under offered-load sweeps

> Notes:
> - Commands assume you run from the repo root: `SLO_Aware_Compression codes/`
> - If you are on Kaggle, use `--time_budget_hours` for trace collection and resume.

---

## 0) Install deps

```bash
pip install -r requirements_.txt
```

---

## 1) Dataset preprocessing

This creates:
`data/processed/{train_data.jsonl,val_data.jsonl,test_data.jsonl}`

```bash
python run_baseline_evaluation.py \
  --preprocess \
  --data_dir data/raw \
  --processed_dir data/processed \
  --seed 0 \
  --skip_load_test \
  --skip_accuracy_eval
```

---

## 2) Calibrate SLO thresholds (p90/p95/p99)

This uses a **concurrency=1** load test and writes:
`<output_dir>/slo_thresholds.json`

Recommended: calibrate using the **base** model only.

```bash
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --service single \
  --variant base \
  --prompt_mode slo \
  --concurrencies 1 \
  --num_requests 200 \
  --output_dir outputs/slo_calibration_base
```

---

## 3) Collect multi-variant traces (resumable)

This runs **each example on each variant** (cheap/med/base) and records:
TTFT, total latency, queue depths, correctness, etc.

It writes/resumes:
`router_models/trainval_traces.jsonl`

```bash
python scripts/train_learned_router.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --processed_dir data/processed \
  --prompt_mode slo \
  --output_root router_models \
  --concurrencies 1 2 4 8 \
  --max_batch_size 8 \
  --batch_wait_ms 8 \
  --time_budget_hours 8 \
  --collect_only
```

Re-run the *same command* to resume.

### 3b) Adapter-aware traces (mixed adapters + setup-aware scheduling)

If you enabled the adapter portfolio (LoRA/PEFT tiers), collect traces with:

- `--enable_adapters` + `--adapter_policy dataset` (mixes adapters across GSM8K/MMLU)
- synthetic setup cost knobs (paper-friendly): `--adapter_synthetic_load_ms`, `--adapter_synthetic_switch_ms`
- setup-aware dispatcher: `--dispatcher_policy setup_lstf` (reduces adapter thrash)

Use a separate output root so you don't overwrite the no-adapter traces:

```bash
python scripts/train_learned_router.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --processed_dir data/processed \
  --prompt_mode slo \
  --output_root router_models_adapters \
  --concurrencies 1 2 4 8 \
  --max_batch_size 8 \
  --batch_wait_ms 8 \
  --dispatcher_policy setup_lstf \
  --enable_adapters \
  --adapter_root adapters \
  --adapter_policy dataset \
  --adapter_rank_policy load \
  --adapter_rank_tiers 8,16,32 \
  --max_loaded_adapters 8 \
  --adapter_synthetic_load_ms 30 \
  --adapter_synthetic_switch_ms 5 \
  --time_budget_hours 8 \
  --collect_only
```

The collected traces now include **router-time** adapter features:
`adapter_state[cheap|med|base] = {resident, hot, num_loaded, setup_est_ms, ...}`.

---

## 4) Train predictors + build the risk-router bundle

This trains per-variant predictors on TRAIN and saves calibration arrays from VAL.

Output folder (bundle):
`router_models/risk_router_bundle/`

```bash
python scripts/train_risk_router.py \
  --trace_jsonl router_models/trainval_traces.jsonl \
  --output_dir router_models/risk_router_bundle \
  --seed 42
```

---

## 5) Evaluate baselines vs risk router (offered-load sweep)

### 5a) Risk router run

```bash
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --service multi \
  --router_mode risk \
  --risk_router_dir router_models/risk_router_bundle \
  --risk_latency_delta 0.05 \
  --risk_quality_epsilon 0.20 \
  --dispatcher_policy lstf \
  --prompt_mode slo \
  --concurrencies 1 2 4 8 \
  --num_requests 200 \
  --output_dir outputs/risk_router_run
```

Adapter-aware evaluation uses the *same* knobs as trace collection:

```bash
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --service multi \
  --router_mode risk \
  --risk_router_dir router_models_adapters/risk_router_bundle \
  --risk_latency_delta 0.05 \
  --risk_quality_epsilon 0.20 \
  --dispatcher_policy setup_lstf \
  --enable_adapters \
  --adapter_root adapters \
  --adapter_policy dataset \
  --adapter_rank_policy load \
  --adapter_rank_tiers 8,16,32 \
  --max_loaded_adapters 8 \
  --adapter_synthetic_load_ms 30 \
  --adapter_synthetic_switch_ms 5 \
  --prompt_mode slo \
  --concurrencies 1 2 4 8 \
  --num_requests 200 \
  --output_dir outputs/risk_router_adapters
```

Copy the calibrated SLO file into this output dir before running (or run the
calibration step once inside `outputs/risk_router_run` so it writes there):

```bash
cp outputs/slo_calibration_base/slo_thresholds.json outputs/risk_router_run/slo_thresholds.json
```

### 5b) Baselines

Always-cheap:

```bash
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --service multi \
  --router_mode always_cheap \
  --prompt_mode slo \
  --concurrencies 1 2 4 8 \
  --num_requests 200 \
  --output_dir outputs/baseline_always_cheap
```

Always-base:

```bash
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --service multi \
  --router_mode always_base \
  --prompt_mode slo \
  --concurrencies 1 2 4 8 \
  --num_requests 200 \
  --output_dir outputs/baseline_always_base
```

Heuristic difficulty router:

```bash
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --service multi \
  --router_mode difficulty \
  --prompt_mode slo \
  --concurrencies 1 2 4 8 \
  --num_requests 200 \
  --output_dir outputs/baseline_difficulty
```

Learned (uncalibrated) routers:

```bash
python run_baseline_evaluation.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --service multi \
  --router_mode learned_ttft \
  --learned_router_dir router_models \
  --prompt_mode slo \
  --concurrencies 1 2 4 8 \
  --num_requests 200 \
  --output_dir outputs/baseline_learned_ttft
```

---

## 6) δ / ε sweeps for paper plots

The server supports changing δ and ε without retraining. For example:

Latency risk sweep (δ):

```bash
for d in 0.01 0.02 0.05 0.10 0.20; do
  python run_baseline_evaluation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --service multi \
    --router_mode risk \
    --risk_router_dir router_models/risk_router_bundle \
    --risk_latency_delta $d \
    --risk_quality_epsilon 0.20 \
    --dispatcher_policy lstf \
    --prompt_mode slo \
    --concurrencies 4 \
    --num_requests 400 \
    --output_dir outputs/sweep_delta_$d
done
```

Quality risk sweep (ε):

```bash
for e in 0.05 0.10 0.15 0.20 0.30; do
  python run_baseline_evaluation.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --service multi \
    --router_mode risk \
    --risk_router_dir router_models/risk_router_bundle \
    --risk_latency_delta 0.05 \
    --risk_quality_epsilon $e \
    --dispatcher_policy lstf \
    --prompt_mode slo \
    --concurrencies 4 \
    --num_requests 400 \
    --output_dir outputs/sweep_epsilon_$e
done
```
