#!/usr/bin/env bash
set -euo pipefail

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_BASE="./kaggle/working"
DATA_SUBSET=200

echo "=== 1) Accuracy baselines (single-variant) ==="
python run_baseline_evaluation.py \
  --backend hf --service single --variant base \
  --model "$MODEL" \
  --prompt_mode accuracy --skip_load_test \
  --data_subset $DATA_SUBSET \
  --output_dir "$OUT_BASE/acc_base_fp16"

python run_baseline_evaluation.py \
  --backend hf --service single --variant med \
  --model "$MODEL" \
  --prompt_mode accuracy --skip_load_test \
  --data_subset $DATA_SUBSET \
  --output_dir "$OUT_BASE/acc_med_int8"

python run_baseline_evaluation.py \
  --backend hf --service single --variant cheap \
  --model "$MODEL" \
  --prompt_mode accuracy --skip_load_test \
  --data_subset $DATA_SUBSET \
  --output_dir "$OUT_BASE/acc_cheap_int4"

echo "=== 2) Accuracy (multi-variant router) ==="
python run_baseline_evaluation.py \
  --backend hf \
  --service multi \
  --router_mode difficulty \
  --router_max_retries 1 \
  --model "$MODEL" \
  --prompt_mode accuracy \
  --skip_load_test \
  --data_subset $DATA_SUBSET \
  --output_dir "$OUT_BASE/acc_multi_router"

echo "=== 3) Calibrate SLO thresholds on BASE ==="
python run_baseline_evaluation.py \
  --backend hf \
  --service single \
  --variant base \
  --model "$MODEL" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 200 \
  --concurrencies 1 2 4 \
  --seed 42 \
  --output_dir "$OUT_BASE/calibration_base"

echo "=== 4) Load test each variant (cost frontier) ==="
for variant in cheap med base; do
  python run_baseline_evaluation.py \
    --backend hf \
    --service single \
    --variant $variant \
    --model "$MODEL" \
    --prompt_mode slo \
    --skip_accuracy_eval \
    --num_requests 400 \
    --concurrencies 1 2 4 8 \
    --seed 42 \
    --output_dir "$OUT_BASE/load_${variant}"
done

echo "=== 5) Multi-variant load test (router stress) ==="
python run_baseline_evaluation.py \
  --backend hf \
  --service multi \
  --router_mode slo_aware \
  --router_calibration_mode base \
  --model "$MODEL" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 400 \
  --concurrencies 1 2 4 8 \
  --seed 42 \
  --output_dir "$OUT_BASE/load_multi_sloaware"

echo "=== DONE ==="
