#!/usr/bin/env bash
set -euo pipefail

# Smoke test: single-variant server + evaluation harness.
# Runs a tiny accuracy eval and a tiny load test.

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
VARIANT="${VARIANT:-med}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTROOT="${OUTROOT:-runs/tests}"

mkdir -p "$OUTROOT"

echo "[RUN] single-variant accuracy smoke"
python run_baseline_evaluation.py \
  --backend hf \
  --service single \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --prompt_mode accuracy \
  --skip_load_test \
  --data_subset 20 \
  --output_dir "$OUTROOT/single_accuracy"

echo "[RUN] single-variant load smoke (TTFT/TPOT)"
python run_baseline_evaluation.py \
  --backend hf \
  --service single \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 10 \
  --concurrencies 1 2 \
  --output_dir "$OUTROOT/single_load"

echo "[OK] single-variant smoke tests done: $OUTROOT"
