#!/usr/bin/env bash
set -euo pipefail

# Smoke test: multi-variant service + heuristic routers.
# - difficulty router
# - slo_aware router

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTROOT="${OUTROOT:-runs/tests}"
# Default to 2 variants to avoid OOM on small GPUs/CPU.
MULTI_VARIANTS=( ${MULTI_VARIANTS:-cheap base} )

mkdir -p "$OUTROOT"

echo "[RUN] multi-variant difficulty router"
python run_baseline_evaluation.py \
  --backend hf \
  --service multi \
  --router_mode difficulty \
  --dispatcher_policy age \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --multi_variants "${MULTI_VARIANTS[@]}" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 10 \
  --concurrencies 1 2 \
  --output_dir "$OUTROOT/multi_difficulty"

echo "[RUN] multi-variant slo_aware router"
python run_baseline_evaluation.py \
  --backend hf \
  --service multi \
  --router_mode slo_aware \
  --dispatcher_policy age \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --multi_variants "${MULTI_VARIANTS[@]}" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 10 \
  --concurrencies 1 2 \
  --output_dir "$OUTROOT/multi_slo_aware"

echo "[OK] multi-variant heuristic router smokes done: $OUTROOT"
