#!/usr/bin/env bash
set -euo pipefail

# Optional smoke test for the vLLM backend.
# Skips automatically if vllm is not installed.

cd "$(dirname "$0")/../.."

if ! python -c "import vllm" >/dev/null 2>&1; then
  echo "[SKIP] vllm not installed. Install it and re-run to test vLLM backend."
  exit 0
fi

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float16}"
OUTROOT="${OUTROOT:-runs/tests}"

mkdir -p "$OUTROOT"

echo "[RUN] vLLM backend load smoke (single-variant)"
python run_baseline_evaluation.py \
  --backend vllm \
  --service single \
  --model "$MODEL" \
  --variant base \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 10 \
  --concurrencies 1 2 \
  --output_dir "$OUTROOT/vllm_load"

echo "[OK] vLLM smoke finished: $OUTROOT/vllm_load"
