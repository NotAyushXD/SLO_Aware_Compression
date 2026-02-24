#!/usr/bin/env bash
set -euo pipefail

# Smoke test: learned-router training + runtime integration.
# 1) Collect a small trace set.
# 2) Train learned routers.
# 3) Run a tiny multi-variant load test using learned_total routing.

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTROOT="${OUTROOT:-runs/tests}"
ARTROOT="${ARTROOT:-router_models/test_learned}"
# Keep tiny by default
MAX_EXAMPLES="${MAX_EXAMPLES:-12}"
CONCURRENCIES="${CONCURRENCIES:-1}"
MULTI_VARIANTS=( ${MULTI_VARIANTS:-cheap base} )

mkdir -p "$OUTROOT"
mkdir -p "$ARTROOT"

echo "[RUN] (optional) preprocessing"
if [[ ! -f data/processed/train_data.jsonl ]]; then
  python preprocessing.py
fi

echo "[RUN] collect+train learned router (small)"
python scripts/train_learned_router.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --processed_dir data/processed \
  --prompt_mode slo \
  --output_root "$ARTROOT" \
  --concurrencies $CONCURRENCIES \
  --max_examples "$MAX_EXAMPLES" \
  --max_batch_size 4 \
  --batch_wait_ms 8

echo "[RUN] multi-variant load smoke with learned_total"
python run_baseline_evaluation.py \
  --backend hf \
  --service multi \
  --router_mode learned_total \
  --learned_router_dir "$ARTROOT" \
  --dispatcher_policy age \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --multi_variants "${MULTI_VARIANTS[@]}" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 10 \
  --concurrencies 1 2 \
  --output_dir "$OUTROOT/multi_learned_total"

echo "[OK] learned-router smoke done. Artifacts: $ARTROOT"
