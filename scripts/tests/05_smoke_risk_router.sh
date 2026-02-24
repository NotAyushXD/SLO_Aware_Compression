#!/usr/bin/env bash
set -euo pipefail

# Smoke test: risk-router training + runtime integration.
# 1) Collect a small trace set (re-uses train_learned_router collector).
# 2) Train risk router bundle (predictors + conformal/quality calibration arrays).
# 3) Run a tiny multi-variant load test using router_mode=risk.

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTROOT="${OUTROOT:-runs/tests}"
TRACE_ROOT="${TRACE_ROOT:-router_models/test_risk}"
BUNDLE_DIR="${BUNDLE_DIR:-router_models/test_risk/risk_router_bundle}"
MAX_EXAMPLES="${MAX_EXAMPLES:-12}"
CONCURRENCIES="${CONCURRENCIES:-1}"
MULTI_VARIANTS=( ${MULTI_VARIANTS:-cheap base} )
DELTA="${DELTA:-0.10}"
EPS="${EPS:-0.25}"

mkdir -p "$OUTROOT" "$TRACE_ROOT" "$BUNDLE_DIR"

echo "[RUN] (optional) preprocessing"
if [[ ! -f data/processed/train_data.jsonl ]]; then
  python preprocessing.py
fi

echo "[RUN] collect traces (small)"
python scripts/train_learned_router.py \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --processed_dir data/processed \
  --prompt_mode slo \
  --output_root "$TRACE_ROOT" \
  --concurrencies $CONCURRENCIES \
  --max_examples "$MAX_EXAMPLES" \
  --max_batch_size 4 \
  --batch_wait_ms 8 \
  --collect_only

TRACE_JSONL="$TRACE_ROOT/trainval_traces.jsonl"
if [[ ! -f "$TRACE_JSONL" ]]; then
  echo "Trace JSONL missing: $TRACE_JSONL" >&2
  exit 2
fi

echo "[RUN] train risk-router bundle"
python scripts/train_risk_router.py \
  --trace_jsonl "$TRACE_JSONL" \
  --output_dir "$BUNDLE_DIR" \
  --min_rows_per_variant 5

echo "[RUN] multi-variant load smoke with risk router (EDF)"
python run_baseline_evaluation.py \
  --backend hf \
  --service multi \
  --router_mode risk \
  --risk_router_dir "$BUNDLE_DIR" \
  --risk_latency_delta "$DELTA" \
  --risk_quality_epsilon "$EPS" \
  --dispatcher_policy edf \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --multi_variants "${MULTI_VARIANTS[@]}" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 10 \
  --concurrencies 1 2 \
  --output_dir "$OUTROOT/multi_risk_edf"

echo "[OK] risk-router smoke done. Bundle: $BUNDLE_DIR"
