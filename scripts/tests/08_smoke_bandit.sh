#!/usr/bin/env bash
set -euo pipefail

# End-to-end bandit smoke test.
# This requires a trained RiskRouter bundle because the bandit uses it as the conservative baseline.
#
# Environment variables:
#   MODEL=gpt2
#   CHEAP_MODEL=distilgpt2
#   RISK_ROUTER_DIR=router_models/risk_router_bundle
#   OUTROOT=runs/tests

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-gpt2}"
CHEAP_MODEL="${CHEAP_MODEL:-distilgpt2}"
RISK_ROUTER_DIR="${RISK_ROUTER_DIR:-router_models/risk_router_bundle}"
OUTROOT="${OUTROOT:-runs/tests}"

mkdir -p "${OUTROOT}"
OUTDIR="${OUTROOT}/bandit_smoke_seed0"

echo "[RUN] Bandit smoke test -> ${OUTDIR}"
echo "  MODEL=${MODEL}"
echo "  CHEAP_MODEL=${CHEAP_MODEL}"
echo "  RISK_ROUTER_DIR=${RISK_ROUTER_DIR}"

python run_baseline_evaluation.py \
  --model "${MODEL}" \
  --cheap_model "${CHEAP_MODEL}" \
  --service multi \
  --router_mode bandit \
  --risk_router_dir "${RISK_ROUTER_DIR}" \
  --prompt_mode slo \
  --processed_dir data/processed \
  --load_test_split val \
  --num_requests 30 \
  --concurrencies 2 \
  --slo_calibration_concurrency 1 \
  --bandit_label_budget_p 1.0 \
  --seed 0 \
  --output_dir "${OUTDIR}" \
  --bandit_keep_learning_during_eval

python scripts/tests/08_smoke_bandit_check.py \
  --requests_jsonl "${OUTDIR}/requests_concurrency_2.jsonl" \
  --min_updates 1

echo "[OK] Bandit smoke test passed."
echo "  Output dir: ${OUTDIR}"
