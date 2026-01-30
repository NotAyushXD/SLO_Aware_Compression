#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_all.sh [MODEL] [OUTROOT] [VARIANT] [DEVICE] [SEED]
#
# Example (Kaggle):
#   bash run_all.sh meta-llama/Llama-3.1-8B-Instruct /kaggle/working/outputs_paper_smoke med auto 0

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
OUTROOT="${2:-./outputs_paper_smoke}"
VARIANT="${3:-med}"
DEVICE="${4:-auto}"
SEED="${5:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p "$OUTROOT"

echo "============================================================"
echo "Running paper smoke suite"
echo "  MODEL   : $MODEL"
echo "  VARIANT : $VARIANT"
echo "  DEVICE  : $DEVICE"
echo "  SEED    : $SEED"
echo "  OUTROOT : $OUTROOT"
echo "============================================================"
echo

# 1) Accuracy smoke (correctness + formatting)
python run_baseline_evaluation.py \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --prompt_mode accuracy \
  --skip_load_test \
  --data_subset 200 \
  --output_dir "$OUTROOT/test_accuracy_smoke"

# 2) SLO-mode quality guardrail (SLO prompt still answers correctly)
python run_baseline_evaluation.py \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --prompt_mode slo \
  --skip_load_test \
  --data_subset 200 \
  --output_dir "$OUTROOT/test_slo_accuracycheck"

# 3) Serving/load smoke (TTFT/TPOT/queue + dynamic SLO calibration)
python run_baseline_evaluation.py \
  --model "$MODEL" \
  --variant "$VARIANT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --prompt_mode slo \
  --skip_accuracy_eval \
  --num_requests 20 \
  --concurrencies 1 4 \
  --output_dir "$OUTROOT/test_slo_load_smoke"

echo
echo "Done. Outputs written under: $OUTROOT"
