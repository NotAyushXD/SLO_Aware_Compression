#!/usr/bin/env bash
set -euo pipefail

# Master smoke suite that exercises all major modules.
#
# Defaults are intentionally small so it can run on Kaggle/Colab.
#
# Environment variables you can set:
#   MODEL=meta-llama/Llama-3.1-8B-Instruct
#   DEVICE=auto
#   DTYPE=auto
#   VARIANT=med
#   OUTROOT=runs/tests
#   RUN_TRAINING_TESTS=1        # to include learned/risk router training
#   RUN_VLLM_TEST=1             # to include vLLM backend test (requires vllm installed)
#   MULTI_VARIANTS="cheap base" # default is cheap+base

cd "$(dirname "$0")/../.."

RUN_TRAINING_TESTS="${RUN_TRAINING_TESTS:-0}"
RUN_VLLM_TEST="${RUN_VLLM_TEST:-0}"

python scripts/tests/00_check_imports.py
python scripts/tests/06_metrics_unit.py
bash scripts/tests/01_smoke_preprocess.sh
bash scripts/tests/02_smoke_single.sh
bash scripts/tests/03_smoke_multi_heuristics.sh

if [[ "$RUN_TRAINING_TESTS" == "1" ]]; then
  bash scripts/tests/04_smoke_learned_router.sh
  bash scripts/tests/05_smoke_risk_router.sh
else
  echo "[SKIP] Training-based tests (set RUN_TRAINING_TESTS=1 to include learned/risk router training)."
fi

if [[ "$RUN_VLLM_TEST" == "1" ]]; then
  bash scripts/tests/07_smoke_vllm_optional.sh
else
  echo "[SKIP] vLLM test (set RUN_VLLM_TEST=1 to include; requires vllm installed)."
fi

echo "[DONE] module smoke suite finished."
