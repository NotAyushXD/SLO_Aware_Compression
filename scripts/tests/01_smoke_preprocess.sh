#!/usr/bin/env bash
set -euo pipefail

# Preprocessing smoke test.
# Produces: data/processed/{train,val,test}_data.jsonl

cd "$(dirname "$0")/../.."

if [[ -f data/processed/train_data.jsonl && -f data/processed/val_data.jsonl ]]; then
  echo "[OK] data/processed already exists; skipping preprocessing."
else
  echo "[RUN] preprocessing.py"
  python preprocessing.py
fi

echo "[OK] preprocessing smoke test passed."
