#!/usr/bin/env bash
set -euo pipefail

# Paper-ready smoke test entrypoint (fast-fail).
# Runs lightweight module checks + smoke suite. Optional: set RUN_TRAINING_TESTS=1
# to include router training smoke tests.

bash scripts/tests/run_all_module_tests.sh
