#!/usr/bin/env python
"""Quick import sanity check for all major modules.

This is the fastest possible "does the code import" test.
"""

import importlib
import os
import sys

# Ensure repo root is on sys.path when invoked as a file path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODULES = [
    'preprocessing',
    'evaluation',
    'server',
    'load_generator',
    'learned_router',
    'risk_router',
    'metrics',
    'prompt_templates',
    'answer_utils',
]


def main() -> None:
    failed = []
    for m in MODULES:
        try:
            importlib.import_module(m)
            print(f"[OK] import {m}")
        except ModuleNotFoundError as e:
            # Allow lightweight environments to run this smoke test without heavyweight deps.
            # In production/Kaggle, install requirements_.txt and this should pass.
            if m == "server" and any(x in str(e) for x in ("transformers", "torch")):
                print(f"[WARN] import {m} skipped (missing deps): {e}", file=sys.stderr)
                continue
            failed.append((m, repr(e)))
            print(f"[FAIL] import {m}: {e}", file=sys.stderr)
        except Exception as e:
            failed.append((m, repr(e)))
            print(f"[FAIL] import {m}: {e}", file=sys.stderr)

    if failed:
        print("\nFAILED IMPORTS:")
        for m, err in failed:
            print(f"  - {m}: {err}")
        raise SystemExit(2)

    # Lightweight functional checks
    from learned_router import LearnedRouter
    from risk_router import _quantile_higher, binom_upper_confidence_bound

    feats = LearnedRouter.extract_features(
        dataset_type='gsm8k',
        difficulty='easy',
        max_tokens=64,
        prompt_tokens=128,
        concurrency=4,
        queue_depths={'cheap': 1, 'med': 2, 'base': 3},
    )
    # Feature dim is allowed to grow as we add router features (e.g., adapter hotness/setup).
    assert feats.shape[1] >= 22, f"Unexpected feature dim: {feats.shape}"
    assert _quantile_higher([1, 2, 3, 4], 0.75) in (3.0, 4.0)
    assert 0.0 <= binom_upper_confidence_bound(0, 10, 0.05) <= 1.0

    print("\nAll import checks passed.")


if __name__ == '__main__':
    main()
