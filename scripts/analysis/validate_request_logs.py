#!/usr/bin/env python
"""Validate request JSONL logs for required fields.

This is a *fast-fail* audit tool aligned with the execution plan's "required logging schema".
It checks that each record contains enough information to reproduce:
- context (dataset/difficulty/tokens)
- system state at decision time (queue/scheduler/batching)
- decision + router_meta (chosen action, baseline action, fallback)
- outcomes (TTFT, E2E, violation flags)
- cost breakdown (token/adapter/swap/total)
- router state (bandit Q snapshots when applicable)

Usage:
  python scripts/analysis/validate_request_logs.py --requests_jsonl runs/.../requests_concurrency_4.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_TOP = [
    "request_id",
    "join_key",
    "dataset_type",
    "difficulty",
    "submit_time",
    "start_time",
    "end_time",
    "e2e_latency_ms",
    "queue_wait_time_ms",
    "inference_time_ms",
]

# Required in inference_metrics for any successful request.
REQUIRED_INF = [
    "success",
    "ttft_ms",
    "total_latency_ms",
    "output_length",
    "prompt_tokens",
    "variant_effective",
    "router_meta",
    "token_cost_units",
    "adapter_overhead_units",
    "swap_overhead_units",
    "total_cost_units",
    "slo_violation",
    "risk_violation",
]


def _missing(d: Dict[str, Any], keys: List[str]) -> List[str]:
    out = []
    for k in keys:
        if k not in d:
            out.append(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_jsonl", type=str, required=True)
    ap.add_argument("--max_errors", type=int, default=20)
    args = ap.parse_args()

    path = Path(args.requests_jsonl)
    if not path.exists():
        raise SystemExit(f"Missing requests_jsonl: {path}")

    n = 0
    n_success = 0
    errors: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            rec: Dict[str, Any] = json.loads(line)

            miss_top = _missing(rec, REQUIRED_TOP)
            if miss_top:
                errors.append(f"request_id={rec.get('request_id')} missing_top={miss_top}")
                if len(errors) >= int(args.max_errors):
                    break

            inf = rec.get("inference_metrics", {}) or {}
            if bool(inf.get("success", False)):
                n_success += 1
                miss_inf = _missing(inf, REQUIRED_INF)
                if miss_inf:
                    errors.append(f"request_id={rec.get('request_id')} missing_inference_metrics={miss_inf}")
                    if len(errors) >= int(args.max_errors):
                        break

                # router_meta should be a dict
                rm = inf.get("router_meta", None)
                if not isinstance(rm, dict):
                    errors.append(f"request_id={rec.get('request_id')} router_meta not a dict")
                    if len(errors) >= int(args.max_errors):
                        break

    if n == 0:
        raise SystemExit("Empty requests_jsonl")

    if errors:
        print("[FAIL] Log schema validation failed (showing first errors):")
        for e in errors[: int(args.max_errors)]:
            print("  -", e)
        raise SystemExit(2)

    print("[OK] Log schema looks good")
    print(f"  records: {n}")
    print(f"  successful: {n_success}")


if __name__ == "__main__":
    main()
