#!/usr/bin/env python
"""Oracle policy analysis from multi-variant traces.

Motivation (reviewer-facing):
  - A common request in NeurIPS/ICML/ICLR reviews is an *upper bound* (clairvoyant)
    to quantify remaining headroom.
  - Our learned/bandit routers operate with partial feedback and uncertainty.
    An oracle that has access to per-request outcomes for *all* actions provides
    a useful reference curve.

This script consumes the multi-variant trace logs produced by:
  - scripts/train_learned_router.py (trainval_traces.jsonl)

and computes the oracle action per request under SLO constraints.

Example:
  python scripts/analysis/oracle_from_traces.py \
    --traces outputs/learned_router/trainval_traces.jsonl \
    --slo_ttft_ms 300 --slo_e2e_ms 1200 \
    --oracle qa \
    --out_json outputs/oracle_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=str, nargs="+", required=True, help="One or more JSONL trace files.")
    ap.add_argument("--slo_ttft_ms", type=float, required=True)
    ap.add_argument("--slo_e2e_ms", type=float, required=True)
    ap.add_argument(
        "--oracle",
        type=str,
        default="qa",
        choices=["slo", "qa"],
        help="Oracle objective: 'slo' = meet SLO at min cost; 'qa' = correct+SLO at min cost.",
    )
    ap.add_argument("--out_json", type=str, required=True)
    return ap.parse_args()


def _slo_ok(r: Dict[str, Any], ttft_ms: float, e2e_ms: float) -> bool:
    try:
        return float(r.get("ttft_ms", 0.0) or 0.0) <= ttft_ms and float(r.get("total_latency_ms", 0.0) or 0.0) <= e2e_ms
    except Exception:
        return False


def _cost(r: Dict[str, Any]) -> float:
    for k in ["total_cost_units", "cost_units", "token_cost_units"]:
        if k in r:
            try:
                return float(r.get(k) or 0.0)
            except Exception:
                pass
    return 0.0


def main() -> None:
    args = parse_args()

    # Group by request id (from trace) + dataset.
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for p in args.traces:
        path = Path(p)
        for r in _iter_jsonl(path):
            rid = str(r.get("id") or r.get("request_id") or "")
            ds = str(r.get("dataset") or r.get("dataset_type") or "")
            if not rid:
                continue
            groups[(ds, rid)].append(r)

    n = 0
    success = 0
    slo_ok = 0
    correct = 0
    total_cost = 0.0
    chosen_variants: Dict[str, int] = defaultdict(int)

    for (_ds, _rid), recs in groups.items():
        if not recs:
            continue
        n += 1

        # Filter to successful records.
        cand = [r for r in recs if int(r.get("success", 1) or 0) == 1]
        if not cand:
            # Treat as failed request.
            continue
        success += 1

        # Build feasibility set.
        feasible = [r for r in cand if _slo_ok(r, args.slo_ttft_ms, args.slo_e2e_ms)]

        # Objective.
        def key_cost(r: Dict[str, Any]) -> float:
            return _cost(r)

        chosen: Dict[str, Any]
        if args.oracle == "slo":
            chosen = min(feasible, key=key_cost) if feasible else min(cand, key=key_cost)
        else:
            # QA oracle: prefer correct+SLO; fall back to SLO; then min-cost.
            feas_good = [r for r in feasible if int(r.get("correct", 0) or 0) == 1]
            if feas_good:
                chosen = min(feas_good, key=key_cost)
            elif feasible:
                chosen = min(feasible, key=key_cost)
            else:
                chosen = min(cand, key=key_cost)

        total_cost += key_cost(chosen)
        v = str(chosen.get("variant") or "base")
        chosen_variants[v] += 1

        ok_slo = _slo_ok(chosen, args.slo_ttft_ms, args.slo_e2e_ms)
        if ok_slo:
            slo_ok += 1
        if int(chosen.get("correct", 0) or 0) == 1:
            correct += 1

    out = {
        "oracle": args.oracle,
        "slo_ttft_ms": float(args.slo_ttft_ms),
        "slo_e2e_ms": float(args.slo_e2e_ms),
        "total_requests": int(n),
        "successful_requests": int(success),
        "slo_compliant_successful_requests": int(slo_ok),
        "correct_successful_requests": int(correct),
        "accuracy_success": float(correct) / float(max(success, 1)),
        "slo_compliance": float(slo_ok) / float(max(success, 1)),
        "violation_rate": 1.0 - (float(slo_ok) / float(max(success, 1))),
        "cost_per_request": float(total_cost) / float(max(success, 1)),
        "cost_per_goodput_request": float(total_cost) / float(max(slo_ok, 1)),
        "cost_per_qa_goodput_request": float(total_cost) / float(max(min(correct, slo_ok), 1)),
        "chosen_variants": dict(sorted(chosen_variants.items(), key=lambda kv: kv[0])),
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
