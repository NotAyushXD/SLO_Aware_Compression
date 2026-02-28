#!/usr/bin/env python
"""Validate that a bandit run produced expected logging and updates.

Checks:
- router_meta.bandit exists
- bandit_update.updated == True sometimes
- Q changes over time (Q_after != Q_before at least once)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_jsonl", type=str, required=True)
    ap.add_argument("--min_updates", type=int, default=1)
    args = ap.parse_args()

    path = Path(args.requests_jsonl)
    if not path.exists():
        raise SystemExit(f"Missing requests_jsonl: {path}")

    n = 0
    n_bandit_meta = 0
    n_updates = 0
    n_q_changed = 0

    last_q = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec: Dict[str, Any] = json.loads(line)
            n += 1
            inf = rec.get("inference_metrics", {}) or {}
            rm = inf.get("router_meta", {}) or {}

            if isinstance(rm, dict) and "bandit" in rm:
                n_bandit_meta += 1

            bu = inf.get("bandit_update", None)
            if isinstance(bu, dict) and bool(bu.get("updated", False)):
                n_updates += 1
                qb = bu.get("Q_before", None)
                qa = bu.get("Q_after", None)
                if qb is not None and qa is not None and float(qa) != float(qb):
                    n_q_changed += 1
                last_q = qa

    if n == 0:
        raise SystemExit("Empty requests file")

    if n_bandit_meta == 0:
        raise SystemExit("router_meta.bandit missing from all requests")

    if n_updates < int(args.min_updates):
        raise SystemExit(f"Expected >= {args.min_updates} bandit updates, saw {n_updates}")

    if n_q_changed == 0:
        raise SystemExit("Q never changed (Q_after == Q_before for all updates)")

    print("[OK] Bandit logs look healthy")
    print(f"  requests: {n}")
    print(f"  router_meta.bandit present: {n_bandit_meta}/{n}")
    print(f"  bandit updates: {n_updates}")
    print(f"  Q changed events: {n_q_changed}")
    print(f"  last Q: {last_q}")


if __name__ == "__main__":
    main()
