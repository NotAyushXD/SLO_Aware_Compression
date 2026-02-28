#!/usr/bin/env python
"""Create a delayed-label (judge) file from request logs.

This is a *debug/validation* helper so you can exercise the delayed-label ingestion pipeline
without deploying an external judge model.

It reads requests_*.jsonl (produced by run_baseline_evaluation/load_generator) and emits:

  labels.jsonl  with fields: {"join_key": <str>, "y": <int>, "source": "gold_from_log"}

By default, y is taken from inference_metrics.correct if present, else inference_metrics.correct_parseable.

Usage:
  python scripts/make_judge_labels_from_logs.py \
    --requests_jsonl runs/.../requests_concurrency_2.jsonl \
    --out labels.jsonl \
    --p 0.1

Then replay:
  python scripts/replay_delayed_labels.py \
    --bandit_state_path <prefix> \
    --judge_file labels.jsonl \
    --requests_jsonl runs/.../requests_concurrency_2.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests_jsonl", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--p", type=float, default=1.0, help="Sample probability (label budget). 1.0 = label all.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    in_path = Path(args.requests_jsonl)
    out_path = Path(args.out)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    n_in = 0
    n_out = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if float(args.p) < 1.0 and rng.random() > float(args.p):
                continue

            rec: Dict[str, Any] = json.loads(line)
            join_key = rec.get("join_key")
            if join_key is None:
                # Backward compat: fall back to request_id
                join_key = str(rec.get("request_id"))

            inf = rec.get("inference_metrics", {}) or {}
            y = inf.get("correct", None)
            if y is None:
                y = inf.get("correct_parseable", None)
            if y is None:
                # If no gold label exists, skip (this shouldn't happen for our built-in datasets).
                continue

            fout.write(json.dumps({"join_key": str(join_key), "y": int(y), "source": "gold_from_log"}) + "\n")
            n_out += 1

    print(f"[OK] wrote {n_out} labels from {n_in} requests -> {out_path}")


if __name__ == "__main__":
    main()
