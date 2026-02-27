#!/usr/bin/env python
"""Replay delayed quality labels into a saved BanditRouter state.

This script supports the paper's "delayed / partial feedback" claim:
- During serving, run with --server_label_mode none so the server does NOT
  receive correctness labels in real-time.
- The bandit stores (join_key -> (action,x)) in a pending buffer.
- Later, obtain a judge file (human/LLM evaluator) that provides quality labels.
- Replay labels into the saved bandit state using this script.

Input formats
-------------
Judge file can be JSONL or CSV.

JSONL: each line must contain one of:
  - {"request_id": 123, "correct": 1}
  - {"join_key": "123", "quality_label": 0}

CSV: must contain columns: request_id (or join_key), and label (or correct).

If the saved state does not contain a pending buffer (or a join_key is missing),
you may provide --requests_jsonl which contains router_meta.bandit_x and the
chosen action so we can ingest labels directly from logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from bandit_router import BanditAction, BanditRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("replay_delayed_labels")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _read_csv(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield dict(row)


def _iter_labels(path: Path) -> Iterable[Tuple[str, int]]:
    if path.suffix.lower() == ".csv":
        rows = _read_csv(path)
    else:
        rows = _read_jsonl(path)

    for r in rows:
        join = r.get("join_key", None)
        if join is None:
            join = r.get("request_id", None)
        if join is None:
            continue
        join_key = str(join)

        y = r.get("quality_label", None)
        if y is None:
            y = r.get("correct", None)
        if y is None:
            y = r.get("label", None)
        if y is None:
            continue
        try:
            y_i = 1 if int(y) != 0 else 0
        except Exception:
            continue

        yield join_key, y_i


def _index_requests_log(requests_jsonl: Path, needed_keys: set[str]) -> Dict[str, Tuple[BanditAction, np.ndarray]]:
    """Build a minimal index from requests_jsonl for the needed join_keys."""

    out: Dict[str, Tuple[BanditAction, np.ndarray]] = {}
    if not needed_keys:
        return out

    n = 0
    hit = 0
    for rec in _read_jsonl(requests_jsonl):
        n += 1
        rid = rec.get("request_id", None)
        if rid is None:
            continue
        jk = str(rid)
        if jk not in needed_keys:
            continue
        inf = rec.get("inference_metrics", {}) or {}
        rm = inf.get("router_meta", {}) or {}
        # bandit_x is stored at router_meta['bandit_x']
        x_list = rm.get("bandit_x", None)
        if x_list is None:
            continue

        act_dict = None
        try:
            act_dict = (rm.get("bandit") or {}).get("chosen_action")
        except Exception:
            act_dict = None

        if isinstance(act_dict, dict):
            action = BanditAction.from_dict(act_dict)
        else:
            # Best-effort fallback
            action = BanditAction(variant=str(inf.get("variant", "base")))

        try:
            x = np.asarray(x_list, dtype=np.float32).reshape(-1)
        except Exception:
            continue

        out[jk] = (action, x)
        hit += 1
        if hit >= len(needed_keys):
            break

    logger.info(f"Indexed requests log: scanned={n}, matched={hit}/{len(needed_keys)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandit_state_path", type=str, required=True, help="Path prefix for BanditRouter.save files.")
    ap.add_argument("--judge_file", type=str, required=True, help="JSONL/CSV file containing delayed labels.")
    ap.add_argument(
        "--output_state_path",
        type=str,
        default=None,
        help="Output path prefix for updated state (default: <bandit_state_path>_ingested).",
    )
    ap.add_argument(
        "--requests_jsonl",
        type=str,
        default=None,
        help="Optional requests JSONL (from load_generator) to recover (action,x) when pending buffer is missing.",
    )

    args = ap.parse_args()

    base = str(args.bandit_state_path)
    judge_path = Path(args.judge_file)
    if not judge_path.exists():
        raise FileNotFoundError(judge_path)

    out_prefix = args.output_state_path or (base + "_ingested")

    logger.info(f"Loading bandit state: {base}")
    router = BanditRouter.load(base)

    # Read labels
    labels: List[Tuple[str, int]] = list(_iter_labels(judge_path))
    if not labels:
        raise RuntimeError(f"No labels found in judge file: {judge_path}")

    needed = {jk for jk, _y in labels}

    # Optional index from request logs for direct ingestion
    log_index: Dict[str, Tuple[BanditAction, np.ndarray]] = {}
    if args.requests_jsonl is not None:
        log_index = _index_requests_log(Path(args.requests_jsonl), needed_keys=needed)

    ok = 0
    miss = 0
    direct = 0

    for jk, y in labels:
        res = router.ingest_quality_label(jk, y)
        if res.get("updated"):
            ok += 1
            continue

        # Fallback: ingest directly from logs
        if jk in log_index:
            action, x = log_index[jk]
            res2 = router.ingest_quality_label_direct(join_key=jk, action=action, x=x, quality_label=y)
            if res2.get("updated"):
                ok += 1
                direct += 1
                continue

        miss += 1

    logger.info(
        f"Ingest complete: updated={ok}/{len(labels)} (direct_from_logs={direct}), missing={miss}"
    )

    router.save(out_prefix)
    summary = {
        "input_state": base,
        "judge_file": str(judge_path),
        "requests_jsonl": str(args.requests_jsonl) if args.requests_jsonl else None,
        "output_state": out_prefix,
        "num_labels": int(len(labels)),
        "num_updated": int(ok),
        "num_direct_from_logs": int(direct),
        "num_missing": int(miss),
    }
    Path(out_prefix + "_ingest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Saved updated bandit state to: {out_prefix}.json/.npz")


if __name__ == "__main__":
    main()
