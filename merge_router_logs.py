# merge_router_logs.py
import argparse, json
from collections import defaultdict

def read_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slo_logs", required=True, help="train_logs_slo.jsonl")
    ap.add_argument("--acc_logs", required=False, help="train_logs_acc.jsonl (optional)")
    ap.add_argument("--output", required=True, help="router_train.jsonl")
    args = ap.parse_args()

    slo = read_jsonl(args.slo_logs)
    acc = read_jsonl(args.acc_logs) if args.acc_logs else []

    acc_map = {r.get("example_id", r.get("request_id")): r for r in acc}

    out = []
    for r in slo:
        key = r.get("example_id", r.get("request_id"))
        prompt = r.get("prompt_text") or r.get("prompt") or ""
        quality = 1 if r.get("is_correct") else 0

        latency = (
            r.get("total_latency_ms")
            or r.get("e2e_latency_ms")
            or r.get("latency_ms")
            or None
        )

        merged = {
            "id": key,
            "prompt": prompt,
            "label_quality": quality,
            "label_latency_ms": latency,
            "slo_ttft_ms": r.get("ttft_ms"),
            "slo_tpot_ms": r.get("tpot_ms"),
            "dataset": r.get("dataset"),
            "difficulty": r.get("difficulty"),
        }

        if key in acc_map:
            merged["acc_is_correct"] = 1 if acc_map[key].get("is_correct") else 0
            merged["acc_total_latency_ms"] = (
                acc_map[key].get("total_latency_ms")
                or acc_map[key].get("latency_ms")
            )
            merged["acc_pred"] = acc_map[key].get("pred")

        out.append(merged)

    with open(args.output, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")

    print(f"[OK] Wrote {len(out)} merged training rows -> {args.output}")

if __name__ == "__main__":
    main()