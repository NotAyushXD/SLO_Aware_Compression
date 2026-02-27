from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_experiment_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "command" not in data:
        raise ValueError(f"Config must be a JSON object with a 'command' list: {path}")
    if not isinstance(data["command"], list):
        raise ValueError("config['command'] must be a list of CLI args")
    return data


def run_baseline_eval(
    *,
    repo_dir: Path,
    base_command: List[str],
    out_dir: Path,
    seed: int,
    extra_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """Run run_baseline_evaluation.py as a subprocess."""

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["python", "run_baseline_evaluation.py"]
    cmd += list(base_command)
    cmd += ["--seed", str(int(seed)), "--output_dir", str(out_dir)]
    if extra_args:
        cmd += list(extra_args)

    subprocess.run(cmd, cwd=str(repo_dir), env=env, check=True)


def find_metrics_file(out_dir: Path) -> Path:
    """Pick a representative metrics file from a run directory."""

    # Prefer schedule metrics if present.
    sched = out_dir / "metrics_schedule.json"
    if sched.exists():
        return sched

    # Otherwise pick highest concurrency metrics.
    cand = sorted(out_dir.glob("metrics_concurrency_*.json"))
    if cand:
        # sort by concurrency number
        def key(p: Path) -> int:
            try:
                s = p.stem.split("_")[-1]
                return int(s)
            except Exception:
                return 0

        cand = sorted(cand, key=key)
        return cand[-1]

    # Or phase metrics
    cand2 = sorted(out_dir.glob("metrics_phase_*_concurrency_*.json"))
    if cand2:
        return cand2[-1]

    raise FileNotFoundError(f"No metrics files found in {out_dir}")


def load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_frontier_point(metrics: Dict[str, Any]) -> Dict[str, float]:
    summ = (metrics.get("summary") or {})
    succ = float(summ.get("successful_requests", 0) or 0.0)
    total_cost = float(summ.get("total_cost_units", 0.0) or 0.0)
    cost_per_req = total_cost / succ if succ > 0 else 0.0

    return {
        "accuracy": float(summ.get("accuracy_success", 0.0) or 0.0),
        "slo_compliance": float(summ.get("slo_compliance", 0.0) or 0.0),
        "violation_rate": 1.0 - float(summ.get("slo_compliance", 0.0) or 0.0),
        "cost_per_request": float(cost_per_req),
    }
