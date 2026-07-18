#!/usr/bin/env python3
"""Print a read-only progress and liveness snapshot for a ToolVista eval run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.io import load_benchmark_registry, load_suites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--suite", default="c0_light")
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/benchmarks.yaml",
    )
    parser.add_argument(
        "--suite-config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation/suites.yaml",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--warning-seconds", type=int, default=300)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def count_jsonl(path: Path) -> int | None:
    try:
        with path.open("rb") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return None


def planned_samples(spec: Any) -> int | None:
    if spec.execution == "swift_eval" and spec.eval_limit is not None:
        return spec.eval_limit * len(spec.swift_datasets)
    if spec.execution == "toolvqa_step":
        return spec.eval_limit
    if spec.dataset_path and str(spec.dataset_path).endswith(".jsonl"):
        path = Path(spec.dataset_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return count_jsonl(path)
    return None


def result_progress(result_path: Path) -> tuple[str, int | None, int | None]:
    result = load_json(result_path)
    if result is None:
        return "invalid result", None, None
    attempted = result.get("attempted")
    completed = result.get("completed")
    status = str(result.get("status", "unknown"))
    if isinstance(attempted, int) and isinstance(completed, int):
        return status, completed, attempted
    return status, None, None


def toolvqa_progress(benchmark_dir: Path, planned: int | None) -> tuple[int, int] | None:
    final_files = sorted(
        path
        for path in benchmark_dir.glob("upstream/*/predictions/*/gta_bench.json")
        if not path.name.startswith("tmp_")
    )
    if final_files and planned is not None:
        return planned, planned
    temporary_files = sorted(benchmark_dir.glob("upstream/*/predictions/*/tmp_gta_bench.json"))
    if not temporary_files or planned is None:
        return None
    payload = load_json(temporary_files[-1])
    return (len(payload), planned) if payload is not None else None


def general_dataset_progress(benchmark_dir: Path, datasets: list[str]) -> tuple[int, int]:
    attempt_dirs = sorted(path for path in (benchmark_dir / "swift_eval" / "vlmeval").glob("*") if path.is_dir())
    if not attempt_dirs:
        return 0, len(datasets)
    names = [path.name.lower() for path in attempt_dirs[-1].rglob("*.xlsx")]
    predicted = sum(
        any(name.endswith(f"_{dataset.lower()}.xlsx") for name in names)
        for dataset in datasets
    )
    return predicted, len(datasets)


def newest_artifact_age(run_dir: Path) -> tuple[int | None, Path | None]:
    newest_path: Path | None = None
    newest_time = -1.0
    try:
        candidates = run_dir.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            if modified > newest_time:
                newest_time = modified
                newest_path = path
    except OSError:
        return None, None
    if newest_path is None:
        return None, None
    return max(0, int(time.time() - newest_time)), newest_path


def evaluator_processes(run_id: str) -> list[str]:
    try:
        output = subprocess.check_output(
            ["ps", "ww", "-eo", "pid=,etimes=,stat=,pcpu=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    marker = f"--run-id {run_id}"
    return [
        line.strip()
        for line in output.splitlines()
        if "scripts/evaluate.py" in line and marker in line
    ]

def api_health(api_base: str) -> str:
    url = f"{api_base.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            return "OK" if 200 <= response.status < 300 else "FAILED"
    except (TimeoutError, socket.timeout):
        return "BUSY"
    except urllib.error.URLError as exc:
        return "BUSY" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "FAILED"
    except (OSError, ValueError):
        return "FAILED"
        return False


def gpu_snapshot() -> tuple[int | None, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).splitlines()[0]
        utilization, used, total = (part.strip() for part in line.split(","))
        return int(utilization), f"{utilization}% | {used}/{total} MiB"
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return None, "unavailable"


def format_progress(done: int | None, total: int | None) -> str:
    if done is None or total is None or total <= 0:
        return "N/A"
    return f"{done}/{total} ({100.0 * done / total:.1f}%)"


def main() -> int:
    args = parse_args()
    run_dir = args.runs_root / args.run_id
    if not run_dir.is_dir():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    registry = load_benchmark_registry(args.benchmark_config)
    suites = load_suites(args.suite_config)
    if args.suite not in suites:
        print(f"Unknown suite: {args.suite}", file=sys.stderr)
        return 2
    benchmark_ids = suites[args.suite]
    processes = evaluator_processes(args.run_id)
    api_status = api_health(args.api_base)
    gpu_utilization, gpu_text = gpu_snapshot()
    artifact_age, artifact_path = newest_artifact_age(run_dir / "eval")

    completed_steps = 0
    current_index: int | None = None
    rows: list[tuple[str, str, str]] = []
    observable_done = 0
    observable_total = 0

    for index, benchmark_id in enumerate(benchmark_ids, start=1):
        spec = registry[benchmark_id]
        benchmark_dir = run_dir / "eval" / benchmark_id
        result_path = benchmark_dir / "result.json"
        planned = planned_samples(spec)
        if result_path.is_file():
            status, done, total = result_progress(result_path)
            rows.append((benchmark_id, status, format_progress(done, total)))
            completed_steps += 1
            if done is not None and total is not None:
                observable_done += done
                observable_total += total
            continue

        if current_index is None:
            current_index = index
            if spec.execution == "toolvqa_step":
                progress = toolvqa_progress(benchmark_dir, planned)
                done, total = progress if progress is not None else (None, planned)
                detail = format_progress(done, total)
            elif spec.execution == "swift_eval":
                dataset_done, dataset_total = general_dataset_progress(
                    benchmark_dir, list(spec.swift_datasets)
                )
                detail = f"prediction files {format_progress(dataset_done, dataset_total)}; sample % N/A"
            else:
                detail = "sample % N/A (predictions are written atomically at completion)"
            rows.append((benchmark_id, "RUNNING" if processes else "INCOMPLETE", detail))
        else:
            rows.append((benchmark_id, "pending", format_progress(0, planned)))

    print(time.strftime("ToolVista eval snapshot | %Y-%m-%d %H:%M:%S %Z"))
    print(f"run_id: {args.run_id}")
    print(f"suite: {args.suite} | completed steps: {completed_steps}/{len(rows)} "
          f"({100.0 * completed_steps / len(rows):.1f}%)")
    if current_index is not None:
        print(f"current step: {current_index}/{len(rows)}")
    elif completed_steps == len(rows):
        print("current step: all benchmarks completed")
    print()
    for index, (benchmark_id, status, detail) in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {benchmark_id:<27} {status:<10} {detail}")

    print()
    print(f"evaluator: {'RUNNING' if processes else 'NOT FOUND'}")
    for process in processes:
        print(f"  {process}")
    print(f"API: {api_status} | GPU: {gpu_text}")
    if artifact_age is None or artifact_path is None:
        print("newest artifact: unavailable")
    else:
        try:
            display_path = artifact_path.relative_to(run_dir)
        except ValueError:
            display_path = artifact_path
        print(f"newest artifact: {artifact_age}s ago | {display_path}")

    if completed_steps == len(rows):
        health = "COMPLETE"
        reason = "all benchmark result files exist"
    elif not processes:
        health = "STOPPED/FAILED"
        reason = "the evaluator process is missing before all results were written"
    elif api_status == "FAILED":
        health = "UNHEALTHY"
        reason = "the model API health check failed"
    elif gpu_utilization is not None and gpu_utilization >= 10:
        health = "WORKING"
        reason = "the evaluator is alive and the GPU is active; API may be busy"
    elif artifact_age is not None and artifact_age >= args.warning_seconds:
        health = "CHECK"
        reason = (
            f"GPU utilization is low and no artifact changed for {artifact_age}s; "
            "inspect the active benchmark logs before stopping anything"
        )
    else:
        health = "WAITING"
        reason = "processes are alive; the API may be busy and scoring/download phases may use little GPU"
    print(f"health: {health} | {reason}")
    print("Note: N/A is intentional when an upstream runner does not expose in-flight counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
