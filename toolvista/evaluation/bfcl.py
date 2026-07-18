"""ToolVista wrapper for the official BFCL v4 generation and evaluator."""

from __future__ import annotations

import csv
import json
import os
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from toolvista.evaluation.io import load_json, write_json
from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
CSV_METRICS = {
    "Non-Live Overall Acc": "bfcl/non_live_overall_accuracy",
    "AST Summary": "bfcl/ast_summary",
    "Simple AST": "bfcl/simple_ast",
    "Python Simple AST": "bfcl/python_simple_ast",
    "Java Simple AST": "bfcl/java_simple_ast",
    "JavaScript Simple AST": "bfcl/javascript_simple_ast",
    "Multiple AST": "bfcl/multiple_ast",
    "Parallel AST": "bfcl/parallel_ast",
    "Parallel Multiple AST": "bfcl/parallel_multiple_ast",
    "Irrelevance Detection": "bfcl/irrelevance_detection",
}


def _percentage(value: str, column: str) -> float:
    normalized = value.strip()
    if not normalized.endswith("%"):
        raise ValueError(f"BFCL CSV metric {column!r} is not a percentage: {value!r}")
    number = float(normalized[:-1]) / 100.0
    if not math.isfinite(number):
        raise ValueError(f"BFCL CSV metric {column!r} must be finite: {value!r}")
    return number


def parse_non_live_csv(path: Path) -> tuple[dict[str, str], dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"BFCL non-live CSV must contain exactly one model row: {path}")
    row = rows[0]
    missing = [column for column in CSV_METRICS if column not in row]
    if missing:
        raise ValueError(f"BFCL non-live CSV is missing columns: {missing}")
    metrics = {
        metric: _percentage(row[column], column)
        for column, metric in CSV_METRICS.items()
    }
    return row, metrics


def _manifest_count(path: Path) -> int:
    value = load_json(path)
    if not value:
        raise ValueError(f"BFCL ID manifest is empty: {path}")
    count = 0
    for category, ids in value.items():
        if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"BFCL manifest category {category!r} is malformed.")
        count += len(ids)
    return count


def _result_stats(result_dir: Path) -> tuple[int, int]:
    generated = 0
    api_failures = 0
    for path in result_dir.rglob("BFCL_v4_*_result.json"):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"Malformed BFCL result at {path}:{line_number}")
                generated += 1
                result = value.get("result")
                if isinstance(result, str) and result.startswith("Error during inference:"):
                    api_failures += 1
    return generated, api_failures


def run_bfcl_benchmark(
    spec: BenchmarkSpec,
    *,
    checkpoint: str,
    run_id: str,
    benchmark_dir: Path,
    api_base: str | None,
    api_model: str | None,
    timeout: float,
    score_only: bool,
    resume: bool,
    display_command: str,
    command_runner: CommandRunner = subprocess.run,
) -> EvaluationResult:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    result_path = benchmark_dir / "result.json"
    if resume and result_path.is_file():
        return EvaluationResult.from_dict(load_json(result_path))
    if score_only:
        raise ValueError("--score-only is not supported for BFCL; use --resume for an existing run.")
    if not api_base or not api_model:
        raise ValueError("BFCL evaluation requires --api-base and --api-model.")
    if not spec.dataset_path:
        raise ValueError("BFCL benchmark requires a frozen ID manifest in dataset_path.")

    manifest = Path(spec.dataset_path).expanduser().resolve()
    bfcl_python = Path(
        os.environ.get(
            "BFCL_PYTHON",
            str(spec.settings.get("python", "/root/autodl-tmp/venvs/bfcl/bin/python")),
        )
    ).expanduser()
    bootstrap = Path(
        str(spec.settings.get("bootstrap", "scripts/bfcl_toolvista_bootstrap.py"))
    ).expanduser().resolve()
    for path, label in ((manifest, "manifest"), (bfcl_python, "Python"), (bootstrap, "bootstrap")):
        if not path.is_file():
            raise FileNotFoundError(f"BFCL {label} is missing: {path}")

    workspace = (benchmark_dir / "bfcl_workspace").resolve()
    command = [
        str(bfcl_python),
        str(bootstrap),
        "--project-root",
        str(workspace),
        "--id-manifest",
        str(manifest),
        "--api-base",
        api_base,
        "--api-model",
        api_model,
        "--num-threads",
        str(int(spec.settings.get("num_threads", 4))),
        "--timeout",
        str(timeout),
    ]
    if spec.profile == "full":
        command.append("--full")

    stdout_path = benchmark_dir / "stdout.log"
    stderr_path = benchmark_dir / "stderr.log"
    start = time.perf_counter()
    launch_error: Exception | None = None
    process: subprocess.CompletedProcess[str] | None = None
    try:
        process = command_runner(
            command,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        launch_error = exc
    duration = time.perf_counter() - start
    stdout_path.write_text(process.stdout or "" if process else "", encoding="utf-8")
    stderr = process.stderr or "" if process else ""
    if launch_error is not None:
        stderr = f"{type(launch_error).__name__}: {launch_error}\n"
    stderr_path.write_text(stderr, encoding="utf-8")

    attempted = _manifest_count(manifest)
    generated, api_failures = _result_stats(workspace / "result") if (workspace / "result").is_dir() else (0, 0)
    csv_path = workspace / "score" / "data_non_live.csv"
    process_exit_code = process.returncode if process is not None else None
    status = "passed"
    upstream: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    if launch_error is not None or process_exit_code != 0 or generated != attempted or api_failures:
        status = "failed"
    elif not csv_path.is_file():
        status = "failed"
        with stderr_path.open("a", encoding="utf-8") as handle:
            handle.write(f"BFCL score CSV is missing: {csv_path}\n")
    else:
        try:
            csv_row, metrics = parse_non_live_csv(csv_path)
            missing_metrics = [name for name in spec.primary_metrics if name not in metrics]
            if missing_metrics:
                raise ValueError(f"BFCL CSV is missing primary metrics: {missing_metrics}")
            upstream = {
                "bfcl_version": "v4",
                "suite": "non_live",
                "partial_eval": spec.profile != "full",
                "csv_row": csv_row,
                "generated": generated,
                "api_failures": api_failures,
            }
        except Exception as exc:
            status = "failed"
            metrics = {}
            with stderr_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{type(exc).__name__}: {exc}\n")

    upstream_path = benchmark_dir / "upstream_result.json"
    write_json(upstream_path, upstream)
    artifacts = {
        "bfcl_results": (workspace / "result").as_posix(),
        "bfcl_scores": (workspace / "score").as_posix(),
        "bfcl_non_live_csv": csv_path.as_posix(),
        "upstream_result": upstream_path.as_posix(),
        "stdout": stdout_path.as_posix(),
        "stderr": stderr_path.as_posix(),
    }
    completed = max(0, generated - api_failures)
    failed = max(api_failures, attempted - completed)
    result = EvaluationResult(
        schema_version="1.0",
        run_id=run_id,
        checkpoint=checkpoint,
        benchmark=spec.benchmark_id,
        profile=spec.profile,
        prediction_mode="fresh",
        test_summary=TestSummary(
            method=spec.method,
            dataset=f"{spec.dataset_name} / {spec.split}",
            source_project=spec.source_project,
            evaluator=spec.evaluator,
            sample_count=attempted,
            command=display_command,
            main_metrics=spec.primary_metrics,
            result_path=result_path.as_posix(),
            artifact_paths=artifacts,
        ),
        status=status,
        attempted=attempted,
        completed=min(completed, attempted),
        failed=min(failed, attempted),
        upstream_metrics=upstream,
        metrics=metrics,
        runtime={
            "duration_seconds": duration,
            "peak_gpu_memory_mb": None,
            "exit_code": 0 if status == "passed" else process_exit_code or 1,
            "process_exit_code": process_exit_code,
        },
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result
