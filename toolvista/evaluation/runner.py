"""Sequential execution for the lightweight evaluation workflow."""

from __future__ import annotations

import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from toolvista.evaluation.io import (
    load_json,
    load_jsonl,
    normalize_swift_metrics,
    read_last_swift_eval_report,
    write_json,
    write_jsonl,
)
from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.scorers import score_mllm_router, score_xlam


PredictionClient = Callable[..., Mapping[str, Any]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def call_chat_completion(
    *,
    api_base: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    timeout: float,
) -> Mapping[str, Any]:
    """Call one OpenAI-compatible completion, retrying one transport failure."""
    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": 0,
        "max_tokens": 512,
    }
    if tools:
        payload["tools"] = list(tools)
        payload["tool_choice"] = "auto"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )

    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 0:
                time.sleep(0.5)
                continue
            raise
        if not isinstance(value, Mapping):
            raise ValueError("Chat completion response root must be an object.")
        return value
    raise AssertionError("unreachable")


def _assistant_message(response: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("Chat completion response has no first choice.")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Chat completion response has no assistant message.")
    return message


def extract_prediction(spec: BenchmarkSpec, response: Mapping[str, Any]) -> Any:
    message = _assistant_message(response)
    if spec.execution == "native_xlam":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return [
                call.get("function", call) if isinstance(call, Mapping) else call
                for call in tool_calls
            ]
        return message.get("content", "")
    if spec.execution == "native_mllm_router":
        return str(message.get("content") or "").strip()
    raise ValueError(f"Not a native benchmark: {spec.benchmark_id}")


def predict_native_rows(
    spec: BenchmarkSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    api_base: str,
    api_model: str,
    timeout: float,
    prediction_client: PredictionClient,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        output = dict(row)
        try:
            response = prediction_client(
                api_base=api_base,
                model=api_model,
                messages=row.get("messages", []),
                tools=row.get("tools", []),
                timeout=timeout,
            )
            if not isinstance(response, Mapping):
                raise ValueError("Chat completion response root must be an object.")
            output["prediction"] = extract_prediction(spec, response)
        except Exception as exc:
            output["error"] = f"{type(exc).__name__}: {exc}"
        predictions.append(output)
    return predictions


def _validate_complete_predictions(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"--score-only predictions must not be empty: {path}")
    for row_number, row in enumerate(rows, 1):
        has_prediction = "prediction" in row
        error = row.get("error")
        has_error = isinstance(error, str) and bool(error.strip())
        if has_prediction == has_error:
            raise ValueError(
                f"--score-only requires complete predictions: row {row_number} in {path} "
                "must contain exactly one of prediction or non-empty error."
            )


def _score_native(spec: BenchmarkSpec, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if spec.execution == "native_xlam":
        return score_xlam(rows)
    if spec.execution == "native_mllm_router":
        return score_mllm_router(rows)
    raise ValueError(f"Not a native benchmark: {spec.benchmark_id}")


def _validate_score_counts(scored: Mapping[str, Any], row_count: int) -> None:
    counts: dict[str, int] = {}
    for name in ("attempted", "completed", "failed"):
        value = scored.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Native scorer returned invalid {name} count: {value!r}")
        counts[name] = value
    if counts["attempted"] != row_count or counts["completed"] + counts["failed"] != row_count:
        raise ValueError(
            "Native scorer count mismatch: "
            f"rows={row_count}, attempted={counts['attempted']}, "
            f"completed={counts['completed']}, failed={counts['failed']}"
        )


def _artifact_paths(benchmark_dir: Path) -> dict[str, str]:
    return {
        "predictions": (benchmark_dir / "predictions.jsonl").as_posix(),
        "upstream_result": (benchmark_dir / "upstream_result.json").as_posix(),
        "stdout": (benchmark_dir / "stdout.log").as_posix(),
        "stderr": (benchmark_dir / "stderr.log").as_posix(),
    }


def run_native_benchmark(
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
    prediction_client: PredictionClient = call_chat_completion,
) -> EvaluationResult:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    result_path = benchmark_dir / "result.json"
    predictions_path = benchmark_dir / "predictions.jsonl"
    if resume and result_path.is_file():
        return EvaluationResult.from_dict(load_json(result_path))

    start = time.perf_counter()
    if score_only:
        if not predictions_path.is_file():
            raise FileNotFoundError(f"--score-only requires complete predictions: {predictions_path}")
        predictions = load_jsonl(predictions_path)
        prediction_mode = "score_only"
    else:
        if not spec.dataset_path:
            raise ValueError(f"Native benchmark has no dataset_path: {spec.benchmark_id}")
        if not api_base or not api_model:
            raise ValueError("Fresh native evaluation requires --api-base and --api-model.")
        manifest_rows = load_jsonl(spec.dataset_path)
        predictions = predict_native_rows(
            spec,
            manifest_rows,
            api_base=api_base,
            api_model=api_model,
            timeout=timeout,
            prediction_client=prediction_client,
        )
        if len(predictions) != len(manifest_rows):
            raise ValueError(
                "Native prediction count mismatch: "
                f"manifest={len(manifest_rows)}, predictions={len(predictions)}"
            )
        write_jsonl(predictions_path, predictions)
        prediction_mode = "fresh"

    _validate_complete_predictions(predictions, predictions_path)
    scored = _score_native(spec, predictions)
    _validate_score_counts(scored, len(predictions))
    upstream_path = benchmark_dir / "upstream_result.json"
    write_json(upstream_path, scored)
    stdout_path = benchmark_dir / "stdout.log"
    stderr_path = benchmark_dir / "stderr.log"
    stdout_path.write_text("native scorer completed\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    attempted = int(scored["attempted"])
    completed = int(scored["completed"])
    failed = int(scored["failed"])
    artifacts = _artifact_paths(benchmark_dir)
    result = EvaluationResult(
        schema_version="1.0",
        run_id=run_id,
        checkpoint=checkpoint,
        benchmark=spec.benchmark_id,
        profile=spec.profile,
        prediction_mode=prediction_mode,
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
        status="passed" if failed == 0 else "failed",
        attempted=attempted,
        completed=completed,
        failed=failed,
        upstream_metrics=dict(scored["upstream_metrics"]),
        metrics={str(key): float(value) for key, value in scored["metrics"].items()},
        runtime={
            "duration_seconds": time.perf_counter() - start,
            "peak_gpu_memory_mb": None,
            "exit_code": 0 if failed == 0 else 1,
        },
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result


def build_swift_eval_command(
    spec: BenchmarkSpec,
    *,
    checkpoint: str,
    eval_output_dir: Path,
    infer_backend: str,
    eval_url: str | None,
    swift_executable: str,
) -> list[str]:
    """Build the argv passed directly to ms-swift's upstream evaluator."""
    if spec.execution != "swift_eval" or not spec.eval_backend or not spec.swift_datasets:
        raise ValueError(f"Invalid swift eval benchmark: {spec.benchmark_id}")
    for name, value in (
        ("checkpoint", checkpoint),
        ("infer_backend", infer_backend),
        ("swift_executable", swift_executable),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"swift eval requires non-empty {name}.")
    command = [
        swift_executable,
        "eval",
        "--model",
        checkpoint,
        "--eval_backend",
        spec.eval_backend,
        "--infer_backend",
        infer_backend,
        "--eval_dataset",
        *spec.swift_datasets,
    ]
    if spec.eval_limit is not None:
        command.extend(["--eval_limit", str(spec.eval_limit)])
    command.extend(
        [
            "--eval_output_dir",
            eval_output_dir.as_posix(),
            "--eval_generation_config",
            '{"max_tokens":512,"temperature":0}',
        ]
    )
    if eval_url:
        command.extend(["--eval_url", eval_url])
    return command


def _resolve_checkpoint_reference(checkpoint: str) -> str:
    source = Path(checkpoint).expanduser()
    if source.exists():
        return source.resolve().as_posix()
    return checkpoint


def _swift_result_candidates(checkpoint: str, benchmark_dir: Path) -> list[Path]:
    source = Path(checkpoint)
    candidates = [source / "eval_result.jsonl"] if source.is_dir() else []
    candidates.extend(sorted((benchmark_dir / "result").rglob("eval_result.jsonl")))
    return candidates


def _swift_result_row_counts(checkpoint: str, benchmark_dir: Path) -> dict[Path, int]:
    counts: dict[Path, int] = {}
    for candidate in _swift_result_candidates(checkpoint, benchmark_dir):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                counts[candidate.resolve()] = sum(1 for line in handle if line.strip())
        except FileNotFoundError:
            continue
    return counts


def _read_swift_report(
    checkpoint: str,
    benchmark_dir: Path,
    *,
    backend: str,
    eval_output_dir: Path,
    previous_row_counts: Mapping[Path, int],
) -> dict[str, Any]:
    errors: list[str] = []
    for candidate in _swift_result_candidates(checkpoint, benchmark_dir):
        try:
            rows = load_jsonl(candidate)
            previous_count = previous_row_counts.get(candidate.resolve(), 0)
            if len(rows) <= previous_count:
                errors.append(f"No new swift eval report rows in {candidate}")
                continue
            report = read_last_swift_eval_report(
                candidate,
                backend=backend,
                eval_output_dir=eval_output_dir,
            )
            if report not in rows[previous_count:]:
                errors.append(f"No new matching swift eval report in {candidate}")
                continue
            return report
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    raise ValueError(f"Could not find matching swift eval report: {errors}")


def run_swift_benchmark(
    spec: BenchmarkSpec,
    *,
    checkpoint: str,
    run_id: str,
    benchmark_dir: Path,
    infer_backend: str,
    eval_url: str | None,
    resume: bool,
    swift_executable: str = "swift",
    command_runner: CommandRunner = subprocess.run,
) -> EvaluationResult:
    """Run ms-swift/VLMEvalKit and wrap its unmodified report and logs."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    result_path = benchmark_dir / "result.json"
    if resume and result_path.is_file():
        return EvaluationResult.from_dict(load_json(result_path))
    if result_path.is_file():
        result_path.unlink()

    resolved_checkpoint = _resolve_checkpoint_reference(checkpoint)
    eval_output_dir = (benchmark_dir / "swift_eval").resolve()
    command = build_swift_eval_command(
        spec,
        checkpoint=resolved_checkpoint,
        eval_output_dir=eval_output_dir,
        infer_backend=infer_backend,
        eval_url=eval_url,
        swift_executable=swift_executable,
    )
    stdout_path = benchmark_dir / "stdout.log"
    stderr_path = benchmark_dir / "stderr.log"
    upstream_path = benchmark_dir / "upstream_result.json"
    if upstream_path.is_file():
        upstream_path.unlink()
    artifacts = {
        "stdout": stdout_path.as_posix(),
        "stderr": stderr_path.as_posix(),
        "swift_eval_output": eval_output_dir.as_posix(),
    }

    start = time.perf_counter()
    previous_row_counts = _swift_result_row_counts(resolved_checkpoint, benchmark_dir)
    launch_error: Exception | None = None
    completed_process: subprocess.CompletedProcess[str] | None = None
    try:
        completed_process = command_runner(
            command,
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        launch_error = exc
    duration = time.perf_counter() - start

    stdout = completed_process.stdout or "" if completed_process is not None else ""
    stderr = completed_process.stderr or "" if completed_process is not None else ""
    if launch_error is not None:
        stderr = f"{type(launch_error).__name__}: {launch_error}\n"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    exit_code = completed_process.returncode if completed_process is not None else None
    status = "failed" if launch_error is not None or exit_code != 0 else "passed"
    upstream_metrics: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    if status == "passed":
        try:
            report = _read_swift_report(
                resolved_checkpoint,
                benchmark_dir,
                backend=str(spec.eval_backend),
                eval_output_dir=eval_output_dir,
                previous_row_counts=previous_row_counts,
            )
            write_json(upstream_path, report)
            artifacts["upstream_result"] = upstream_path.as_posix()
            upstream_metrics, metrics = normalize_swift_metrics(
                report,
                backend=str(spec.eval_backend),
                metric_scales=spec.metric_scales,
            )
            missing_metrics = [metric for metric in spec.primary_metrics if metric not in metrics]
            if missing_metrics:
                raise ValueError(f"swift eval report is missing primary metrics: {missing_metrics}")
        except Exception as exc:
            status = "failed"
            upstream_metrics = {}
            metrics = {}
            separator = "\n" if stderr else ""
            with stderr_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{separator}{type(exc).__name__}: {exc}\n")

    planned = (spec.eval_limit or 0) * len(spec.swift_datasets)
    process_exit_code = completed_process.returncode if completed_process is not None else None
    benchmark_exit_code = 0 if status == "passed" else process_exit_code or 1
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
            sample_count=spec.eval_limit,
            command=shlex.join(command),
            main_metrics=spec.primary_metrics,
            result_path=result_path.as_posix(),
            artifact_paths=artifacts,
        ),
        status=status,  # type: ignore[arg-type]
        attempted=planned,
        completed=planned if status == "passed" else 0,
        failed=0 if status == "passed" else planned,
        upstream_metrics=upstream_metrics,
        metrics=metrics,
        runtime={
            "duration_seconds": duration,
            "peak_gpu_memory_mb": None,
            "exit_code": benchmark_exit_code,
            "process_exit_code": process_exit_code,
        },
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result
