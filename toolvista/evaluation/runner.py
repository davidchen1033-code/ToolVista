"""Sequential execution for the lightweight evaluation workflow."""

from __future__ import annotations

import datetime as dt
import importlib
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from toolvista.evaluation.io import (
    load_json,
    load_jsonl,
    normalize_swift_metrics,
    normalize_vlmevalkit_artifact_metrics,
    read_last_swift_eval_report,
    write_json,
    write_jsonl,
)
from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.scorers import score_mllm_router, score_xlam


PredictionClient = Callable[..., Mapping[str, Any]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _chat_completions_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


def _swift_subprocess_environment(
    *,
    judge_api_base: str | None = None,
    judge_model: str | None = None,
) -> dict[str, str]:
    """Route VLMEvalKit LLM-as-judge scoring through one fixed endpoint."""
    environment = os.environ.copy()
    if bool(judge_api_base) != bool(judge_model):
        raise ValueError("Set both judge_api_base and judge_model, or neither.")
    if judge_api_base and judge_model:
        environment["OPENAI_API_KEY"] = "EMPTY"
        environment["OPENAI_API_BASE"] = _chat_completions_endpoint(judge_api_base)
        environment["LOCAL_LLM"] = judge_model
        return environment
    openai_key = environment.get("OPENAI_API_KEY")
    if openai_key is not None and openai_key.strip().upper() in {"", "EMPTY"}:
        environment.pop("OPENAI_API_KEY", None)
    return environment


def _openai_multimodal_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Evaluation row messages must be a list.")
    media = {
        "image": iter(row.get("images", [])),
        "video": iter(row.get("videos", [])),
        "audio": iter(row.get("audios", [])),
    }
    tags = {"<image>": "image", "<video>": "video", "<audio>": "audio"}
    converted: list[dict[str, Any]] = []
    for message in messages:
        output = dict(message)
        content = output.get("content")
        if not isinstance(content, str) or not any(tag in content for tag in tags):
            converted.append(output)
            continue
        parts: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(content):
            matches = [(content.find(tag, cursor), tag) for tag in tags]
            matches = [(index, tag) for index, tag in matches if index >= 0]
            if not matches:
                if content[cursor:]:
                    parts.append({"type": "text", "text": content[cursor:]})
                break
            index, tag = min(matches)
            if index > cursor:
                parts.append({"type": "text", "text": content[cursor:index]})
            modality = tags[tag]
            try:
                path = next(media[modality])
            except StopIteration as exc:
                raise ValueError(f"Missing media path for {tag} in evaluation row.") from exc
            url_key = f"{modality}_url"
            parts.append({"type": url_key, url_key: {"url": str(path)}})
            cursor = index + len(tag)
        output["content"] = parts
        converted.append(output)
    for modality, values in media.items():
        try:
            extra = next(values)
        except StopIteration:
            continue
        raise ValueError(f"Unreferenced {modality} media path in evaluation row: {extra}")
    return converted


def call_chat_completion(
    *,
    api_base: str,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    timeout: float,
) -> Mapping[str, Any]:
    """Call one OpenAI-compatible completion, retrying one transport failure."""
    endpoint = _chat_completions_endpoint(api_base)
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
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                detail = ""
            message = f"HTTP {exc.code}: {exc.reason}"
            if detail:
                message += f": {detail}"
            raise RuntimeError(message) from exc
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
    progress_path: Path | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    if progress_path is not None:
        write_json(
            progress_path,
            {"phase": "predicting", "completed": 0, "total": len(rows)},
        )
    for row in rows:
        output = dict(row)
        try:
            messages = (
                _openai_multimodal_messages(row)
                if spec.execution == "native_mllm_router"
                else row.get("messages", [])
            )
            response = prediction_client(
                api_base=api_base,
                model=api_model,
                messages=messages,
                tools=row.get("tools", []),
                timeout=timeout,
            )
            if not isinstance(response, Mapping):
                raise ValueError("Chat completion response root must be an object.")
            output["prediction"] = extract_prediction(spec, response)
        except Exception as exc:
            output["error"] = f"{type(exc).__name__}: {exc}"
        predictions.append(output)
        if progress_path is not None:
            write_json(
                progress_path,
                {
                    "phase": "predicting",
                    "completed": len(predictions),
                    "total": len(rows),
                },
            )
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
            progress_path=benchmark_dir / "progress.json",
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
    served_model_name: str | None = None,
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
    if served_model_name:
        command.extend(["--served_model_name", served_model_name])
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


def _vlmeval_run_dirs(eval_output_dir: Path) -> set[Path]:
    root = eval_output_dir / "vlmeval"
    if not root.is_dir():
        return set()
    return {path.resolve() for path in root.iterdir() if path.is_dir()}


def _new_vlmeval_run_dir(eval_output_dir: Path, *, previous: set[Path]) -> Path:
    created = sorted(_vlmeval_run_dirs(eval_output_dir) - previous)
    if len(created) != 1:
        raise ValueError(
            f"Expected exactly one new VLMEvalKit timestamp directory in "
            f"{eval_output_dir / 'vlmeval'}, found {len(created)}."
        )
    return created[0]


def _check_fixed_judge(*, api_base: str, model: str, timeout: float = 15.0) -> None:
    response = call_chat_completion(
        api_base=api_base,
        model=model,
        messages=[{"role": "user", "content": "Reply with OK."}],
        tools=[],
        timeout=timeout,
    )
    message = _assistant_message(response)
    if not str(message.get("content") or "").strip():
        raise ValueError("Fixed judge health check returned empty content.")


def run_swift_benchmark(
    spec: BenchmarkSpec,
    *,
    checkpoint: str,
    run_id: str,
    benchmark_dir: Path,
    infer_backend: str,
    eval_url: str | None,
    api_model: str | None = None,
    judge_api_base: str | None = None,
    judge_model: str | None = None,
    resume: bool,
    swift_executable: str = "swift",
    command_runner: CommandRunner = subprocess.run,
) -> EvaluationResult:
    """Run ms-swift/VLMEvalKit and wrap its unmodified report and logs."""
    if "MathVision_MINI" in spec.swift_datasets and not (
        judge_api_base and judge_model
    ):
        raise ValueError(
            "MathVision_MINI requires a fixed LLM judge. Set both "
            "--judge-api-base and --judge-model; do not use the evaluated candidate "
            "as a changing judge across C0-C4."
        )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    result_path = benchmark_dir / "result.json"
    if resume and result_path.is_file():
        resumed = EvaluationResult.from_dict(load_json(result_path))
        if resumed.status == "passed":
            return resumed
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
        served_model_name=api_model,
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
    previous_vlmeval_dirs = _vlmeval_run_dirs(eval_output_dir)
    if "MathVision_MINI" in spec.swift_datasets:
        assert judge_api_base is not None and judge_model is not None
        _check_fixed_judge(api_base=judge_api_base, model=judge_model)
    launch_error: Exception | None = None
    completed_process: subprocess.CompletedProcess[str] | None = None
    try:
        completed_process = command_runner(
            command,
            cwd=benchmark_dir,
            capture_output=True,
            text=True,
            check=False,
            env=_swift_subprocess_environment(
                judge_api_base=judge_api_base,
                judge_model=judge_model,
            ),
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
            if missing_metrics and str(spec.eval_backend) == "VLMEvalKit":
                try:
                    vlmeval_run_dir = _new_vlmeval_run_dir(
                        eval_output_dir,
                        previous=previous_vlmeval_dirs,
                    )
                    upstream_metrics, metrics, raw_artifacts = (
                        normalize_vlmevalkit_artifact_metrics(
                            vlmeval_run_dir,
                            datasets=spec.swift_datasets,
                        )
                    )
                    artifacts.update(raw_artifacts)
                    missing_metrics = [
                        metric for metric in spec.primary_metrics if metric not in metrics
                    ]
                except Exception as artifact_exc:
                    raise ValueError(
                        f"swift eval report is missing primary metrics: {missing_metrics}; "
                        f"VLMEvalKit artifact fallback failed: "
                        f"{type(artifact_exc).__name__}: {artifact_exc}"
                    ) from artifact_exc
            if missing_metrics:
                raise ValueError(f"swift eval report is missing primary metrics: {missing_metrics}")
            write_json(
                upstream_path,
                {
                    **report,
                    "toolvista_upstream_metrics": upstream_metrics,
                    "toolvista_metrics": metrics,
                },
            )
        except Exception as exc:
            status = "failed"
            upstream_metrics = {}
            metrics = {}
            separator = "\n" if stderr else ""
            with stderr_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{separator}{type(exc).__name__}: {exc}\n")

    planned = spec.eval_limit * len(spec.swift_datasets) if spec.eval_limit is not None else None
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
        completed=planned if planned is not None and status == "passed" else (0 if planned is not None else None),
        failed=0 if planned is not None and status == "passed" else planned,
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


def _finite_metric(value: Any, *, name: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} metric {name!r} must be a finite number.")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{source} metric {name!r} must be finite.")
    return numeric


def calculate_comparisons(
    current_metrics: Mapping[str, float],
    *,
    baseline_metrics: Mapping[str, float],
    historical_metrics: Sequence[Mapping[str, float]],
    metric_directions: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Compare current metrics with one baseline and prior summaries."""
    comparisons: dict[str, dict[str, float]] = {}
    for metric_name, direction in sorted(metric_directions.items()):
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"Metric {metric_name!r} has invalid direction: {direction!r}")
        if metric_name not in current_metrics:
            continue

        current = _finite_metric(
            current_metrics[metric_name], name=metric_name, source="current"
        )
        row: dict[str, float] = {"current": current}
        base: float | None = None
        if metric_name in baseline_metrics:
            base = _finite_metric(
                baseline_metrics[metric_name], name=metric_name, source="baseline"
            )
            row["base"] = base
            row["gain_vs_base"] = current - base if direction == "maximize" else base - current
            if direction == "maximize" and base != 0.0:
                row["retention"] = current / base
            elif direction == "minimize" and current != 0.0:
                row["retention"] = base / current

        history = [
            _finite_metric(values[metric_name], name=metric_name, source="historical")
            for values in historical_metrics
            if metric_name in values
        ]
        if base is not None:
            history.append(base)
        if history:
            historical_best = max(history) if direction == "maximize" else min(history)
            row["historical_best"] = historical_best
            raw_forgetting = (
                historical_best - current
                if direction == "maximize"
                else current - historical_best
            )
            row["forgetting"] = max(0.0, raw_forgetting)
        comparisons[metric_name] = row
    return comparisons


def _exception_result(
    spec: BenchmarkSpec,
    *,
    checkpoint: str,
    run_id: str,
    benchmark_dir: Path,
    display_command: str,
    score_only: bool,
    error: Exception,
) -> EvaluationResult:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = benchmark_dir / "stderr.log"
    stdout_path = benchmark_dir / "stdout.log"
    result_path = benchmark_dir / "result.json"
    wrapper_error = f"{type(error).__name__}: {error}\n"
    if stderr_path.is_file() and stderr_path.stat().st_size:
        with stderr_path.open("a", encoding="utf-8") as handle:
            if not stderr_path.read_bytes().endswith(b"\n"):
                handle.write("\n")
            handle.write(wrapper_error)
    else:
        stderr_path.write_text(wrapper_error, encoding="utf-8")
    if not stdout_path.is_file():
        stdout_path.write_text("", encoding="utf-8")
    artifacts = {"stdout": stdout_path.as_posix(), "stderr": stderr_path.as_posix()}
    result = EvaluationResult(
        schema_version="1.0",
        run_id=run_id,
        checkpoint=checkpoint,
        benchmark=spec.benchmark_id,
        profile=spec.profile,
        prediction_mode="score_only" if score_only else "fresh",
        test_summary=TestSummary(
            method=spec.method,
            dataset=f"{spec.dataset_name} / {spec.split}",
            source_project=spec.source_project,
            evaluator=spec.evaluator,
            sample_count=spec.eval_limit,
            command=display_command,
            main_metrics=spec.primary_metrics,
            result_path=result_path.as_posix(),
            artifact_paths=artifacts,
        ),
        status="failed",
        attempted=0,
        completed=0,
        failed=0,
        upstream_metrics={},
        metrics={},
        runtime={"duration_seconds": 0.0, "peak_gpu_memory_mb": None, "exit_code": 1},
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result


def _suite_status(results: Sequence[EvaluationResult]) -> str:
    if any(result.status in {"failed", "blocked"} for result in results):
        return "failed"
    if all(result.status == "skipped" for result in results):
        return "skipped"
    return "passed"


def _validate_suite_specs(specs: Sequence[BenchmarkSpec]) -> None:
    if not specs:
        raise ValueError("An evaluation suite requires at least one benchmark.")
    benchmark_ids: set[str] = set()
    metric_owners: dict[str, str] = {}
    for spec in specs:
        if spec.benchmark_id in benchmark_ids:
            raise ValueError(f"Evaluation suite contains duplicate benchmark: {spec.benchmark_id}")
        benchmark_ids.add(spec.benchmark_id)
        for metric_name in spec.metric_directions:
            owner = metric_owners.get(metric_name)
            if owner is not None:
                raise ValueError(
                    f"Evaluation suite contains duplicate metric {metric_name!r} in "
                    f"{owner!r} and {spec.benchmark_id!r}."
                )
            metric_owners[metric_name] = spec.benchmark_id


def _validate_judge_configuration(
    specs: Sequence[BenchmarkSpec],
    *,
    judge_api_base: str | None,
    judge_model: str | None,
) -> None:
    if bool(judge_api_base) != bool(judge_model):
        raise ValueError("Set both judge_api_base and judge_model, or neither.")
    requires_judge = any(
        spec.execution == "swift_eval" and "MathVision_MINI" in spec.swift_datasets
        for spec in specs
    )
    if requires_judge and not judge_api_base:
        raise ValueError(
            "MathVision_MINI requires a fixed LLM judge. Set both "
            "--judge-api-base and --judge-model before starting the suite."
        )


def _validate_result_identity(
    result: EvaluationResult,
    spec: BenchmarkSpec,
    *,
    checkpoint: str,
    run_id: str,
    benchmark_dir: Path,
) -> None:
    mismatches: list[str] = []
    expected_values = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "benchmark": spec.benchmark_id,
        "profile": spec.profile,
    }
    for name, expected in expected_values.items():
        actual = getattr(result, name)
        if actual != expected:
            mismatches.append(f"{name}={actual!r}, expected {expected!r}")
    if result.test_summary.main_metrics != spec.primary_metrics:
        mismatches.append(
            f"main_metrics={result.test_summary.main_metrics!r}, expected {spec.primary_metrics!r}"
        )
    expected_result_path = (benchmark_dir / "result.json").resolve()
    if Path(result.test_summary.result_path).resolve() != expected_result_path:
        mismatches.append(
            f"result_path={result.test_summary.result_path!r}, expected {expected_result_path.as_posix()!r}"
        )
    if result.status == "passed":
        missing_metrics = [metric for metric in spec.primary_metrics if metric not in result.metrics]
        if missing_metrics:
            mismatches.append(f"missing primary metrics={missing_metrics!r}")
    if mismatches:
        raise ValueError("Evaluation result does not match requested benchmark: " + "; ".join(mismatches))


def _flat_metrics(results: Sequence[EvaluationResult]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    owners: dict[str, str] = {}
    for result in results:
        for metric_name, value in result.metrics.items():
            owner = owners.get(metric_name)
            if owner is not None:
                raise ValueError(
                    f"Evaluation results contain duplicate metric {metric_name!r} in "
                    f"{owner!r} and {result.benchmark!r}."
                )
            metrics[metric_name] = _finite_metric(
                value, name=metric_name, source=f"benchmark {result.benchmark!r}"
            )
            owners[metric_name] = result.benchmark
    return metrics


def _metric_directions(specs: Sequence[BenchmarkSpec]) -> dict[str, str]:
    return {
        metric_name: direction
        for spec in specs
        for metric_name, direction in spec.metric_directions.items()
    }


def _toolvista_commit() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def _safe_endpoint(value: str | None) -> str | None:
    """Keep endpoint provenance without persisting credentials or query tokens."""
    if not value:
        return None
    try:
        parsed = urllib.parse.urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return None
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return None


def evaluate_suite(
    specs: Sequence[BenchmarkSpec],
    *,
    checkpoint: str,
    run_id: str,
    runs_root: Path,
    api_base: str | None,
    api_model: str | None,
    infer_backend: str,
    eval_url: str | None,
    timeout: float,
    score_only: bool,
    resume: bool,
    display_command: str,
    stage: str | None,
    algorithm: str | None,
    seed: int,
    baseline_metrics: Mapping[str, float],
    historical_metrics: Sequence[Mapping[str, float]],
    judge_api_base: str | None = None,
    judge_model: str | None = None,
    swift_executable: str = "swift",
    native_runner: Callable[..., EvaluationResult] = run_native_benchmark,
    swift_runner: Callable[..., EvaluationResult] = run_swift_benchmark,
    toolvqa_runner: Callable[..., EvaluationResult] | None = None,
    bfcl_runner: Callable[..., EvaluationResult] | None = None,
    visual_rft_runner: Callable[..., EvaluationResult] | None = None,
) -> dict[str, Any]:
    """Run a suite sequentially and write one local manifest and summary."""
    _validate_suite_specs(specs)
    _validate_judge_configuration(
        specs,
        judge_api_base=judge_api_base,
        judge_model=judge_model,
    )
    run_dir = runs_root / run_id
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (run_dir / "manifest.json", run_dir / "summary.json"):
        if stale_path.is_file():
            stale_path.unlink()
    if not resume:
        for spec in specs:
            stale_result = eval_dir / spec.benchmark_id / "result.json"
            if stale_result.is_file():
                stale_result.unlink()
    start_time = dt.datetime.now(dt.timezone.utc)
    results: list[EvaluationResult] = []

    for spec in specs:
        benchmark_dir = eval_dir / spec.benchmark_id
        try:
            if spec.execution.startswith("native_"):
                result = native_runner(
                    spec,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    benchmark_dir=benchmark_dir,
                    api_base=api_base,
                    api_model=api_model,
                    timeout=timeout,
                    score_only=score_only,
                    resume=resume,
                    display_command=display_command,
                )
            elif spec.execution == "toolvqa_step":
                if toolvqa_runner is None:
                    from toolvista.evaluation.toolvqa_benchmark import run_toolvqa_benchmark

                    selected_toolvqa_runner = run_toolvqa_benchmark
                else:
                    selected_toolvqa_runner = toolvqa_runner
                result = selected_toolvqa_runner(
                    spec,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    benchmark_dir=benchmark_dir,
                    api_base=api_base,
                    api_model=api_model,
                    timeout=timeout,
                    score_only=score_only,
                    resume=resume,
                    display_command=display_command,
                )
            elif spec.execution == "visual_rft_coco":
                if visual_rft_runner is None:
                    from toolvista.evaluation.visual_rft_coco import run_visual_rft_coco

                    selected_visual_rft_runner = run_visual_rft_coco
                else:
                    selected_visual_rft_runner = visual_rft_runner
                result = selected_visual_rft_runner(
                    spec,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    benchmark_dir=benchmark_dir,
                    api_base=api_base,
                    api_model=api_model,
                    timeout=timeout,
                    score_only=score_only,
                    resume=resume,
                    display_command=display_command,
                )
            elif spec.execution == "bfcl":
                if bfcl_runner is None:
                    from toolvista.evaluation.bfcl import run_bfcl_benchmark

                    selected_bfcl_runner = run_bfcl_benchmark
                else:
                    selected_bfcl_runner = bfcl_runner
                result = selected_bfcl_runner(
                    spec,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    benchmark_dir=benchmark_dir,
                    api_base=api_base,
                    api_model=api_model,
                    timeout=timeout,
                    score_only=score_only,
                    resume=resume,
                    display_command=display_command,
                )
            elif spec.execution == "swift_eval":
                if score_only:
                    raise ValueError("--score-only is not supported for swift eval benchmarks.")
                result = swift_runner(
                    spec,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    benchmark_dir=benchmark_dir,
                    infer_backend=infer_backend,
                    eval_url=eval_url,
                    api_model=api_model,
                    judge_api_base=judge_api_base,
                    judge_model=judge_model,
                    resume=resume,
                    swift_executable=swift_executable,
                )
            else:
                raise ValueError(f"Unsupported benchmark execution: {spec.execution}")
            _validate_result_identity(
                result,
                spec,
                checkpoint=checkpoint,
                run_id=run_id,
                benchmark_dir=benchmark_dir,
            )
        except Exception as exc:
            result = _exception_result(
                spec,
                checkpoint=checkpoint,
                run_id=run_id,
                benchmark_dir=benchmark_dir,
                display_command=display_command,
                score_only=score_only,
                error=exc,
            )
        results.append(result)

    end_time = dt.datetime.now(dt.timezone.utc)
    duration_seconds = (end_time - start_time).total_seconds()
    metrics = _flat_metrics(results)
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "checkpoint": checkpoint,
        "stage": stage,
        "algorithm": algorithm,
        "seed": seed,
        "toolvista_commit": _toolvista_commit(),
        "benchmarks": [spec.benchmark_id for spec in specs],
        "benchmark_specs": [
            {
                "benchmark": spec.benchmark_id,
                "source_project": spec.source_project,
                "dataset": spec.dataset_name,
                "split": spec.split,
                "evaluator": spec.evaluator,
                "sample_limit": spec.eval_limit,
                "primary_metrics": list(spec.primary_metrics),
                "metric_directions": dict(spec.metric_directions),
            }
            for spec in specs
        ],
        "inference": {
            "api_base": _safe_endpoint(api_base),
            "api_model": api_model,
            "infer_backend": infer_backend,
            "eval_url": _safe_endpoint(eval_url),
            "chat_template": None,
            "system_prompt": "from frozen manifest",
            "temperature": 0,
            "top_p": None,
            "top_k": None,
            "max_tokens": 512,
            "image_preprocessing": "upstream default",
            "tool_order": "frozen manifest order",
            "stop_words": [],
            "reasoning_visibility": "upstream default",
        },
        "evaluator": {
            "judge_api_base": _safe_endpoint(judge_api_base),
            "judge_model": judge_model,
            "policy": "fixed_across_c0_c4",
        },
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration_seconds,
    }
    summary = {
        "schema_version": "1.0",
        "run_id": run_id,
        "checkpoint": checkpoint,
        "status": _suite_status(results),
        "metrics": metrics,
        "comparisons": calculate_comparisons(
            metrics,
            baseline_metrics=baseline_metrics,
            historical_metrics=historical_metrics,
            metric_directions=_metric_directions(specs),
        ),
        "benchmarks": {
            result.benchmark: {
                "status": result.status,
                "attempted": result.attempted,
                "completed": result.completed,
                "failed": result.failed,
                "result_path": result.test_summary.result_path,
                "main_metrics": list(result.test_summary.main_metrics),
                "artifacts": dict(result.artifacts),
            }
            for result in results
        },
        "runtime": {
            "duration_seconds": duration_seconds,
            "benchmark_duration_seconds": sum(
                float(result.runtime.get("duration_seconds", 0.0)) for result in results
            ),
        },
    }
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "summary.json", summary)
    return summary


def upload_summary_to_wandb(
    summary: Mapping[str, Any],
    *,
    project: str,
    group: str | None,
    wandb_module: Any = None,
) -> str | None:
    """Best-effort upload of final numeric suite metrics."""
    run: Any = None
    failure: Exception | None = None
    try:
        wandb = wandb_module or importlib.import_module("wandb")
        run = wandb.init(
            project=project,
            group=group,
            name=str(summary["run_id"]),
            job_type="evaluation",
        )
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError("Evaluation summary metrics must be a mapping.")
        run.log(
            {
                f"eval/{name}": float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(float(value))
            }
        )
    except Exception as exc:
        failure = exc
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as exc:
                if failure is None:
                    failure = exc
    if failure is None:
        return None
    return f"W&B upload failed: {type(failure).__name__}: {failure}"
