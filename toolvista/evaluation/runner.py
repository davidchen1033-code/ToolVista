"""Sequential execution for the lightweight evaluation workflow."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from toolvista.evaluation.io import load_json, load_jsonl, write_json, write_jsonl
from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.scorers import score_mllm_router, score_xlam


PredictionClient = Callable[..., Mapping[str, Any]]


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
