"""Qwen3-VL API bridge for Visual-RFT's COCO open-vocabulary evaluator."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from toolvista.evaluation.io import load_jsonl, write_json, write_jsonl
from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.runner import (
    PredictionClient,
    _assistant_message,
    _openai_multimodal_messages,
    _validate_complete_predictions,
    call_chat_completion,
)


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NOVEL15 = (
    "mouse",
    "fork",
    "hot dog",
    "cat",
    "airplane",
    "suitcase",
    "parking meter",
    "sandwich",
    "train",
    "hair drier",
    "toilet",
    "toaster",
    "snowboard",
    "frisbee",
    "bear",
)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_boxes(text: Any) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        return []
    match = ANSWER_PATTERN.search(text)
    payload = match.group(1) if match else text
    payload = payload.strip().replace("[[", "[").replace("]]", "]")
    if not payload or "no objects" in payload.lower():
        return []
    try:
        value = json.loads(payload.replace("'", '"'))
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []

    boxes: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for item in value:
        if not isinstance(item, Mapping):
            return []
        position = item.get("Position")
        confidence = item.get("Confidence")
        if (
            not isinstance(position, list)
            or len(position) != 4
            or not all(_finite_number(number) for number in position)
            or not _finite_number(confidence)
        ):
            return []
        normalized = tuple(float(number) for number in position)
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            continue
        if normalized not in seen:
            seen.add(normalized)
            boxes.append({"Position": list(normalized), "Confidence": float(confidence)})
    return boxes


def _negative_prediction_kind(text: Any) -> str:
    if not isinstance(text, str):
        return "other"
    match = ANSWER_PATTERN.search(text)
    if match:
        payload = match.group(1).strip()
    elif "<answer>" in text or "</answer>" in text:
        return "other"
    else:
        payload = text.strip()
    normalized = re.sub(r"\s+", " ", payload).strip().casefold()
    if normalized == "no objects":
        return "abstention"
    try:
        value = json.loads(payload.replace("'", '"'))
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list) and not value:
        return "abstention"
    if parse_boxes(text):
        return "false_positive"
    return "other"


def score_negative_behavior(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    if not rows or not all("has_ground_truth" in row for row in rows):
        return {}, {
            "available": False,
            "reason": "has_ground_truth is not available on every prediction row",
        }

    negative_rows = 0
    abstentions = 0
    false_positives = 0
    for index, row in enumerate(rows, 1):
        has_ground_truth = row["has_ground_truth"]
        if not isinstance(has_ground_truth, bool):
            raise ValueError(
                f"Visual-RFT prediction row {index} has non-boolean has_ground_truth."
            )
        if has_ground_truth:
            continue
        negative_rows += 1
        kind = _negative_prediction_kind(row.get("prediction"))
        abstentions += int(kind == "abstention")
        false_positives += int(kind == "false_positive")

    if negative_rows == 0:
        return {}, {
            "available": False,
            "reason": "the evaluated split contains no negative queries",
            "negative_rows": 0,
        }

    metrics = {
        "visual_rft_coco/negative_abstention_accuracy": abstentions / negative_rows,
        "visual_rft_coco/negative_false_positive_rate": false_positives / negative_rows,
    }
    details = {
        "available": True,
        "negative_rows": negative_rows,
        "abstention_rows": abstentions,
        "false_positive_rows": false_positives,
        "other_rows": negative_rows - abstentions - false_positives,
        "definitions": {
            "negative_abstention_accuracy": (
                "negative queries parsed as explicit No Objects or an empty list "
                "/ all negative queries"
            ),
            "negative_false_positive_rate": (
                "negative queries parsed with one or more valid boxes "
                "/ all negative queries"
            ),
        },
    }
    return metrics, details


def _to_coco_detection(row: Mapping[str, Any], box: Mapping[str, Any]) -> dict[str, Any]:
    width = float(row["image_width"])
    height = float(row["image_height"])
    x1, y1, x2, y2 = (float(value) for value in box["Position"])
    x1 = min(max(x1, 0.0), 1000.0) / 1000.0 * width
    y1 = min(max(y1, 0.0), 1000.0) / 1000.0 * height
    x2 = min(max(x2, 0.0), 1000.0) / 1000.0 * width
    y2 = min(max(y2, 0.0), 1000.0) / 1000.0 * height
    return {
        "image_id": int(row["image_id"]),
        "category_id": int(row["category_id"]),
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "score": min(max(float(box["Confidence"]), 0.0), 1.0),
    }


def predict_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    api_base: str,
    api_model: str,
    timeout: float,
    prediction_client: PredictionClient = call_chat_completion,
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
            response = prediction_client(
                api_base=api_base,
                model=api_model,
                messages=_openai_multimodal_messages(row),
                tools=[],
                timeout=timeout,
            )
            output["prediction"] = str(_assistant_message(response).get("content") or "")
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


def _load_upstream_evaluator(path: Path) -> type:
    spec = importlib.util.spec_from_file_location("toolvista_visual_rft_coco_evaluation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Visual-RFT COCO evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CocoDetectionEvaluator


def _filtered_annotations(source: Path, image_ids: set[int], destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["images"] = [item for item in payload["images"] if int(item["id"]) in image_ids]
    payload["annotations"] = [
        item for item in payload["annotations"] if int(item["image_id"]) in image_ids
    ]
    write_json(destination, payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _normalize_evaluator_output(
    output: Any,
    *,
    evaluator: Any,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if isinstance(output, tuple) and len(output) == 2:
        aggregate, per_class = output
    elif isinstance(output, Mapping):
        aggregate = output
        per_class = [{name: 0.0} for name in evaluator.class_names]
    else:
        raise ValueError(
            "Visual-RFT COCO evaluator must return an aggregate mapping or "
            "an (aggregate, per_class) pair."
        )
    if not isinstance(aggregate, Mapping) or not isinstance(per_class, list):
        raise ValueError("Visual-RFT COCO evaluator returned malformed metrics.")
    normalized_aggregate = {
        str(name): float(value)
        for name, value in aggregate.items()
        if _finite_number(value)
    }
    missing_aggregate = [name for name in evaluator.metric_names if name not in normalized_aggregate]
    if missing_aggregate:
        raise ValueError(f"Visual-RFT COCO evaluator omitted aggregate metrics: {missing_aggregate}")
    normalized_per_class: list[dict[str, float]] = []
    for item in per_class:
        if not isinstance(item, Mapping):
            raise ValueError("Visual-RFT COCO evaluator returned malformed per-class metrics.")
        normalized_per_class.append({str(name): float(value) for name, value in item.items()})
    return normalized_aggregate, normalized_per_class


def score_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    annotations: Path,
    upstream_evaluator: Path,
    benchmark_dir: Path,
    categories: Sequence[str] = NOVEL15,
) -> tuple[dict[str, Any], dict[str, float], int]:
    failures = sum(1 for row in rows if isinstance(row.get("error"), str) and row["error"].strip())
    detections: list[dict[str, Any]] = []
    for row in rows:
        if "prediction" not in row:
            continue
        detections.extend(_to_coco_detection(row, box) for box in parse_boxes(row["prediction"]))

    predictions_path = benchmark_dir / "coco_predictions.json"
    predictions_path.write_text(json.dumps(detections, ensure_ascii=False) + "\n", encoding="utf-8")
    filtered_annotations = benchmark_dir / "coco_annotations.json"
    _filtered_annotations(
        annotations,
        {int(row["image_id"]) for row in rows},
        filtered_annotations,
    )

    evaluator_cls = _load_upstream_evaluator(upstream_evaluator)
    evaluator = evaluator_cls(str(filtered_annotations))
    evaluation_dir = benchmark_dir / "coco_evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator_output = evaluator.evaluate(str(predictions_path), str(evaluation_dir))
    aggregate, per_class = _normalize_evaluator_output(evaluator_output, evaluator=evaluator)
    per_class_map = {name: value for item in per_class for name, value in item.items()}
    missing = sorted(set(categories) - set(per_class_map))
    if missing:
        raise ValueError(f"Visual-RFT COCO evaluator omitted selected categories: {missing}")
    selected_values = [per_class_map[name] for name in categories]
    nonfinite = [
        name for name, value in zip(categories, selected_values) if not math.isfinite(value)
    ]
    if nonfinite:
        raise ValueError(
            "Visual-RFT COCO selected categories have no evaluable ground truth: "
            f"{nonfinite}"
        )
    selected_map = sum(selected_values) / len(selected_values) / 100.0
    negative_metrics, negative_behavior = score_negative_behavior(rows)
    metrics = {
        "visual_rft_coco/novel15_map": selected_map,
        "visual_rft_coco/map": float(aggregate["mAP"]),
        "visual_rft_coco/ap50": float(aggregate["AP_50"]),
        **negative_metrics,
    }
    upstream = {
        "aggregate": _json_safe(aggregate),
        "per_class": _json_safe(per_class),
        "selected_categories": list(categories),
        "selected_map_percent": selected_map * 100.0,
        "prediction_count": len(detections),
        "negative_behavior": negative_behavior,
    }
    return upstream, metrics, failures


def run_visual_rft_coco(
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
        return EvaluationResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))
    if not spec.dataset_path:
        raise ValueError("Visual-RFT COCO benchmark requires dataset_path.")

    start = time.perf_counter()
    if score_only:
        if not predictions_path.is_file():
            raise FileNotFoundError(f"--score-only requires predictions: {predictions_path}")
        predictions = load_jsonl(predictions_path)
        prediction_mode = "score_only"
    else:
        if not api_base or not api_model:
            raise ValueError("Visual-RFT COCO evaluation requires --api-base and --api-model.")
        manifest = load_jsonl(spec.dataset_path)
        predictions = predict_rows(
            manifest,
            api_base=api_base,
            api_model=api_model,
            timeout=timeout,
            prediction_client=prediction_client,
            progress_path=benchmark_dir / "progress.json",
        )
        write_jsonl(predictions_path, predictions)
        prediction_mode = "fresh"

    _validate_complete_predictions(predictions, predictions_path)

    annotations = Path(str(spec.settings.get("annotations", ""))).expanduser()
    evaluator_path = Path(str(spec.settings.get("upstream_evaluator", ""))).expanduser()
    if not annotations.is_file() or not evaluator_path.is_file():
        raise FileNotFoundError(
            f"Visual-RFT COCO annotations/evaluator missing: {annotations}, {evaluator_path}"
        )
    upstream, metrics, failures = score_predictions(
        predictions,
        annotations=annotations,
        upstream_evaluator=evaluator_path,
        benchmark_dir=benchmark_dir,
    )
    upstream_path = benchmark_dir / "upstream_result.json"
    write_json(upstream_path, upstream)
    stdout_path = benchmark_dir / "stdout.log"
    stderr_path = benchmark_dir / "stderr.log"
    stdout_path.write_text("Visual-RFT COCO scorer completed\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    artifacts = {
        "predictions": predictions_path.as_posix(),
        "coco_predictions": (benchmark_dir / "coco_predictions.json").as_posix(),
        "upstream_result": upstream_path.as_posix(),
        "stdout": stdout_path.as_posix(),
        "stderr": stderr_path.as_posix(),
    }
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
            sample_count=len(predictions),
            command=display_command,
            main_metrics=spec.primary_metrics,
            result_path=result_path.as_posix(),
            artifact_paths=artifacts,
        ),
        status="passed" if failures == 0 else "failed",
        attempted=len(predictions),
        completed=len(predictions) - failures,
        failed=failures,
        upstream_metrics=upstream,
        metrics=metrics,
        runtime={
            "duration_seconds": time.perf_counter() - start,
            "peak_gpu_memory_mb": None,
            "exit_code": 0 if failures == 0 else 1,
        },
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result
