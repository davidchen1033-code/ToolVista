"""Small data contracts for ToolVista evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal, Mapping, cast


ExecutionMode = Literal[
    "native_xlam",
    "native_mllm_router",
    "toolvqa_step",
    "bfcl",
    "visual_rft_coco",
    "swift_eval",
]
BenchmarkStatus = Literal["passed", "failed", "blocked", "skipped"]
PredictionMode = Literal["fresh", "score_only"]
MetricDirection = Literal["maximize", "minimize"]
VALID_EXECUTIONS = {
    "native_xlam",
    "native_mllm_router",
    "toolvqa_step",
    "bfcl",
    "visual_rft_coco",
    "swift_eval",
}
VALID_METRIC_DIRECTIONS = {"maximize", "minimize"}
VALID_PREDICTION_MODES = {"fresh", "score_only"}
VALID_BENCHMARK_STATUSES = {"passed", "failed", "blocked", "skipped"}


def _required_string(data: Mapping[str, Any], key: str, benchmark_id: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Benchmark {benchmark_id!r} requires non-empty {key}.")
    return value.strip()


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    execution: ExecutionMode
    profile: str
    method: str
    source_project: str
    dataset_name: str
    split: str
    evaluator: str
    primary_metrics: tuple[str, ...]
    dataset_path: str | None = None
    swift_datasets: tuple[str, ...] = ()
    eval_backend: str | None = None
    eval_limit: int | None = None
    metric_scales: dict[str, float] = field(default_factory=dict)
    metric_directions: dict[str, MetricDirection] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, benchmark_id: str, data: Mapping[str, Any]) -> "BenchmarkSpec":
        execution = _required_string(data, "execution", benchmark_id)
        if execution not in VALID_EXECUTIONS:
            raise ValueError(f"Benchmark {benchmark_id!r} has unsupported execution: {execution}")

        raw_primary_metrics = data.get("primary_metrics")
        if (
            not isinstance(raw_primary_metrics, list)
            or not raw_primary_metrics
            or not all(isinstance(item, str) and item.strip() for item in raw_primary_metrics)
        ):
            raise ValueError(f"Benchmark {benchmark_id!r} requires primary_metrics as a non-empty string list.")
        primary_metrics = [item.strip() for item in raw_primary_metrics]

        raw_dataset_path = data.get("dataset_path")
        if execution != "swift_eval" and (
            not isinstance(raw_dataset_path, str) or not raw_dataset_path.strip()
        ):
            raise ValueError(f"Benchmark {benchmark_id!r} requires dataset_path for {execution} execution.")
        dataset_path = raw_dataset_path.strip() if isinstance(raw_dataset_path, str) else None

        raw_swift_datasets = data.get("swift_datasets", [])
        if not isinstance(raw_swift_datasets, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_swift_datasets
        ):
            raise ValueError(f"Benchmark {benchmark_id!r} swift_datasets must contain non-empty strings.")
        if execution == "swift_eval" and not raw_swift_datasets:
            raise ValueError(f"Benchmark {benchmark_id!r} requires non-empty swift_datasets.")
        swift_datasets = [item.strip() for item in raw_swift_datasets]

        raw_eval_backend = data.get("eval_backend")
        if execution == "swift_eval" and (
            not isinstance(raw_eval_backend, str) or not raw_eval_backend.strip()
        ):
            raise ValueError(f"Benchmark {benchmark_id!r} requires non-empty eval_backend.")
        eval_backend = raw_eval_backend.strip() if isinstance(raw_eval_backend, str) else None

        raw_metric_scales = data.get("metric_scales", {})
        if not isinstance(raw_metric_scales, Mapping):
            raise ValueError(f"Benchmark {benchmark_id!r} metric_scales must be a mapping.")
        metric_scales: dict[str, float] = {}
        for key, value in raw_metric_scales.items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ValueError(
                    f"Benchmark {benchmark_id!r} metric_scales requires non-empty string keys "
                    "and finite numeric values."
                )
            metric_scales[key.strip()] = float(value)

        raw_metric_directions = data.get("metric_directions", {})
        if not isinstance(raw_metric_directions, Mapping):
            raise ValueError(f"Benchmark {benchmark_id!r} metric_directions must be a mapping.")
        metric_directions: dict[str, MetricDirection] = {}
        invalid_directions: dict[str, Any] = {}
        for metric, direction in raw_metric_directions.items():
            if not isinstance(metric, str) or not metric.strip():
                invalid_directions[str(metric)] = direction
                continue
            normalized_direction = direction.strip() if isinstance(direction, str) else direction
            if not isinstance(normalized_direction, str) or normalized_direction not in VALID_METRIC_DIRECTIONS:
                invalid_directions[metric.strip()] = direction
                continue
            metric_directions[metric.strip()] = cast(MetricDirection, normalized_direction)
        if invalid_directions:
            raise ValueError(f"Benchmark {benchmark_id!r} has invalid metric_directions: {invalid_directions}")
        missing_directions = [metric for metric in primary_metrics if metric not in metric_directions]
        if missing_directions:
            raise ValueError(
                f"Benchmark {benchmark_id!r} requires metric_directions for primary metrics: {missing_directions}"
            )

        eval_limit = data.get("eval_limit")
        if eval_limit is not None and (not isinstance(eval_limit, int) or isinstance(eval_limit, bool) or eval_limit <= 0):
            raise ValueError(f"Benchmark {benchmark_id!r} eval_limit must be a positive integer.")

        raw_settings = data.get("settings", {})
        if not isinstance(raw_settings, Mapping) or not all(
            isinstance(key, str) and key.strip() and key == key.strip()
            for key in raw_settings
        ):
            raise ValueError(f"Benchmark {benchmark_id!r} settings must be a mapping with normalized string keys.")

        return cls(
            benchmark_id=benchmark_id,
            execution=cast(ExecutionMode, execution),
            profile=_required_string(data, "profile", benchmark_id),
            method=_required_string(data, "method", benchmark_id),
            source_project=_required_string(data, "source_project", benchmark_id),
            dataset_name=_required_string(data, "dataset_name", benchmark_id),
            split=_required_string(data, "split", benchmark_id),
            evaluator=_required_string(data, "evaluator", benchmark_id),
            primary_metrics=tuple(primary_metrics),
            dataset_path=dataset_path,
            swift_datasets=tuple(swift_datasets),
            eval_backend=eval_backend,
            eval_limit=eval_limit,
            metric_scales=metric_scales,
            metric_directions=metric_directions,
            settings=dict(raw_settings),
        )


@dataclass(frozen=True)
class TestSummary:
    method: str
    dataset: str
    source_project: str
    evaluator: str
    sample_count: int | None
    command: str
    main_metrics: tuple[str, ...]
    result_path: str
    artifact_paths: dict[str, str]


@dataclass
class EvaluationResult:
    schema_version: str
    run_id: str
    checkpoint: str
    benchmark: str
    profile: str
    prediction_mode: PredictionMode
    test_summary: TestSummary
    status: BenchmarkStatus
    attempted: int | None
    completed: int | None
    failed: int | None
    upstream_metrics: dict[str, Any]
    metrics: dict[str, float]
    runtime: dict[str, Any]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["test_summary"]["main_metrics"] = list(self.test_summary.main_metrics)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        payload = dict(data)
        raw_summary = payload.pop("test_summary", None)
        if not isinstance(raw_summary, Mapping):
            raise ValueError("Evaluation result test_summary must be a mapping.")
        summary = dict(raw_summary)

        main_metrics = summary.get("main_metrics")
        if (
            not isinstance(main_metrics, (list, tuple))
            or not main_metrics
            or not all(isinstance(item, str) and item.strip() for item in main_metrics)
        ):
            raise ValueError("Evaluation result test_summary.main_metrics must contain non-empty strings.")
        summary["main_metrics"] = tuple(item.strip() for item in main_metrics)

        summary_artifacts = summary.get("artifact_paths")
        if not isinstance(summary_artifacts, Mapping):
            raise ValueError("Evaluation result test_summary.artifact_paths must be a mapping.")
        summary["artifact_paths"] = dict(summary_artifacts)

        prediction_mode = payload.get("prediction_mode")
        if prediction_mode not in VALID_PREDICTION_MODES:
            raise ValueError(f"Evaluation result has invalid prediction_mode: {prediction_mode!r}")
        status = payload.get("status")
        if status not in VALID_BENCHMARK_STATUSES:
            raise ValueError(f"Evaluation result has invalid status: {status!r}")

        count_values = [payload.get(field_name) for field_name in ("attempted", "completed", "failed")]
        if any(value is None for value in count_values) and not all(value is None for value in count_values):
            raise ValueError("Evaluation result attempted/completed/failed must be known or unknown together.")
        for field_name, value in zip(("attempted", "completed", "failed"), count_values):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"Evaluation result {field_name} must be a nonnegative integer or null.")

        for field_name in ("upstream_metrics", "metrics", "runtime", "artifacts"):
            value = payload.get(field_name)
            if not isinstance(value, Mapping):
                raise ValueError(f"Evaluation result {field_name} must be a mapping.")
            payload[field_name] = dict(value)

        normalized_metrics: dict[str, float] = {}
        for key, value in payload["metrics"].items():
            if (
                not isinstance(key, str)
                or not key.strip()
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ValueError(
                    "Evaluation result metrics requires non-empty string keys and finite numeric values."
                )
            normalized_metrics[key.strip()] = float(value)
        payload["metrics"] = normalized_metrics

        return cls(test_summary=TestSummary(**summary), **payload)
