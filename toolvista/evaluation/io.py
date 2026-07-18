"""Configuration and artifact I/O for the lightweight evaluator."""

from __future__ import annotations

import csv
import json
import posixpath
import re
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from toolvista.evaluation.models import BenchmarkSpec


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError(
                f"unhashable YAML mapping key at line {key_node.start_mark.line + 1}"
            ) from exc
        if duplicate:
            raise ValueError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}, "
                f"column {key_node.start_mark.column + 1}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    try:
        data = yaml.load(source.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader) or {}
    except ValueError as exc:
        raise ValueError(f"Invalid YAML configuration {source}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    return dict(data)


def load_benchmark_registry(path: str | Path) -> dict[str, BenchmarkSpec]:
    root = load_yaml_mapping(path).get("benchmarks")
    if not isinstance(root, Mapping) or not root:
        raise ValueError(f"Benchmark config requires a non-empty benchmarks mapping: {path}")

    registry: dict[str, BenchmarkSpec] = {}
    for benchmark_id, value in root.items():
        if not isinstance(benchmark_id, str) or not benchmark_id.strip():
            raise ValueError(f"Benchmark config contains an invalid benchmark ID: {benchmark_id!r}")
        normalized_id = benchmark_id.strip()
        if normalized_id in registry:
            raise ValueError(f"Benchmark config contains duplicate benchmark ID after normalization: {normalized_id!r}")
        if not isinstance(value, Mapping):
            raise ValueError(f"Benchmark {normalized_id!r} must be a mapping.")
        registry[normalized_id] = BenchmarkSpec.from_mapping(normalized_id, value)
    return registry


def load_suites(path: str | Path) -> dict[str, list[str]]:
    root = load_yaml_mapping(path).get("suites")
    if not isinstance(root, Mapping) or not root:
        raise ValueError(f"Suite config requires a non-empty suites mapping: {path}")

    suites: dict[str, list[str]] = {}
    for name, benchmark_ids in root.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Suite config contains an invalid suite name: {name!r}")
        normalized_name = name.strip()
        if normalized_name in suites:
            raise ValueError(f"Suite config contains duplicate suite name after normalization: {normalized_name!r}")
        if (
            not isinstance(benchmark_ids, list)
            or not benchmark_ids
            or not all(isinstance(item, str) and item.strip() for item in benchmark_ids)
        ):
            raise ValueError(f"Suite {normalized_name!r} must be a non-empty list of benchmark IDs.")
        normalized_ids = [item.strip() for item in benchmark_ids]
        seen_ids: set[str] = set()
        for benchmark_id in normalized_ids:
            if benchmark_id in seen_ids:
                raise ValueError(
                    f"Suite {normalized_name!r} contains duplicate benchmark ID after normalization: "
                    f"{benchmark_id!r}"
                )
            seen_ids.add(benchmark_id)
        suites[normalized_name] = normalized_ids
    return suites


def load_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(data)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_number}: {exc.msg}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row {line_number} must be an object: {path}")
            rows.append(dict(value))
    return rows


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_last_swift_eval_report(
    path: str | Path,
    *,
    backend: str,
    eval_output_dir: str | Path,
) -> dict[str, Any]:
    expected_dir = _normalize_path_text(eval_output_dir)
    match: dict[str, Any] | None = None
    for row in load_jsonl(path):
        row_dir = _normalize_path_text(str(row.get("eval_output_dir", "")))
        if backend in row and row_dir == expected_dir:
            match = row
    if match is None:
        raise ValueError(f"No {backend} report for eval_output_dir={expected_dir} in {path}")
    return match


def _normalize_path_text(value: str | Path) -> str:
    normalized = re.sub(r"/+", "/", str(value).strip().replace("\\", "/"))
    normalized = posixpath.normpath(normalized)
    if re.match(r"^[a-zA-Z]:/", normalized):
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


def _metric_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _finite_artifact_number(value: Any, *, source: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VLMEvalKit artifact {source} must be numeric: {value!r}") from exc
    if not isfinite(numeric):
        raise ValueError(f"VLMEvalKit artifact {source} must be finite: {value!r}")
    return numeric


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _single_artifact(model_dir: Path, suffix: str) -> Path:
    matches = sorted(model_dir.glob(f"*_{suffix}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one VLMEvalKit artifact '*_{suffix}' in {model_dir}, "
            f"found {len(matches)}."
        )
    return matches[0]


def normalize_vlmevalkit_artifact_metrics(
    vlmeval_run_dir: str | Path,
    *,
    datasets: Sequence[str],
) -> tuple[dict[str, Any], dict[str, float], dict[str, str]]:
    """Extract stable metrics from VLMEvalKit's native CSV/JSON/XLSX artifacts."""
    run_dir = Path(vlmeval_run_dir)
    model_dirs = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and path.name not in {"configs", "logs"}
    )
    if len(model_dirs) != 1:
        raise ValueError(
            f"Expected exactly one VLMEvalKit model artifact directory in {run_dir}, "
            f"found {len(model_dirs)}."
        )
    model_dir = model_dirs[0]
    requested = set(datasets)
    supported = {"RealWorldQA", "MathVision_MINI", "MMMU_DEV_VAL", "OCRBench"}
    unsupported = sorted(requested - supported)
    if unsupported:
        raise ValueError(f"Unsupported VLMEvalKit artifact datasets: {unsupported}")

    upstream: dict[str, Any] = {}
    normalized: dict[str, float] = {}
    artifacts: dict[str, str] = {}

    if "RealWorldQA" in requested:
        path = _single_artifact(model_dir, "RealWorldQA_acc.csv")
        rows = _read_csv_rows(path)
        if len(rows) != 1 or "Overall" not in rows[0]:
            raise ValueError(f"Invalid RealWorldQA score artifact: {path}")
        value = _finite_artifact_number(rows[0]["Overall"], source=f"{path}:Overall")
        upstream["RealWorldQA/acc/Overall"] = value
        normalized["vlmevalkit/realworldqa/acc"] = value
        artifacts["realworldqa_score"] = path.as_posix()

    if "MathVision_MINI" in requested:
        path = _single_artifact(model_dir, "MathVision_MINI_*_score.csv")
        rows = _read_csv_rows(path)
        overall = next((row for row in rows if row.get("Subject") == "Overall"), None)
        if overall is None or "acc" not in overall:
            raise ValueError(f"Invalid MathVision MINI score artifact: {path}")
        value = _finite_artifact_number(overall["acc"], source=f"{path}:Overall/acc") * 0.01
        upstream["MathVision_MINI/accuracy/Overall"] = value
        normalized["vlmevalkit/mathvision_mini/accuracy"] = value
        artifacts["mathvision_mini_score"] = path.as_posix()

    if "MMMU_DEV_VAL" in requested:
        path = _single_artifact(model_dir, "MMMU_DEV_VAL_*_result.xlsx")
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas with Excel support is required for MMMU artifact scoring.") from exc
        table = pd.read_excel(path)
        if "hit" not in table.columns or len(table) == 0:
            raise ValueError(f"Invalid MMMU result artifact: {path}")
        hits = [_finite_artifact_number(value, source=f"{path}:hit") for value in table["hit"]]
        if any(value not in {0.0, 1.0} for value in hits):
            raise ValueError(f"MMMU hit values must be binary: {path}")
        value = sum(hits) / len(hits)
        upstream["MMMU_DEV_VAL/accuracy/micro"] = value
        normalized["vlmevalkit/mmmu_dev_val/accuracy"] = value
        artifacts["mmmu_dev_val_result"] = path.as_posix()

    if "OCRBench" in requested:
        path = _single_artifact(model_dir, "OCRBench_score.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or "Final Score" not in payload:
            raise ValueError(f"Invalid OCRBench score artifact: {path}")
        value = _finite_artifact_number(payload["Final Score"], source=f"{path}:Final Score")
        upstream["OCRBench/score/Final Score"] = value
        normalized["vlmevalkit/ocrbench/score"] = value
        artifacts["ocrbench_score"] = path.as_posix()

    return upstream, normalized, artifacts


def normalize_swift_metrics(
    report: Mapping[str, Any],
    *,
    backend: str,
    metric_scales: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, float]]:
    backend_result = report.get(backend)
    if not isinstance(backend_result, Mapping):
        raise ValueError(f"swift eval report has no {backend} metric mapping.")

    upstream: dict[str, Any] = {}
    normalized: dict[str, float] = {}
    normalized_sources: dict[str, str] = {}
    for dataset, metric_values in backend_result.items():
        if not isinstance(metric_values, Mapping):
            continue
        for metric_name, raw_value in metric_values.items():
            source_key = f"{dataset}/{metric_name}"
            upstream[source_key] = raw_value
            if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                numeric_value = float(raw_value)
                if not isfinite(numeric_value):
                    raise ValueError(f"swift eval metric {source_key!r} has non-finite raw value: {raw_value!r}")
                raw_scale = metric_scales.get(source_key, 1.0)
                if (
                    isinstance(raw_scale, bool)
                    or not isinstance(raw_scale, (int, float))
                    or not isfinite(float(raw_scale))
                ):
                    raise ValueError(f"swift eval metric {source_key!r} has non-finite or invalid scale: {raw_scale!r}")
                scale = float(raw_scale)
                metric_key = (
                    f"{_metric_slug(backend)}/"
                    f"{_metric_slug(str(dataset))}/"
                    f"{_metric_slug(str(metric_name))}"
                )
                previous_source = normalized_sources.get(metric_key)
                if previous_source is not None and previous_source != source_key:
                    raise ValueError(
                        f"swift eval normalized metric key collision for {metric_key!r}: "
                        f"{previous_source!r} and {source_key!r}"
                    )
                product = numeric_value * scale
                if not isfinite(product):
                    raise ValueError(f"swift eval metric {source_key!r} has non-finite scaled product: {product!r}")
                normalized_sources[metric_key] = source_key
                normalized[metric_key] = product
    return upstream, normalized
