# Lightweight Evaluation Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, sequential ToolVista evaluation command that reuses `swift eval`/VLMEvalKit, adds only the missing xLAM and MLLM-Tool Router scorers, preserves readable local artifacts, computes basic retention/forgetting, and optionally uploads summary metrics to W&B.

**Architecture:** Keep evaluation separate from the existing `raw -> adapter -> CanonicalSample -> exporter` training-data path. A minimal YAML registry describes how each benchmark runs; `scripts/evaluate.py` executes one suite sequentially through either an OpenAI-compatible prediction path plus a native scorer, or the upstream `swift eval` command, then writes one small result envelope per benchmark and one suite summary. Recovery is intentionally coarse: `result.json + --resume` skips a whole benchmark, `predictions.jsonl + --score-only` reruns only a native scorer, and every other incomplete state reruns the benchmark.

**Tech Stack:** Python 3.12 standard library, PyYAML, `unittest`, ms-swift `swift deploy`/`swift eval`, EvalScope/VLMEvalKit, optional W&B.

---

## Locked Scope / 已锁定范围

Phase 1 includes only:

- one unified CLI with sequential execution;
- minimal benchmark and suite YAML loading;
- frozen xLAM and MLLM-Tool text manifests;
- OpenAI-compatible prediction for the two native specialty benchmarks;
- xLAM function-call scoring;
- MLLM-Tool Router scoring;
- swift eval / VLMEvalKit integration through `swift eval --eval_backend VLMEvalKit` command construction and result parsing;
- `manifest.json`, per-benchmark artifacts, and `summary.json`;
- whole-benchmark `--resume` and native-only `--score-only`;
- basic primary-metric retention and forgetting summaries;
- optional best-effort W&B summary upload;
- documentation of rich training metrics already emitted by ms-swift/W&B.

Phase 1 explicitly defers ToolVQA/OpenCompass, AgentLego tool execution, Visual-RFT/COCO/LISA, MAT, official BFCL, multi-seed RL aggregation, sample-level resume, prediction caches, job queues, service lifecycle management, automatic model restarts, and a second training-metrics collector.

The native xLAM manifest is deliberately a one-step function-calling protocol test: preparation keeps tool-call targets whose prompt history contains only system/user messages. Multi-turn tool execution belongs to the later ToolVQA/MAT phases. If xLAM ActionStudio or MLLM-Tool does not provide an official held-out split, these fixed-seed manifests are internal ToolVista regression sets rather than upstream leaderboard test sets; reports must say that explicitly.

## File Map / 文件职责

| Path | Responsibility |
|---|---|
| `toolvista/evaluation/models.py` | Small dataclasses and validated benchmark/result contracts. |
| `toolvista/evaluation/io.py` | YAML/JSON/JSONL helpers, registry/suite loading, and `swift eval` report parsing. |
| `toolvista/evaluation/scorers.py` | The two scorers that do not exist upstream: xLAM and MLLM Router. |
| `toolvista/evaluation/runner.py` | Sequential API prediction, native scoring, `swift eval` subprocess execution, artifact writing, resume logic, comparison summary, and optional W&B upload. |
| `scripts/prepare_eval_data.py` | One-time deterministic materialization of frozen xLAM and MLLM Router evaluation manifests. |
| `scripts/evaluate.py` | The single user-facing evaluation CLI. |
| `configs/evaluation/benchmarks.yaml` | Minimal benchmark registry. |
| `configs/evaluation/suites.yaml` | Phase 1 cumulative lightweight suites. |
| `examples/evaluation/*.jsonl` | Frozen scorer and upstream-parser contract fixtures. |
| `tests/test_evaluation_io.py` | Dataclass, config, file I/O, and ms-swift result-parser tests. |
| `tests/test_evaluation_scorers.py` | Exact xLAM and MLLM Router metric tests. |
| `tests/test_evaluation_runner.py` | API prediction, subprocess, artifacts, resume, and comparison tests. |
| `tests/test_evaluate_script.py` | Manifest preparation and unified CLI smoke tests. |
| `docs/phase1_evaluation_usage.md` | AutoDL preparation, serving, evaluation, metric-source, and artifact commands. |

## Public Contracts / 公共契约

The implementation must keep these names and shapes consistent across all tasks:

```python
ExecutionMode = Literal["native_xlam", "native_mllm_router", "swift_eval"]
BenchmarkStatus = Literal["passed", "failed", "blocked", "skipped"]
PredictionMode = Literal["fresh", "score_only"]

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
    metric_directions: dict[str, Literal["maximize", "minimize"]] = field(default_factory=dict)

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
    attempted: int
    completed: int
    failed: int
    upstream_metrics: dict[str, Any]
    metrics: dict[str, float]
    runtime: dict[str, Any]
    artifacts: dict[str, str]
```

Every `result.json.test_summary` must therefore contain all information the user requested after a real test:

```text
测试方式
测评集及 split
来源项目
使用的上游 evaluator 或 ToolVista scorer
样本数或限制
实际启动命令
主要指标
结果与 artifact 路径
```

### Task 1: Define The Minimal Result Contracts

**Files:**
- Create: `toolvista/evaluation/models.py`
- Create: `tests/test_evaluation_io.py`

- [ ] **Step 1: Write the failing dataclass contract tests**

Create `tests/test_evaluation_io.py` with the initial tests:

```python
from __future__ import annotations

import unittest

from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary


class EvaluationModelTest(unittest.TestCase):
    def test_benchmark_spec_parses_and_validates_native_entry(self) -> None:
        spec = BenchmarkSpec.from_mapping(
            "xlam_function_calling",
            {
                "execution": "native_xlam",
                "profile": "light",
                "method": "OpenAI-compatible prediction plus ToolVista xLAM scorer",
                "source_project": "xLAM ActionStudio",
                "dataset_name": "xLAM fixed-seed dev",
                "split": "fixed_seed_dev",
                "evaluator": "toolvista.evaluation.scorers.score_xlam",
                "primary_metrics": ["xlam/function_name_accuracy"],
                "metric_directions": {"xlam/function_name_accuracy": "maximize"},
                "dataset_path": "data/eval/xlam/dev.jsonl",
            },
        )

        self.assertEqual(spec.execution, "native_xlam")
        self.assertEqual(spec.primary_metrics, ("xlam/function_name_accuracy",))
        self.assertEqual(spec.dataset_path, "data/eval/xlam/dev.jsonl")
        self.assertEqual(spec.metric_directions, {"xlam/function_name_accuracy": "maximize"})

    def test_benchmark_spec_rejects_missing_mode_specific_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset_path"):
            BenchmarkSpec.from_mapping(
                "broken_native",
                {
                    "execution": "native_xlam",
                    "profile": "light",
                    "method": "native",
                    "source_project": "xLAM",
                    "dataset_name": "dev",
                    "split": "dev",
                    "evaluator": "score_xlam",
                    "primary_metrics": ["xlam/function_name_accuracy"],
                },
            )

        with self.assertRaisesRegex(ValueError, "swift_datasets"):
            BenchmarkSpec.from_mapping(
                "broken_swift",
                {
                    "execution": "swift_eval",
                    "profile": "light",
                    "method": "swift eval",
                    "source_project": "ms-swift/VLMEvalKit",
                    "dataset_name": "general VL",
                    "split": "upstream default",
                    "evaluator": "swift eval",
                    "primary_metrics": ["vlmevalkit/realworldqa/acc"],
                    "eval_backend": "VLMEvalKit",
                },
            )

    def test_result_serialization_keeps_required_test_summary(self) -> None:
        summary = TestSummary(
            method="ToolVista native xLAM scoring",
            dataset="xLAM fixed-seed dev / fixed_seed_dev",
            source_project="xLAM ActionStudio",
            evaluator="toolvista.evaluation.scorers.score_xlam",
            sample_count=4,
            command="python scripts/evaluate.py --benchmark xlam_function_calling",
            main_metrics=("xlam/function_name_accuracy",),
            result_path="runs/test/eval/xlam_function_calling/result.json",
            artifact_paths={"predictions": "predictions.jsonl"},
        )
        result = EvaluationResult(
            schema_version="1.0",
            run_id="test",
            checkpoint="checkpoint",
            benchmark="xlam_function_calling",
            profile="light",
            prediction_mode="score_only",
            test_summary=summary,
            status="passed",
            attempted=4,
            completed=4,
            failed=0,
            upstream_metrics={"function_name_accuracy": 0.75},
            metrics={"xlam/function_name_accuracy": 0.75},
            runtime={"duration_seconds": 0.1, "peak_gpu_memory_mb": None, "exit_code": 0},
            artifacts={"predictions": "predictions.jsonl"},
        )

        payload = result.to_dict()

        self.assertEqual(payload["test_summary"]["main_metrics"], ["xlam/function_name_accuracy"])
        self.assertEqual(payload["test_summary"]["result_path"], summary.result_path)
        self.assertEqual(payload["test_summary"]["artifact_paths"], summary.artifact_paths)
        self.assertEqual(EvaluationResult.from_dict(payload).to_dict(), payload)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
python -m unittest tests.test_evaluation_io -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'toolvista.evaluation'`.

- [ ] **Step 3: Implement the exact minimal dataclasses and validation**

Create `toolvista/evaluation/models.py`:

```python
"""Small data contracts for ToolVista evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


ExecutionMode = Literal["native_xlam", "native_mllm_router", "swift_eval"]
BenchmarkStatus = Literal["passed", "failed", "blocked", "skipped"]
PredictionMode = Literal["fresh", "score_only"]
VALID_EXECUTIONS = {"native_xlam", "native_mllm_router", "swift_eval"}


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
    metric_directions: dict[str, Literal["maximize", "minimize"]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, benchmark_id: str, data: Mapping[str, Any]) -> "BenchmarkSpec":
        execution = _required_string(data, "execution", benchmark_id)
        if execution not in VALID_EXECUTIONS:
            raise ValueError(f"Benchmark {benchmark_id!r} has unsupported execution: {execution}")
        primary_metrics = data.get("primary_metrics")
        if not isinstance(primary_metrics, list) or not all(isinstance(item, str) and item for item in primary_metrics):
            raise ValueError(f"Benchmark {benchmark_id!r} requires primary_metrics as a non-empty string list.")
        dataset_path = data.get("dataset_path")
        swift_datasets = data.get("swift_datasets", [])
        if execution.startswith("native_") and (not isinstance(dataset_path, str) or not dataset_path):
            raise ValueError(f"Benchmark {benchmark_id!r} requires dataset_path for native execution.")
        if execution == "swift_eval" and (
            not isinstance(swift_datasets, list)
            or not swift_datasets
            or not all(isinstance(item, str) and item for item in swift_datasets)
        ):
            raise ValueError(f"Benchmark {benchmark_id!r} requires non-empty swift_datasets.")
        metric_scales = data.get("metric_scales", {})
        if not isinstance(metric_scales, Mapping):
            raise ValueError(f"Benchmark {benchmark_id!r} metric_scales must be a mapping.")
        metric_directions = data.get("metric_directions", {})
        if not isinstance(metric_directions, Mapping):
            raise ValueError(f"Benchmark {benchmark_id!r} metric_directions must be a mapping.")
        missing_directions = [metric for metric in primary_metrics if metric not in metric_directions]
        if missing_directions:
            raise ValueError(
                f"Benchmark {benchmark_id!r} requires metric_directions for primary metrics: {missing_directions}"
            )
        invalid_directions = {
            str(metric): direction
            for metric, direction in metric_directions.items()
            if direction not in {"maximize", "minimize"}
        }
        if invalid_directions:
            raise ValueError(f"Benchmark {benchmark_id!r} has invalid metric_directions: {invalid_directions}")
        eval_limit = data.get("eval_limit")
        if eval_limit is not None and (not isinstance(eval_limit, int) or eval_limit <= 0):
            raise ValueError(f"Benchmark {benchmark_id!r} eval_limit must be a positive integer.")
        return cls(
            benchmark_id=benchmark_id,
            execution=execution,  # type: ignore[arg-type]
            profile=_required_string(data, "profile", benchmark_id),
            method=_required_string(data, "method", benchmark_id),
            source_project=_required_string(data, "source_project", benchmark_id),
            dataset_name=_required_string(data, "dataset_name", benchmark_id),
            split=_required_string(data, "split", benchmark_id),
            evaluator=_required_string(data, "evaluator", benchmark_id),
            primary_metrics=tuple(primary_metrics),
            dataset_path=dataset_path,
            swift_datasets=tuple(swift_datasets),
            eval_backend=data.get("eval_backend"),
            eval_limit=eval_limit,
            metric_scales={str(key): float(value) for key, value in metric_scales.items()},
            metric_directions={str(key): value for key, value in metric_directions.items()},  # type: ignore[misc]
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
    attempted: int
    completed: int
    failed: int
    upstream_metrics: dict[str, Any]
    metrics: dict[str, float]
    runtime: dict[str, Any]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        payload = dict(data)
        summary = dict(payload.pop("test_summary"))
        summary["main_metrics"] = tuple(summary["main_metrics"])
        return cls(test_summary=TestSummary(**summary), **payload)
```

- [ ] **Step 4: Run the dataclass tests and verify they pass**

Run:

```powershell
python -m unittest tests.test_evaluation_io -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit the contracts**

```bash
git add toolvista/evaluation/models.py tests/test_evaluation_io.py
git commit -m "feat: define lightweight evaluation contracts"
```

### Task 2: Add Registry, JSONL, And ms-swift Report I/O

**Files:**
- Modify: `tests/test_evaluation_io.py`
- Create: `toolvista/evaluation/io.py`
- Create: `configs/evaluation/benchmarks.yaml`
- Create: `configs/evaluation/suites.yaml`
- Create: `examples/evaluation/swift_eval_result.jsonl`

- [ ] **Step 1: Add failing configuration and upstream-parser tests**

Append these imports and tests to `tests/test_evaluation_io.py`:

```python
import json
import tempfile
from pathlib import Path

from toolvista.evaluation.io import (
    load_benchmark_registry,
    load_suites,
    normalize_swift_metrics,
    read_last_swift_eval_report,
    write_json,
    write_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class EvaluationIOTest(unittest.TestCase):
    def test_registry_and_suite_configs_load_phase1_entries(self) -> None:
        registry = load_benchmark_registry(REPO_ROOT / "configs/evaluation/benchmarks.yaml")
        suites = load_suites(REPO_ROOT / "configs/evaluation/suites.yaml")

        self.assertEqual(registry["xlam_function_calling"].execution, "native_xlam")
        self.assertEqual(registry["mllm_router"].execution, "native_mllm_router")
        self.assertEqual(registry["qwen3vl_general_light"].eval_backend, "VLMEvalKit")
        self.assertEqual(
            suites["c3_light"],
            ["xlam_function_calling", "mllm_router", "qwen3vl_general_light"],
        )

    def test_swift_parser_selects_last_matching_report_and_keeps_scales_explicit(self) -> None:
        fixture = REPO_ROOT / "examples/evaluation/swift_eval_result.jsonl"

        report = read_last_swift_eval_report(
            fixture,
            backend="VLMEvalKit",
            eval_output_dir="runs/demo/eval/qwen3vl_general_light/swift_eval",
        )
        upstream, normalized = normalize_swift_metrics(
            report,
            backend="VLMEvalKit",
            metric_scales={"RealWorldQA/acc": 0.01},
        )

        self.assertEqual(upstream["RealWorldQA/acc"], 67.5)
        self.assertEqual(upstream["OCRBench/score"], 700)
        self.assertEqual(normalized["vlmevalkit/realworldqa/acc"], 0.675)
        self.assertEqual(normalized["vlmevalkit/ocrbench/score"], 700.0)

    def test_json_helpers_write_utf8_with_trailing_newlines(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
            root = Path(temp_dir)
            json_path = root / "result.json"
            jsonl_path = root / "rows.jsonl"

            write_json(json_path, {"name": "测试"})
            write_jsonl(jsonl_path, [{"id": 1}, {"id": 2}])

            self.assertTrue(json_path.read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(
                [json.loads(line)["id"] for line in jsonl_path.read_text(encoding="utf-8").splitlines()],
                [1, 2],
            )
```

- [ ] **Step 2: Create the frozen `swift eval` JSONL fixture**

Create `examples/evaluation/swift_eval_result.jsonl`:

```jsonl
{"Native":{"dummy":{"acc":1.0}},"eval_output_dir":"runs/other","eval_limit":1}
{"VLMEvalKit":{"RealWorldQA":{"acc":50.0}},"eval_output_dir":"runs/old","eval_limit":10}
{"VLMEvalKit":{"RealWorldQA":{"acc":67.5},"OCRBench":{"score":700}},"time":"2026-07-13 12:00:00","model":"checkpoint","adapters":[],"result_path":null,"eval_output_dir":"runs/demo/eval/qwen3vl_general_light/swift_eval","eval_limit":100}
```

- [ ] **Step 3: Create the minimal benchmark registry**

Create `configs/evaluation/benchmarks.yaml`:

```yaml
benchmarks:
  xlam_function_calling:
    execution: native_xlam
    profile: light
    method: OpenAI-compatible prediction plus ToolVista xLAM function-call scoring
    source_project: xLAM ActionStudio
    dataset_name: xLAM fixed-seed one-step function-calling manifest
    split: fixed_seed_dev
    evaluator: toolvista.evaluation.scorers.score_xlam
    dataset_path: data/eval/xlam/dev.jsonl
    primary_metrics:
      - xlam/function_name_accuracy
      - xlam/argument_exact_match
    metric_directions:
      xlam/function_name_accuracy: maximize
      xlam/argument_exact_match: maximize

  mllm_router:
    execution: native_mllm_router
    profile: light
    method: OpenAI-compatible prediction plus ToolVista MLLM Router scoring
    source_project: MLLM-Tool
    dataset_name: MLLM-Tool text Router fixed-seed manifest
    split: fixed_seed_dev
    evaluator: toolvista.evaluation.scorers.score_mllm_router
    dataset_path: data/eval/mllm_router/dev.jsonl
    primary_metrics:
      - mllm_router/top1_accuracy
      - mllm_router/macro_f1
    metric_directions:
      mllm_router/top1_accuracy: maximize
      mllm_router/macro_f1: maximize

  qwen3vl_general_light:
    execution: swift_eval
    profile: light
    method: ms-swift swift eval through EvalScope VLMEvalKit backend
    source_project: ms-swift and VLMEvalKit
    dataset_name: RealWorldQA, MathVision_MINI, MMMU_DEV_VAL, OCRBench
    split: upstream validation or benchmark default
    evaluator: swift eval --eval_backend VLMEvalKit
    eval_backend: VLMEvalKit
    eval_limit: 100
    swift_datasets:
      - RealWorldQA
      - MathVision_MINI
      - MMMU_DEV_VAL
      - OCRBench
    primary_metrics:
      - vlmevalkit/realworldqa/acc
      - vlmevalkit/mathvision_mini/accuracy
      - vlmevalkit/mmmu_dev_val/accuracy
      - vlmevalkit/ocrbench/score
    metric_scales:
      RealWorldQA/acc: 0.01
    metric_directions:
      vlmevalkit/realworldqa/acc: maximize
      vlmevalkit/mathvision_mini/accuracy: maximize
      vlmevalkit/mmmu_dev_val/accuracy: maximize
      vlmevalkit/ocrbench/score: maximize
```

Create `configs/evaluation/suites.yaml`:

```yaml
suites:
  c1_light:
    - xlam_function_calling
    - qwen3vl_general_light

  c3_light:
    - xlam_function_calling
    - mllm_router
    - qwen3vl_general_light

  phase1_smoke:
    - xlam_function_calling
    - mllm_router
```

Do not add ToolVQA to these Phase 1 suites. The later ToolVQA plan will extend `c2_light`, `c3_light`, and `c4_light` after the upstream OpenCompass bridge exists.

- [ ] **Step 4: Run the tests and verify I/O functions are missing**

Run:

```powershell
python -m unittest tests.test_evaluation_io -v
```

Expected: FAIL importing `toolvista.evaluation.io`.

- [ ] **Step 5: Implement the I/O and parser functions**

Create `toolvista/evaluation/io.py`:

```python
"""Configuration and artifact I/O for the lightweight evaluator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from toolvista.evaluation.models import BenchmarkSpec


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    return dict(data)


def load_benchmark_registry(path: str | Path) -> dict[str, BenchmarkSpec]:
    root = load_yaml_mapping(path).get("benchmarks")
    if not isinstance(root, Mapping) or not root:
        raise ValueError(f"Benchmark config requires a non-empty benchmarks mapping: {path}")
    return {
        str(benchmark_id): BenchmarkSpec.from_mapping(str(benchmark_id), value)
        for benchmark_id, value in root.items()
        if isinstance(value, Mapping)
    }


def load_suites(path: str | Path) -> dict[str, list[str]]:
    root = load_yaml_mapping(path).get("suites")
    if not isinstance(root, Mapping) or not root:
        raise ValueError(f"Suite config requires a non-empty suites mapping: {path}")
    suites: dict[str, list[str]] = {}
    for name, benchmark_ids in root.items():
        if not isinstance(benchmark_ids, list) or not all(isinstance(item, str) and item for item in benchmark_ids):
            raise ValueError(f"Suite {name!r} must be a list of benchmark IDs.")
        suites[str(name)] = list(benchmark_ids)
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
            value = json.loads(line)
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
    expected_dir = Path(eval_output_dir).as_posix()
    match: dict[str, Any] | None = None
    for row in load_jsonl(path):
        row_dir = Path(str(row.get("eval_output_dir", ""))).as_posix()
        if backend in row and row_dir == expected_dir:
            match = row
    if match is None:
        raise ValueError(f"No {backend} report for eval_output_dir={expected_dir} in {path}")
    return match


def _metric_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


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
    for dataset, metric_values in backend_result.items():
        if not isinstance(metric_values, Mapping):
            continue
        for metric_name, raw_value in metric_values.items():
            source_key = f"{dataset}/{metric_name}"
            upstream[source_key] = raw_value
            if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                scale = float(metric_scales.get(source_key, 1.0))
                normalized[f"{_metric_slug(backend)}/{_metric_slug(str(dataset))}/{_metric_slug(str(metric_name))}"] = (
                    float(raw_value) * scale
                )
    return upstream, normalized
```

- [ ] **Step 6: Run the complete I/O tests**

Run:

```powershell
python -m unittest tests.test_evaluation_io -v
```

Expected: 6 tests PASS.

- [ ] **Step 7: Commit registry and I/O support**

```bash
git add toolvista/evaluation/io.py tests/test_evaluation_io.py configs/evaluation/benchmarks.yaml configs/evaluation/suites.yaml examples/evaluation/swift_eval_result.jsonl
git commit -m "feat: add lightweight evaluation registry and IO"
```

### Task 3: Implement The xLAM Function-Calling Scorer

**Files:**
- Create: `toolvista/evaluation/scorers.py`
- Create: `tests/test_evaluation_scorers.py`
- Create: `examples/evaluation/xlam_predictions.jsonl`

- [ ] **Step 1: Create a fixture covering exact, invalid, parallel, and partial predictions**

Create `examples/evaluation/xlam_predictions.jsonl`:

```jsonl
{"sample_id":"x1","target_tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}],"prediction":"{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"}
{"sample_id":"x2","target_tool_calls":[{"name":"get_weather","arguments":{"city":"Tokyo"}}],"prediction":"not-json"}
{"sample_id":"x3","target_tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}},{"name":"get_time","arguments":{"zone":"UTC"}}],"prediction":[{"name":"get_time","arguments":{"zone":"UTC"}},{"name":"get_weather","arguments":{"city":"Paris"}}]}
{"sample_id":"x4","target_tool_calls":[{"name":"get_weather","arguments":{"city":"Paris","unit":"celsius"}}],"prediction":{"name":"get_weather","arguments":{"city":"London"}}}
```

- [ ] **Step 2: Write the failing scorer test**

Create `tests/test_evaluation_scorers.py`:

```python
from __future__ import annotations

import unittest
from pathlib import Path

from toolvista.evaluation.io import load_jsonl
from toolvista.evaluation.scorers import score_xlam


REPO_ROOT = Path(__file__).resolve().parents[1]


class XLAMScorerTest(unittest.TestCase):
    def test_xlam_metrics_cover_format_schema_names_arguments_and_parallel_calls(self) -> None:
        rows = load_jsonl(REPO_ROOT / "examples/evaluation/xlam_predictions.jsonl")

        result = score_xlam(rows)

        self.assertEqual(result["attempted"], 4)
        self.assertEqual(result["completed"], 4)
        self.assertEqual(result["failed"], 0)
        metrics = result["upstream_metrics"]
        self.assertAlmostEqual(metrics["json_valid_rate"], 0.75)
        self.assertAlmostEqual(metrics["schema_valid_rate"], 0.75)
        self.assertAlmostEqual(metrics["function_name_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["argument_key_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["argument_value_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["argument_exact_match"], 0.5)
        self.assertAlmostEqual(metrics["parallel_call_exact_match"], 1.0)
        self.assertEqual(result["metrics"]["xlam/argument_exact_match"], 0.5)

    def test_transport_error_is_failed_and_not_scored_as_model_output(self) -> None:
        result = score_xlam(
            [
                {
                    "sample_id": "failed",
                    "target_tool_calls": [{"name": "f", "arguments": {}}],
                    "error": "HTTP 500",
                }
            ]
        )

        self.assertEqual(result["attempted"], 1)
        self.assertEqual(result["completed"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["upstream_metrics"]["json_valid_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test and verify the scorer is missing**

Run:

```powershell
python -m unittest tests.test_evaluation_scorers.XLAMScorerTest -v
```

Expected: FAIL importing `toolvista.evaluation.scorers`.

- [ ] **Step 4: Implement the xLAM parser and scorer**

Create `toolvista/evaluation/scorers.py` with this first implementation:

```python
"""Minimal native scorers used only when an upstream scorer is absent."""

from __future__ import annotations

from collections import Counter
import json
from typing import Any, Iterable, Mapping, Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _decode_call_prediction(value: Any) -> tuple[list[Any], bool]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [], False
    if isinstance(value, Mapping):
        return [value], True
    if isinstance(value, list):
        return list(value), True
    return [], False


def _normalize_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    function = value.get("function", value)
    if not isinstance(function, Mapping):
        return None
    name = function.get("name")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not name or not isinstance(arguments, Mapping):
        return None
    return {"name": name, "arguments": dict(arguments)}


def _canonical_calls(values: Iterable[Any]) -> list[dict[str, Any]]:
    calls = [_normalize_call(value) for value in values]
    if any(call is None for call in calls):
        return []
    return sorted(
        (call for call in calls if call is not None),
        key=lambda call: (call["name"], json.dumps(call["arguments"], sort_keys=True, ensure_ascii=False)),
    )


def _function_name_score(target: Sequence[Mapping[str, Any]], prediction: Sequence[Mapping[str, Any]]) -> float:
    target_names = Counter(str(call["name"]) for call in target)
    predicted_names = Counter(str(call["name"]) for call in prediction)
    denominator = max(sum(target_names.values()), sum(predicted_names.values()), 1)
    return sum((target_names & predicted_names).values()) / denominator


def _argument_tokens(calls: Sequence[Mapping[str, Any]]) -> tuple[set[tuple[int, str]], dict[tuple[int, str], Any]]:
    keys: set[tuple[int, str]] = set()
    values: dict[tuple[int, str], Any] = {}
    for call_index, call in enumerate(calls):
        arguments = call["arguments"]
        for key, value in arguments.items():
            token = (call_index, str(key))
            keys.add(token)
            values[token] = value
    return keys, values


def _f1(target: set[Any], prediction: set[Any]) -> float:
    if not target and not prediction:
        return 1.0
    overlap = len(target & prediction)
    precision = overlap / len(prediction) if prediction else 0.0
    recall = overlap / len(target) if target else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_xlam(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = completed = failed = 0
    json_valid: list[float] = []
    schema_valid: list[float] = []
    name_scores: list[float] = []
    key_f1_scores: list[float] = []
    value_scores: list[float] = []
    argument_exact: list[float] = []
    parallel_exact: list[float] = []

    for row in rows:
        attempted += 1
        if row.get("error"):
            failed += 1
            continue
        completed += 1
        raw_prediction, is_json_valid = _decode_call_prediction(row.get("prediction"))
        json_valid.append(float(is_json_valid))
        target = _canonical_calls(row.get("target_tool_calls", []))
        prediction = _canonical_calls(raw_prediction) if is_json_valid else []
        is_schema_valid = is_json_valid and len(prediction) == len(raw_prediction)
        schema_valid.append(float(is_schema_valid))
        if not is_schema_valid:
            name_scores.append(0.0)
            key_f1_scores.append(0.0)
            value_scores.append(0.0)
            argument_exact.append(0.0)
            if len(target) > 1:
                parallel_exact.append(0.0)
            continue

        name_scores.append(_function_name_score(target, prediction))
        target_keys, target_values = _argument_tokens(target)
        predicted_keys, predicted_values = _argument_tokens(prediction)
        key_f1_scores.append(_f1(target_keys, predicted_keys))
        all_keys = target_keys | predicted_keys
        value_scores.append(
            sum(target_values.get(key) == predicted_values.get(key) for key in all_keys) / len(all_keys)
            if all_keys
            else 1.0
        )
        exact = float(target == prediction)
        argument_exact.append(exact)
        if len(target) > 1:
            parallel_exact.append(exact)

    upstream_metrics = {
        "json_valid_rate": _mean(json_valid),
        "schema_valid_rate": _mean(schema_valid),
        "function_name_accuracy": _mean(name_scores),
        "argument_key_f1": _mean(key_f1_scores),
        "argument_value_accuracy": _mean(value_scores),
        "argument_exact_match": _mean(argument_exact),
        "parallel_call_exact_match": _mean(parallel_exact),
        "parallel_samples": len(parallel_exact),
    }
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "upstream_metrics": upstream_metrics,
        "metrics": {
            f"xlam/{name}": float(value)
            for name, value in upstream_metrics.items()
            if name != "parallel_samples"
        },
    }
```

- [ ] **Step 5: Run the xLAM scorer tests**

Run:

```powershell
python -m unittest tests.test_evaluation_scorers.XLAMScorerTest -v
```

Expected: 2 tests PASS.

- [ ] **Step 6: Commit the xLAM scorer**

```bash
git add toolvista/evaluation/scorers.py tests/test_evaluation_scorers.py examples/evaluation/xlam_predictions.jsonl
git commit -m "feat: add xlam function calling scorer"
```

### Task 4: Implement The MLLM-Tool Router Scorer

**Files:**
- Modify: `toolvista/evaluation/scorers.py`
- Modify: `tests/test_evaluation_scorers.py`
- Create: `examples/evaluation/mllm_router_predictions.jsonl`

- [ ] **Step 1: Create a multi-label Router fixture**

Create `examples/evaluation/mllm_router_predictions.jsonl`:

```jsonl
{"sample_id":"m1","input_modality":"text","target_tools":["route_a","route_alt"],"prediction":"route_alt"}
{"sample_id":"m2","input_modality":"image","target_tools":["route_b"],"prediction":["route_x","route_b"]}
{"sample_id":"m3","input_modality":"text","target_tools":["route_a"],"prediction":""}
{"sample_id":"m4","input_modality":"image","target_tools":["route_b"],"prediction":"route_b"}
```

- [ ] **Step 2: Add the failing Router metric test**

Append to `tests/test_evaluation_scorers.py`:

```python
from toolvista.evaluation.scorers import score_mllm_router


class MLLMRouterScorerTest(unittest.TestCase):
    def test_router_metrics_allow_multiple_targets_and_report_modalities(self) -> None:
        rows = load_jsonl(REPO_ROOT / "examples/evaluation/mllm_router_predictions.jsonl")

        result = score_mllm_router(rows)

        self.assertEqual(result["attempted"], 4)
        self.assertEqual(result["completed"], 4)
        self.assertEqual(result["failed"], 0)
        metrics = result["upstream_metrics"]
        self.assertAlmostEqual(metrics["format_valid_rate"], 0.75)
        self.assertAlmostEqual(metrics["top1_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["candidate_hit_rate"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["invalid_route_rate"], 0.5)
        self.assertEqual(metrics["accuracy_by_modality"], {"image": 0.5, "text": 0.5})
        self.assertEqual(result["metrics"]["mllm_router/accuracy_by_modality/text"], 0.5)
```

- [ ] **Step 3: Run the Router test and verify the function is missing**

Run:

```powershell
python -m unittest tests.test_evaluation_scorers.MLLMRouterScorerTest -v
```

Expected: FAIL importing `score_mllm_router`.

- [ ] **Step 4: Add the compact multi-label Router scorer**

Append to `toolvista/evaluation/scorers.py`:

```python
def _route_predictions(value: Any) -> list[str]:
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def score_mllm_router(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    valid_route_universe = {
        str(tool)
        for row in materialized
        for tool in row.get("target_tools", [])
        if isinstance(tool, str) and tool
    }
    attempted = completed = failed = 0
    format_valid: list[float] = []
    top1_scores: list[float] = []
    candidate_hits: list[float] = []
    invalid_routes: list[float] = []
    modality_scores: dict[str, list[float]] = {}
    targets_by_label: dict[str, list[int]] = {}
    predictions_by_label: dict[str, list[int]] = {}

    for row_index, row in enumerate(materialized):
        attempted += 1
        if row.get("error"):
            failed += 1
            continue
        completed += 1
        target_tools = {str(tool) for tool in row.get("target_tools", []) if isinstance(tool, str) and tool}
        predictions = _route_predictions(row.get("prediction"))
        predicted_set = set(predictions)
        is_valid = bool(predictions)
        top1_correct = float(bool(predictions) and predictions[0] in target_tools)
        candidate_hit = float(bool(target_tools & predicted_set))
        invalid_route = float(not predictions or any(route not in valid_route_universe for route in predictions))
        format_valid.append(float(is_valid))
        top1_scores.append(top1_correct)
        candidate_hits.append(candidate_hit)
        invalid_routes.append(invalid_route)
        modality = str(row.get("input_modality") or "unknown")
        modality_scores.setdefault(modality, []).append(top1_correct)

        for label in target_tools | predicted_set:
            targets_by_label.setdefault(label, []).append(row_index if label in target_tools else -1)
            predictions_by_label.setdefault(label, []).append(row_index if label in predicted_set else -1)

    label_f1: list[float] = []
    for label in sorted(set(targets_by_label) | set(predictions_by_label)):
        target_indices = {index for index in targets_by_label.get(label, []) if index >= 0}
        predicted_indices = {index for index in predictions_by_label.get(label, []) if index >= 0}
        label_f1.append(_f1(target_indices, predicted_indices))

    accuracy_by_modality = {name: _mean(scores) for name, scores in sorted(modality_scores.items())}
    upstream_metrics: dict[str, Any] = {
        "format_valid_rate": _mean(format_valid),
        "top1_accuracy": _mean(top1_scores),
        "candidate_hit_rate": _mean(candidate_hits),
        "macro_f1": _mean(label_f1),
        "invalid_route_rate": _mean(invalid_routes),
        "accuracy_by_modality": accuracy_by_modality,
    }
    normalized = {
        f"mllm_router/{name}": float(value)
        for name, value in upstream_metrics.items()
        if isinstance(value, (int, float))
    }
    normalized.update(
        {f"mllm_router/accuracy_by_modality/{modality}": score for modality, score in accuracy_by_modality.items()}
    )
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": failed,
        "upstream_metrics": upstream_metrics,
        "metrics": normalized,
    }
```

- [ ] **Step 5: Run both native scorer test classes**

Run:

```powershell
python -m unittest tests.test_evaluation_scorers -v
```

Expected: all scorer tests PASS.

- [ ] **Step 6: Commit the Router scorer**

```bash
git add toolvista/evaluation/scorers.py tests/test_evaluation_scorers.py examples/evaluation/mllm_router_predictions.jsonl
git commit -m "feat: add mllm router scorer"
```

### Task 5: Materialize Frozen Specialty Evaluation Manifests

**Files:**
- Create: `scripts/prepare_eval_data.py`
- Create: `tests/test_evaluate_script.py`

- [ ] **Step 1: Write failing preparation, determinism, and leakage tests**

Create `tests/test_evaluate_script.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_eval_data import (
    check_no_exact_overlap,
    filter_training_rows,
    prepare_xlam_rows,
    prepare_mllm_router_rows,
)
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_paths() -> ToolVistaPaths:
    examples = REPO_ROOT / "examples" / "data"
    return ToolVistaPaths(
        config_path=Path("test"),
        raw={
            "datasets": {
                "xlam": {
                    "root": str(examples / "xlam"),
                    "unified_data_dir": ".",
                    "default_subset": "actionstudio-98k",
                    "default_split": "train",
                },
                "mllm_tool": {
                    "root": str(examples / "mllm_tool"),
                    "instruction_dir": "data/IT_data_ins/T+X-T_data",
                    "media_dir": "data/IT_data_ins/T+X-T_data/mm_dataset",
                    "files": {
                        "image": "image_tx2t.json",
                        "video": "video_tx2t.json",
                        "audio": "audio_tx2t.json",
                        "text": "text_t2t.json",
                        "combined": "combined_data.json",
                    },
                },
            }
        },
    )


class PrepareEvaluationDataTest(unittest.TestCase):
    def test_xlam_preparation_keeps_only_one_step_tool_call_targets(self) -> None:
        rows = prepare_xlam_rows(make_paths(), limit=10, seed=42)

        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(all(row["target_tool_calls"] for row in rows))
        self.assertTrue(all({message["role"] for message in row["messages"]} <= {"system", "user"} for row in rows))
        self.assertTrue(all(row["tools"] for row in rows))
        self.assertEqual(rows, prepare_xlam_rows(make_paths(), limit=10, seed=42))

    def test_mllm_preparation_builds_a_shared_text_router_catalog(self) -> None:
        rows = prepare_mllm_router_rows(make_paths(), limit=10, seed=42)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_modality"], "text")
        self.assertIn(rows[0]["target_tools"][0], rows[0]["router_catalog"])
        self.assertIn("Return exactly one tool ID", rows[0]["messages"][0]["content"])

    def test_exact_training_overlap_is_reported_before_evaluation(self) -> None:
        eval_rows = prepare_xlam_rows(make_paths(), limit=1, seed=42)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as temp_dir:
            train_path = Path(temp_dir) / "train.jsonl"
            train_path.write_text(
                json.dumps(
                    {
                        "metadata": {"sample_id": eval_rows[0]["sample_id"]},
                        "messages": eval_rows[0]["messages"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "overlap"):
                check_no_exact_overlap([train_path], eval_rows)

    def test_training_filter_removes_eval_ids_and_exact_text(self) -> None:
        eval_rows = prepare_xlam_rows(make_paths(), limit=1, seed=42)
        keep_row = {
            "metadata": {"sample_id": "keep"},
            "messages": [{"role": "user", "content": "A different prompt"}],
        }
        overlapping_row = {
            "metadata": {"sample_id": eval_rows[0]["sample_id"]},
            "messages": eval_rows[0]["messages"],
        }

        filtered = filter_training_rows([overlapping_row, keep_row], eval_rows)

        self.assertEqual(filtered, [keep_row])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the preparation script is missing**

Run:

```powershell
python -m unittest tests.test_evaluate_script.PrepareEvaluationDataTest -v
```

Expected: FAIL importing `scripts.prepare_eval_data`.

- [ ] **Step 3: Implement deterministic manifest preparation**

Create `scripts/prepare_eval_data.py`:

```python
"""Materialize frozen xLAM and MLLM-Tool text evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.adapters.mllm_tool import MLLMToolAdapter
from toolvista.adapters.xlam import XLAMAdapter
from toolvista.evaluation.io import load_jsonl, write_json, write_jsonl
from toolvista.schema.canonical import CanonicalMessage, CanonicalSample
from toolvista.utils.paths import ToolVistaPaths


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _message_dict(message: CanonicalMessage) -> dict[str, Any]:
    return {key: value for key, value in message.to_dict().items() if key in {"role", "content"}}


def _last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _selection_key(sample_id: str, *, seed: int, namespace: str) -> str:
    digest = hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode("utf-8")).hexdigest()
    return f"{digest}:{sample_id}"


def _fixed_seed_select(rows: Iterable[dict[str, Any]], *, limit: int, seed: int, namespace: str) -> list[dict[str, Any]]:
    if limit <= 0:
        return sorted(rows, key=lambda row: row["sample_id"])
    selected: list[tuple[int, str, int, dict[str, Any]]] = []
    for source_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        digest = hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode("utf-8")).digest()
        score = int.from_bytes(digest, "big")
        item = (-score, sample_id, source_index, row)
        if len(selected) < limit:
            heapq.heappush(selected, item)
        elif item > selected[0]:
            heapq.heapreplace(selected, item)
    return [item[3] for item in sorted(selected, key=lambda item: (-item[0], item[1], item[2]))]


def _xlam_row(sample: CanonicalSample) -> dict[str, Any] | None:
    target_calls = sample.labels.get("target_tool_calls")
    if sample.labels.get("target_kind") != "tool_calls" or not isinstance(target_calls, list) or not target_calls:
        return None
    prompt_messages = sample.messages[: -len(target_calls)]
    if not prompt_messages or any(message.role not in {"system", "user"} for message in prompt_messages):
        return None
    messages = [_message_dict(message) for message in prompt_messages]
    normalized = normalize_text(_last_user_text(messages))
    return {
        "sample_id": sample.sample_id,
        "source_project": "xLAM ActionStudio",
        "split": "fixed_seed_dev",
        "messages": messages,
        "tools": [tool.to_openai_tool() for tool in sample.tools],
        "target_tool_calls": target_calls,
        "normalized_text": normalized,
    }


def prepare_xlam_rows(paths: ToolVistaPaths, *, limit: int, seed: int) -> list[dict[str, Any]]:
    adapter = XLAMAdapter(paths)
    rows = (
        row
        for result in adapter.iter_results()
        if result.ok and result.sample is not None
        for row in [_xlam_row(result.sample)]
        if row is not None
    )
    selected = _fixed_seed_select(rows, limit=limit, seed=seed, namespace="xlam")
    validate_manifest(selected)
    return selected


def prepare_mllm_router_rows(paths: ToolVistaPaths, *, limit: int, seed: int) -> list[dict[str, Any]]:
    adapter = MLLMToolAdapter(paths, file_keys=("text",), validate_media_files=False)
    samples = [result.sample for result in adapter.iter_results() if result.ok and result.sample is not None]
    catalog = sorted(
        {
            str(tool)
            for sample in samples
            for tool in sample.labels.get("target_tools", [])
            if isinstance(tool, str) and tool
        }
    )
    system_prompt = (
        "You are a tool router. Choose the best tool ID from this catalog: "
        + json.dumps(catalog, ensure_ascii=False)
        + ". Return exactly one tool ID and no explanation."
    )
    rows: list[dict[str, Any]] = []
    for sample in samples:
        target_tools = sample.labels.get("target_tools")
        if not isinstance(target_tools, list) or not target_tools or not sample.messages:
            continue
        prompt_messages = sample.messages[:-1]
        messages = [{"role": "system", "content": system_prompt}] + [
            _message_dict(message) for message in prompt_messages if message.role in {"system", "user"}
        ]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "source_project": "MLLM-Tool",
                "split": "fixed_seed_dev",
                "input_modality": "text",
                "messages": messages,
                "router_catalog": catalog,
                "target_tools": target_tools,
                "normalized_text": normalize_text(_last_user_text(messages)),
            }
        )
    selected = _fixed_seed_select(rows, limit=limit, seed=seed, namespace="mllm_router")
    validate_manifest(selected)
    return selected


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("Every evaluation row requires a non-empty sample_id.")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Evaluation manifest contains duplicate sample IDs.")
    normalized = [row.get("normalized_text") for row in rows if row.get("normalized_text")]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Evaluation manifest contains duplicate normalized text.")


def _training_identity(row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    metadata = row.get("metadata")
    sample_id = metadata.get("sample_id") if isinstance(metadata, Mapping) else row.get("sample_id")
    normalized = row.get("normalized_text")
    if not isinstance(normalized, str):
        messages = row.get("messages")
        normalized = normalize_text(_last_user_text(messages)) if isinstance(messages, list) else None
    return sample_id if isinstance(sample_id, str) else None, normalized


def check_no_exact_overlap(training_files: Sequence[Path], eval_rows: Sequence[Mapping[str, Any]]) -> None:
    eval_ids = {str(row["sample_id"]) for row in eval_rows}
    eval_text = {str(row["normalized_text"]) for row in eval_rows if row.get("normalized_text")}
    overlapping_ids: set[str] = set()
    overlapping_text: set[str] = set()
    for path in training_files:
        for row in load_jsonl(path):
            sample_id, normalized = _training_identity(row)
            if sample_id in eval_ids:
                overlapping_ids.add(str(sample_id))
            if normalized and normalized in eval_text:
                overlapping_text.add(normalized)
    if overlapping_ids or overlapping_text:
        raise ValueError(
            f"Training/evaluation overlap detected: sample_ids={len(overlapping_ids)}, normalized_text={len(overlapping_text)}"
        )


def filter_training_rows(
    training_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    eval_ids = {str(row["sample_id"]) for row in eval_rows}
    eval_text = {str(row["normalized_text"]) for row in eval_rows if row.get("normalized_text")}
    filtered: list[dict[str, Any]] = []
    for row in training_rows:
        sample_id, normalized = _training_identity(row)
        if sample_id in eval_ids or (normalized and normalized in eval_text):
            continue
        filtered.append(dict(row))
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl.yaml")
    parser.add_argument("--xlam-output", type=Path, default=Path("data/eval/xlam/dev.jsonl"))
    parser.add_argument("--mllm-output", type=Path, default=Path("data/eval/mllm_router/dev.jsonl"))
    parser.add_argument("--xlam-limit", type=int, default=200)
    parser.add_argument("--mllm-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--filter-training",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("INPUT", "OUTPUT"),
        help="Filter one ms-swift training JSONL into OUTPUT using the frozen evaluation IDs/text.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_manifest(path: Path, rows: list[dict[str, Any]], *, seed: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Frozen evaluation manifest already exists: {path}. Use --overwrite explicitly.")
    write_jsonl(path, rows)
    write_json(
        path.with_suffix(".report.json"),
        {
            "output": path.as_posix(),
            "sample_count": len(rows),
            "seed": seed,
            "sample_ids": [row["sample_id"] for row in rows],
            "selection_keys": [
                _selection_key(
                    str(row["sample_id"]),
                    seed=seed,
                    namespace="xlam" if "target_tool_calls" in row else "mllm_router",
                )
                for row in rows
            ],
        },
    )


def main() -> None:
    args = parse_args()
    paths = ToolVistaPaths.from_yaml(args.config)
    xlam_rows = prepare_xlam_rows(paths, limit=args.xlam_limit, seed=args.seed)
    mllm_rows = prepare_mllm_router_rows(paths, limit=args.mllm_limit, seed=args.seed)
    filtered_reports: list[dict[str, Any]] = []
    for input_path, output_path in args.filter_training:
        source_rows = load_jsonl(input_path)
        filtered_rows = filter_training_rows(source_rows, [*xlam_rows, *mllm_rows])
        write_jsonl(output_path, filtered_rows)
        check_no_exact_overlap([output_path], [*xlam_rows, *mllm_rows])
        filtered_reports.append(
            {
                "input": input_path.as_posix(),
                "output": output_path.as_posix(),
                "input_rows": len(source_rows),
                "output_rows": len(filtered_rows),
                "removed_rows": len(source_rows) - len(filtered_rows),
            }
        )
    if args.training_file:
        check_no_exact_overlap(args.training_file, [*xlam_rows, *mllm_rows])
    _write_manifest(args.xlam_output, xlam_rows, seed=args.seed, overwrite=args.overwrite)
    _write_manifest(args.mllm_output, mllm_rows, seed=args.seed, overwrite=args.overwrite)
    print(
        json.dumps(
            {"xlam": len(xlam_rows), "mllm_router": len(mllm_rows), "filtered_training": filtered_reports},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
```

The generated `.report.json` sample ID list is the explicit exclusion list when the upstream source has no official dev split. Use `--filter-training INPUT OUTPUT` after a normal `--include-metadata` training export to create the non-overlapping training JSONL, then use `--training-file OUTPUT` as the simple enforcement check before evaluation. This keeps split logic outside the existing adapters/exporters.

- [ ] **Step 4: Run preparation tests**

Run:

```powershell
python -m unittest tests.test_evaluate_script.PrepareEvaluationDataTest -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Run a local sample preparation smoke test**

Create a temporary local config in the test itself rather than modifying `configs/autodl.yaml`, then run:

```powershell
python -m unittest tests.test_evaluate_script.PrepareEvaluationDataTest.test_xlam_preparation_keeps_only_one_step_tool_call_targets tests.test_evaluate_script.PrepareEvaluationDataTest.test_mllm_preparation_builds_a_shared_text_router_catalog -v
```

Expected: 2 tests PASS without loading a model or accessing the network.

- [ ] **Step 6: Commit the frozen-manifest builder**

```bash
git add scripts/prepare_eval_data.py tests/test_evaluate_script.py
git commit -m "feat: prepare frozen specialty evaluation data"
```

### Task 6: Add OpenAI-Compatible Prediction And Native Benchmark Execution

**Files:**
- Create: `toolvista/evaluation/runner.py`
- Create: `tests/test_evaluation_runner.py`

- [ ] **Step 1: Write failing native runner tests**

Create `tests/test_evaluation_runner.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from toolvista.evaluation.io import load_json, load_jsonl, write_jsonl
from toolvista.evaluation.models import BenchmarkSpec
from toolvista.evaluation.runner import run_native_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def xlam_spec(dataset_path: Path) -> BenchmarkSpec:
    return BenchmarkSpec.from_mapping(
        "xlam_function_calling",
        {
            "execution": "native_xlam",
            "profile": "light",
            "method": "OpenAI-compatible prediction plus ToolVista xLAM scorer",
            "source_project": "xLAM ActionStudio",
            "dataset_name": "xLAM fixed-seed dev",
            "split": "fixed_seed_dev",
            "evaluator": "toolvista.evaluation.scorers.score_xlam",
            "dataset_path": dataset_path.as_posix(),
            "primary_metrics": ["xlam/function_name_accuracy"],
            "metric_directions": {"xlam/function_name_accuracy": "maximize"},
        },
    )


class NativeEvaluationRunnerTest(unittest.TestCase):
    def test_score_only_reuses_predictions_and_writes_required_artifacts(self) -> None:
        fixture = REPO_ROOT / "examples/evaluation/xlam_predictions.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_dir = root / "run" / "eval" / "xlam_function_calling"
            benchmark_dir.mkdir(parents=True)
            predictions = benchmark_dir / "predictions.jsonl"
            predictions.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

            result = run_native_benchmark(
                xlam_spec(root / "unused.jsonl"),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=benchmark_dir,
                api_base=None,
                api_model=None,
                timeout=10.0,
                score_only=True,
                resume=False,
                display_command="python scripts/evaluate.py --benchmark xlam_function_calling --score-only",
            )

            self.assertEqual(result.prediction_mode, "score_only")
            self.assertEqual(result.status, "passed")
            self.assertEqual(result.test_summary.sample_count, 4)
            self.assertTrue((benchmark_dir / "result.json").is_file())
            self.assertTrue((benchmark_dir / "upstream_result.json").is_file())
            self.assertEqual(load_json(benchmark_dir / "result.json")["test_summary"]["source_project"], "xLAM ActionStudio")

    def test_fresh_prediction_sends_tools_and_scores_returned_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "xlam.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "sample_id": "x1",
                        "messages": [{"role": "user", "content": "Weather?"}],
                        "tools": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ],
                        "target_tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}],
                    }
                ],
            )
            requests: list[dict[str, object]] = []

            def fake_client(**kwargs):
                requests.append(kwargs)
                return {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": "{\"city\":\"Paris\"}",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }

            result = run_native_benchmark(
                xlam_spec(manifest),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=root / "run/eval/xlam_function_calling",
                api_base="http://127.0.0.1:8000/v1",
                api_model="checkpoint",
                timeout=10.0,
                score_only=False,
                resume=False,
                display_command="python scripts/evaluate.py --benchmark xlam_function_calling",
                prediction_client=fake_client,
            )

            self.assertEqual(result.metrics["xlam/argument_exact_match"], 1.0)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["tools"], load_jsonl(manifest)[0]["tools"])

    def test_resume_returns_existing_result_without_calling_model(self) -> None:
        fixture = REPO_ROOT / "examples/evaluation/xlam_predictions.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_dir = root / "run/eval/xlam_function_calling"
            benchmark_dir.mkdir(parents=True)
            (benchmark_dir / "predictions.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            first = run_native_benchmark(
                xlam_spec(root / "unused.jsonl"),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=benchmark_dir,
                api_base=None,
                api_model=None,
                timeout=10.0,
                score_only=True,
                resume=False,
                display_command="first",
            )

            def fail_client(**kwargs):
                raise AssertionError("resume must not call the model")

            resumed = run_native_benchmark(
                xlam_spec(root / "missing.jsonl"),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=benchmark_dir,
                api_base="unused",
                api_model="unused",
                timeout=10.0,
                score_only=False,
                resume=True,
                display_command="second",
                prediction_client=fail_client,
            )

            self.assertEqual(resumed.to_dict(), first.to_dict())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the runner is missing**

Run:

```powershell
python -m unittest tests.test_evaluation_runner.NativeEvaluationRunnerTest -v
```

Expected: FAIL importing `toolvista.evaluation.runner`.

- [ ] **Step 3: Implement API calls, prediction extraction, scoring, and artifacts**

Create `toolvista/evaluation/runner.py` with the native portion:

```python
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
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, Mapping):
                    raise ValueError("Chat completion response root must be an object.")
                return value
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.5)
    assert last_error is not None
    raise last_error


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
            return [call.get("function", call) if isinstance(call, Mapping) else call for call in tool_calls]
        return message.get("content", "")
    return str(message.get("content") or "").strip()


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
            output["prediction"] = extract_prediction(spec, response)
        except Exception as exc:
            output["error"] = f"{type(exc).__name__}: {exc}"
        predictions.append(output)
    return predictions


def _score_native(spec: BenchmarkSpec, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if spec.execution == "native_xlam":
        return score_xlam(rows)
    if spec.execution == "native_mllm_router":
        return score_mllm_router(rows)
    raise ValueError(f"Not a native benchmark: {spec.benchmark_id}")


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
        write_jsonl(predictions_path, predictions)
        prediction_mode = "fresh"

    scored = _score_native(spec, predictions)
    upstream_path = benchmark_dir / "upstream_result.json"
    write_json(upstream_path, scored)
    stdout_path = benchmark_dir / "stdout.log"
    stderr_path = benchmark_dir / "stderr.log"
    stdout_path.write_text("native scorer completed\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    artifacts = _artifact_paths(benchmark_dir)
    result = EvaluationResult(
        schema_version="1.0",
        run_id=run_id,
        checkpoint=checkpoint,
        benchmark=spec.benchmark_id,
        profile=spec.profile,
        prediction_mode=prediction_mode,  # type: ignore[arg-type]
        test_summary=TestSummary(
            method=spec.method,
            dataset=f"{spec.dataset_name} / {spec.split}",
            source_project=spec.source_project,
            evaluator=spec.evaluator,
            sample_count=int(scored["attempted"]),
            command=display_command,
            main_metrics=spec.primary_metrics,
            result_path=result_path.as_posix(),
            artifact_paths=artifacts,
        ),
        status="passed" if int(scored["failed"]) == 0 else "failed",
        attempted=int(scored["attempted"]),
        completed=int(scored["completed"]),
        failed=int(scored["failed"]),
        upstream_metrics=dict(scored["upstream_metrics"]),
        metrics={str(key): float(value) for key, value in scored["metrics"].items()},
        runtime={
            "duration_seconds": time.perf_counter() - start,
            "peak_gpu_memory_mb": None,
            "exit_code": 0 if int(scored["failed"]) == 0 else 1,
        },
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result
```

- [ ] **Step 4: Run the native runner tests**

Run:

```powershell
python -m unittest tests.test_evaluation_runner.NativeEvaluationRunnerTest -v
```

Expected: 3 tests PASS. These are contract tests with a fake API client; they do not load Qwen3-VL.

- [ ] **Step 5: Commit native execution**

```bash
git add toolvista/evaluation/runner.py tests/test_evaluation_runner.py
git commit -m "feat: run native specialty evaluations"
```

### Task 7: Bridge `swift eval` And Preserve Upstream Results

**Files:**
- Modify: `toolvista/evaluation/runner.py`
- Modify: `tests/test_evaluation_runner.py`

- [ ] **Step 1: Add failing command, parser, log, and failure tests**

Append these imports and tests to `tests/test_evaluation_runner.py`:

```python
import subprocess

from toolvista.evaluation.runner import build_swift_eval_command, run_swift_benchmark


def swift_spec() -> BenchmarkSpec:
    return BenchmarkSpec.from_mapping(
        "qwen3vl_general_light",
        {
            "execution": "swift_eval",
            "profile": "light",
            "method": "ms-swift swift eval through VLMEvalKit",
            "source_project": "ms-swift and VLMEvalKit",
            "dataset_name": "RealWorldQA and OCRBench",
            "split": "upstream default",
            "evaluator": "swift eval --eval_backend VLMEvalKit",
            "primary_metrics": ["vlmevalkit/realworldqa/acc"],
            "metric_directions": {"vlmevalkit/realworldqa/acc": "maximize"},
            "eval_backend": "VLMEvalKit",
            "eval_limit": 10,
            "swift_datasets": ["RealWorldQA", "OCRBench"],
            "metric_scales": {"RealWorldQA/acc": 0.01},
        },
    )


class SwiftEvaluationRunnerTest(unittest.TestCase):
    def test_command_uses_upstream_swift_eval_flags(self) -> None:
        command = build_swift_eval_command(
            swift_spec(),
            checkpoint="/checkpoints/c3",
            eval_output_dir=Path("runs/c3/eval/general/swift_eval"),
            infer_backend="vllm",
            eval_url="http://127.0.0.1:8000/v1",
            swift_executable="swift",
        )

        self.assertEqual(command[:2], ["swift", "eval"])
        self.assertIn("VLMEvalKit", command)
        self.assertEqual(command[command.index("--eval_dataset") + 1 : command.index("--eval_limit")], ["RealWorldQA", "OCRBench"])
        self.assertIn('{"max_tokens":512,"temperature":0}', command)

    def test_success_preserves_raw_report_stdout_stderr_and_normalized_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            benchmark_dir = root / "run/eval/qwen3vl_general_light"
            eval_output_dir = (benchmark_dir / "swift_eval").resolve()

            def fake_run(command, **kwargs):
                self.assertEqual(Path(kwargs["cwd"]), benchmark_dir)
                write_jsonl(
                    checkpoint / "eval_result.jsonl",
                    [
                        {
                            "VLMEvalKit": {"RealWorldQA": {"acc": 67.5}, "OCRBench": {"score": 700}},
                            "eval_output_dir": eval_output_dir.as_posix(),
                            "eval_limit": 10,
                        }
                    ],
                )
                return subprocess.CompletedProcess(command, 0, stdout="upstream stdout", stderr="upstream stderr")

            result = run_swift_benchmark(
                swift_spec(),
                checkpoint=checkpoint.as_posix(),
                run_id="run",
                benchmark_dir=benchmark_dir,
                infer_backend="vllm",
                eval_url=None,
                resume=False,
                command_runner=fake_run,
            )

            self.assertEqual(result.status, "passed")
            self.assertEqual(result.metrics["vlmevalkit/realworldqa/acc"], 0.675)
            self.assertEqual(load_json(benchmark_dir / "upstream_result.json")["VLMEvalKit"]["OCRBench"]["score"], 700)
            self.assertEqual((benchmark_dir / "stdout.log").read_text(encoding="utf-8"), "upstream stdout")
            self.assertEqual((benchmark_dir / "stderr.log").read_text(encoding="utf-8"), "upstream stderr")

    def test_nonzero_exit_writes_failed_result_without_synthetic_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()

            def fake_run(command, **kwargs):
                return subprocess.CompletedProcess(command, 2, stdout="", stderr="evaluator crashed")

            result = run_swift_benchmark(
                swift_spec(),
                checkpoint=checkpoint.as_posix(),
                run_id="run",
                benchmark_dir=root / "run/eval/qwen3vl_general_light",
                infer_backend="vllm",
                eval_url=None,
                resume=False,
                command_runner=fake_run,
            )

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.metrics, {})
            self.assertEqual(result.runtime["exit_code"], 2)
```

- [ ] **Step 2: Run the tests and verify swift runner functions are missing**

Run:

```powershell
python -m unittest tests.test_evaluation_runner.SwiftEvaluationRunnerTest -v
```

Expected: FAIL importing `build_swift_eval_command` or `run_swift_benchmark`.

- [ ] **Step 3: Add exact command construction and subprocess execution**

Add these imports to `toolvista/evaluation/runner.py`:

```python
import shlex
import subprocess

from toolvista.evaluation.io import normalize_swift_metrics, read_last_swift_eval_report
```

Append these functions:

```python
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def build_swift_eval_command(
    spec: BenchmarkSpec,
    *,
    checkpoint: str,
    eval_output_dir: Path,
    infer_backend: str,
    eval_url: str | None,
    swift_executable: str,
) -> list[str]:
    if spec.execution != "swift_eval" or not spec.eval_backend or not spec.swift_datasets:
        raise ValueError(f"Invalid swift eval benchmark: {spec.benchmark_id}")
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


def _swift_result_candidates(checkpoint: str, benchmark_dir: Path) -> list[Path]:
    source = Path(checkpoint)
    candidates = [source / "eval_result.jsonl"] if source.is_dir() else []
    candidates.extend(sorted((benchmark_dir / "result").rglob("eval_result.jsonl")))
    return candidates


def _read_swift_report(
    checkpoint: str,
    benchmark_dir: Path,
    *,
    backend: str,
    eval_output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    for candidate in _swift_result_candidates(checkpoint, benchmark_dir):
        try:
            return read_last_swift_eval_report(
                candidate,
                backend=backend,
                eval_output_dir=eval_output_dir,
            )
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
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    result_path = benchmark_dir / "result.json"
    if resume and result_path.is_file():
        return EvaluationResult.from_dict(load_json(result_path))

    eval_output_dir = (benchmark_dir / "swift_eval").resolve()
    command = build_swift_eval_command(
        spec,
        checkpoint=checkpoint,
        eval_output_dir=eval_output_dir,
        infer_backend=infer_backend,
        eval_url=eval_url,
        swift_executable=swift_executable,
    )
    start = time.perf_counter()
    completed_process = command_runner(command, cwd=benchmark_dir, capture_output=True, text=True, check=False)
    duration = time.perf_counter() - start
    stdout_path = benchmark_dir / "stdout.log"
    stderr_path = benchmark_dir / "stderr.log"
    stdout_path.write_text(completed_process.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed_process.stderr or "", encoding="utf-8")
    upstream_path = benchmark_dir / "upstream_result.json"
    artifacts = {
        "upstream_result": upstream_path.as_posix(),
        "stdout": stdout_path.as_posix(),
        "stderr": stderr_path.as_posix(),
        "swift_eval_output": eval_output_dir.as_posix(),
    }
    planned = (spec.eval_limit or 0) * len(spec.swift_datasets)
    upstream_metrics: dict[str, Any] = {}
    metrics: dict[str, float] = {}
    status = "failed" if completed_process.returncode else "passed"
    if completed_process.returncode == 0:
        try:
            report = _read_swift_report(
                checkpoint,
                benchmark_dir,
                backend=str(spec.eval_backend),
                eval_output_dir=eval_output_dir,
            )
            write_json(upstream_path, report)
            upstream_metrics, metrics = normalize_swift_metrics(
                report,
                backend=str(spec.eval_backend),
                metric_scales=spec.metric_scales,
            )
        except Exception as exc:
            status = "failed"
            with stderr_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n{type(exc).__name__}: {exc}\n")

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
        runtime={"duration_seconds": duration, "peak_gpu_memory_mb": None, "exit_code": completed_process.returncode},
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result
```

For Phase 1, `sample_count` is the configured per-dataset `eval_limit`, and `attempted` is the simple planned upper bound `eval_limit * dataset_count`. The raw `swift eval` report remains authoritative when upstream later exposes exact per-dataset counts.

- [ ] **Step 4: Run all runner tests**

Run:

```powershell
python -m unittest tests.test_evaluation_runner -v
```

Expected: 6 tests PASS, with no ms-swift installation or model required because the command runner is injected.

- [ ] **Step 5: Commit the ms-swift bridge**

```bash
git add toolvista/evaluation/runner.py tests/test_evaluation_runner.py
git commit -m "feat: bridge ms-swift vlmevalkit evaluation"
```

### Task 8: Add Sequential Suite Summaries, Retention, Forgetting, And Optional W&B

**Files:**
- Modify: `toolvista/evaluation/runner.py`
- Modify: `tests/test_evaluation_runner.py`

- [ ] **Step 1: Add failing comparison and suite-continuation tests**

Append these imports and tests to `tests/test_evaluation_runner.py`:

```python
from unittest.mock import patch

from toolvista.evaluation.models import EvaluationResult, TestSummary
from toolvista.evaluation.runner import calculate_comparisons, evaluate_suite, upload_summary_to_wandb


def passed_result(benchmark: str, metric_name: str, value: float, root: Path) -> EvaluationResult:
    result_path = root / benchmark / "result.json"
    return EvaluationResult(
        schema_version="1.0",
        run_id="run",
        checkpoint="checkpoint",
        benchmark=benchmark,
        profile="light",
        prediction_mode="fresh",
        test_summary=TestSummary(
            method="test",
            dataset="dataset / split",
            source_project="source",
            evaluator="evaluator",
            sample_count=1,
            command="command",
            main_metrics=(metric_name,),
            result_path=result_path.as_posix(),
            artifact_paths={},
        ),
        status="passed",
        attempted=1,
        completed=1,
        failed=0,
        upstream_metrics={metric_name: value},
        metrics={metric_name: value},
        runtime={"duration_seconds": 1.0, "peak_gpu_memory_mb": None, "exit_code": 0},
        artifacts={},
    )


class EvaluationSummaryTest(unittest.TestCase):
    def test_comparisons_report_gain_retention_and_historical_forgetting(self) -> None:
        comparisons = calculate_comparisons(
            {"xlam/function_name_accuracy": 0.72},
            baseline_metrics={"xlam/function_name_accuracy": 0.60},
            historical_metrics=[
                {"xlam/function_name_accuracy": 0.80},
                {"xlam/function_name_accuracy": 0.75},
            ],
            metric_directions={"xlam/function_name_accuracy": "maximize"},
        )

        row = comparisons["xlam/function_name_accuracy"]
        self.assertAlmostEqual(row["gain_vs_base"], 0.12)
        self.assertAlmostEqual(row["retention"], 1.2)
        self.assertAlmostEqual(row["historical_best"], 0.8)
        self.assertAlmostEqual(row["forgetting"], 0.08)

    def test_suite_is_sequential_and_continues_after_one_benchmark_raises(self) -> None:
        order: list[str] = []
        specs = [
            xlam_spec(Path("unused-xlam.jsonl")),
            BenchmarkSpec.from_mapping(
                "mllm_router",
                {
                    "execution": "native_mllm_router",
                    "profile": "light",
                    "method": "native",
                    "source_project": "MLLM-Tool",
                    "dataset_name": "router",
                    "split": "dev",
                    "evaluator": "score_mllm_router",
                    "dataset_path": "unused-mllm.jsonl",
                    "primary_metrics": ["mllm_router/top1_accuracy"],
                    "metric_directions": {"mllm_router/top1_accuracy": "maximize"},
                },
            ),
        ]

        def fake_native(spec, **kwargs):
            order.append(spec.benchmark_id)
            if spec.benchmark_id == "xlam_function_calling":
                raise RuntimeError("first failed")
            return passed_result(spec.benchmark_id, "mllm_router/top1_accuracy", 0.75, kwargs["benchmark_dir"])

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = evaluate_suite(
                specs,
                checkpoint="checkpoint",
                run_id="run",
                runs_root=Path(temp_dir),
                api_base="http://127.0.0.1:8000/v1",
                api_model="checkpoint",
                infer_backend="vllm",
                eval_url=None,
                timeout=10.0,
                score_only=False,
                resume=False,
                display_command="python scripts/evaluate.py",
                stage="c3",
                algorithm=None,
                seed=42,
                baseline_metrics={},
                historical_metrics=[],
                native_runner=fake_native,
            )

            self.assertEqual(order, ["xlam_function_calling", "mllm_router"])
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["benchmarks"]["xlam_function_calling"]["status"], "failed")
            self.assertEqual(summary["benchmarks"]["mllm_router"]["status"], "passed")
            self.assertTrue((Path(temp_dir) / "run/manifest.json").is_file())
            self.assertTrue((Path(temp_dir) / "run/summary.json").is_file())

    def test_wandb_upload_flattens_numeric_metrics_and_warning_is_nonfatal(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.logged = None

            def log(self, values) -> None:
                self.logged = values

            def finish(self) -> None:
                pass

        class FakeWandb:
            def __init__(self) -> None:
                self.run = FakeRun()

            def init(self, **kwargs):
                return self.run

        fake = FakeWandb()
        warning = upload_summary_to_wandb(
            {"run_id": "run", "metrics": {"xlam/acc": 0.75}, "status": "passed"},
            project="toolvista-qwen3vl",
            group="c3-mllm-router-sft",
            wandb_module=fake,
        )

        self.assertIsNone(warning)
        self.assertEqual(fake.run.logged, {"eval/xlam/acc": 0.75})

        class BrokenWandb:
            def init(self, **kwargs):
                raise RuntimeError("offline")

        self.assertIn(
            "offline",
            upload_summary_to_wandb(
                {"run_id": "run", "metrics": {}},
                project="toolvista-qwen3vl",
                group=None,
                wandb_module=BrokenWandb(),
            ),
        )
```

- [ ] **Step 2: Run the summary tests and verify the functions are missing**

Run:

```powershell
python -m unittest tests.test_evaluation_runner.EvaluationSummaryTest -v
```

Expected: FAIL importing `calculate_comparisons`, `evaluate_suite`, or `upload_summary_to_wandb`.

- [ ] **Step 3: Add manifest, exception-result, comparison, and W&B helpers**

Add these imports to `toolvista/evaluation/runner.py`:

```python
import datetime as dt
import importlib
```

Append these functions:

```python
def calculate_comparisons(
    current_metrics: Mapping[str, float],
    *,
    baseline_metrics: Mapping[str, float],
    historical_metrics: Sequence[Mapping[str, float]],
    metric_directions: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    comparisons: dict[str, dict[str, float]] = {}
    for metric_name, direction in sorted(metric_directions.items()):
        if metric_name not in current_metrics:
            continue
        current = current_metrics[metric_name]
        row: dict[str, float] = {"current": float(current)}
        base = baseline_metrics.get(metric_name)
        if base is not None:
            row["base"] = float(base)
            row["gain_vs_base"] = (
                float(current) - float(base) if direction == "maximize" else float(base) - float(current)
            )
            if direction == "maximize" and base != 0:
                row["retention"] = float(current) / float(base)
        history = [float(values[metric_name]) for values in historical_metrics if metric_name in values]
        if base is not None:
            history.append(float(base))
        if history:
            row["historical_best"] = max(history) if direction == "maximize" else min(history)
            raw_forgetting = (
                row["historical_best"] - float(current)
                if direction == "maximize"
                else float(current) - row["historical_best"]
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
    error: Exception,
) -> EvaluationResult:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = benchmark_dir / "stderr.log"
    stdout_path = benchmark_dir / "stdout.log"
    result_path = benchmark_dir / "result.json"
    stderr_path.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
    stdout_path.write_text("", encoding="utf-8")
    artifacts = {"stdout": stdout_path.as_posix(), "stderr": stderr_path.as_posix()}
    result = EvaluationResult(
        schema_version="1.0",
        run_id=run_id,
        checkpoint=checkpoint,
        benchmark=spec.benchmark_id,
        profile=spec.profile,
        prediction_mode="score_only" if "score-only" in display_command else "fresh",
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
    if results and all(result.status == "skipped" for result in results):
        return "skipped"
    return "passed"


def _flat_metrics(results: Sequence[EvaluationResult]) -> dict[str, float]:
    return {
        metric_name: value
        for result in results
        for metric_name, value in result.metrics.items()
    }


def _metric_directions(specs: Sequence[BenchmarkSpec]) -> dict[str, str]:
    return {
        metric_name: direction
        for spec in specs
        for metric_name, direction in spec.metric_directions.items()
    }


def _toolvista_commit() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


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
    swift_executable: str = "swift",
    native_runner: Callable[..., EvaluationResult] = run_native_benchmark,
    swift_runner: Callable[..., EvaluationResult] = run_swift_benchmark,
) -> dict[str, Any]:
    run_dir = runs_root / run_id
    eval_dir = run_dir / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
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
            else:
                if score_only:
                    raise ValueError("--score-only is supported only for one native specialty benchmark in Phase 1.")
                result = swift_runner(
                    spec,
                    checkpoint=checkpoint,
                    run_id=run_id,
                    benchmark_dir=benchmark_dir,
                    infer_backend=infer_backend,
                    eval_url=eval_url,
                    resume=resume,
                    swift_executable=swift_executable,
                )
        except Exception as exc:
            result = _exception_result(
                spec,
                checkpoint=checkpoint,
                run_id=run_id,
                benchmark_dir=benchmark_dir,
                display_command=display_command,
                error=exc,
            )
        results.append(result)

    end_time = dt.datetime.now(dt.timezone.utc)
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
            "api_base": api_base,
            "api_model": api_model,
            "infer_backend": infer_backend,
            "eval_url": eval_url,
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
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
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
                "result_path": result.test_summary.result_path,
                "main_metrics": list(result.test_summary.main_metrics),
            }
            for result in results
        },
        "runtime": {
            "duration_seconds": (end_time - start_time).total_seconds(),
            "benchmark_duration_seconds": sum(float(result.runtime.get("duration_seconds", 0.0)) for result in results),
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
    try:
        wandb = wandb_module or importlib.import_module("wandb")
        run = wandb.init(project=project, group=group, name=str(summary["run_id"]), job_type="evaluation")
        metrics = summary.get("metrics", {})
        run.log(
            {
                f"eval/{name}": float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        )
        run.finish()
        return None
    except Exception as exc:
        return f"W&B upload failed: {type(exc).__name__}: {exc}"
```

This is the only W&B code in Phase 1. It uploads final evaluation metrics; it does not mirror training histories, queue retries, or reconstruct offline runs.

- [ ] **Step 4: Run summary and complete runner tests**

Run:

```powershell
python -m unittest tests.test_evaluation_runner -v
```

Expected: all 9 runner tests PASS.

- [ ] **Step 5: Commit sequential suite summaries**

```bash
git add toolvista/evaluation/runner.py tests/test_evaluation_runner.py
git commit -m "feat: summarize sequential evaluation suites"
```

### Task 9: Add The Unified `scripts/evaluate.py` Command

**Files:**
- Create: `scripts/evaluate.py`
- Modify: `tests/test_evaluate_script.py`

- [ ] **Step 1: Add a failing end-to-end score-only CLI test**

Append these imports and tests to `tests/test_evaluate_script.py`:

```python
import subprocess
import sys


class EvaluateCLITest(unittest.TestCase):
    def test_score_only_cli_runs_one_native_benchmark_end_to_end(self) -> None:
        fixture = REPO_ROOT / "examples/evaluation/xlam_predictions.jsonl"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_dir = root / "runs/smoke/eval/xlam_function_calling"
            benchmark_dir.mkdir(parents=True)
            (benchmark_dir / "predictions.jsonl").write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")

            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/evaluate.py"),
                    "--config",
                    str(REPO_ROOT / "configs/evaluation/benchmarks.yaml"),
                    "--suite-config",
                    str(REPO_ROOT / "configs/evaluation/suites.yaml"),
                    "--benchmark",
                    "xlam_function_calling",
                    "--checkpoint",
                    "checkpoint",
                    "--run-id",
                    "smoke",
                    "--runs-root",
                    str(root / "runs"),
                    "--score-only",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(process.returncode, 0, process.stderr)
            summary = json.loads((root / "runs/smoke/summary.json").read_text(encoding="utf-8"))
            result = json.loads((benchmark_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "passed")
            self.assertEqual(result["prediction_mode"], "score_only")
            self.assertEqual(result["test_summary"]["sample_count"], 4)
            self.assertIn("测试方式", process.stdout)
            self.assertIn("结果与 artifact 路径", process.stdout)

    def test_score_only_rejects_a_suite_or_swift_benchmark(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/evaluate.py"),
                "--suite",
                "c1_light",
                "--checkpoint",
                "checkpoint",
                "--run-id",
                "invalid",
                "--score-only",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("one native --benchmark", process.stderr)
```

- [ ] **Step 2: Run the CLI tests and verify the script is missing**

Run:

```powershell
python -m unittest tests.test_evaluate_script.EvaluateCLITest -v
```

Expected: FAIL because `scripts/evaluate.py` does not exist.

- [ ] **Step 3: Implement the exact unified CLI**

Create `scripts/evaluate.py`:

```python
"""Run one lightweight ToolVista evaluation benchmark or suite."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.io import load_benchmark_registry, load_json, load_suites
from toolvista.evaluation.runner import evaluate_suite, upload_summary_to_wandb


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/benchmarks.yaml")
    parser.add_argument("--suite-config", default="configs/evaluation/suites.yaml")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--suite")
    selection.add_argument("--benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--api-base")
    parser.add_argument("--api-model")
    parser.add_argument("--infer-backend", default="vllm")
    parser.add_argument("--eval-url")
    parser.add_argument("--swift-executable", default="swift")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--stage")
    parser.add_argument("--algorithm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--history-summary", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="toolvista-qwen3vl")
    parser.add_argument("--wandb-group")
    return parser.parse_args(argv)


def _metrics_from_summary(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    metrics = load_json(path).get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Summary metrics must be a mapping: {path}")
    return {
        str(name): float(value)
        for name, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _print_test_handoff(summary: Mapping[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for benchmark_id, benchmark in summary.get("benchmarks", {}).items():
        result = load_json(benchmark["result_path"])
        test = result["test_summary"]
        print(f"\n[{benchmark_id}]")
        print(f"测试方式: {test['method']}")
        print(f"测评集及 split: {test['dataset']}")
        print(f"来源项目: {test['source_project']}")
        print(f"使用的上游 evaluator 或 ToolVista scorer: {test['evaluator']}")
        print(f"样本数或限制: {test['sample_count']}")
        print(f"实际启动命令: {test['command']}")
        print(f"主要指标: {', '.join(test['main_metrics'])}")
        print(f"结果与 artifact 路径: {test['result_path']} | {json.dumps(test['artifact_paths'], ensure_ascii=False)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    registry = load_benchmark_registry(args.config)
    suites = load_suites(args.suite_config)
    if args.benchmark:
        if args.benchmark not in registry:
            raise SystemExit(f"Unknown benchmark: {args.benchmark}")
        selected_ids = [args.benchmark]
    else:
        if args.suite not in suites:
            raise SystemExit(f"Unknown suite: {args.suite}")
        selected_ids = suites[args.suite]
    missing = [benchmark_id for benchmark_id in selected_ids if benchmark_id not in registry]
    if missing:
        raise SystemExit(f"Suite references unknown benchmarks: {', '.join(missing)}")
    specs = [registry[benchmark_id] for benchmark_id in selected_ids]
    if args.score_only and (len(specs) != 1 or not specs[0].execution.startswith("native_")):
        raise SystemExit("--score-only requires one native --benchmark in Phase 1.")

    display_command = shlex.join([sys.executable, *sys.argv])
    summary = evaluate_suite(
        specs,
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        runs_root=args.runs_root,
        api_base=args.api_base,
        api_model=args.api_model,
        infer_backend=args.infer_backend,
        eval_url=args.eval_url,
        timeout=args.timeout,
        score_only=args.score_only,
        resume=args.resume,
        display_command=display_command,
        stage=args.stage,
        algorithm=args.algorithm,
        seed=args.seed,
        baseline_metrics=_metrics_from_summary(args.baseline_summary),
        historical_metrics=[_metrics_from_summary(path) for path in args.history_summary],
        swift_executable=args.swift_executable,
    )
    if args.wandb:
        warning = upload_summary_to_wandb(
            summary,
            project=args.wandb_project,
            group=args.wandb_group or args.stage,
        )
        if warning:
            print(warning, file=sys.stderr)
    _print_test_handoff(summary)
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the end-to-end CLI tests**

Run:

```powershell
python -m unittest tests.test_evaluate_script.EvaluateCLITest -v
```

Expected: 2 tests PASS. The successful test exercises `predictions.jsonl -> scorer -> upstream_result.json -> result.json -> summary.json` without a model.

- [ ] **Step 5: Commit the unified command**

```bash
git add scripts/evaluate.py tests/test_evaluate_script.py
git commit -m "feat: add unified evaluation command"
```

### Task 10: Document AutoDL Commands, Test Sources, And Existing Training Metrics

**Files:**
- Create: `docs/phase1_evaluation_usage.md`
- Modify: `tests/test_evaluate_script.py`

- [ ] **Step 1: Add a failing documentation contract test**

Append to `tests/test_evaluate_script.py`:

```python
class EvaluationDocumentationTest(unittest.TestCase):
    def test_usage_doc_names_sources_commands_metrics_and_artifacts(self) -> None:
        text = (REPO_ROOT / "docs/phase1_evaluation_usage.md").read_text(encoding="utf-8")

        for required in (
            "xLAM ActionStudio",
            "MLLM-Tool",
            "ms-swift",
            "VLMEvalKit",
            "python scripts/prepare_eval_data.py",
            "swift deploy",
            "python scripts/evaluate.py",
            "--report_to wandb",
            "--log_completions true",
            "--log_entropy true",
            "--log_rollout_offpolicy_metrics true",
            "loss",
            "grad_norm",
            "reward_std",
            "KL",
            "runs/<run_id>/summary.json",
        ):
            self.assertIn(required, text)
```

- [ ] **Step 2: Run the documentation test and verify the file is missing**

Run:

```powershell
python -m unittest tests.test_evaluate_script.EvaluationDocumentationTest -v
```

Expected: ERROR with `FileNotFoundError` for `docs/phase1_evaluation_usage.md`.

- [ ] **Step 3: Create the exact Phase 1 usage guide**

Create `docs/phase1_evaluation_usage.md` with these sections and commands:

````markdown
# Phase 1 轻量评测使用说明

Phase 1 只负责三个当前可稳定执行的评测入口。专项评分尽量复用上游数据定义；只有 xLAM 和 MLLM-Tool 仓库缺少可直接复用的 held-out scorer，因此 ToolVista 提供两个小型 scorer。通用视觉评测直接调用 ms-swift 已集成的 VLMEvalKit，不复制 evaluator 源码。

| 测试 | 测试方式 | 测评集及 split | 来源项目 | evaluator/scorer | 轻量限制 | 主要指标 |
|---|---|---|---|---|---:|---|
| xLAM 函数调用 | OpenAI-compatible 推理后做协议评分 | 固定 seed 的 one-step dev manifest | xLAM ActionStudio | `toolvista.evaluation.scorers.score_xlam` | 建议 200 | JSON/schema 合法率、函数名准确率、参数 F1/准确率/精确匹配、并行调用精确匹配 |
| MLLM Router | OpenAI-compatible 推理后做多标签 Router 评分 | MLLM-Tool `text` 固定 seed dev manifest | MLLM-Tool | `toolvista.evaluation.scorers.score_mllm_router` | 建议 200 | top-1、candidate hit、macro F1、非法 route、按模态准确率 |
| 通用视觉理解 | `swift eval` 调用 EvalScope/VLMEvalKit | RealWorldQA、MathVision_MINI、MMMU_DEV_VAL、OCRBench 的上游 split | ms-swift + VLMEvalKit | 上游 `swift eval --eval_backend VLMEvalKit` | 每数据集 100 | 保留上游指标名和量纲，并按 registry 显式映射 ToolVista 指标 |

ToolVQA、Visual-RFT、MAT 和 BFCL 不属于 Phase 1；对应 evaluator 集成后再加入累计 suite。

## 1. 准备冻结评测文件

AutoDL 中先确认 `configs/autodl.yaml` 指向实际 xLAM 和 MLLM-Tool 数据根目录。然后运行：

```bash
cd /root/autodl-tmp/ToolVista
python scripts/prepare_eval_data.py \
  --config configs/autodl.yaml \
  --xlam-output data/eval/xlam/dev.jsonl \
  --mllm-output data/eval/mllm_router/dev.jsonl \
  --xlam-limit 200 \
  --mllm-limit 200 \
  --seed 42
```

如果已经生成训练 JSONL，把它们逐个传给 `--training-file` 做 sample ID 和规范化文本精确重合检查：

```bash
python scripts/prepare_eval_data.py \
  --config configs/autodl.yaml \
  --filter-training data/exports/msswift_sft/xlam_train_with_metadata.jsonl data/exports/msswift_sft/xlam_train_filtered.jsonl \
  --filter-training data/exports/msswift_sft/mllm_tool_text_with_metadata.jsonl data/exports/msswift_sft/mllm_tool_text_filtered.jsonl \
  --training-file data/exports/msswift_sft/xlam_train_filtered.jsonl \
  --training-file data/exports/msswift_sft/mllm_tool_text_filtered.jsonl \
  --overwrite
```

过滤输入必须来自带 `--include-metadata` 的训练导出，这样 JSONL 内保留 `metadata.sample_id`。冻结文件生成后不要在每次评测时重新随机切分。`.report.json` 中的 `sample_ids` 是训练导出时要排除的评测 ID，正式 SFT 使用 `_filtered.jsonl` 文件。

## 2. 启动专项评测所需模型 API

xLAM 和 MLLM Router 通过现成的 ms-swift 部署能力启动 Qwen3-VL，不在 ToolVista 重写推理服务器：

```bash
CHECKPOINT=/root/autodl-tmp/checkpoints/c3_mllm_router_sft
API_MODEL=$(basename "$CHECKPOINT")
swift deploy \
  --model "$CHECKPOINT" \
  --infer_backend vllm \
  --host 0.0.0.0 \
  --port 8000
```

这个进程由用户显式启动和停止。每次切换被测 checkpoint 时重启一次，并让 `API_MODEL` 保持为该 checkpoint 的目录名；这也与 `swift eval` 使用的 `model_suffix` 一致。Phase 1 不做后台生命周期管理或自动重启。

## 3. 每次训练后的轻量回测

C1 xLAM SFT 后：

```bash
CHECKPOINT=/root/autodl-tmp/checkpoints/c1_xlam_sft
API_MODEL=$(basename "$CHECKPOINT")
python scripts/evaluate.py \
  --suite c1_light \
  --checkpoint "$CHECKPOINT" \
  --run-id c1_xlam_sft_seed42 \
  --stage c1-xlam-sft \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model "$API_MODEL" \
  --eval-url http://127.0.0.1:8000/v1 \
  --seed 42 \
  --wandb \
  --wandb-group c1-xlam-sft
```

C3 MLLM Router SFT 后：

```bash
CHECKPOINT=/root/autodl-tmp/checkpoints/c3_mllm_router_sft
API_MODEL=$(basename "$CHECKPOINT")
python scripts/evaluate.py \
  --suite c3_light \
  --checkpoint "$CHECKPOINT" \
  --run-id c3_mllm_router_sft_seed42 \
  --stage c3-mllm-router-sft \
  --api-base http://127.0.0.1:8000/v1 \
  --api-model "$API_MODEL" \
  --eval-url http://127.0.0.1:8000/v1 \
  --seed 42 \
  --baseline-summary runs/c0_base_seed42/summary.json \
  --history-summary runs/c1_xlam_sft_seed42/summary.json \
  --wandb \
  --wandb-group c3-mllm-router-sft
```

`qwen3vl_general_light` 会在同一命令中调用等价的上游命令：

```bash
swift eval \
  --model /root/autodl-tmp/checkpoints/c3_mllm_router_sft \
  --eval_backend VLMEvalKit \
  --infer_backend vllm \
  --eval_dataset RealWorldQA MathVision_MINI MMMU_DEV_VAL OCRBench \
  --eval_limit 100 \
  --eval_output_dir runs/c3_mllm_router_sft_seed42/eval/qwen3vl_general_light/swift_eval \
  --eval_generation_config '{"max_tokens":512,"temperature":0}' \
  --eval_url http://127.0.0.1:8000/v1
```

当专项评测已经使用 `swift deploy` 启动共享 API 时，suite 命令必须同时传入 `--eval-url`，避免 `swift eval` 在同一张 GPU 上再次加载 checkpoint。只运行通用视觉 benchmark、且没有外部 API 时，才省略 `--eval-url` 让 ms-swift 自行部署。

## 4. 简单恢复与重新评分

整项结果已存在时跳过：

```bash
python scripts/evaluate.py <原参数> --resume
```

只有本地专项 benchmark 支持使用完整 `predictions.jsonl` 重新评分：

```bash
python scripts/evaluate.py \
  --benchmark xlam_function_calling \
  --checkpoint /root/autodl-tmp/checkpoints/c1_xlam_sft \
  --run-id c1_xlam_sft_seed42 \
  --score-only
```

损坏或部分写入的文件直接手动改名或删除，然后整项重跑。Phase 1 不做逐样本 resume。

## 5. 结果与 artifact

```text
runs/<run_id>/
|-- manifest.json
|-- eval/
|   `-- <benchmark_id>/
|       |-- predictions.jsonl
|       |-- upstream_result.json
|       |-- result.json
|       |-- stdout.log
|       `-- stderr.log
`-- summary.json
```

`result.json.test_summary` 和命令行末尾都必须报告：测试方式、测评集及 split、来源项目、上游 evaluator 或 ToolVista scorer、样本数或限制、实际命令、主要指标、结果与 artifact 路径。`runs/<run_id>/summary.json` 是跨 checkpoint 比较的简化入口；原始上游结果仍保留在各 benchmark 目录。

## 6. 训练指标直接复用 ms-swift/W&B

ToolVista 不再实现第二套训练监控器。SFT 至少启用：

```bash
--report_to wandb \
--logging_steps 5
```

ms-swift/Transformers 能直接记录的常用指标包括 `loss`、学习率、epoch、global step、`grad_norm`、token accuracy、训练速度、耗时和显存日志；实际字段取决于训练后端。

GRPO、BNPO、DAPO 可在正式诊断时增加：

```bash
--report_to wandb \
--log_completions true \
--log_entropy true \
--log_rollout_offpolicy_metrics true
```

RL 可直接复用的指标包括总 reward、各 reward function 的均值/标准差、`reward_std`、零方差 group 比例、completion length、KL、entropy，以及启用后产生的 rollout/off-policy 诊断。`--log_rollout_offpolicy_metrics true` 只用于需要详细训推差异诊断的正式实验；简单 smoke test 不必默认打开。

ToolVista 外层只补充每个 benchmark 的开始/结束时间、持续时间和退出码。`peak_gpu_memory_mb` 在 Phase 1 可为 `null`，因为不为它引入轮询守护进程。
````

- [ ] **Step 4: Run the documentation contract test**

Run:

```powershell
python -m unittest tests.test_evaluate_script.EvaluationDocumentationTest -v
```

Expected: 1 test PASS.

- [ ] **Step 5: Commit the usage guide**

```bash
git add docs/phase1_evaluation_usage.md tests/test_evaluate_script.py
git commit -m "docs: explain phase1 evaluation workflow"
```

### Task 11: Run Phase 1 Verification And Record The Test Handoff

**Files:**
- Modify only if a verification failure identifies a defect in: `toolvista/evaluation/*.py`, `scripts/prepare_eval_data.py`, `scripts/evaluate.py`, `configs/evaluation/*.yaml`, `tests/test_evaluation_*.py`, `tests/test_evaluate_script.py`, `docs/phase1_evaluation_usage.md`

- [ ] **Step 1: Run the focused Phase 1 test suite**

Run:

```powershell
python -m unittest tests.test_evaluation_io tests.test_evaluation_scorers tests.test_evaluation_runner tests.test_evaluate_script -v
```

Expected: all Phase 1 unit, contract, scorer, resume, CLI, and documentation tests PASS.

- [ ] **Step 2: Run Python syntax compilation**

Run:

```powershell
python -m py_compile toolvista/evaluation/models.py toolvista/evaluation/io.py toolvista/evaluation/scorers.py toolvista/evaluation/runner.py scripts/prepare_eval_data.py scripts/evaluate.py tests/test_evaluation_io.py tests/test_evaluation_scorers.py tests/test_evaluation_runner.py tests/test_evaluate_script.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Run repository regression tests**

Run the explicit current test modules so inaccessible user-owned temporary directories do not break discovery:

```powershell
python -m unittest tests.test_xlam_adapter tests.test_export_mllm_tool tests.test_toolvqa_adapter tests.test_visual_rft_adapter tests.test_export_visual_rft tests.test_ms_swift_grpo_exporter tests.test_visual_rft_reward_plugin tests.test_evaluation_io tests.test_evaluation_scorers tests.test_evaluation_runner tests.test_evaluate_script -v
```

Expected: all existing adapter/exporter/reward tests and all new evaluation tests PASS.

- [ ] **Step 4: Run an offline scorer smoke test through the real CLI**

First place the frozen fixture where the selected run expects it, then run:

```powershell
$runDir = "runs/phase1_score_only_smoke/eval/xlam_function_calling"
New-Item -ItemType Directory -Force $runDir | Out-Null
Copy-Item examples/evaluation/xlam_predictions.jsonl "$runDir/predictions.jsonl"
python scripts/evaluate.py --benchmark xlam_function_calling --checkpoint checkpoint --run-id phase1_score_only_smoke --score-only
```

Expected: exit code 0; the console prints all eight required Chinese test-handoff fields; these files exist:

```text
runs/phase1_score_only_smoke/eval/xlam_function_calling/predictions.jsonl
runs/phase1_score_only_smoke/eval/xlam_function_calling/upstream_result.json
runs/phase1_score_only_smoke/eval/xlam_function_calling/result.json
runs/phase1_score_only_smoke/eval/xlam_function_calling/stdout.log
runs/phase1_score_only_smoke/eval/xlam_function_calling/stderr.log
runs/phase1_score_only_smoke/manifest.json
runs/phase1_score_only_smoke/summary.json
```

- [ ] **Step 5: Run the ms-swift bridge contract test without loading a model**

Run:

```powershell
python -m unittest tests.test_evaluation_runner.SwiftEvaluationRunnerTest.test_success_preserves_raw_report_stdout_stderr_and_normalized_metrics -v
```

Expected: PASS. Test method is a frozen `swift eval` report contract test; source is local ms-swift `swift/pipelines/eval/eval.py`; evaluator semantics are VLMEvalKit; sample limit is 10 in the fixture; no GPU or benchmark download is used.

- [ ] **Step 6: Perform a real AutoDL smoke test when the environment is available**

Run the commands from `docs/phase1_evaluation_usage.md` with `--xlam-limit 2`, `--mllm-limit 2`, and a lightweight `--benchmark` selection before running `c1_light` or `c3_light`.

Expected: real model inference produces `predictions.jsonl`; native specialty scores complete; `swift eval` downloads/uses its upstream benchmark data and writes its raw report. This external GPU/network smoke is required before claiming the bridge is ready for a formal checkpoint, but it is not required for the local implementation commit.

- [ ] **Step 7: Report the exact test provenance after verification**

The implementation handoff must include one compact table or list with these rows:

```text
xLAM scorer contract/smoke
  测试方式: frozen predictions -> ToolVista native scorer
  测评集及 split: examples/evaluation fixture; real run uses fixed_seed_dev
  来源项目: xLAM ActionStudio
  evaluator/scorer: toolvista.evaluation.scorers.score_xlam
  样本数: fixture 4; real smoke limit used in the command
  启动命令: the exact unittest and scripts/evaluate.py commands actually run
  主要指标: JSON/schema validity, function name, arguments, parallel exact match
  路径: result.json, upstream_result.json, stdout.log, stderr.log, summary.json

MLLM Router scorer contract/smoke
  测试方式: frozen predictions -> ToolVista native scorer
  测评集及 split: examples/evaluation fixture; real run uses MLLM-Tool text fixed_seed_dev
  来源项目: MLLM-Tool
  evaluator/scorer: toolvista.evaluation.scorers.score_mllm_router
  样本数: fixture 4; real smoke limit used in the command
  主要指标: top-1, candidate hit, macro F1, invalid route, modality accuracy

ms-swift/VLMEvalKit bridge contract/smoke
  测试方式: frozen eval_result.jsonl parser contract; real run invokes swift eval
  测评集及 split: RealWorldQA, MathVision_MINI, MMMU_DEV_VAL, OCRBench upstream splits
  来源项目: ms-swift + EvalScope/VLMEvalKit
  evaluator: swift eval --eval_backend VLMEvalKit
  样本数: fixture limit 10; configured light run limit 100 per dataset
  主要指标: preserved upstream metrics plus explicit registry mappings
```

Do not claim a real Qwen3-VL benchmark score from the local contract tests. Distinguish clearly between format/scorer verification and a model-backed AutoDL result.

- [ ] **Step 8: Commit any verification-only corrections**

If verification required code corrections, commit only those scoped corrections:

```bash
git add toolvista/evaluation scripts/prepare_eval_data.py scripts/evaluate.py configs/evaluation tests/test_evaluation_io.py tests/test_evaluation_scorers.py tests/test_evaluation_runner.py tests/test_evaluate_script.py docs/phase1_evaluation_usage.md
git commit -m "test: verify phase1 evaluation workflow"
```

If no corrections were needed, do not create an empty commit.

## Plan Self-Review / 计划自检

- Spec coverage: Tasks 1-2 define the small result protocol and registry; Tasks 3-4 implement only the two missing native scorers; Task 5 freezes manifests and checks simple leakage; Tasks 6-8 implement sequential execution, upstream preservation, whole-benchmark resume, comparison, and optional W&B; Task 9 exposes one CLI; Task 10 documents upstream sources and training metrics; Task 11 verifies contracts and records the required handoff fields.
- Scope boundary: ToolVQA, AgentLego, Visual-RFT, COCO/LISA, MAT, BFCL, service management, sample-level recovery, caching, queues, and multi-seed RL aggregation are not implemented here.
- Recovery boundary: `--resume` only reads an existing per-benchmark `result.json`; `--score-only` only accepts one native benchmark and a complete `predictions.jsonl`; every other state reruns or fails visibly.
- Metric boundary: ms-swift/VLMEvalKit names and scales are preserved in `upstream_metrics`; scale changes are declared in YAML; the runner does not infer percent-vs-ratio semantics.
- Training metrics boundary: W&B/ms-swift remains responsible for loss, gradient, throughput, memory, token, reward, KL, entropy, and rollout metrics. ToolVista only records evaluation duration and exit code; GPU peak is nullable.
- Type consistency: `BenchmarkSpec`, `TestSummary`, `EvaluationResult`, runner signatures, YAML field names, `result.json`, and CLI arguments use the same names throughout all tasks.
- Test reporting: every benchmark result and console handoff carries method, dataset/split, source project, evaluator/scorer, sample count/limit, actual command, main metrics, and result/artifact paths.
