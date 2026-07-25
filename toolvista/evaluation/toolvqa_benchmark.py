"""EvaluationResult wrapper around ToolVQA's upstream OpenCompass step evaluator."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from toolvista.evaluation.io import write_json
from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.toolvqa import run_toolvqa


def run_toolvqa_benchmark(
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
) -> EvaluationResult:
    del timeout
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    result_path = benchmark_dir / "result.json"
    if resume and result_path.is_file():
        resumed = EvaluationResult.from_dict(
            json.loads(result_path.read_text(encoding="utf-8"))
        )
        if resumed.status == "passed":
            return resumed
    if result_path.is_file():
        result_path.unlink()
    if score_only:
        raise ValueError("ToolVQA OpenCompass step benchmark does not support --score-only.")
    if not spec.dataset_path:
        raise ValueError("ToolVQA benchmark requires dataset_path.")
    if not api_base or not api_model:
        raise ValueError("ToolVQA benchmark requires --api-base and --api-model.")

    toolvqa_root = Path(str(spec.settings.get("toolvqa_root", ""))).expanduser()
    media_root_value = spec.settings.get("media_root")
    media_root = Path(str(media_root_value)).expanduser() if media_root_value else None
    python_executable = os.environ.get(
        "TOOLVQA_PYTHON",
        str(spec.settings.get("python_executable") or sys.executable),
    )
    workers = int(spec.settings.get("workers", 2))
    start = time.perf_counter()
    upstream = run_toolvqa(
        mode="step",
        toolvqa_root=toolvqa_root,
        data_root=Path(spec.dataset_path),
        media_root=media_root,
        output_dir=benchmark_dir,
        api_base=api_base,
        api_model=api_model,
        python_executable=python_executable,
        workers=workers,
        limit=spec.eval_limit,
    )
    metrics = {str(key): float(value) for key, value in upstream["metrics"].items()}
    missing = [metric for metric in spec.primary_metrics if metric not in metrics]
    if missing:
        raise ValueError(f"ToolVQA result is missing primary metrics: {missing}")

    artifacts = {
        "runtime_config": (benchmark_dir / "runtime_config.py").as_posix(),
        "prepared_dataset": (benchmark_dir / "data/dataset.json").as_posix(),
        "upstream_root": (benchmark_dir / "upstream").as_posix(),
        "upstream_result": str(upstream["upstream_result"]),
        "stdout": (benchmark_dir / "stdout.log").as_posix(),
        "stderr": (benchmark_dir / "stderr.log").as_posix(),
    }
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
            sample_count=int(upstream["sample_count"]),
            command=str(upstream["command"]),
            main_metrics=spec.primary_metrics,
            result_path=result_path.as_posix(),
            artifact_paths=artifacts,
        ),
        status="passed",
        attempted=int(upstream["sample_count"]),
        completed=int(upstream["sample_count"]),
        failed=0,
        upstream_metrics=dict(upstream["upstream_metrics"]),
        metrics=metrics,
        runtime={
            "duration_seconds": time.perf_counter() - start,
            "peak_gpu_memory_mb": None,
            "exit_code": 0,
        },
        artifacts=artifacts,
    )
    write_json(result_path, result.to_dict())
    return result
