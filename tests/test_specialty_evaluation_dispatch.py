from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.runner import evaluate_suite


REPO_ROOT = Path(__file__).resolve().parents[1]


def specialty_spec(benchmark_id: str, execution: str, metric: str) -> BenchmarkSpec:
    return BenchmarkSpec.from_mapping(
        benchmark_id,
        {
            "execution": execution,
            "profile": "light",
            "method": "specialty",
            "source_project": "upstream",
            "dataset_name": "fixture",
            "split": "fixture",
            "evaluator": "upstream evaluator",
            "dataset_path": "fixture",
            "primary_metrics": [metric],
            "metric_directions": {metric: "maximize"},
        },
    )


class SpecialtyEvaluationDispatchTest(unittest.TestCase):
    def test_suite_dispatches_toolvqa_and_visual_rft_to_specialty_runners(self) -> None:
        root = REPO_ROOT / "tests" / f"tmp-specialty-dispatch-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        calls: list[str] = []

        def runner(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            calls.append(spec.execution)
            benchmark_dir = Path(kwargs["benchmark_dir"])
            metric = spec.primary_metrics[0]
            return EvaluationResult(
                schema_version="1.0",
                run_id="run",
                checkpoint="checkpoint",
                benchmark=spec.benchmark_id,
                profile=spec.profile,
                prediction_mode="fresh",
                test_summary=TestSummary(
                    method=spec.method,
                    dataset=f"{spec.dataset_name} / {spec.split}",
                    source_project=spec.source_project,
                    evaluator=spec.evaluator,
                    sample_count=1,
                    command="command",
                    main_metrics=spec.primary_metrics,
                    result_path=(benchmark_dir / "result.json").as_posix(),
                    artifact_paths={},
                ),
                status="passed",
                attempted=1,
                completed=1,
                failed=0,
                upstream_metrics={},
                metrics={metric: 0.5},
                runtime={"duration_seconds": 0.1, "exit_code": 0},
                artifacts={},
            )

        specs = [
            specialty_spec("toolvqa", "toolvqa_step", "toolvqa_sbs/tool_accuracy"),
            specialty_spec("visual_rft", "visual_rft_coco", "visual_rft_coco/novel15_map"),
        ]
        summary = evaluate_suite(
            specs,
            checkpoint="checkpoint",
            run_id="run",
            runs_root=root,
            api_base="http://127.0.0.1:8000/v1",
            api_model="checkpoint",
            infer_backend="vllm",
            eval_url=None,
            timeout=10.0,
            score_only=False,
            resume=False,
            display_command="python scripts/evaluate.py",
            stage="c0",
            algorithm=None,
            seed=42,
            baseline_metrics={},
            historical_metrics=[],
            toolvqa_runner=runner,
            visual_rft_runner=runner,
        )

        self.assertEqual(calls, ["toolvqa_step", "visual_rft_coco"])
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(
            set(summary["metrics"]),
            {"toolvqa_sbs/tool_accuracy", "visual_rft_coco/novel15_map"},
        )


if __name__ == "__main__":
    unittest.main()
