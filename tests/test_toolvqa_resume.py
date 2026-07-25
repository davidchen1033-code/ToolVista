from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.runner import evaluate_suite
from toolvista.evaluation.toolvqa_benchmark import run_toolvqa_benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]
METRIC = "toolvqa_sbs/tool_accuracy"


def toolvqa_spec() -> BenchmarkSpec:
    return BenchmarkSpec.from_mapping(
        "toolvqa",
        {
            "execution": "toolvqa_step",
            "profile": "light",
            "method": "specialty",
            "source_project": "ToolVQA",
            "dataset_name": "fixture",
            "split": "fixture",
            "evaluator": "upstream evaluator",
            "dataset_path": "fixture",
            "primary_metrics": [METRIC],
            "metric_directions": {METRIC: "maximize"},
        },
    )


def existing_result(status: str, benchmark_dir: Path) -> EvaluationResult:
    passed = status == "passed"
    return EvaluationResult(
        schema_version="1.0",
        run_id="run",
        checkpoint="checkpoint",
        benchmark="toolvqa",
        profile="light",
        prediction_mode="fresh",
        test_summary=TestSummary(
            method="specialty",
            dataset="fixture / fixture",
            source_project="ToolVQA",
            evaluator="upstream evaluator",
            sample_count=1,
            command="command",
            main_metrics=(METRIC,),
            result_path=(benchmark_dir / "result.json").as_posix(),
            artifact_paths={},
        ),
        status=status,
        attempted=1 if passed else 0,
        completed=1 if passed else 0,
        failed=0,
        upstream_metrics={},
        metrics={METRIC: 0.5} if passed else {},
        runtime={"duration_seconds": 1.0, "exit_code": 0 if passed else 1},
        artifacts={},
    )


class ToolVQAResumeTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-toolvqa-resume-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_resume_only_reuses_passed_result(self) -> None:
        spec = toolvqa_spec()
        for initial_status in ("passed", "failed"):
            with self.subTest(initial_status=initial_status):
                benchmark_dir = self.make_temp_dir() / "eval/toolvqa"
                benchmark_dir.mkdir(parents=True)
                previous = existing_result(initial_status, benchmark_dir)
                (benchmark_dir / "result.json").write_text(
                    json.dumps(previous.to_dict()),
                    encoding="utf-8",
                )
                upstream = {
                    "metrics": {METRIC: 0.75},
                    "upstream_metrics": {"tool_acc": 75.0},
                    "upstream_result": (benchmark_dir / "upstream/result.json").as_posix(),
                    "sample_count": 1,
                    "command": "opencompass run.py",
                }
                with patch(
                    "toolvista.evaluation.toolvqa_benchmark.run_toolvqa",
                    return_value=upstream,
                ) as mocked:
                    resumed = run_toolvqa_benchmark(
                        spec,
                        checkpoint="checkpoint",
                        run_id="run",
                        benchmark_dir=benchmark_dir,
                        api_base="http://127.0.0.1:8000/v1",
                        api_model="candidate",
                        timeout=10.0,
                        score_only=False,
                        resume=True,
                        display_command="evaluate --resume",
                    )

                if initial_status == "passed":
                    mocked.assert_not_called()
                    self.assertEqual(resumed.metrics[METRIC], 0.5)
                else:
                    mocked.assert_called_once()
                    self.assertEqual(resumed.metrics[METRIC], 0.75)

    def test_suite_exception_preserves_upstream_logs(self) -> None:
        root = self.make_temp_dir()

        def failed_runner(_spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            benchmark_dir = Path(kwargs["benchmark_dir"])
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            (benchmark_dir / "stdout.log").write_text("upstream stdout", encoding="utf-8")
            (benchmark_dir / "stderr.log").write_text("upstream stderr", encoding="utf-8")
            raise RuntimeError("ToolVQA wrapper failure")

        summary = evaluate_suite(
            [toolvqa_spec()],
            checkpoint="checkpoint",
            run_id="run",
            runs_root=root,
            api_base="http://127.0.0.1:8000/v1",
            api_model="candidate",
            infer_backend="vllm",
            eval_url=None,
            timeout=10.0,
            score_only=False,
            resume=False,
            display_command="evaluate",
            stage="c4",
            algorithm="bnpo",
            seed=42,
            baseline_metrics={},
            historical_metrics=[],
            toolvqa_runner=failed_runner,
        )

        benchmark_dir = root / "run/eval/toolvqa"
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            (benchmark_dir / "stdout.log").read_text(encoding="utf-8"),
            "upstream stdout",
        )
        stderr = (benchmark_dir / "stderr.log").read_text(encoding="utf-8")
        self.assertIn("upstream stderr", stderr)
        self.assertIn("RuntimeError: ToolVQA wrapper failure", stderr)


if __name__ == "__main__":
    unittest.main()
