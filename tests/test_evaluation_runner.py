from __future__ import annotations

import io
import json
import shlex
import shutil
import subprocess
import unittest
import urllib.error
import uuid
from pathlib import Path
from unittest.mock import patch

from toolvista.evaluation.io import load_json, load_jsonl, write_jsonl
from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary
from toolvista.evaluation.runner import (
    _swift_subprocess_environment,
    build_swift_eval_command,
    calculate_comparisons,
    call_chat_completion,
    evaluate_suite,
    run_native_benchmark,
    run_swift_benchmark,
    upload_summary_to_wandb,
)


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


def mllm_router_spec(dataset_path: Path) -> BenchmarkSpec:
    return BenchmarkSpec.from_mapping(
        "mllm_router",
        {
            "execution": "native_mllm_router",
            "profile": "light",
            "method": "OpenAI-compatible prediction plus ToolVista MLLM Router scorer",
            "source_project": "MLLM-Tool",
            "dataset_name": "MLLM-Tool text Router fixed-seed dev",
            "split": "fixed_seed_dev",
            "evaluator": "toolvista.evaluation.scorers.score_mllm_router",
            "dataset_path": dataset_path.as_posix(),
            "primary_metrics": ["mllm_router/top1_accuracy"],
            "metric_directions": {"mllm_router/top1_accuracy": "maximize"},
        },
    )


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


def mathvision_swift_spec() -> BenchmarkSpec:
    return BenchmarkSpec.from_mapping(
        "mathvision_light",
        {
            "execution": "swift_eval",
            "profile": "light",
            "method": "VLMEvalKit",
            "source_project": "VLMEvalKit",
            "dataset_name": "MathVision MINI",
            "split": "mini",
            "evaluator": "VLMEvalKit fixed judge",
            "primary_metrics": ["vlmevalkit/mathvision_mini/accuracy"],
            "metric_directions": {"vlmevalkit/mathvision_mini/accuracy": "maximize"},
            "eval_backend": "VLMEvalKit",
            "eval_limit": 10,
            "swift_datasets": ["MathVision_MINI"],
        },
    )


def passed_result(
    benchmark: str,
    metric_name: str,
    value: float,
    benchmark_dir: Path,
) -> EvaluationResult:
    result_path = benchmark_dir / "result.json"
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


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class _TemporaryDirectoryTestCase(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-evaluation-runner-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path


class NativeEvaluationRunnerTest(_TemporaryDirectoryTestCase):
    def test_score_only_reuses_predictions_and_writes_required_artifacts(self) -> None:
        root = self.make_temp_dir()
        fixture = REPO_ROOT / "examples/evaluation/xlam_predictions.jsonl"
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
            display_command=(
                "python scripts/evaluate.py --benchmark xlam_function_calling --score-only"
            ),
        )

        self.assertEqual(result.prediction_mode, "score_only")
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.test_summary.sample_count, 4)
        self.assertTrue((benchmark_dir / "result.json").is_file())
        self.assertTrue((benchmark_dir / "upstream_result.json").is_file())
        self.assertTrue((benchmark_dir / "stdout.log").is_file())
        self.assertTrue((benchmark_dir / "stderr.log").is_file())
        stored = load_json(benchmark_dir / "result.json")
        self.assertEqual(stored["test_summary"]["source_project"], "xLAM ActionStudio")
        self.assertEqual(stored["runtime"]["exit_code"], 0)

    def test_fresh_prediction_sends_tools_and_scores_returned_tool_calls(self) -> None:
        root = self.make_temp_dir()
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
                    "target_tool_calls": [
                        {"name": "get_weather", "arguments": {"city": "Paris"}}
                    ],
                }
            ],
        )
        requests: list[dict[str, object]] = []

        def fake_client(**kwargs: object) -> dict[str, object]:
            requests.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Paris"}',
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
        self.assertEqual(
            load_json(root / "run/eval/xlam_function_calling/progress.json"),
            {"phase": "predicting", "completed": 1, "total": 1},
        )

    def test_mllm_router_fresh_prediction_scores_assistant_content(self) -> None:
        root = self.make_temp_dir()
        manifest = root / "mllm_router.jsonl"
        write_jsonl(
            manifest,
            [
                {
                    "sample_id": "m1",
                    "messages": [{"role": "user", "content": "Route this request"}],
                    "tools": [],
                    "input_modality": "text",
                    "target_tools": ["route_a"],
                    "router_catalog": ["route_a", "route_b"],
                }
            ],
        )
        requests: list[dict[str, object]] = []

        def fake_client(**kwargs: object) -> dict[str, object]:
            requests.append(kwargs)
            return {"choices": [{"message": {"content": "  route_a  "}}]}

        result = run_native_benchmark(
            mllm_router_spec(manifest),
            checkpoint="checkpoint",
            run_id="run",
            benchmark_dir=root / "run/eval/mllm_router",
            api_base="http://127.0.0.1:8000/v1",
            api_model="checkpoint",
            timeout=10.0,
            score_only=False,
            resume=False,
            display_command="python scripts/evaluate.py --benchmark mllm_router",
            prediction_client=fake_client,
        )

        self.assertEqual(result.metrics["mllm_router/top1_accuracy"], 1.0)
        self.assertEqual(requests[0]["tools"], [])
        stored = load_jsonl(root / "run/eval/mllm_router/predictions.jsonl")
        self.assertEqual(stored[0]["prediction"], "route_a")

    def test_mllm_router_converts_manifest_media_to_openai_content_parts(self) -> None:
        root = self.make_temp_dir()
        image = root / "sample.jpg"
        image.write_bytes(b"image")
        manifest = root / "mllm_router.jsonl"
        write_jsonl(
            manifest,
            [
                {
                    "sample_id": "m-image",
                    "messages": [{"role": "user", "content": "<image>Route this request"}],
                    "images": [image.as_posix()],
                    "tools": [],
                    "input_modality": "image",
                    "target_tools": ["route_a"],
                    "router_catalog": ["route_a"],
                }
            ],
        )
        requests: list[dict[str, object]] = []

        def fake_client(**kwargs: object) -> dict[str, object]:
            requests.append(kwargs)
            return {"choices": [{"message": {"content": "route_a"}}]}

        run_native_benchmark(
            mllm_router_spec(manifest),
            checkpoint="checkpoint",
            run_id="run",
            benchmark_dir=root / "run/eval/mllm_router",
            api_base="http://127.0.0.1:8000/v1",
            api_model="checkpoint",
            timeout=10.0,
            score_only=False,
            resume=False,
            display_command="python scripts/evaluate.py --benchmark mllm_router",
            prediction_client=fake_client,
        )

        messages = requests[0]["messages"]
        self.assertEqual(
            messages[0]["content"],
            [
                {"type": "image_url", "image_url": {"url": image.as_posix()}},
                {"type": "text", "text": "Route this request"},
            ],
        )

    def test_malformed_model_response_is_recorded_without_retry(self) -> None:
        root = self.make_temp_dir()
        manifest = root / "xlam.jsonl"
        write_jsonl(
            manifest,
            [
                {
                    "sample_id": "bad-response",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [],
                    "target_tool_calls": [
                        {"name": "get_weather", "arguments": {"city": "Paris"}}
                    ],
                }
            ],
        )
        calls = 0

        def malformed_client(**kwargs: object) -> dict[str, object]:
            nonlocal calls
            del kwargs
            calls += 1
            return {"choices": [{"finish_reason": "stop"}]}

        benchmark_dir = root / "run/eval/xlam_function_calling"
        result = run_native_benchmark(
            xlam_spec(manifest),
            checkpoint="checkpoint",
            run_id="run",
            benchmark_dir=benchmark_dir,
            api_base="http://127.0.0.1:8000/v1",
            api_model="checkpoint",
            timeout=10.0,
            score_only=False,
            resume=False,
            display_command="python scripts/evaluate.py --benchmark xlam_function_calling",
            prediction_client=malformed_client,
        )

        self.assertEqual(calls, 1)
        self.assertEqual((result.attempted, result.completed, result.failed), (1, 0, 1))
        self.assertEqual(result.status, "failed")
        row = load_jsonl(benchmark_dir / "predictions.jsonl")[0]
        self.assertIn("no assistant message", row["error"])
        self.assertNotIn("prediction", row)

    def test_score_only_rejects_missing_prediction_and_scorer_count_mismatch(self) -> None:
        root = self.make_temp_dir()
        benchmark_dir = root / "run/eval/xlam_function_calling"
        write_jsonl(benchmark_dir / "predictions.jsonl", [])

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            run_native_benchmark(
                xlam_spec(root / "unused.jsonl"),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=benchmark_dir,
                api_base=None,
                api_model=None,
                timeout=10.0,
                score_only=True,
                resume=False,
                display_command="score-only",
            )

        write_jsonl(
            benchmark_dir / "predictions.jsonl",
            [{"sample_id": "missing", "target_tool_calls": [{"name": "f", "arguments": {}}]}],
        )

        with self.assertRaisesRegex(ValueError, "complete predictions"):
            run_native_benchmark(
                xlam_spec(root / "unused.jsonl"),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=benchmark_dir,
                api_base=None,
                api_model=None,
                timeout=10.0,
                score_only=True,
                resume=False,
                display_command="score-only",
            )

        write_jsonl(
            benchmark_dir / "predictions.jsonl",
            [
                {
                    "sample_id": "present",
                    "target_tool_calls": [{"name": "f", "arguments": {}}],
                    "prediction": {"name": "f", "arguments": {}},
                }
            ],
        )
        invalid_counts = {
            "attempted": 2,
            "completed": 2,
            "failed": 0,
            "upstream_metrics": {},
            "metrics": {"xlam/function_name_accuracy": 1.0},
        }
        with patch("toolvista.evaluation.runner._score_native", return_value=invalid_counts):
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                run_native_benchmark(
                    xlam_spec(root / "unused.jsonl"),
                    checkpoint="checkpoint",
                    run_id="run",
                    benchmark_dir=benchmark_dir,
                    api_base=None,
                    api_model=None,
                    timeout=10.0,
                    score_only=True,
                    resume=False,
                    display_command="score-only",
                )

    def test_fresh_prediction_rejects_manifest_prediction_count_mismatch(self) -> None:
        root = self.make_temp_dir()
        manifest = root / "xlam.jsonl"
        write_jsonl(
            manifest,
            [
                {
                    "sample_id": "x1",
                    "messages": [{"role": "user", "content": "Weather?"}],
                    "tools": [],
                    "target_tool_calls": [{"name": "f", "arguments": {}}],
                }
            ],
        )
        benchmark_dir = root / "run/eval/xlam_function_calling"

        with patch("toolvista.evaluation.runner.predict_native_rows", return_value=[]):
            with self.assertRaisesRegex(ValueError, "prediction count mismatch"):
                run_native_benchmark(
                    xlam_spec(manifest),
                    checkpoint="checkpoint",
                    run_id="run",
                    benchmark_dir=benchmark_dir,
                    api_base="http://127.0.0.1:8000/v1",
                    api_model="checkpoint",
                    timeout=10.0,
                    score_only=False,
                    resume=False,
                    display_command="fresh",
                )

        self.assertFalse((benchmark_dir / "predictions.jsonl").exists())

    def test_resume_returns_existing_result_without_calling_model(self) -> None:
        root = self.make_temp_dir()
        fixture = REPO_ROOT / "examples/evaluation/xlam_predictions.jsonl"
        benchmark_dir = root / "run/eval/xlam_function_calling"
        benchmark_dir.mkdir(parents=True)
        (benchmark_dir / "predictions.jsonl").write_text(
            fixture.read_text(encoding="utf-8"), encoding="utf-8"
        )
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

        def fail_client(**kwargs: object) -> dict[str, object]:
            del kwargs
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


class SwiftEvaluationRunnerTest(_TemporaryDirectoryTestCase):
    def test_swift_environment_drops_only_placeholder_openai_keys(self) -> None:
        for placeholder in ("", "EMPTY", " empty "):
            with self.subTest(placeholder=placeholder):
                with patch.dict("os.environ", {"OPENAI_API_KEY": placeholder}, clear=False):
                    environment = _swift_subprocess_environment()
                self.assertNotIn("OPENAI_API_KEY", environment)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-real-key"}, clear=False):
            environment = _swift_subprocess_environment()
        self.assertEqual(environment["OPENAI_API_KEY"], "sk-real-key")

        with patch.dict("os.environ", {}, clear=True):
            environment = _swift_subprocess_environment(
                judge_api_base="http://127.0.0.1:8000/v1/",
                judge_model="Qwen3-VL-4B-Instruct",
            )
        self.assertEqual(environment["OPENAI_API_KEY"], "EMPTY")
        self.assertEqual(
            environment["OPENAI_API_BASE"],
            "http://127.0.0.1:8000/v1/chat/completions",
        )
        self.assertEqual(environment["LOCAL_LLM"], "Qwen3-VL-4B-Instruct")

        with self.assertRaisesRegex(ValueError, "both judge_api_base and judge_model"):
            _swift_subprocess_environment(judge_api_base="http://127.0.0.1:8000/v1")

    def test_full_swift_eval_reports_unknown_counts_instead_of_zero(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_full"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()
        spec = BenchmarkSpec.from_mapping(
            "qwen3vl_general_full",
            {
                "execution": "swift_eval",
                "profile": "full",
                "method": "swift eval",
                "source_project": "ms-swift",
                "dataset_name": "RealWorldQA",
                "split": "full",
                "evaluator": "VLMEvalKit",
                "primary_metrics": ["vlmevalkit/realworldqa/acc"],
                "metric_directions": {"vlmevalkit/realworldqa/acc": "maximize"},
                "eval_backend": "VLMEvalKit",
                "swift_datasets": ["RealWorldQA"],
                "metric_scales": {"RealWorldQA/acc": 0.01},
            },
        )

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            write_jsonl(
                checkpoint / "eval_result.jsonl",
                [{
                    "VLMEvalKit": {"RealWorldQA": {"acc": 50.0}},
                    "eval_output_dir": eval_output_dir.as_posix(),
                }],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        result = run_swift_benchmark(
            spec,
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=False,
            command_runner=fake_run,
        )

        self.assertEqual(result.status, "passed")
        self.assertIsNone(result.test_summary.sample_count)
        self.assertEqual((result.attempted, result.completed, result.failed), (None, None, None))

    def test_command_uses_upstream_swift_eval_flags(self) -> None:
        command = build_swift_eval_command(
            swift_spec(),
            checkpoint="C:/checkpoints/c3 adapter",
            eval_output_dir=Path("runs/c3/eval/general/swift eval"),
            infer_backend="vllm",
            eval_url="http://127.0.0.1:8000/v1",
            swift_executable="swift",
        )

        self.assertEqual(command[:2], ["swift", "eval"])
        self.assertEqual(command[command.index("--model") + 1], "C:/checkpoints/c3 adapter")
        self.assertEqual(command[command.index("--eval_backend") + 1], "VLMEvalKit")
        self.assertEqual(
            command[command.index("--eval_dataset") + 1 : command.index("--eval_limit")],
            ["RealWorldQA", "OCRBench"],
        )
        self.assertEqual(command[command.index("--eval_limit") + 1], "10")
        self.assertEqual(
            command[command.index("--eval_generation_config") + 1],
            '{"max_tokens":512,"temperature":0}',
        )
        self.assertEqual(command[command.index("--eval_url") + 1], "http://127.0.0.1:8000/v1")

        candidate_command = build_swift_eval_command(
            swift_spec(),
            checkpoint="base-model",
            eval_output_dir=Path("swift_eval"),
            infer_backend="vllm",
            eval_url="http://127.0.0.1:8000/v1",
            served_model_name="c1-candidate",
            swift_executable="swift",
        )
        self.assertEqual(
            candidate_command[candidate_command.index("--served_model_name") + 1],
            "c1-candidate",
        )

    def test_command_rejects_non_swift_spec(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid swift eval benchmark"):
            build_swift_eval_command(
                xlam_spec(Path("unused.jsonl")),
                checkpoint="checkpoint",
                eval_output_dir=Path("swift_eval"),
                infer_backend="vllm",
                eval_url=None,
                swift_executable="swift",
            )

        with self.assertRaisesRegex(ValueError, "non-empty checkpoint"):
            build_swift_eval_command(
                swift_spec(),
                checkpoint=" ",
                eval_output_dir=Path("swift_eval"),
                infer_backend="vllm",
                eval_url=None,
                swift_executable="swift",
            )

        def fail_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            raise AssertionError("invalid spec must not launch swift eval")

        with self.assertRaisesRegex(ValueError, "Invalid swift eval benchmark"):
            run_swift_benchmark(
                xlam_spec(Path("unused.jsonl")),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=self.make_temp_dir() / "invalid",
                infer_backend="vllm",
                eval_url=None,
                resume=False,
                command_runner=fail_run,
            )

    def test_success_preserves_raw_report_logs_and_normalized_metrics(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint adapter"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()
        launched_commands: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            launched_commands.append(command)
            self.assertEqual(Path(str(kwargs["cwd"])), benchmark_dir)
            self.assertEqual(kwargs["capture_output"], True)
            self.assertEqual(kwargs["text"], True)
            self.assertEqual(kwargs["check"], False)
            write_jsonl(
                checkpoint / "eval_result.jsonl",
                [
                    {
                        "VLMEvalKit": {
                            "RealWorldQA": {"acc": 67.5},
                            "OCRBench": {"score": 700},
                        },
                        "eval_output_dir": eval_output_dir.as_posix(),
                        "eval_limit": 10,
                    }
                ],
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="upstream stdout",
                stderr="upstream stderr",
            )

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
        self.assertEqual(result.metrics["vlmevalkit/ocrbench/score"], 700.0)
        self.assertEqual(
            load_json(benchmark_dir / "upstream_result.json")["VLMEvalKit"]["OCRBench"]["score"],
            700,
        )
        self.assertEqual((benchmark_dir / "stdout.log").read_text(encoding="utf-8"), "upstream stdout")
        self.assertEqual((benchmark_dir / "stderr.log").read_text(encoding="utf-8"), "upstream stderr")
        self.assertEqual(shlex.split(result.test_summary.command), launched_commands[0])
        self.assertEqual((result.attempted, result.completed, result.failed), (20, 20, 0))

    def test_directory_checkpoint_falls_back_to_benchmark_result_report(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()
        write_jsonl(
            checkpoint / "eval_result.jsonl",
            [
                {
                    "VLMEvalKit": {"RealWorldQA": {"acc": 99.0}},
                    "eval_output_dir": eval_output_dir.as_posix(),
                }
            ],
        )

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            write_jsonl(
                benchmark_dir / "result/nested/eval_result.jsonl",
                [
                    {
                        "VLMEvalKit": {"RealWorldQA": {"acc": 50.0}},
                        "eval_output_dir": eval_output_dir.as_posix(),
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        result = run_swift_benchmark(
            swift_spec(),
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="pt",
            eval_url=None,
            resume=False,
            command_runner=fake_run,
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.metrics["vlmevalkit/realworldqa/acc"], 0.5)

    def test_relative_local_checkpoint_is_resolved_before_subprocess_cwd_changes(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "relative-checkpoint"
        checkpoint.mkdir()
        relative_checkpoint = checkpoint.relative_to(Path.cwd()).as_posix()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            self.assertEqual(command[command.index("--model") + 1], checkpoint.resolve().as_posix())
            write_jsonl(
                checkpoint / "eval_result.jsonl",
                [
                    {
                        "VLMEvalKit": {
                            "RealWorldQA": {"acc": 50.0},
                            "OCRBench": {"score": 700},
                        },
                        "eval_output_dir": eval_output_dir.as_posix(),
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        result = run_swift_benchmark(
            swift_spec(),
            checkpoint=relative_checkpoint,
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="pt",
            eval_url=None,
            resume=False,
            command_runner=fake_run,
        )

        self.assertEqual(result.status, "passed")
        self.assertEqual(result.checkpoint, relative_checkpoint)

    def test_adapter_file_checkpoint_uses_only_benchmark_result_candidates(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "adapter.safetensors"
        checkpoint.write_text("adapter", encoding="utf-8")
        write_jsonl(
            root / "eval_result.jsonl",
            [
                {
                    "VLMEvalKit": {"RealWorldQA": {"acc": 99.0}},
                    "eval_output_dir": "wrong",
                }
            ],
        )
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            write_jsonl(
                benchmark_dir / "result/eval_result.jsonl",
                [
                    {
                        "VLMEvalKit": {"RealWorldQA": {"acc": 25.0}},
                        "eval_output_dir": eval_output_dir.as_posix(),
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        result = run_swift_benchmark(
            swift_spec(),
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="pt",
            eval_url=None,
            resume=False,
            command_runner=fake_run,
        )

        self.assertEqual(result.metrics["vlmevalkit/realworldqa/acc"], 0.25)

    def test_successful_process_with_missing_or_mismatched_report_is_failed(self) -> None:
        for report_kind in ("missing", "mismatched"):
            with self.subTest(report_kind=report_kind):
                root = self.make_temp_dir()
                checkpoint = root / "checkpoint"
                checkpoint.mkdir()
                benchmark_dir = root / "run/eval/qwen3vl_general_light"

                def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                    del kwargs
                    if report_kind == "mismatched":
                        write_jsonl(
                            checkpoint / "eval_result.jsonl",
                            [
                                {
                                    "VLMEvalKit": {"RealWorldQA": {"acc": 67.5}},
                                    "eval_output_dir": "another/run",
                                }
                            ],
                        )
                    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

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

                self.assertEqual(result.status, "failed")
                self.assertEqual(result.metrics, {})
                self.assertEqual(result.upstream_metrics, {})
                self.assertEqual(result.runtime["exit_code"], 1)
                self.assertEqual(result.runtime["process_exit_code"], 0)
                stderr = (benchmark_dir / "stderr.log").read_text(encoding="utf-8")
                self.assertIn("Could not find matching swift eval report", stderr)
                self.assertTrue((benchmark_dir / "result.json").is_file())

    def test_rerun_does_not_reuse_matching_report_from_previous_run(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()
        write_jsonl(
            checkpoint / "eval_result.jsonl",
            [
                {
                    "VLMEvalKit": {"RealWorldQA": {"acc": 99.0}},
                    "eval_output_dir": eval_output_dir.as_posix(),
                }
            ],
        )

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

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

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics, {})
        self.assertIn(
            "No new swift eval report rows",
            (benchmark_dir / "stderr.log").read_text(encoding="utf-8"),
        )

    def test_missing_primary_metric_fails_and_preserves_current_raw_report(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            write_jsonl(
                checkpoint / "eval_result.jsonl",
                [
                    {
                        "VLMEvalKit": {"UnrelatedDataset": {"score": 1.0}},
                        "eval_output_dir": eval_output_dir.as_posix(),
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

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

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics, {})
        self.assertEqual(result.runtime["exit_code"], 1)
        self.assertIn("upstream_result", result.artifacts)
        self.assertTrue((benchmark_dir / "upstream_result.json").is_file())
        self.assertIn(
            "missing primary metrics",
            (benchmark_dir / "stderr.log").read_text(encoding="utf-8"),
        )

    def test_nonzero_exit_writes_failed_result_without_synthetic_metrics(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="evaluator crashed")

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

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics, {})
        self.assertEqual(result.upstream_metrics, {})
        self.assertEqual(result.runtime["exit_code"], 2)
        self.assertEqual(result.runtime["process_exit_code"], 2)
        self.assertEqual((benchmark_dir / "stderr.log").read_text(encoding="utf-8"), "evaluator crashed")
        self.assertTrue((benchmark_dir / "result.json").is_file())

    def test_subprocess_launch_exception_writes_failed_result_and_logs(self) -> None:
        root = self.make_temp_dir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            raise FileNotFoundError("swift executable not found")

        result = run_swift_benchmark(
            swift_spec(),
            checkpoint=(root / "checkpoint").as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=False,
            command_runner=fake_run,
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.runtime["exit_code"], 1)
        self.assertEqual(result.runtime["process_exit_code"], None)
        self.assertEqual(result.metrics, {})
        self.assertEqual((benchmark_dir / "stdout.log").read_text(encoding="utf-8"), "")
        stderr = (benchmark_dir / "stderr.log").read_text(encoding="utf-8")
        self.assertIn("FileNotFoundError: swift executable not found", stderr)
        self.assertTrue((benchmark_dir / "result.json").is_file())

    def test_failed_rerun_removes_previous_upstream_result_artifact(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()

        def first_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            write_jsonl(
                checkpoint / "eval_result.jsonl",
                [
                    {
                        "VLMEvalKit": {
                            "RealWorldQA": {"acc": 50.0},
                            "OCRBench": {"score": 700},
                        },
                        "eval_output_dir": eval_output_dir.as_posix(),
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        first = run_swift_benchmark(
            swift_spec(),
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=False,
            command_runner=first_run,
        )
        self.assertEqual(first.status, "passed")
        self.assertTrue((benchmark_dir / "upstream_result.json").is_file())

        def failed_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="failed")

        failed = run_swift_benchmark(
            swift_spec(),
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=False,
            command_runner=failed_run,
        )

        self.assertEqual(failed.status, "failed")
        self.assertNotIn("upstream_result", failed.artifacts)
        self.assertNotIn("upstream_result", failed.test_summary.artifact_paths)
        self.assertFalse((benchmark_dir / "upstream_result.json").exists())
        self.assertTrue((checkpoint / "eval_result.jsonl").is_file())

    def test_interrupted_fresh_rerun_invalidates_previous_result_before_resume(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()

        def write_report(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            report_path = checkpoint / "eval_result.jsonl"
            rows = load_jsonl(report_path) if report_path.is_file() else []
            write_jsonl(
                report_path,
                [
                    *rows,
                    {
                        "VLMEvalKit": {
                            "RealWorldQA": {"acc": 50.0},
                            "OCRBench": {"score": 700},
                        },
                        "eval_output_dir": eval_output_dir.as_posix(),
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        first = run_swift_benchmark(
            swift_spec(),
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=False,
            command_runner=write_report,
        )
        self.assertEqual(first.status, "passed")

        def interrupt(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            run_swift_benchmark(
                swift_spec(),
                checkpoint=checkpoint.as_posix(),
                run_id="run",
                benchmark_dir=benchmark_dir,
                infer_backend="vllm",
                eval_url=None,
                resume=False,
                command_runner=interrupt,
            )

        self.assertFalse((benchmark_dir / "result.json").exists())
        reran = False

        def rerun(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal reran
            reran = True
            return write_report(command, **kwargs)

        resumed = run_swift_benchmark(
            swift_spec(),
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=True,
            command_runner=rerun,
        )

        self.assertTrue(reran)
        self.assertEqual(resumed.status, "passed")

    def test_resume_returns_existing_swift_result_without_subprocess(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"
        eval_output_dir = (benchmark_dir / "swift_eval").resolve()

        def first_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            write_jsonl(
                checkpoint / "eval_result.jsonl",
                [
                    {
                        "VLMEvalKit": {"RealWorldQA": {"acc": 67.5}},
                        "eval_output_dir": eval_output_dir.as_posix(),
                    }
                ],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        first = run_swift_benchmark(
            swift_spec(),
            checkpoint=checkpoint.as_posix(),
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=False,
            command_runner=first_run,
        )

        def fail_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del command, kwargs
            raise AssertionError("resume must not launch swift eval")

        resumed = run_swift_benchmark(
            swift_spec(),
            checkpoint="missing",
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="pt",
            eval_url="unused",
            resume=True,
            command_runner=fail_run,
        )

        self.assertEqual(resumed.to_dict(), first.to_dict())

    def test_resume_reruns_failed_swift_result(self) -> None:
        root = self.make_temp_dir()
        benchmark_dir = root / "run/eval/qwen3vl_general_light"

        failed = run_swift_benchmark(
            swift_spec(),
            checkpoint="checkpoint",
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=False,
            command_runner=lambda command, **kwargs: subprocess.CompletedProcess(
                command, 2, stdout="", stderr="failed"
            ),
        )
        self.assertEqual(failed.status, "failed")
        reran = False

        def rerun(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal reran
            del kwargs
            reran = True
            return subprocess.CompletedProcess(command, 2, stdout="", stderr="failed again")

        resumed = run_swift_benchmark(
            swift_spec(),
            checkpoint="checkpoint",
            run_id="run",
            benchmark_dir=benchmark_dir,
            infer_backend="vllm",
            eval_url=None,
            resume=True,
            command_runner=rerun,
        )
        self.assertTrue(reran)
        self.assertEqual(resumed.status, "failed")



class EvaluationSummaryTest(_TemporaryDirectoryTestCase):

    def test_mathvision_suite_requires_complete_fixed_judge_pair(self) -> None:
        common = {
            "checkpoint": "checkpoint",
            "run_id": "run",
            "runs_root": self.make_temp_dir(),
            "api_base": "http://127.0.0.1:8000/v1",
            "api_model": "c1-candidate",
            "infer_backend": "vllm",
            "eval_url": "http://127.0.0.1:8000/v1",
            "timeout": 10.0,
            "score_only": False,
            "resume": False,
            "display_command": "evaluate",
            "stage": "c1",
            "algorithm": None,
            "seed": 42,
            "baseline_metrics": {},
            "historical_metrics": [],
        }
        with self.assertRaisesRegex(ValueError, "fixed LLM judge"):
            evaluate_suite([mathvision_swift_spec()], **common)
        with self.assertRaisesRegex(ValueError, "both judge_api_base and judge_model"):
            evaluate_suite(
                [mathvision_swift_spec()],
                judge_api_base="http://127.0.0.1:8000/v1",
                **common,
            )

    def test_candidate_and_fixed_judge_routes_are_independent(self) -> None:
        root = self.make_temp_dir()
        captured: dict[str, object] = {}

        def fake_swift(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            captured.update(kwargs)
            return passed_result(
                spec.benchmark_id,
                "vlmevalkit/mathvision_mini/accuracy",
                0.5,
                Path(kwargs["benchmark_dir"]),
            )

        summary = evaluate_suite(
            [mathvision_swift_spec()],
            checkpoint="checkpoint",
            run_id="run",
            runs_root=root,
            api_base="http://127.0.0.1:8000/v1",
            api_model="c1-candidate",
            infer_backend="vllm",
            eval_url="http://127.0.0.1:8000/v1",
            judge_api_base="http://fixed-judge.example/v1?token=secret",
            judge_model="Qwen3-VL-4B-Instruct",
            timeout=10.0,
            score_only=False,
            resume=False,
            display_command="evaluate",
            stage="c1",
            algorithm=None,
            seed=42,
            baseline_metrics={},
            historical_metrics=[],
            swift_runner=fake_swift,
        )

        self.assertEqual(captured["api_model"], "c1-candidate")
        self.assertEqual(captured["judge_model"], "Qwen3-VL-4B-Instruct")
        manifest = load_json(root / "run/manifest.json")
        self.assertEqual(manifest["evaluator"]["judge_model"], "Qwen3-VL-4B-Instruct")
        self.assertEqual(
            manifest["evaluator"]["judge_api_base"],
            "http://fixed-judge.example/v1",
        )
        self.assertEqual(summary["status"], "passed")

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

    def test_comparisons_are_direction_aware_for_minimized_metrics(self) -> None:
        comparisons = calculate_comparisons(
            {"latency_seconds": 3.0},
            baseline_metrics={"latency_seconds": 5.0},
            historical_metrics=[{"latency_seconds": 2.0}, {"latency_seconds": 4.0}],
            metric_directions={"latency_seconds": "minimize"},
        )

        row = comparisons["latency_seconds"]
        self.assertEqual(row["gain_vs_base"], 2.0)
        self.assertEqual(row["retention"], 5.0 / 3.0)
        self.assertEqual(row["historical_best"], 2.0)
        self.assertEqual(row["forgetting"], 1.0)

    def test_comparisons_omit_retention_for_zero_baseline(self) -> None:
        comparisons = calculate_comparisons(
            {"accuracy": 0.2},
            baseline_metrics={"accuracy": 0.0},
            historical_metrics=[],
            metric_directions={"accuracy": "maximize"},
        )

        self.assertNotIn("retention", comparisons["accuracy"])

    def test_comparisons_reject_nonfinite_inputs(self) -> None:
        cases = [
            ({"metric": float("nan")}, {}, []),
            ({"metric": 1.0}, {"metric": float("inf")}, []),
            ({"metric": 1.0}, {}, [{"metric": float("-inf")}]),
        ]
        for current, baseline, history in cases:
            with self.subTest(current=current, baseline=baseline, history=history):
                with self.assertRaisesRegex(ValueError, "finite"):
                    calculate_comparisons(
                        current,
                        baseline_metrics=baseline,
                        historical_metrics=history,
                        metric_directions={"metric": "maximize"},
                    )

    def test_suite_is_sequential_and_continues_after_one_benchmark_raises(self) -> None:
        order: list[str] = []
        specs = [xlam_spec(Path("unused-xlam.jsonl")), mllm_router_spec(Path("unused-mllm.jsonl"))]

        def fake_native(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            order.append(spec.benchmark_id)
            if spec.benchmark_id == "xlam_function_calling":
                raise RuntimeError("first failed")
            return passed_result(
                spec.benchmark_id,
                "mllm_router/top1_accuracy",
                0.75,
                Path(kwargs["benchmark_dir"]),
            )

        root = self.make_temp_dir()
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
            stage="c3",
            algorithm=None,
            seed=42,
            baseline_metrics={},
            historical_metrics=[],
            native_runner=fake_native,
        )

        self.assertEqual(order, ["xlam_function_calling", "mllm_router"])
        self.assertEqual(summary["status"], "failed")
        failed = summary["benchmarks"]["xlam_function_calling"]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["attempted"], 0)
        self.assertEqual(failed["completed"], 0)
        self.assertEqual(failed["failed"], 0)
        result = load_json(failed["result_path"])
        self.assertEqual(result["runtime"]["exit_code"], 1)
        self.assertIn(
            "RuntimeError: first failed",
            Path(result["artifacts"]["stderr"]).read_text(encoding="utf-8"),
        )
        self.assertEqual(summary["benchmarks"]["mllm_router"]["status"], "passed")
        self.assertTrue((root / "run/manifest.json").is_file())
        self.assertTrue((root / "run/summary.json").is_file())

    def test_suite_rejects_resumed_result_from_another_checkpoint(self) -> None:
        root = self.make_temp_dir()

        def stale_runner(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            return passed_result(
                spec.benchmark_id,
                "xlam/function_name_accuracy",
                0.75,
                Path(kwargs["benchmark_dir"]),
            )

        summary = evaluate_suite(
            [xlam_spec(Path("unused.jsonl"))],
            checkpoint="checkpoint-B",
            run_id="run",
            runs_root=root,
            api_base=None,
            api_model=None,
            infer_backend="vllm",
            eval_url=None,
            timeout=10.0,
            score_only=False,
            resume=True,
            display_command="python scripts/evaluate.py --resume",
            stage=None,
            algorithm=None,
            seed=42,
            baseline_metrics={},
            historical_metrics=[],
            native_runner=stale_runner,
        )

        self.assertEqual(summary["checkpoint"], "checkpoint-B")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["metrics"], {})
        result = load_json(summary["benchmarks"]["xlam_function_calling"]["result_path"])
        self.assertEqual(result["checkpoint"], "checkpoint-B")
        self.assertIn(
            "checkpoint='checkpoint', expected 'checkpoint-B'",
            Path(result["artifacts"]["stderr"]).read_text(encoding="utf-8"),
        )

    def test_suite_rejects_empty_benchmark_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one benchmark"):
            evaluate_suite(
                [], checkpoint="checkpoint", run_id="run", runs_root=self.make_temp_dir(),
                api_base=None, api_model=None, infer_backend="vllm", eval_url=None,
                timeout=10.0, score_only=False, resume=False,
                display_command="python scripts/evaluate.py", stage=None, algorithm=None, seed=42,
                baseline_metrics={}, historical_metrics=[],
            )

    def test_suite_rejects_duplicate_metric_names_across_benchmarks(self) -> None:
        duplicate = BenchmarkSpec.from_mapping(
            "duplicate_metric",
            {
                "execution": "native_mllm_router", "profile": "light", "method": "native",
                "source_project": "source", "dataset_name": "dataset", "split": "dev",
                "evaluator": "scorer", "dataset_path": "unused.jsonl",
                "primary_metrics": ["xlam/function_name_accuracy"],
                "metric_directions": {"xlam/function_name_accuracy": "maximize"},
            },
        )
        with self.assertRaisesRegex(ValueError, "duplicate metric"):
            evaluate_suite(
                [xlam_spec(Path("unused.jsonl")), duplicate], checkpoint="checkpoint",
                run_id="run", runs_root=self.make_temp_dir(), api_base=None, api_model=None,
                infer_backend="vllm", eval_url=None, timeout=10.0, score_only=False,
                resume=False, display_command="python scripts/evaluate.py", stage=None,
                algorithm=None, seed=42, baseline_metrics={}, historical_metrics=[],
            )

    def test_suite_rejects_runtime_metric_collisions(self) -> None:
        specs = [xlam_spec(Path("unused-xlam.jsonl")), mllm_router_spec(Path("unused-mllm.jsonl"))]

        def fake_native(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            result = passed_result(
                spec.benchmark_id,
                spec.primary_metrics[0],
                0.75,
                Path(kwargs["benchmark_dir"]),
            )
            result.metrics["unexpected/shared_metric"] = 0.5
            result.upstream_metrics["unexpected/shared_metric"] = 0.5
            return result

        with self.assertRaisesRegex(ValueError, "duplicate metric"):
            evaluate_suite(
                specs, checkpoint="checkpoint", run_id="run", runs_root=self.make_temp_dir(),
                api_base=None, api_model=None, infer_backend="vllm", eval_url=None,
                timeout=10.0, score_only=False, resume=False,
                display_command="python scripts/evaluate.py", stage=None, algorithm=None, seed=42,
                baseline_metrics={}, historical_metrics=[], native_runner=fake_native,
            )

    def test_interrupted_suite_rerun_invalidates_previous_summary(self) -> None:
        root = self.make_temp_dir()
        benchmark_dir = root / "run/eval/xlam_function_calling"

        def passing_runner(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            return passed_result(
                spec.benchmark_id, "xlam/function_name_accuracy", 0.75,
                Path(kwargs["benchmark_dir"]),
            )

        common = {
            "checkpoint": "checkpoint", "run_id": "run", "runs_root": root,
            "api_base": None, "api_model": None, "infer_backend": "vllm", "eval_url": None,
            "timeout": 10.0, "score_only": False, "resume": False,
            "display_command": "python scripts/evaluate.py", "stage": None,
            "algorithm": None, "seed": 42, "baseline_metrics": {}, "historical_metrics": [],
        }
        evaluate_suite([xlam_spec(Path("unused.jsonl"))], native_runner=passing_runner, **common)
        self.assertTrue((root / "run/manifest.json").is_file())
        self.assertTrue((root / "run/summary.json").is_file())
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        (benchmark_dir / "result.json").write_text(
            json.dumps(
                passed_result(
                    "xlam_function_calling",
                    "xlam/function_name_accuracy",
                    0.75,
                    benchmark_dir,
                ).to_dict()
            ),
            encoding="utf-8",
        )

        def interrupted_runner(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            evaluate_suite(
                [xlam_spec(Path("unused.jsonl"))], native_runner=interrupted_runner, **common
            )

        self.assertFalse((root / "run/manifest.json").exists())
        self.assertFalse((root / "run/summary.json").exists())
        self.assertFalse((benchmark_dir / "result.json").exists())

        reran = False

        def resume_aware_runner(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            nonlocal reran
            result_path = Path(kwargs["benchmark_dir"]) / "result.json"
            if kwargs["resume"] and result_path.is_file():
                return EvaluationResult.from_dict(load_json(result_path))
            reran = True
            return passed_result(
                spec.benchmark_id,
                "xlam/function_name_accuracy",
                0.80,
                Path(kwargs["benchmark_dir"]),
            )

        resumed = evaluate_suite(
            [xlam_spec(Path("unused.jsonl"))],
            native_runner=resume_aware_runner,
            **{**common, "resume": True},
        )
        self.assertTrue(reran)
        self.assertEqual(resumed["status"], "passed")
        self.assertEqual(resumed["metrics"]["xlam/function_name_accuracy"], 0.80)

    def test_manifest_omits_api_credentials(self) -> None:
        def fake_native(spec: BenchmarkSpec, **kwargs: object) -> EvaluationResult:
            return passed_result(
                spec.benchmark_id, "xlam/function_name_accuracy", 0.75,
                Path(kwargs["benchmark_dir"]),
            )

        root = self.make_temp_dir()
        evaluate_suite(
            [xlam_spec(Path("unused.jsonl"))], checkpoint="checkpoint", run_id="run",
            runs_root=root, api_base="https://secret-token@example.test/v1?api_key=secret",
            api_model="checkpoint", infer_backend="vllm",
            eval_url="https://eval-secret@example.test", timeout=10.0, score_only=False,
            resume=False, display_command="python scripts/evaluate.py --api-key secret",
            stage="c1", algorithm=None, seed=42, baseline_metrics={}, historical_metrics=[],
            judge_api_base="https://judge-secret@example.test/v1?token=secret",
            judge_model="fixed-judge",
            native_runner=fake_native,
        )
        serialized = json.dumps(load_json(root / "run/manifest.json"))
        self.assertNotIn("secret-token", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("eval-secret", serialized)
        self.assertNotIn("judge-secret", serialized)

    def test_wandb_upload_flattens_numeric_metrics_and_warning_is_nonfatal(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.logged: dict[str, float] | None = None
                self.finished = False

            def log(self, values: dict[str, float]) -> None:
                self.logged = values

            def finish(self) -> None:
                self.finished = True

        class FakeWandb:
            def __init__(self) -> None:
                self.run = FakeRun()

            def init(self, **kwargs: object) -> FakeRun:
                return self.run

        fake = FakeWandb()
        warning = upload_summary_to_wandb(
            {"run_id": "run", "metrics": {"xlam/acc": 0.75}, "status": "passed"},
            project="toolvista-qwen3vl", group="c3-mllm-router-sft", wandb_module=fake,
        )
        self.assertIsNone(warning)
        self.assertEqual(fake.run.logged, {"eval/xlam/acc": 0.75})
        self.assertTrue(fake.run.finished)

        class BrokenWandb:
            def init(self, **kwargs: object) -> object:
                raise RuntimeError("offline")

        warning = upload_summary_to_wandb(
            {"run_id": "run", "metrics": {}}, project="toolvista-qwen3vl",
            group=None, wandb_module=BrokenWandb(),
        )
        self.assertIn("offline", warning)

    def test_wandb_finishes_run_when_log_fails(self) -> None:
        class FailingRun:
            def __init__(self) -> None:
                self.finished = False

            def log(self, values: dict[str, float]) -> None:
                raise RuntimeError("log failed")

            def finish(self) -> None:
                self.finished = True

        class FakeWandb:
            def __init__(self) -> None:
                self.run = FailingRun()

            def init(self, **kwargs: object) -> FailingRun:
                return self.run

        fake = FakeWandb()
        warning = upload_summary_to_wandb(
            {"run_id": "run", "metrics": {"xlam/acc": 0.75}},
            project="toolvista-qwen3vl", group=None, wandb_module=fake,
        )
        self.assertIn("log failed", warning)
        self.assertTrue(fake.run.finished)


class ChatCompletionClientTest(unittest.TestCase):
    def test_transport_failure_is_retried_once(self) -> None:
        response = _Response(b'{"choices":[{"message":{"content":"route_a"}}]}')
        with (
            patch(
                "toolvista.evaluation.runner.urllib.request.urlopen",
                side_effect=[urllib.error.URLError("connection refused"), response],
            ) as urlopen,
            patch("toolvista.evaluation.runner.time.sleep") as sleep,
        ):
            result = call_chat_completion(
                api_base="http://127.0.0.1:8000/v1",
                model="checkpoint",
                messages=[{"role": "user", "content": "Route"}],
                tools=[],
                timeout=10.0,
            )

        self.assertEqual(result["choices"][0]["message"]["content"], "route_a")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_malformed_json_response_is_not_retried(self) -> None:
        with (
            patch(
                "toolvista.evaluation.runner.urllib.request.urlopen",
                return_value=_Response(b"not-json"),
            ) as urlopen,
            patch("toolvista.evaluation.runner.time.sleep") as sleep,
        ):
            with self.assertRaises(json.JSONDecodeError):
                call_chat_completion(
                    api_base="http://127.0.0.1:8000/v1",
                    model="checkpoint",
                    messages=[{"role": "user", "content": "Route"}],
                    tools=[],
                    timeout=10.0,
                )

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()

    def test_http_error_response_is_not_retried(self) -> None:
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8000/v1/chat/completions",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"message":"maximum context length exceeded"}}'),
        )
        with (
            patch(
                "toolvista.evaluation.runner.urllib.request.urlopen",
                side_effect=error,
            ) as urlopen,
            patch("toolvista.evaluation.runner.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "HTTP 400: Bad Request:.*maximum context length exceeded",
            ):
                call_chat_completion(
                    api_base="http://127.0.0.1:8000/v1",
                    model="checkpoint",
                    messages=[{"role": "user", "content": "Route"}],
                    tools=[],
                    timeout=10.0,
                )

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
