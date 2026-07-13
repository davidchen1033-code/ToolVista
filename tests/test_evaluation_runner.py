from __future__ import annotations

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
from toolvista.evaluation.models import BenchmarkSpec
from toolvista.evaluation.runner import (
    build_swift_eval_command,
    call_chat_completion,
    run_native_benchmark,
    run_swift_benchmark,
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
            None,
        )
        with (
            patch(
                "toolvista.evaluation.runner.urllib.request.urlopen",
                side_effect=error,
            ) as urlopen,
            patch("toolvista.evaluation.runner.time.sleep") as sleep,
        ):
            with self.assertRaises(urllib.error.HTTPError):
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
