from __future__ import annotations

import json
import shutil
import unittest
import urllib.error
import uuid
from pathlib import Path
from unittest.mock import patch

from toolvista.evaluation.io import load_json, load_jsonl, write_jsonl
from toolvista.evaluation.models import BenchmarkSpec
from toolvista.evaluation.runner import call_chat_completion, run_native_benchmark


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


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class NativeEvaluationRunnerTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-evaluation-runner-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

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
