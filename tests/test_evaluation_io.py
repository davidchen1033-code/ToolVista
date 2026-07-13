from __future__ import annotations

import json
import unittest

from toolvista.evaluation.models import BenchmarkSpec, EvaluationResult, TestSummary


class EvaluationModelTest(unittest.TestCase):
    @staticmethod
    def _result_payload() -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "run_id": "test",
            "checkpoint": "checkpoint",
            "benchmark": "xlam_function_calling",
            "profile": "light",
            "prediction_mode": "score_only",
            "test_summary": {
                "method": "ToolVista native xLAM scoring",
                "dataset": "xLAM fixed-seed dev / fixed_seed_dev",
                "source_project": "xLAM ActionStudio",
                "evaluator": "toolvista.evaluation.scorers.score_xlam",
                "sample_count": 4,
                "command": "python scripts/evaluate.py --benchmark xlam_function_calling",
                "main_metrics": ["xlam/function_name_accuracy"],
                "result_path": "runs/test/eval/xlam_function_calling/result.json",
                "artifact_paths": {"predictions": "predictions.jsonl"},
            },
            "status": "passed",
            "attempted": 4,
            "completed": 4,
            "failed": 0,
            "upstream_metrics": {"function_name_accuracy": 0.75},
            "metrics": {"xlam/function_name_accuracy": 0.75},
            "runtime": {"duration_seconds": 0.1, "peak_gpu_memory_mb": None, "exit_code": 0},
            "artifacts": {"predictions": "predictions.jsonl"},
        }

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

    def test_benchmark_spec_rejects_unhashable_metric_direction_as_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid metric_directions"):
            BenchmarkSpec.from_mapping(
                "broken_direction",
                {
                    "execution": "native_xlam",
                    "profile": "light",
                    "method": "native",
                    "source_project": "xLAM",
                    "dataset_name": "dev",
                    "split": "dev",
                    "evaluator": "score_xlam",
                    "primary_metrics": ["xlam/function_name_accuracy"],
                    "metric_directions": {"xlam/function_name_accuracy": ["maximize"]},
                    "dataset_path": "data/eval/xlam/dev.jsonl",
                },
            )

    def test_swift_eval_requires_non_empty_eval_backend(self) -> None:
        for eval_backend in (None, "   "):
            with self.subTest(eval_backend=eval_backend), self.assertRaisesRegex(ValueError, "eval_backend"):
                BenchmarkSpec.from_mapping(
                    "broken_swift_backend",
                    {
                        "execution": "swift_eval",
                        "profile": "light",
                        "method": "swift eval",
                        "source_project": "ms-swift/VLMEvalKit",
                        "dataset_name": "general VL",
                        "split": "upstream default",
                        "evaluator": "swift eval",
                        "primary_metrics": ["vlmevalkit/realworldqa/acc"],
                        "metric_directions": {"vlmevalkit/realworldqa/acc": "maximize"},
                        "swift_datasets": ["RealWorldQA"],
                        "eval_backend": eval_backend,
                    },
                )

    def test_benchmark_spec_normalizes_metric_and_path_strings(self) -> None:
        native = BenchmarkSpec.from_mapping(
            "normalized_native",
            {
                "execution": "native_xlam",
                "profile": "light",
                "method": "native",
                "source_project": "xLAM",
                "dataset_name": "dev",
                "split": "dev",
                "evaluator": "score_xlam",
                "primary_metrics": ["  xlam/function_name_accuracy  "],
                "metric_directions": {" xlam/function_name_accuracy ": " maximize "},
                "metric_scales": {" xlam/function_name_accuracy ": 100},
                "dataset_path": "  data/eval/xlam/dev.jsonl  ",
            },
        )
        swift = BenchmarkSpec.from_mapping(
            "normalized_swift",
            {
                "execution": "swift_eval",
                "profile": "light",
                "method": "swift eval",
                "source_project": "ms-swift/VLMEvalKit",
                "dataset_name": "general VL",
                "split": "upstream default",
                "evaluator": "swift eval",
                "primary_metrics": [" vlmevalkit/realworldqa/acc "],
                "metric_directions": {" vlmevalkit/realworldqa/acc ": "maximize"},
                "swift_datasets": [" RealWorldQA ", " OCRBench "],
                "eval_backend": " VLMEvalKit ",
            },
        )

        self.assertEqual(native.primary_metrics, ("xlam/function_name_accuracy",))
        self.assertEqual(native.dataset_path, "data/eval/xlam/dev.jsonl")
        self.assertEqual(native.metric_directions, {"xlam/function_name_accuracy": "maximize"})
        self.assertEqual(native.metric_scales, {"xlam/function_name_accuracy": 100.0})
        self.assertEqual(swift.swift_datasets, ("RealWorldQA", "OCRBench"))
        self.assertEqual(swift.eval_backend, "VLMEvalKit")

    def test_benchmark_spec_rejects_whitespace_only_contract_strings(self) -> None:
        base = {
            "execution": "native_xlam",
            "profile": "light",
            "method": "native",
            "source_project": "xLAM",
            "dataset_name": "dev",
            "split": "dev",
            "evaluator": "score_xlam",
            "primary_metrics": ["xlam/function_name_accuracy"],
            "metric_directions": {"xlam/function_name_accuracy": "maximize"},
            "dataset_path": "data/eval/xlam/dev.jsonl",
        }
        invalid_updates = (
            ({"primary_metrics": ["   "]}, "primary_metrics"),
            ({"dataset_path": "   "}, "dataset_path"),
            ({"metric_directions": {"   ": "maximize"}}, "metric_directions"),
        )
        for update, expected in invalid_updates:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, expected):
                BenchmarkSpec.from_mapping("blank_string", {**base, **update})

        with self.assertRaisesRegex(ValueError, "swift_datasets"):
            BenchmarkSpec.from_mapping(
                "blank_swift_dataset",
                {
                    **base,
                    "execution": "swift_eval",
                    "swift_datasets": ["   "],
                    "eval_backend": "VLMEvalKit",
                },
            )

    def test_benchmark_spec_rejects_invalid_metric_scales(self) -> None:
        for key, value in (("metric", True), ("metric", float("nan")), ("metric", float("inf")), ("   ", 1.0)):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "scale_benchmark.*metric_scales"):
                BenchmarkSpec.from_mapping(
                    "scale_benchmark",
                    {
                        "execution": "native_xlam",
                        "profile": "light",
                        "method": "native",
                        "source_project": "xLAM",
                        "dataset_name": "dev",
                        "split": "dev",
                        "evaluator": "score_xlam",
                        "primary_metrics": ["metric"],
                        "metric_directions": {"metric": "maximize"},
                        "metric_scales": {key: value},
                        "dataset_path": "data/eval/xlam/dev.jsonl",
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

        payload = json.loads(json.dumps(result.to_dict()))

        self.assertEqual(payload["test_summary"]["main_metrics"], ["xlam/function_name_accuracy"])
        self.assertEqual(payload["test_summary"]["result_path"], summary.result_path)
        self.assertEqual(payload["test_summary"]["artifact_paths"], summary.artifact_paths)
        self.assertEqual(EvaluationResult.from_dict(payload).to_dict(), payload)

    def test_result_from_dict_rejects_malformed_summary_metrics_and_counts(self) -> None:
        invalid_values = (
            ("test_summary", "not-a-mapping", "test_summary"),
            ("main_metrics", [], "main_metrics"),
            ("main_metrics", ["   "], "main_metrics"),
            ("attempted", -1, "attempted"),
            ("completed", True, "completed"),
            ("failed", 1.5, "failed"),
        )
        for field, value, expected in invalid_values:
            payload = self._result_payload()
            if field == "main_metrics":
                payload["test_summary"][field] = value  # type: ignore[index]
            else:
                payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, expected):
                EvaluationResult.from_dict(payload)

    def test_result_from_dict_validates_enums_and_copies_nested_mappings(self) -> None:
        for field, value in (("prediction_mode", "resume"), ("status", "unknown"), ("metrics", [])):
            payload = self._result_payload()
            payload[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                EvaluationResult.from_dict(payload)

        payload = self._result_payload()
        result = EvaluationResult.from_dict(payload)
        payload["metrics"]["xlam/function_name_accuracy"] = 0.0  # type: ignore[index]
        payload["runtime"]["duration_seconds"] = 9.0  # type: ignore[index]
        result.runtime["details"] = {"changed": False}

        self.assertEqual(result.metrics["xlam/function_name_accuracy"], 0.75)
        self.assertEqual(result.runtime["duration_seconds"], 0.1)
        self.assertNotIn("details", payload["runtime"])  # type: ignore[operator]

    def test_result_from_dict_normalizes_and_validates_metrics(self) -> None:
        payload = self._result_payload()
        payload["metrics"] = {" xlam/function_name_accuracy ": 1}

        result = EvaluationResult.from_dict(payload)

        self.assertEqual(result.metrics, {"xlam/function_name_accuracy": 1.0})
        self.assertIsInstance(result.metrics["xlam/function_name_accuracy"], float)

        for key, value in (
            ("   ", 1.0),
            ("metric", True),
            ("metric", float("nan")),
            ("metric", float("inf")),
            ("metric", "1.0"),
        ):
            invalid = self._result_payload()
            invalid["metrics"] = {key: value}
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "metrics"):
                EvaluationResult.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
