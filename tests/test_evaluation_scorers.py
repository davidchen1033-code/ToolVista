from __future__ import annotations

import unittest
from pathlib import Path

from toolvista.evaluation.io import load_jsonl
from toolvista.evaluation.scorers import score_xlam


REPO_ROOT = Path(__file__).resolve().parents[1]


class XLAMScorerTest(unittest.TestCase):
    @staticmethod
    def _score(target_tool_calls: list[dict[str, object]], prediction: object) -> dict[str, object]:
        return score_xlam(
            [
                {
                    "sample_id": "focused",
                    "target_tool_calls": target_tool_calls,
                    "prediction": prediction,
                }
            ]
        )["upstream_metrics"]

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

    def test_malformed_target_is_rejected_instead_of_counted_as_model_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "broken.*malformed target_tool_calls"):
            score_xlam(
                [
                    {
                        "sample_id": "broken",
                        "target_tool_calls": [{"name": "f", "arguments": "not-json"}],
                        "prediction": {"name": "f", "arguments": {}},
                    }
                ]
            )

    def test_argument_metrics_pair_calls_by_function_name_and_count_unmatched_arguments(self) -> None:
        wrong_function = self._score(
            [{"name": "expected", "arguments": {"city": "Paris"}}],
            {"name": "other", "arguments": {"city": "Paris"}},
        )
        wrong_function_without_arguments = self._score(
            [{"name": "expected", "arguments": {}}],
            {"name": "other", "arguments": {}},
        )
        extra_call = self._score(
            [{"name": "f", "arguments": {"a": 1}}],
            [
                {"name": "f", "arguments": {"a": 1}},
                {"name": "f", "arguments": {"b": 2}},
            ],
        )

        self.assertEqual(wrong_function["argument_key_f1"], 0.0)
        self.assertEqual(wrong_function["argument_value_accuracy"], 0.0)
        self.assertEqual(wrong_function_without_arguments["argument_key_f1"], 0.0)
        self.assertEqual(wrong_function_without_arguments["argument_value_accuracy"], 0.0)
        self.assertAlmostEqual(extra_call["argument_key_f1"], 2.0 / 3.0)
        self.assertEqual(extra_call["argument_value_accuracy"], 0.5)

    def test_duplicate_function_names_use_best_deterministic_argument_pairing(self) -> None:
        metrics = self._score(
            [
                {"name": "f", "arguments": {"value": 1}},
                {"name": "f", "arguments": {"value": 2}},
            ],
            [
                {"name": "f", "arguments": {"value": 2}},
                {"name": "f", "arguments": {"value": 3}},
            ],
        )

        self.assertEqual(metrics["argument_key_f1"], 1.0)
        self.assertEqual(metrics["argument_value_accuracy"], 0.5)

    def test_many_duplicate_function_calls_are_scored_deterministically(self) -> None:
        targets = [{"name": "f", "arguments": {"value": index}} for index in range(24)]
        predictions = list(reversed(targets)) + [
            {"name": "f", "arguments": {"extra": index}} for index in range(16)
        ]

        first = self._score(targets, predictions)
        second = self._score(targets, predictions)

        self.assertEqual(first, second)
        self.assertEqual(first["function_name_accuracy"], 0.6)
        self.assertEqual(first["argument_key_f1"], 0.75)
        self.assertEqual(first["argument_value_accuracy"], 0.6)
        self.assertEqual(first["argument_exact_match"], 0.0)

    def test_argument_values_distinguish_missing_null_and_json_types_recursively(self) -> None:
        missing = self._score(
            [{"name": "f", "arguments": {"value": None}}],
            {"name": "f", "arguments": {}},
        )
        type_mismatch = self._score(
            [{"name": "f", "arguments": {"config": {"enabled": True, "values": [True]}}}],
            {"name": "f", "arguments": {"config": {"enabled": 1, "values": [1]}}},
        )
        numeric_equivalent = self._score(
            [{"name": "f", "arguments": {"value": 1}}],
            {"name": "f", "arguments": {"value": 1.0}},
        )

        self.assertEqual(missing["argument_value_accuracy"], 0.0)
        self.assertEqual(type_mismatch["argument_value_accuracy"], 0.0)
        self.assertEqual(type_mismatch["argument_exact_match"], 0.0)
        self.assertEqual(numeric_equivalent["argument_value_accuracy"], 1.0)
        self.assertEqual(numeric_equivalent["argument_exact_match"], 1.0)

    def test_function_names_are_stripped_and_blank_names_are_rejected(self) -> None:
        stripped = self._score(
            [{"name": " f ", "arguments": {}}],
            {"name": "f", "arguments": {}},
        )
        blank_prediction = self._score(
            [{"name": "f", "arguments": {}}],
            {"name": "   ", "arguments": {}},
        )

        self.assertEqual(stripped["argument_exact_match"], 1.0)
        self.assertEqual(blank_prediction["json_valid_rate"], 1.0)
        self.assertEqual(blank_prediction["schema_valid_rate"], 0.0)
        with self.assertRaisesRegex(ValueError, "blank-target.*malformed target_tool_calls"):
            score_xlam(
                [
                    {
                        "sample_id": "blank-target",
                        "target_tool_calls": [{"name": "   ", "arguments": {}}],
                        "prediction": {"name": "f", "arguments": {}},
                    }
                ]
            )

    def test_empty_call_list_is_json_valid_but_schema_invalid(self) -> None:
        metrics = self._score(
            [{"name": "f", "arguments": {}}],
            "[]",
        )

        self.assertEqual(metrics["json_valid_rate"], 1.0)
        self.assertEqual(metrics["schema_valid_rate"], 0.0)
        self.assertEqual(metrics["argument_key_f1"], 0.0)

    def test_non_finite_numbers_are_rejected_in_strings_and_parsed_structures(self) -> None:
        predictions = [
            f'{{"name":"f","arguments":{{"value":{constant}}}}}'
            for constant in ("NaN", "Infinity", "-Infinity")
        ]
        predictions.extend(
            [
                {"name": "f", "arguments": {"value": float("nan")}},
                {"name": "f", "arguments": {"value": [float("inf")]}},
            ]
        )

        for prediction in predictions:
            with self.subTest(prediction=prediction):
                metrics = self._score(
                    [{"name": "f", "arguments": {"value": 1}}],
                    prediction,
                )
                self.assertEqual(metrics["json_valid_rate"], 0.0)
                self.assertEqual(metrics["schema_valid_rate"], 0.0)

        nested_string = self._score(
            [{"name": "f", "arguments": {"value": 1}}],
            {"name": "f", "arguments": '{"value":NaN}'},
        )
        self.assertEqual(nested_string["schema_valid_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
