from __future__ import annotations

import unittest
from pathlib import Path

from toolvista.evaluation.io import load_jsonl
from toolvista.evaluation.scorers import score_mllm_router, score_xlam


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
        self.assertAlmostEqual(metrics["top_k_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["candidate_hit_rate"], 0.75)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["invalid_route_rate"], 0.5)
        self.assertEqual(metrics["accuracy_by_modality"], {"image": 0.5, "text": 0.5})
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                '["route_a","route_alt"]': {'"route_alt"': 1},
                '["route_a"]': {"null": 1},
                '["route_b"]': {'"route_b"': 1, '"route_x"': 1},
            },
        )
        self.assertEqual(result["metrics"]["mllm_router/top_k_accuracy"], 0.75)
        self.assertEqual(result["metrics"]["mllm_router/accuracy_by_modality/text"], 0.5)

    def test_shared_catalog_defines_valid_routes_beyond_observed_targets(self) -> None:
        result = score_mllm_router(
            [
                {
                    "sample_id": "catalog-route",
                    "input_modality": "text",
                    "router_catalog": [" route_a ", "route_unused"],
                    "target_tools": ["route_a"],
                    "prediction": "route_unused",
                }
            ]
        )

        self.assertEqual(result["upstream_metrics"]["invalid_route_rate"], 0.0)
        self.assertEqual(
            result["upstream_metrics"]["confusion_matrix"],
            {'["route_a"]': {'"route_unused"': 1}},
        )

    def test_invalid_candidate_lists_are_not_filtered_or_repaired(self) -> None:
        malformed_predictions = [
            ["route_a", 1],
            ["route_a", ""],
            ["route_a", "   "],
            ["route_a", "route_a"],
            [" route_a ", "route_a"],
        ]

        for prediction in malformed_predictions:
            with self.subTest(prediction=prediction):
                result = score_mllm_router(
                    [
                        {
                            "sample_id": "bad-prediction",
                            "input_modality": "text",
                            "target_tools": ["route_a"],
                            "prediction": prediction,
                        }
                    ]
                )
                metrics = result["upstream_metrics"]
                self.assertEqual(metrics["format_valid_rate"], 0.0)
                self.assertEqual(metrics["top1_accuracy"], 0.0)
                self.assertEqual(metrics["top_k_accuracy"], 0.0)
                self.assertEqual(metrics["candidate_hit_rate"], 0.0)
                self.assertEqual(metrics["invalid_route_rate"], 1.0)
                self.assertEqual(
                    metrics["confusion_matrix"],
                    {'["route_a"]': {"null": 1}},
                )

    def test_confusion_matrix_preserves_full_multilabel_target_when_missed(self) -> None:
        result = score_mllm_router(
            [
                {
                    "sample_id": "missed-multilabel",
                    "input_modality": "text",
                    "router_catalog": ["route_a", "route_b", "route_x"],
                    "target_tools": ["route_b", "route_a"],
                    "prediction": "route_x",
                }
            ]
        )

        self.assertEqual(
            result["upstream_metrics"]["confusion_matrix"],
            {'["route_a","route_b"]': {'"route_x"': 1}},
        )

    def test_confusion_matrix_json_keys_avoid_target_and_invalid_sentinel_collisions(self) -> None:
        result = score_mllm_router(
            [
                {
                    "sample_id": "single-pipe-label",
                    "input_modality": "text",
                    "target_tools": ["a|b"],
                    "prediction": "<invalid>",
                },
                {
                    "sample_id": "two-labels",
                    "input_modality": "text",
                    "target_tools": ["a", "b"],
                    "prediction": "<invalid>",
                },
                {
                    "sample_id": "real-invalid-route-name",
                    "input_modality": "text",
                    "target_tools": ["<invalid>"],
                    "prediction": "<invalid>",
                },
                {
                    "sample_id": "invalid-format",
                    "input_modality": "text",
                    "target_tools": ["<invalid>"],
                    "prediction": "",
                },
            ]
        )

        self.assertEqual(
            result["upstream_metrics"]["confusion_matrix"],
            {
                '["<invalid>"]': {'"<invalid>"': 1, "null": 1},
                '["a","b"]': {'"<invalid>"': 1},
                '["a|b"]': {'"<invalid>"': 1},
            },
        )

    def test_catalog_is_shared_and_contains_every_target(self) -> None:
        malformed_catalogs = ["route_a", [], ["route_a", 1], ["   "]]
        for router_catalog in malformed_catalogs:
            with self.subTest(router_catalog=router_catalog):
                with self.assertRaisesRegex(
                    ValueError,
                    "bad-catalog.*non-empty list of non-blank strings",
                ):
                    score_mllm_router(
                        [
                            {
                                "sample_id": "bad-catalog",
                                "input_modality": "text",
                                "router_catalog": router_catalog,
                                "target_tools": ["route_a"],
                                "prediction": "route_a",
                            }
                        ]
                    )

        with self.assertRaisesRegex(ValueError, "catalog-b.*same router_catalog"):
            score_mllm_router(
                [
                    {
                        "sample_id": "catalog-a",
                        "input_modality": "text",
                        "router_catalog": ["route_a", "route_b"],
                        "target_tools": ["route_a"],
                        "prediction": "route_a",
                    },
                    {
                        "sample_id": "catalog-b",
                        "input_modality": "text",
                        "router_catalog": ["route_a", "route_c"],
                        "target_tools": ["route_a"],
                        "prediction": "route_a",
                    },
                ]
            )

        normalized = score_mllm_router(
            [
                {
                    "sample_id": "normalized-a",
                    "input_modality": "text",
                    "router_catalog": ["route_a", "route_b"],
                    "target_tools": ["route_a"],
                    "prediction": "route_a",
                },
                {
                    "sample_id": "normalized-b",
                    "input_modality": "text",
                    "router_catalog": [" route_b ", "route_a"],
                    "target_tools": ["route_b"],
                    "prediction": "route_b",
                },
            ]
        )
        self.assertEqual(normalized["upstream_metrics"]["invalid_route_rate"], 0.0)

        with self.assertRaisesRegex(ValueError, "missing-catalog.*same router_catalog"):
            score_mllm_router(
                [
                    {
                        "sample_id": "has-catalog",
                        "input_modality": "text",
                        "router_catalog": ["route_a"],
                        "target_tools": ["route_a"],
                        "prediction": "route_a",
                    },
                    {
                        "sample_id": "missing-catalog",
                        "input_modality": "text",
                        "target_tools": ["route_a"],
                        "prediction": "route_a",
                    },
                ]
            )

        with self.assertRaisesRegex(ValueError, "target-outside.*not present in router_catalog"):
            score_mllm_router(
                [
                    {
                        "sample_id": "target-outside",
                        "input_modality": "text",
                        "router_catalog": ["route_a"],
                        "target_tools": ["route_b"],
                        "prediction": "route_a",
                    }
                ]
            )

    def test_malformed_target_tools_are_rejected_instead_of_corrupting_metrics(self) -> None:
        malformed_values = ["route_a", [], ["route_a", 1], ["   "]]

        for target_tools in malformed_values:
            with self.subTest(target_tools=target_tools):
                with self.assertRaisesRegex(
                    ValueError,
                    "bad-target.*non-empty list of non-blank strings",
                ):
                    score_mllm_router(
                        [
                            {
                                "sample_id": "bad-target",
                                "input_modality": "text",
                                "target_tools": target_tools,
                                "prediction": "route_a",
                            }
                        ]
                    )

    def test_modality_is_required_and_normalized_before_metric_keys_are_created(self) -> None:
        result = score_mllm_router(
            [
                {
                    "sample_id": "spaced-modality",
                    "input_modality": " Image-2 ",
                    "target_tools": ["route_a"],
                    "prediction": "route_a",
                }
            ]
        )

        self.assertEqual(result["upstream_metrics"]["accuracy_by_modality"], {"image-2": 1.0})
        self.assertIn("mllm_router/accuracy_by_modality/image-2", result["metrics"])
        for modality in (None, "", "   ", ["text"]):
            with self.subTest(modality=modality):
                with self.assertRaisesRegex(
                    ValueError,
                    "bad-modality.*non-blank input_modality string",
                ):
                    score_mllm_router(
                        [
                            {
                                "sample_id": "bad-modality",
                                "input_modality": modality,
                                "target_tools": ["route_a"],
                                "prediction": "route_a",
                            }
                        ]
                    )

        for modality in ("text/image", "multi modal", "text\n", "text\tmode", "_text"):
            with self.subTest(modality=modality):
                with self.assertRaisesRegex(ValueError, "compact identifier"):
                    score_mllm_router(
                        [
                            {
                                "sample_id": "bad-modality-grammar",
                                "input_modality": modality,
                                "target_tools": ["route_a"],
                                "prediction": "route_a",
                            }
                        ]
                    )


if __name__ == "__main__":
    unittest.main()
