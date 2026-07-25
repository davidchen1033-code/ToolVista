from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analyze_visual_rft_withnegative_predictions.py"
REWARD_PATH = REPO_ROOT / "toolvista" / "rewards" / "visual_rft_withnegative.py"


class StubORM:
    pass


def load_module():
    rewards_module = types.ModuleType("swift.rewards")
    rewards_module.ORM = StubORM
    rewards_module.orms = {}
    swift_module = types.ModuleType("swift")
    swift_module.rewards = rewards_module
    module_name = "toolvista_visual_rft_withnegative_analysis_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"swift": swift_module, "swift.rewards": rewards_module, module_name: module},
    ):
        spec.loader.exec_module(module)
    return module


def reward_contract(module):
    module_name = "toolvista_visual_rft_withnegative_analysis_reward_test"
    reward_spec = importlib.util.spec_from_file_location(module_name, REWARD_PATH)
    assert reward_spec is not None and reward_spec.loader is not None
    reward_module = importlib.util.module_from_spec(reward_spec)
    rewards_module = types.ModuleType("swift.rewards")
    rewards_module.ORM = StubORM
    rewards_module.orms = {}
    swift_module = types.ModuleType("swift")
    swift_module.rewards = rewards_module
    with patch.dict(
        sys.modules,
        {
            "swift": swift_module,
            "swift.rewards": rewards_module,
            module_name: reward_module,
        },
    ):
        reward_spec.loader.exec_module(reward_module)
    return module.RewardContract(
        parse=reward_module._parse_detection,
        score=reward_module._score_detection,
    )


def answer(payload) -> str:
    value = payload if isinstance(payload, str) else repr(payload)
    return f"<think>audit</think><answer>{value}</answer>"


def box(position=(0, 0, 100, 100), confidence=1.0):
    return {"Position": list(position), "Confidence": confidence}


class VisualRFTWithNegativeAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.contract = reward_contract(cls.module)

    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-withnegative-analysis-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    @staticmethod
    def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_inline_gold_reports_requested_metrics_and_exact_reward(self) -> None:
        rows = [
            {
                "example_type": "negative",
                "solution": answer("No Objects"),
                "prediction": answer("No Objects"),
            },
            {
                "example_type": "negative",
                "solution": answer("No Objects"),
                "prediction": answer([]),
            },
            {
                "example_type": "negative",
                "solution": answer("No Objects"),
                "prediction": answer([box(confidence=0.5)]),
            },
            {
                "example_type": "positive",
                "solution": answer([box()]),
                "prediction": answer([box()]),
            },
            {
                "example_type": "positive",
                "solution": answer([box()]),
                "prediction": answer("No Objects"),
            },
            {
                "example_type": "positive",
                "solution": answer([box()]),
                "prediction": answer("not-json"),
            },
        ]

        result = self.module.analyze_predictions(rows, contract=self.contract)

        self.assertEqual(result["counts"]["positive_rows"], 3)
        self.assertEqual(result["counts"]["negative_rows"], 3)
        self.assertEqual(result["parse_kind_counts"]["malformed"], 1)
        self.assertAlmostEqual(
            result["metrics"]["negative_explicit_no_objects_accuracy"], 1 / 3
        )
        self.assertAlmostEqual(
            result["metrics"]["negative_abstention_accuracy"], 2 / 3
        )
        self.assertAlmostEqual(
            result["metrics"]["negative_false_positive_rate"], 1 / 3
        )
        self.assertAlmostEqual(
            result["metrics"]["negative_any_box_false_positive_rate"], 1 / 3
        )
        self.assertAlmostEqual(result["metrics"]["positive_any_box_rate"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["positive_reward_ap_proxy_mean"], -0.25)
        self.assertAlmostEqual(result["metrics"]["malformed_rate"], 1 / 6)
        self.assertEqual(result["counts"]["predicted_boxes"], 2)
        self.assertAlmostEqual(result["metrics"]["predicted_confidence_mean"], 0.75)
        self.assertTrue(result["coverage"]["positive_reward_complete"])

    def test_summary_auto_discovers_predictions_and_sibling_coco_gold(self) -> None:
        root = self.make_temp_dir()
        predictions = root / "run/eval/visual_rft_coco_light/predictions.jsonl"
        self.write_jsonl(
            predictions,
            [
                {
                    "image_id": 7,
                    "category_id": 3,
                    "image_width": 200,
                    "image_height": 100,
                    "has_ground_truth": True,
                    "prediction": answer([box((100, 200, 900, 800), 1.0)]),
                },
                {
                    "image_id": 7,
                    "category_id": 4,
                    "image_width": 200,
                    "image_height": 100,
                    "has_ground_truth": False,
                    "prediction": "No Objects",
                },
            ],
        )
        annotations = {
            "images": [{"id": 7, "width": 200, "height": 100}],
            "annotations": [
                {"image_id": 7, "category_id": 3, "bbox": [20, 20, 160, 60], "iscrowd": 0}
            ],
        }
        predictions.with_name("coco_annotations.json").write_text(
            json.dumps(annotations), encoding="utf-8"
        )
        summary = root / "run/summary.json"
        summary.write_text(
            json.dumps(
                {
                    "benchmarks": {
                        "visual_rft_coco_light": {
                            "artifacts": {"predictions": str(predictions)}
                        },
                        "xlam_function_calling": {
                            "artifacts": {"predictions": str(root / "unused.jsonl")}
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        args = self.module.parse_args([str(summary)])

        report, protected = self.module.build_report(args)

        self.assertEqual(report["source"]["benchmark_id"], "visual_rft_coco_light")
        self.assertEqual(
            report["reward_contract"]["registry_name"], "visual_rft_withnegative"
        )
        self.assertEqual(len(report["reward_contract"]["plugin_sha256"]), 64)
        self.assertEqual(report["gold_resolution"]["coco_positive_gold_rows"], 1)
        self.assertEqual(report["metrics"]["positive_reward_ap_proxy_mean"], 1.0)
        self.assertEqual(report["metrics"]["negative_explicit_no_objects_accuracy"], 1.0)
        self.assertEqual(report["counts"]["coco_target_boxes"], 1)
        self.assertIn(predictions.resolve(), protected)
        self.assertIn(summary.parent.resolve(), protected)
        self.assertIn(self.module.REWARD_PLUGIN_PATH.resolve(), protected)

    def test_custom_nested_fields_and_labels_are_supported(self) -> None:
        rows = [
            {
                "model": {"text": answer("No Objects")},
                "target": answer("No Objects"),
                "kind": "empty",
            },
            {
                "model": {"text": answer([box()])},
                "target": answer([box()]),
                "kind": "object",
            },
        ]

        result = self.module.analyze_predictions(
            rows,
            contract=self.contract,
            prediction_field="model.text",
            gold_field="target",
            label_field="kind",
            positive_label="object",
            negative_label="empty",
        )

        self.assertEqual(result["fields"]["prediction"], "model.text")
        self.assertEqual(result["metrics"]["positive_reward_ap_proxy_mean"], 1.0)
        self.assertEqual(result["metrics"]["negative_explicit_no_objects_accuracy"], 1.0)

    def test_unresolved_positive_gold_does_not_fabricate_complete_metric(self) -> None:
        rows = [{"has_ground_truth": True, "prediction": answer([box()])}]

        result = self.module.analyze_predictions(rows, contract=self.contract)

        self.assertEqual(result["gold_resolution"]["unresolved_positive_gold_rows"], 1)
        self.assertEqual(result["coverage"]["positive_reward_rate"], 0.0)
        self.assertFalse(result["coverage"]["positive_reward_complete"])
        self.assertIsNone(result["metrics"]["positive_reward_ap_proxy_mean"])
        self.assertIsNone(
            result["metrics"]["positive_reward_ap_proxy_mean_on_resolved_gold"]
        )

    def test_transport_error_is_not_malformed_model_output(self) -> None:
        rows = [
            {
                "has_ground_truth": False,
                "failure": "TimeoutError: timed out",
            }
        ]

        result = self.module.analyze_predictions(
            rows, contract=self.contract, error_field="failure"
        )

        self.assertEqual(result["counts"]["transport_error_rows"], 1)
        self.assertEqual(result["counts"]["missing_prediction_rows"], 1)
        self.assertEqual(result["counts"]["malformed_prediction_rows"], 0)
        self.assertEqual(result["metrics"]["malformed_rate"], 0.0)
        self.assertEqual(result["metrics"]["missing_or_malformed_rate"], 1.0)

    def test_label_gold_conflict_has_line_specific_error(self) -> None:
        rows = [
            {
                "example_type": "negative",
                "solution": answer([box()]),
                "prediction": answer([box()]),
            }
        ]

        with self.assertRaisesRegex(ValueError, "line 1: label field.*conflicts"):
            self.module.analyze_predictions(rows, contract=self.contract)

    def test_refuses_to_overwrite_input_even_with_overwrite_flag(self) -> None:
        root = self.make_temp_dir()
        predictions = root / "predictions.jsonl"
        self.write_jsonl(
            predictions,
            [
                {
                    "has_ground_truth": False,
                    "prediction": answer("No Objects"),
                }
            ],
        )
        args = self.module.parse_args(
            [str(predictions), "--output", str(predictions), "--overwrite"]
        )
        report, protected = self.module.build_report(args)

        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            self.module._write_report(
                predictions,
                report,
                overwrite=True,
                protected_paths=protected,
            )

    def test_protected_run_root_allows_new_report_but_never_overwrites_files(self) -> None:
        root = self.make_temp_dir()
        artifact = root / "result.json"
        artifact.write_text("immutable\n", encoding="utf-8")
        report_path = root / "analysis.json"

        self.module._write_report(
            report_path, {"metric": 1.0}, overwrite=False, protected_paths={root}
        )
        self.assertTrue(report_path.is_file())

        with self.assertRaisesRegex(ValueError, "existing run artifact"):
            self.module._write_report(
                artifact,
                {},
                overwrite=True,
                protected_paths={root},
            )
        self.assertEqual(artifact.read_text(encoding="utf-8"), "immutable\n")

        with self.assertRaisesRegex(ValueError, "existing run artifact"):
            self.module._write_report(
                report_path, {}, overwrite=True, protected_paths={root}
            )

    def test_writes_json_atomically_and_requires_overwrite_for_existing_report(self) -> None:
        root = self.make_temp_dir()
        output = root / "nested/report.json"
        report = {"schema_version": "1.0", "metric": 0.5}

        self.module._write_report(
            output, report, overwrite=False, protected_paths=set()
        )

        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
            self.module._write_report(
                output, report, overwrite=False, protected_paths=set()
            )
        self.module._write_report(
            output, {"metric": 1.0}, overwrite=True, protected_paths=set()
        )
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"metric": 1.0})

    def test_multiple_visual_benchmarks_require_explicit_id(self) -> None:
        root = self.make_temp_dir()
        summary = root / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "benchmarks": {
                        "visual_rft_coco_light": {},
                        "visual_rft_coco_full": {},
                    }
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "use --benchmark-id"):
            self.module.resolve_source(summary, source_kind="summary", benchmark_id=None)


if __name__ == "__main__":
    unittest.main()
