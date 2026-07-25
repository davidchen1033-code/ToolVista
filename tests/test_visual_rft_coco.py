from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.prepare_visual_rft_coco_eval import NOVEL15, build_queries, select_light
from toolvista.evaluation.visual_rft_coco import (
    parse_boxes,
    score_negative_behavior,
    score_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class VisualRftCocoTest(unittest.TestCase):
    def test_negative_behavior_separates_abstention_false_positive_and_other(self) -> None:
        metrics, details = score_negative_behavior(
            [
                {"has_ground_truth": False, "prediction": "<answer>No Objects</answer>"},
                {"has_ground_truth": False, "prediction": "<answer>[]</answer>"},
                {
                    "has_ground_truth": False,
                    "prediction": (
                        "<answer>[{'Position':[10,20,110,220],"
                        "'Confidence':0.75}]</answer>"
                    ),
                },
                {"has_ground_truth": False, "prediction": "<answer>broken"},
                {"has_ground_truth": True, "prediction": "<answer>No Objects</answer>"},
            ]
        )

        self.assertEqual(
            metrics["visual_rft_coco/negative_abstention_accuracy"], 0.5
        )
        self.assertEqual(
            metrics["visual_rft_coco/negative_false_positive_rate"], 0.25
        )
        self.assertEqual(details["negative_rows"], 4)
        self.assertEqual(details["abstention_rows"], 2)
        self.assertEqual(details["false_positive_rows"], 1)
        self.assertEqual(details["other_rows"], 1)

    def test_parser_accepts_answer_payload_and_rejects_malformed_boxes(self) -> None:
        boxes = parse_boxes(
            "<think>done</think><answer>[{'Position':[10,20,110,220],"
            "'Confidence':0.75}]</answer>"
        )
        self.assertEqual(boxes, [{"Position": [10.0, 20.0, 110.0, 220.0], "Confidence": 0.75}])
        self.assertEqual(parse_boxes("<answer>No Objects</answer>"), [])
        self.assertEqual(parse_boxes("<answer>[{'Position':[0,0,1], 'Confidence':1}]</answer>"), [])

    def test_light_selection_is_deterministic_and_keeps_positive_gt_for_every_category(self) -> None:
        rows = []
        for index, category in enumerate(NOVEL15):
            rows.extend(
                [
                    {
                        "image_id": index * 10 + 1,
                        "category": category,
                        "category_id": index + 1,
                        "has_ground_truth": True,
                    },
                    {
                        "image_id": index * 10 + 2,
                        "category": category,
                        "category_id": index + 1,
                        "has_ground_truth": False,
                    },
                ]
            )
        first = select_light(rows, seed=42, query_limit=20)
        second = select_light(rows, seed=42, query_limit=20)
        self.assertEqual(first, second)
        positives = Counter(row["category"] for row in first if row["has_ground_truth"])
        self.assertEqual(set(positives), set(NOVEL15))
        selected_images = {row["image_id"] for row in first}
        for image_id in selected_images:
            self.assertEqual(
                [row for row in first if row["image_id"] == image_id],
                [row for row in rows if row["image_id"] == image_id],
            )

    def test_build_queries_marks_ground_truth_pairs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tests") as directory:
            images_dir = Path(directory)
            (images_dir / "one.jpg").write_bytes(b"image")
            annotations = {
                "images": [{"id": 1, "file_name": "one.jpg", "width": 10, "height": 20}],
                "categories": [{"id": 7, "name": "cat"}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 7, "bbox": [1, 1, 2, 2]}],
            }
            rows = build_queries(
                annotations,
                {"1": ["cat"]},
                images_dir=images_dir,
                categories=("cat",),
            )
            self.assertTrue(rows[0]["has_ground_truth"])

    def test_empty_predictions_use_upstream_single_mapping_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tests") as directory:
            root = Path(directory)
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "width": 10, "height": 10}],
                        "categories": [{"id": 1, "name": "cat"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            evaluator = root / "evaluator.py"
            evaluator.write_text(
                "class CocoDetectionEvaluator:\n"
                "    metric_names = ['mAP', 'AP_50']\n"
                "    class_names = ['cat']\n"
                "    def __init__(self, path): self.path = path\n"
                "    def evaluate(self, results, save_dir): return {'mAP': 0.0, 'AP_50': 0.0}\n",
                encoding="utf-8",
            )
            benchmark_dir = root / "benchmark"
            benchmark_dir.mkdir()
            upstream, metrics, failures = score_predictions(
                [
                    {
                        "image_id": 1,
                        "image_width": 10,
                        "image_height": 10,
                        "category_id": 1,
                        "prediction": "<answer>No Objects</answer>",
                    }
                ],
                annotations=annotations,
                upstream_evaluator=evaluator,
                benchmark_dir=benchmark_dir,
                categories=("cat",),
            )
            self.assertEqual(failures, 0)
            self.assertEqual(metrics["visual_rft_coco/novel15_map"], 0.0)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
            filtered = json.loads((benchmark_dir / "coco_annotations.json").read_text())
            self.assertEqual([item["id"] for item in filtered["images"]], [1])
            self.assertEqual(upstream["prediction_count"], 0)


if __name__ == "__main__":
    unittest.main()
