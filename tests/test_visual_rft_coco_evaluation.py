from __future__ import annotations

import unittest

from scripts.prepare_visual_rft_coco_eval import NOVEL15, select_light
from toolvista.evaluation.visual_rft_coco import _to_coco_detection, parse_boxes


class VisualRFTCocoEvaluationTest(unittest.TestCase):
    def test_light_selection_keeps_complete_images_and_positive_category_coverage(self) -> None:
        rows = []
        for index, category in enumerate(NOVEL15, 1):
            rows.extend(
                [
                    {
                        "sample_id": f"{index}-positive",
                        "image_id": index,
                        "category_id": index,
                        "category": category,
                        "has_ground_truth": True,
                    },
                    {
                        "sample_id": f"{index}-extra",
                        "image_id": index,
                        "category_id": 999,
                        "category": "extra",
                        "has_ground_truth": False,
                    },
                ]
            )

        first = select_light(rows, seed=42, query_limit=20)
        second = select_light(rows, seed=42, query_limit=20)

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 20)
        selected_ids = {row["image_id"] for row in first}
        for image_id in selected_ids:
            self.assertEqual(
                [row for row in first if row["image_id"] == image_id],
                [row for row in rows if row["image_id"] == image_id],
            )
        covered = {
            row["category"]
            for row in first
            if row["has_ground_truth"] and row["category"] in NOVEL15
        }
        self.assertEqual(covered, set(NOVEL15))

    def test_parse_boxes_and_convert_normalized_coordinates(self) -> None:
        boxes = parse_boxes(
            "<think>done</think><answer>"
            "[{'Position':[100,200,900,800],'Confidence':0.75},"
            "{'Position':[100,200,900,800],'Confidence':0.50}]"
            "</answer>"
        )

        self.assertEqual(len(boxes), 1)
        detection = _to_coco_detection(
            {
                "image_id": 7,
                "category_id": 3,
                "image_width": 200,
                "image_height": 100,
            },
            boxes[0],
        )
        self.assertEqual(detection["bbox"], [20.0, 20.0, 160.0, 60.0])
        self.assertEqual(detection["score"], 0.75)
        self.assertEqual(parse_boxes("<answer>No Objects</answer>"), [])
        self.assertEqual(parse_boxes("<answer>not-json</answer>"), [])


if __name__ == "__main__":
    unittest.main()
