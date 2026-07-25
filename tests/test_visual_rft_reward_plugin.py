from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = REPO_ROOT / "toolvista" / "rewards" / "visual_rft.py"


class StubORM:
    def __init__(self, args=None, **kwargs):
        self.args = args


def load_plugin():
    registry = {}
    rewards_module = types.ModuleType("swift.rewards")
    rewards_module.ORM = StubORM
    rewards_module.orms = registry
    swift_module = types.ModuleType("swift")
    swift_module.rewards = rewards_module

    spec = importlib.util.spec_from_file_location("toolvista_visual_rft_reward_test", PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"swift": swift_module, "swift.rewards": rewards_module}):
        spec.loader.exec_module(module)
    return module, registry


def answer(boxes) -> str:
    return f"<think>reason</think><answer>{boxes!r}</answer>"


class VisualRFTRewardPluginTest(unittest.TestCase):
    def test_import_registers_routed_and_legacy_ms_swift_orms(self) -> None:
        module, registry = load_plugin()

        self.assertEqual(
            registry,
            {
                "visual_rft": module.VisualRFTORM,
                "visual_rft_iou": module.VisualRFTIoUORM,
            },
        )
        self.assertTrue(issubclass(module.VisualRFTORM, StubORM))
        self.assertTrue(issubclass(module.VisualRFTIoUORM, StubORM))

    def test_routed_reward_scores_visual_rft_classification_like_upstream(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft"]()

        rewards = orm(
            [
                "<think>shape</think><answer>FGVC Aircraft</answer>",
                "<think>shape</think><answer>automobile</answer>",
            ],
            [
                "<think>gold</think><answer>fgvc_aircraft</answer>",
                "<think>gold</think><answer>aircraft</answer>",
            ],
            task=["classification", "classification"],
        )

        self.assertEqual(rewards, [1.0, 0.0])

    def test_routed_reward_dispatches_grounding_and_ignores_unknown_tasks(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft"]()
        boxes = [{"Position": [10, 20, 30, 40], "Confidence": 0.9}]

        rewards = orm(
            [answer(boxes), "<answer>anything</answer>"],
            [answer(boxes), "<answer>anything</answer>"],
            task=["grounding", "unsupported_task"],
        )

        self.assertEqual(rewards, [1.0, None])

    def test_perfect_box_receives_full_reward(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        boxes = [{"Position": [10, 20, 30, 40], "Confidence": 0.9}]

        rewards = orm([answer(boxes)], [answer(boxes)], task=["grounding"])

        self.assertEqual(rewards, [1.0])

    def test_missing_or_extra_boxes_are_penalized(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        gold = [
            {"Position": [0, 0, 10, 10], "Confidence": 1.0},
            {"Position": [20, 20, 30, 30], "Confidence": 1.0},
        ]
        one_prediction = [{"Position": [0, 0, 10, 10], "Confidence": 0.9}]
        extra_prediction = one_prediction + [{"Position": [40, 40, 50, 50], "Confidence": 0.8}]

        rewards = orm(
            [answer(one_prediction), answer(extra_prediction)],
            [answer(gold), answer(one_prediction)],
            task=["detection", "object_detection"],
        )

        self.assertEqual(rewards, [0.5, 0.5])

    def test_iou_at_or_below_threshold_receives_zero(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        gold = [{"Position": [0, 0, 10, 10], "Confidence": 1.0}]
        prediction = [{"Position": [0, 0, 5, 10], "Confidence": 1.0}]

        rewards = orm([answer(prediction)], [answer(gold)], task=["gui_grounding"])

        self.assertEqual(rewards, [0.0])

    def test_duplicate_predictions_are_removed_before_scoring(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        box = {"Position": [0, 0, 10, 10], "Confidence": 0.9}

        rewards = orm([answer([box, dict(box)])], [answer([box])], task=["grounding"])

        self.assertEqual(rewards, [1.0])

    def test_malformed_bbox_output_receives_zero(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        gold = [{"Position": [0, 0, 10, 10], "Confidence": 1.0}]

        rewards = orm(
            ["<think>reason</think><answer>not-json</answer>"],
            [answer(gold)],
            task=["grounding"],
        )

        self.assertEqual(rewards, [0.0])

    def test_unhashable_bbox_literal_receives_zero(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        malformed = (
            "<think>reason</think><answer>"
            "{{'Position': [0, 0, 10, 10], 'Confidence': 1}}"
            "</answer>"
        )
        gold = [{"Position": [0, 0, 10, 10], "Confidence": 1.0}]

        rewards = orm([malformed], [answer(gold)], task=["detection"])

        self.assertEqual(rewards, [0.0])

    def test_truncated_tail_keeps_complete_boxes_like_visual_rft(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        gold = [{"Position": [0, 0, 10, 10], "Confidence": 1.0}]
        truncated = (
            "<think>reason</think><answer>"
            "[{'Position': [0, 0, 10, 10], 'Confidence': 1.0}, "
            "{'Position': [20, 20"
        )

        rewards = orm([truncated], [answer(gold)], task=["grounding"])

        self.assertEqual(rewards, [1.0])

    def test_mismatched_reward_columns_fail_before_ms_swift_tensor_assignment(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()
        boxes = [{"Position": [0, 0, 10, 10], "Confidence": 1.0}]

        with self.assertRaisesRegex(ValueError, "batch length mismatch"):
            orm([answer(boxes), answer(boxes)], [answer(boxes)], task=["grounding", "grounding"])

    def test_non_grounding_task_returns_none_for_ms_swift_to_ignore(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_iou"]()

        rewards = orm(
            ["<think>reason</think><answer>aircraft</answer>"],
            ["<think>gold</think><answer>aircraft</answer>"],
            task=["classification"],
        )

        self.assertEqual(rewards, [None])


if __name__ == "__main__":
    unittest.main()
