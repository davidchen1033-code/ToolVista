from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATH = REPO_ROOT / "toolvista" / "rewards" / "visual_rft_withnegative.py"


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

    module_name = "toolvista_visual_rft_withnegative_test"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"swift": swift_module, "swift.rewards": rewards_module, module_name: module},
    ):
        spec.loader.exec_module(module)
    return module, registry


def answer(payload) -> str:
    value = payload if isinstance(payload, str) else repr(payload)
    return f"<think>reason</think><answer>{value}</answer>"


def box(position=(0, 0, 100, 100), confidence=1.0):
    return {"Position": list(position), "Confidence": confidence}


class VisualRFTWithNegativeRewardPluginTest(unittest.TestCase):
    def test_import_registers_only_the_new_ms_swift_orm(self) -> None:
        module, registry = load_plugin()

        self.assertEqual(
            registry,
            {
                "visual_rft_withnegative": module.VisualRFTWithNegativeORM,
                "visual_rft_withnegative_format": module.VisualRFTWithNegativeFormatORM,
            },
        )
        self.assertTrue(issubclass(module.VisualRFTWithNegativeORM, StubORM))
        self.assertTrue(issubclass(module.VisualRFTWithNegativeFormatORM, StubORM))

    def test_format_reward_accepts_non_thinking_and_thinking_qwen_completions(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative_format"]()

        rewards = orm(
            [
                "<answer>No Objects</answer>",
                "<answer>\n[{'Position': [0, 0, 100, 100], 'Confidence': 1}]\n</answer>",
                "<think>localized the target</think><answer>No Objects</answer>",
                "<think>line one\nline two</think>\n<answer>No Objects</answer>",
                "<think></think><answer>No Objects</answer>",
                "<think> \n </think>  <answer>No Objects</answer>",
            ],
            solution=["ignored"] * 6,
            task=["detection"] * 6,
        )

        self.assertEqual(rewards, [0.20, 0.20, 0.25, 0.25, 0.20, 0.20])

    def test_format_reward_accepts_bare_detection_payloads(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative_format"]()

        completions = [
            "No Objects",
            "[]",
            "[{'Position': [0, 0, 100, 100], 'Confidence': 1}]",
            "ordinary unsupported text",
        ]

        self.assertEqual(orm(completions), [0.20, 0.20, 0.20, 0.0])

    def test_format_reward_rejects_extra_empty_and_malformed_output(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative_format"]()

        completions = [
            " <answer>No Objects</answer>",
            "<answer>No Objects</answer>\n",
            "prefix<answer>No Objects</answer>",
            "<answer>No Objects</answer>suffix",
            "<answer></answer>",
            "<answer> \n </answer>",
            "<think>reason</think><answer></answer>",
            "<think>reason</think>extra<answer>No Objects</answer>",
            "<think>reason</think><answer>No Objects</answer>extra",
            "<think>reason<answer>No Objects</answer>",
            "<think>reason</think><answer>No Objects",
            "<answer>first</answer><answer>second</answer>",
            "<think>one</think><think>two</think><answer>No Objects</answer>",
        ]

        self.assertEqual(orm(completions), [0.0] * len(completions))

    def test_empty_gold_distinguishes_explicit_empty_list_boxes_and_malformed(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        gold = answer("No Objects")

        rewards = orm(
            [
                answer("No Objects"),
                answer([]),
                answer([box(confidence=0.0)]),
                answer([box(confidence=1.0)]),
                answer("not-json"),
            ],
            [gold] * 5,
            task=["detection"] * 5,
        )

        self.assertEqual(rewards, [1.0, 0.75, -0.25, -1.0, -1.0])

    def test_empty_gold_penalizes_the_number_of_false_positive_boxes(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        low_confidence_boxes = [box((0, 0, 100, 100), 0.0), box((200, 200, 300, 300), 0.0)]

        rewards = orm([answer(low_confidence_boxes)], [answer([])], task=["object_detection"])

        self.assertEqual(rewards, [-0.5])

    def test_perfect_positive_detection_receives_full_reward(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        expected = [box()]

        rewards = orm([answer(expected)], [answer(expected)], task=["grounding"])

        self.assertEqual(rewards, [1.0])

    def test_multi_threshold_proxy_rewards_calibrated_localization_confidence(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        gold = [box()]
        iou_point_seven_five_calibrated = [box((0, 0, 75, 100), 0.6)]
        iou_point_seven_five_overconfident = [box((0, 0, 75, 100), 1.0)]

        rewards = orm(
            [answer(iou_point_seven_five_calibrated), answer(iou_point_seven_five_overconfident)],
            [answer(gold), answer(gold)],
            task=["detection", "detection"],
        )

        self.assertAlmostEqual(rewards[0], 0.528, places=12)
        self.assertAlmostEqual(rewards[1], 0.48, places=12)
        self.assertGreater(rewards[0], rewards[1])

    def test_positive_proxy_penalizes_false_positives_false_negatives_and_duplicates(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        perfect = box()
        extra = box((200, 200, 300, 300), 1.0)
        two_gold = [perfect, box((400, 400, 500, 500), 1.0)]

        rewards = orm(
            [answer([perfect, extra]), answer([perfect]), answer([perfect, dict(perfect)])],
            [answer([perfect]), answer(two_gold), answer([perfect])],
            task=["detection"] * 3,
        )

        self.assertAlmostEqual(rewards[0], 0.7 * (2.0 / 3.0), places=12)
        self.assertAlmostEqual(rewards[1], 0.7 * (2.0 / 3.0) + 0.15, places=12)
        self.assertAlmostEqual(rewards[2], 0.7 * (2.0 / 3.0), places=12)

    def test_positive_gold_penalizes_abstention_malformed_and_high_confidence_miss(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        gold = answer([box()])
        miss = [box((200, 200, 300, 300), 1.0)]

        rewards = orm(
            [answer("No Objects"), answer([]), answer("not-json"), answer(miss)],
            [gold] * 4,
            task=["detection"] * 4,
        )

        self.assertEqual(rewards[:3], [-0.75, -0.75, -1.0])
        self.assertAlmostEqual(rewards[3], -0.3, places=12)

    def test_invalid_box_values_and_truncated_payload_are_malformed(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        gold = answer([box()])
        invalid = [
            answer([box(confidence=1.1)]),
            answer([box((-1, 0, 100, 100), 0.5)]),
            answer([box((100, 0, 0, 100), 0.5)]),
            "<think>x</think><answer>[{'Position': [0, 0, 100, 100], 'Confidence': 1.0},",
        ]

        rewards = orm(invalid, [gold] * len(invalid), task=["detection"] * len(invalid))

        self.assertEqual(rewards, [-1.0] * len(invalid))

    def test_unknown_task_returns_none_and_default_task_is_detection(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        payload = answer([box()])

        self.assertEqual(orm([payload], [payload], task=["classification"]), [None])
        self.assertEqual(orm([payload], [payload]), [1.0])

    def test_batch_length_mismatch_fails_fast(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        payload = answer([box()])

        with self.assertRaisesRegex(ValueError, "batch length mismatch"):
            orm([payload, payload], [payload], task=["detection", "detection"])

    def test_malformed_gold_fails_fast(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()

        with self.assertRaisesRegex(ValueError, "malformed gold solution at batch index 0"):
            orm([answer("No Objects")], [answer("not-json")], task=["detection"])

    def test_all_emitted_rewards_are_finite_and_bounded(self) -> None:
        _, registry = load_plugin()
        orm = registry["visual_rft_withnegative"]()
        gold = answer([box()])
        many_false_positives = [box((index * 10, 200, index * 10 + 5, 210), 1.0) for index in range(20)]

        rewards = orm(
            [answer([box()]), answer(many_false_positives), answer("No Objects"), answer("bad")],
            [gold] * 4,
            task=["detection"] * 4,
        )

        self.assertTrue(all(math.isfinite(reward) and -1.0 <= reward <= 1.0 for reward in rewards))


if __name__ == "__main__":
    unittest.main()
