from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_agentrl_data import task_rows
from scripts.prepare_gui_sft import gui_rows, mix_replay
from toolvista.agentrl.actions import PROTOCOL_VERSION
from toolvista.agentrl.checkpoints import sha256
from toolvista.agentrl.gate import SMOKE_SUITE, evaluate_gate


def success_record(task="test", seed=0):
    return {"protocol": PROTOCOL_VERSION, "task_id": task, "seed": seed,
            "success": True, "failure_reason": None,
            "steps": [{"valid": True, "terminated": True, "success": True,
                       "assistant_action_text": '{"action":"press","key":"Enter"}'}]}


class GateTest(unittest.TestCase):
    def test_gate_requires_success_coverage_and_95_percent_format(self):
        records = [success_record(task, seed) for task, seed in SMOKE_SUITE]
        self.assertTrue(evaluate_gate(records, "hash")["passed"])
        self.assertFalse(evaluate_gate(records[:-1], "hash")["passed"])
        records[0]["steps"][0]["assistant_action_text"] = "bad"
        self.assertTrue(evaluate_gate(records, "hash")["passed"])
        records[1]["steps"][0]["assistant_action_text"] = "bad"
        self.assertFalse(evaluate_gate(records, "hash")["passed"])

    def test_zero_success_is_blocking_and_infrastructure_is_distinguished(self):
        records = [success_record(task, seed) for task, seed in SMOKE_SUITE]
        for record in records:
            record["success"] = False
            record["steps"][0]["success"] = False
        report = evaluate_gate(records, "hash")
        self.assertFalse(report["passed"])
        self.assertEqual(report["recommendation"], "gui_sft_or_curriculum_required")
        records[0]["failure_reason"] = "runtime:TimeoutError"
        self.assertEqual(evaluate_gate(records, "hash")["recommendation"], "fix_environment_or_inference")

    def test_range_validation_denominator_is_parsed_actions(self):
        record = success_record()
        record["steps"][0]["assistant_action_text"] = '{"action":"click","x":2,"y":0}'
        report = evaluate_gate([record], "hash", expected_suite=[("test", 0)])
        self.assertEqual(report["json_parse_rate"], 1)
        self.assertEqual(report["action_validity_rate"], 0)
        self.assertFalse(report["passed"])

    def test_seed_splits_are_deterministic_and_disjoint(self):
        from toolvista.agentrl.gate import TASKS, TRAIN_TASKS
        train = task_rows(range(100, 200), tasks=TASKS)
        val = task_rows(range(100000, 100010), tasks=TASKS)
        self.assertEqual(train, task_rows(range(100, 200), tasks=TASKS))
        identity = lambda rows: {(r["extra_info"]["task_id"], r["extra_info"]["seed"]) for r in rows}
        self.assertFalse(identity(train) & identity(val))
        self.assertFalse(identity(train) & set(SMOKE_SUITE))
        self.assertEqual(len(train), 300)
        self.assertEqual(len(task_rows(range(100, 200))), 1200)
        self.assertTrue(set(TRAIN_TASKS).issuperset(TASKS))
        self.assertNotIn("browsergym/miniwob.click-checkboxes", TRAIN_TASKS)
        self.assertIn("browsergym/miniwob.login-user", TRAIN_TASKS)


class GuiSftTest(unittest.TestCase):
    def test_history_actions_are_not_supervised_again(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "screen.png"
            image.write_bytes(b"image")
            record = success_record()
            first = {**record["steps"][0], "terminated": False, "success": False,
                     "observation": {"image_path": str(image), "image_sha256": sha256(image), "goal": "Click", "step": 0}}
            last = {**record["steps"][0], "observation": {**first["observation"], "step": 1}}
            record["steps"] = [first, last]
            rows = gui_rows([record])
            self.assertEqual(rows[0]["messages"][-1].get("loss_scale"), 1.0)
            assistant = [m for m in rows[1]["messages"] if m["role"] == "assistant"]
            self.assertEqual([m.get("loss_scale") for m in assistant], [0.0, 1.0])

    def test_deduplicates_verified_images_actions_and_includes_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "screen.png"
            image.write_bytes(b"image")
            record = success_record()
            record["steps"][0]["observation"] = {"image_path": str(image), "image_sha256": sha256(image),
                                                   "goal": "Press enter", "step": 0}
            rows = gui_rows([record, copy.deepcopy(record)])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["images"], [str(image)])
            self.assertIn("<image>", rows[0]["messages"][-2]["content"])
            replay = [{"messages": [{"role": "assistant", "content": "existing ToolVista answer"}]}]
            mixed = mix_replay(rows, replay)
            self.assertEqual(len(mixed), 2)
            self.assertIn(replay[0], mixed)
            image.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                gui_rows([record])

    def test_unverified_episodes_and_missing_replay_are_rejected(self):
        record = success_record()
        record["steps"][0]["terminated"] = False
        with self.assertRaisesRegex(ValueError, "verified successful"):
            gui_rows([record])
        with self.assertRaises(ValueError):
            mix_replay([{}], [])


if __name__ == "__main__":
    unittest.main()
