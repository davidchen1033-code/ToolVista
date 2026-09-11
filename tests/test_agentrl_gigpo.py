from __future__ import annotations

import copy
import importlib.util
import math
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

from toolvista.agentrl.gigpo import episode_metadata, state_anchor, step_advantages, validate_metadata
from toolvista.agentrl.rollout import run_episode
from tests.test_agentrl_core import FakeEnv, FakePolicy


def episode(anchors, success, *, seed=100, task="click"):
    return {"task_id": task, "seed": seed, "steps": [
        {"anchor": anchor, "start": j * 4, "end": j * 4 + 2,
         "reward": float(success) if j == len(anchors) - 1 else 0.0}
        for j, anchor in enumerate(anchors)]}


class GiGPOTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Hydra config composition requires hydra-core")
    def test_config_composition_preserves_ablation_controls(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        with tempfile.TemporaryDirectory() as tmp:
            # Isolate our composition from optional installed verl packages.
            (Path(tmp) / "ppo_trainer.yaml").write_text("{}")
            with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "configs/verl"),
                                       version_base=None):
                configs = [OmegaConf.to_container(compose(config_name="miniwob_" + name,
                    overrides=["hydra.searchpath=[file://" + Path(tmp).as_posix() + "]"]))
                    for name in ("grpo", "gigpo")]
        first, second = configs
        self.assertEqual(second.pop("gigpo"), {"step_weight": 1.0})
        self.assertEqual(first["algorithm"].pop("adv_estimator"), "grpo")
        self.assertEqual(second["algorithm"].pop("adv_estimator"), "gigpo")
        first["trainer"].pop("experiment_name")
        second["trainer"].pop("experiment_name")
        self.assertEqual(first, second)

    def test_terminal_returns_compare_same_state_and_discount_in_action_units(self):
        good, bad = episode(["initial", "submit"], True), episode(["initial", "elsewhere"], False)
        values, metrics = step_advantages([good, bad], ["group", "group"], gamma=0.9, normalize=False)
        self.assertEqual(values, [[0.45, 0.0], [-0.45, 0.0]])
        self.assertEqual(metrics["matched_fraction"], 0.5)
        self.assertEqual(metrics["nonzero_fraction"], 0.5)

    def test_uid_isolation_singletons_and_all_failure(self):
        good, bad = episode(["same"], True), episode(["same"], False)
        self.assertEqual(step_advantages([good, bad], ["a", "b"])[0], [[0], [0]])
        self.assertEqual(step_advantages([bad, bad], ["a", "a"])[0], [[0], [0]])
        with self.assertRaisesRegex(ValueError, "tasks/seeds"):
            step_advantages([good, episode(["same"], False, seed=101)], ["a", "a"])

    def test_repeated_visits_and_reordering(self):
        a, b = episode(["same", "same"], True), episode(["same"], False)
        values, _ = step_advantages([a, b], ["a", "a"], normalize=False)
        self.assertAlmostEqual(values[0][0], 1 / 3)
        self.assertAlmostEqual(values[1][0], -2 / 3)
        reordered, _ = step_advantages([b, a], ["a", "a"], normalize=False)
        self.assertEqual(reordered, list(reversed(values)))

    def test_sample_standard_deviation_matches_grpo_convention(self):
        values, _ = step_advantages([episode(["x"], True), episode(["x"], False)], ["a", "a"])
        self.assertAlmostEqual(values[0][0], 0.5 / (math.sqrt(0.5) + 1e-6))
        self.assertEqual(values[1][0], -values[0][0])

    def test_metadata_rejects_dropped_tokens_overlap_and_reward_mismatch(self):
        ep = episode(["initial", "submit"], True)
        mask = [1, 1, 0, 0, 1, 1, 0, 0]
        validate_metadata([ep], [mask], [1])
        for altered in ([1, 1, 0, 0, 1, 0, 0, 0], [1, 1, 1, 0, 1, 1, 0, 0]):
            with self.assertRaisesRegex(ValueError, "response_mask"):
                validate_metadata([ep], [altered], [1])
        for change in ({"start": 1}, {"end": 20}, {"anchor": ""}):
            broken = copy.deepcopy(ep)
            broken["steps"][1].update(change)
            with self.assertRaises(ValueError):
                validate_metadata([broken], [mask], [1])
        with self.assertRaisesRegex(ValueError, "terminal-only"):
            validate_metadata([ep], [mask], [0])

    def test_state_requires_browser_evidence_and_distinguishes_live_controls(self):
        obs = {"goal": "log in", "image_sha256": "same-pixels"}
        state = {"dom": "<input/>", "focus": 0, "controls": [{"value": ""}]}
        key = state_anchor(obs, state)
        self.assertEqual(key, state_anchor({**obs, "step": 5, "image_path": "different"}, state))
        for modified in ({**state, "focus": 1}, {**state, "controls": [{"value": "password"}]}):
            self.assertNotEqual(key, state_anchor(obs, modified))
        with self.assertRaises(ValueError):
            state_anchor(obs, {})


class TerminalRolloutTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_then_success_uses_only_terminal_reward(self):
        record = await run_episode(FakeEnv(success_at=1), FakePolicy([
            "bad", '{"action":"press","key":"Enter"}']), task_id="click", seed=100,
            reward_mode="terminal")
        self.assertEqual(record["reward"], 1)
        self.assertEqual([s["reward"] for s in record["steps"]], [0, 1])
        for step in record["steps"]:
            step["observation"]["state_anchor"] = "state"
        metadata = episode_metadata(record, [(0, 2), (4, 6)])
        validate_metadata([metadata], [[1, 1, 0, 0, 1, 1]], [1])
        with self.assertRaisesRegex(ValueError, "spans"):
            episode_metadata(record, [(0, 2)])

    async def test_failed_loop_has_no_shaping_penalty(self):
        record = await run_episode(FakeEnv(success_at=99, repeat=True), FakePolicy(),
                                   task_id="click", seed=100, reward_mode="terminal")
        self.assertEqual(record["reward"], 0)
        self.assertEqual([s["reward"] for s in record["steps"]], [0, 0, 0])


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("verl"),
                     "optional pinned verl/torch integration test")
class VerlGiGPOIntegrationTest(unittest.TestCase):
    def test_bridge_matches_grpo_at_zero_weight_and_masks_observations(self):
        import numpy as np
        import torch
        from verl.trainer.ppo.ray_trainer import compute_advantage as upstream
        from toolvista.agentrl.train_gigpo import compute_advantage
        mask = torch.tensor([[1., 1, 0, 0, 1, 1, 0, 0]] * 2)
        rewards = torch.zeros_like(mask)
        rewards[0, 5] = 1
        metadata = [episode(["same", "only-good"], True), episode(["same", "only-bad"], False)]
        def batch():
            return SimpleNamespace(batch={"response_mask": mask, "token_level_rewards": rewards.clone()},
                non_tensor_batch={"uid": np.array(["a", "a"], dtype=object), "gui_episode": metadata})
        baseline = upstream(batch(), adv_estimator="grpo").batch["advantages"]
        zero = compute_advantage(upstream, batch(), "gigpo", step_weight=0).batch["advantages"]
        torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
        result = compute_advantage(upstream, batch(), "gigpo").batch["advantages"]
        torch.testing.assert_close(result[:, :2], baseline[:, :2] * 2)
        torch.testing.assert_close(result[:, 4:6], baseline[:, 4:6])
        self.assertEqual(result[:, [2, 3, 6, 7]].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
