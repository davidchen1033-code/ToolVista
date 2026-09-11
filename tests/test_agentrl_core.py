from __future__ import annotations

import json
import unittest

from toolvista.agentrl.actions import ActionError, browsergym_action, parse_action
from toolvista.agentrl.rewards import episode_reward
from toolvista.agentrl.rollout import Generation, run_episode


def observation(index=0):
    return {"image_path": "unused.png", "image_sha256": str(index), "goal": "Click target", "step": index}


class FakeEnv:
    def __init__(self, success_at=2, crash=False, repeat=False):
        self.success_at, self.crash, self.repeat = success_at, crash, repeat
        self.steps, self.closed = 0, False

    async def reset(self, **kwargs):
        return observation()

    async def step(self, action):
        self.steps += 1
        if self.crash:
            raise TimeoutError("browser timed out")
        done = self.steps == self.success_at
        return {"observation": observation(0 if self.repeat else self.steps), "terminated": done,
                "success": done, "environment_reward": float(done)}

    async def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, actions=None):
        self.actions = iter(actions) if actions else None

    async def generate(self, observation, history):
        return Generation(next(self.actions) if self.actions else '{"action":"click","x":0.5,"y":0.5}')


class ActionsTest(unittest.TestCase):
    def test_valid_actions_and_injection_is_only_text(self):
        for value in [{"action": "click", "x": 0, "y": 1}, {"action": "type", "text": "hello"},
                      {"action": "scroll", "dx": -1000, "dy": 1000}, {"action": "press", "key": "Enter"},
                      {"action": "finish", "status": "success"}]:
            self.assertEqual(parse_action(json.dumps(value)), value)
        self.assertEqual(browsergym_action({"action": "click", "x": 1, "y": 1}, 500, 320), "mouse_click(499, 319)")
        import ast
        payload = "'); malicious() #\n"
        tree = ast.parse(browsergym_action({"action": "type", "text": payload}, 500, 320))
        self.assertEqual(len(tree.body), 1)
        self.assertEqual(tree.body[0].value.args[0].value, payload)

    def test_bad_json_and_bad_schema_are_different(self):
        for text in ['```json\n{}\n```', '{"action":"finish","action":"click"}', '{"x":NaN}', 'hello']:
            with self.assertRaises(ActionError) as exc:
                parse_action(text)
            self.assertFalse(exc.exception.parsed)
        for value in [{"action": "click", "x": True, "y": 0}, {"action": "click", "x": -0.01, "y": 0},
                      {"action": "scroll", "dx": 0, "dy": 1001}, {"action": "finish", "status": "done"},
                      {"action": "press", "key": "F12"}, {"action": []}, [],
                      {"action": "type", "text": ""}, {"action": "click", "x": 0.2, "y": 0.3, "code": "x"}]:
            with self.assertRaises(ActionError) as exc:
                parse_action(json.dumps(value))
            self.assertTrue(exc.exception.parsed)


class EpisodeTest(unittest.IsolatedAsyncioTestCase):
    async def test_success_reward_and_close(self):
        env = FakeEnv()
        record = await run_episode(env, FakePolicy(), task_id="test", seed=42)
        self.assertTrue(env.closed)
        self.assertEqual(len(record["steps"]), 2)
        self.assertEqual(record["reward"], 1.0)
        self.assertTrue(record["success"])

    async def test_invalid_json_then_success(self):
        env = FakeEnv(success_at=1)
        record = await run_episode(env, FakePolicy(['bad', '{"action":"press","key":"Enter"}']), task_id="test", seed=0)
        self.assertEqual(env.steps, 1)
        self.assertEqual(record["reward"], 0.98)
        self.assertFalse(record["steps"][0]["parsed"])

    async def test_model_cannot_award_success(self):
        record = await run_episode(FakeEnv(success_at=99), FakePolicy(['{"action":"finish","status":"success"}']), task_id="test", seed=0)
        self.assertEqual(record["reward"], 0)
        self.assertEqual(record["failure_reason"], "premature_finish")

    async def test_browser_exception_is_recorded_and_closed(self):
        env = FakeEnv(crash=True)
        record = await run_episode(env, FakePolicy(), task_id="test", seed=0)
        self.assertTrue(env.closed)
        self.assertEqual(record["failure_reason"], "runtime:TimeoutError")
        self.assertEqual(record["reward"], 0)

    async def test_repeated_action_is_bounded(self):
        record = await run_episode(FakeEnv(success_at=99, repeat=True), FakePolicy(), task_id="test", seed=0)
        self.assertEqual(len(record["steps"]), 3)
        self.assertEqual(record["reward"], -0.11)
        self.assertEqual(record["failure_reason"], "action_loop")

    async def test_step_limit_and_reward_caps(self):
        record = await run_episode(FakeEnv(success_at=99), FakePolicy(), task_id="test", seed=0, max_steps=2)
        self.assertEqual(record["failure_reason"], "step_limit")
        self.assertEqual(episode_reward({"steps": [{"valid": False, "no_op": True}] * 100, "loop_detected": True}), -0.25)

    async def test_success_on_last_step_wins_over_gym_time_limit(self):
        class LastStepEnv(FakeEnv):
            async def step(self, action):
                result = await super().step(action)
                result["truncated"] = True
                return result
        record = await run_episode(LastStepEnv(success_at=1), FakePolicy(), task_id="test", seed=0, max_steps=1)
        self.assertTrue(record["success"])
        self.assertEqual(record["reward"], 1.0)
        self.assertIsNone(record["failure_reason"])


if __name__ == "__main__":
    unittest.main()
