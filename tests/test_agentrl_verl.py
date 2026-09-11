from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from toolvista.agentrl.dependencies import load_verl_api
from toolvista.agentrl.rollout import ContextLimit
from toolvista.agentrl.verl_agent_loop import VerlPolicy, create_agent_loop


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for screenshot/token alignment tests")
class VerlPolicyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image = Path(self.temp.name) / "image.png"
        Image.new("RGB", (32, 32), "white").save(self.image)
        self.observation = {"image_path": str(self.image), "goal": "Click", "step": 0}
        self.loop = SimpleNamespace(
            config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(prompt_length=10, response_length=20))),
            tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: '{"action":"press","key":"Enter"}'),
            process_vision_info=AsyncMock(return_value={"images": ["processed-image"]}),
            apply_chat_template=AsyncMock(side_effect=[[1, 2, 3], [8, 9]]),
            server_manager=SimpleNamespace(generate=AsyncMock(side_effect=[
                SimpleNamespace(token_ids=[4, 5], log_probs=[-0.1, -0.2]),
                SimpleNamespace(token_ids=[6, 7], log_probs=[-0.3, -0.4])]))
        )

    async def test_sampled_ids_survive_and_observations_are_masked(self):
        policy = VerlPolicy(self.loop, {"temperature": 1.0})
        await policy.generate(self.observation, [])
        await policy.generate(self.observation, ["previous"])
        self.assertEqual(policy.prompt_ids, [1, 2, 3])
        self.assertEqual(policy.response_ids, [4, 5, 8, 9, 6, 7])
        self.assertEqual(policy.response_mask, [1, 1, 0, 0, 1, 1])
        self.assertEqual(policy.action_spans, [(0, 2), (4, 6)])
        self.assertEqual(policy.logprobs, [-0.1, -0.2, 0, 0, -0.3, -0.4])
        self.assertEqual(len(policy.images), 2)
        calls = self.loop.server_manager.generate.call_args_list
        self.assertEqual(calls[1].kwargs["prompt_ids"], [1, 2, 3, 4, 5, 8, 9])
        self.assertEqual(calls[1].kwargs["sampling_params"]["max_tokens"], 16)
        self.assertTrue(self.loop.apply_chat_template.call_args.kwargs["remove_system_prompt"])

    async def test_context_exhaustion_does_not_append_unused_image(self):
        self.loop.config.actor_rollout_ref.rollout.response_length = 4
        policy = VerlPolicy(self.loop, {})
        await policy.generate(self.observation, [])
        with self.assertRaises(ContextLimit):
            await policy.generate(self.observation, ["previous"])
        self.assertEqual(policy.response_ids, [4, 5])
        self.assertEqual(len(policy.images), 1)
        self.assertEqual(self.loop.server_manager.generate.call_count, 1)

    async def test_logprob_mismatch_is_not_silently_accepted(self):
        self.loop.server_manager.generate = AsyncMock(return_value=SimpleNamespace(token_ids=[1, 2], log_probs=[-1.0]))
        with self.assertRaisesRegex(RuntimeError, "lengths differ"):
            await VerlPolicy(self.loop, {}).generate(self.observation, [])

    async def test_failed_followup_generation_rolls_back_observation(self):
        policy = VerlPolicy(self.loop, {})
        await policy.generate(self.observation, [])
        self.loop.server_manager.generate = AsyncMock(side_effect=TimeoutError("generation failed"))
        with self.assertRaises(TimeoutError):
            await policy.generate(self.observation, ["previous"])
        self.assertEqual(policy.response_ids, [4, 5])
        self.assertEqual(policy.response_mask, [1, 1])
        self.assertEqual(policy.action_spans, [(0, 2)])
        self.assertEqual(len(policy.images), 1)

    async def test_agent_output_transports_steps_without_reading_worker_files(self):
        import json
        from toolvista.agentrl.gigpo import validate_metadata
        loop = self.loop
        class Base:
            def __init__(self):
                self.__dict__.update(vars(loop))
                self.processor = object()
        first = {**self.observation, "image_sha256": "first", "state_anchor": "start"}
        next_obs = {**first, "image_sha256": "second", "state_anchor": "end"}
        env = SimpleNamespace(reset=AsyncMock(return_value=first), close=AsyncMock(),
            step=AsyncMock(return_value={"observation": next_obs, "terminated": True,
                "success": True, "environment_reward": 1.0}))
        api = SimpleNamespace(AgentLoopBase=Base, AgentLoopOutput=SimpleNamespace)
        with patch("toolvista.agentrl.verl_agent_loop.load_verl_api", return_value=api), patch(
                "toolvista.agentrl.verl_agent_loop.AsyncMiniWoBEnv", return_value=env):
            agent = create_agent_loop(artifact_dir=self.temp.name)
            output = await agent.run({}, extra_info={"task_id": "click", "seed": 100})
        self.assertEqual(output.reward_score, 1)
        metadata = output.extra_fields["gui_episode"]
        validate_metadata([metadata], [output.response_mask], [output.reward_score])
        self.assertEqual(metadata["steps"], [{"start": 0, "end": 2, "anchor": "start", "reward": 1}])
        path = Path(output.extra_fields["episode_path"]).with_name("training.json")
        self.assertEqual(json.loads(path.read_text()), metadata)
        env.close.assert_awaited_once()

    async def test_infrastructure_error_after_generation_cannot_become_training_failure(self):
        loop = self.loop
        class Base:
            def __init__(self):
                self.__dict__.update(vars(loop))
                self.processor = object()
        env = SimpleNamespace(reset=AsyncMock(return_value={**self.observation, "image_sha256": "x"}),
            step=AsyncMock(side_effect=TimeoutError("browser unavailable")), close=AsyncMock())
        api = SimpleNamespace(AgentLoopBase=Base, AgentLoopOutput=SimpleNamespace)
        with patch("toolvista.agentrl.verl_agent_loop.load_verl_api", return_value=api), patch(
                "toolvista.agentrl.verl_agent_loop.AsyncMiniWoBEnv", return_value=env):
            agent = create_agent_loop(artifact_dir=self.temp.name)
            with self.assertRaisesRegex(RuntimeError, "infrastructure failure"):
                await agent.run({}, extra_info={"task_id": "click", "seed": 100})
        env.close.assert_awaited_once()


class VerlImportTest(unittest.TestCase):
    def test_missing_or_changed_verl_fails_clearly(self):
        with patch("toolvista.agentrl.dependencies.metadata.version", return_value="0.6.0"):
            with self.assertRaisesRegex(RuntimeError, "expected 0.7.0"):
                load_verl_api()
        with patch("toolvista.agentrl.dependencies.check_versions"), patch(
                "toolvista.agentrl.dependencies.import_module", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "Incompatible verl AgentLoop API"):
                load_verl_api()

    def test_factory_constructs_real_subclass_with_forwarded_arguments(self):
        class Base:
            def __init__(self, *args, **kwargs):
                self.received = (args, kwargs)
        with patch("toolvista.agentrl.verl_agent_loop.load_verl_api", return_value=SimpleNamespace(AgentLoopBase=Base)):
            instance = create_agent_loop(trainer_config="trainer", server_manager="manager", name="toolvista_miniwob")
        self.assertIsInstance(instance, Base)
        self.assertEqual(instance.received[1]["trainer_config"], "trainer")


if __name__ == "__main__":
    unittest.main()
