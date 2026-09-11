from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.test_agentrl_checkpoint import write_model
from toolvista.agentrl.checkpoints import export_checkpoint
from toolvista.agentrl.launch import launch_plan, main
from toolvista.agentrl.miniwob_env import AsyncMiniWoBEnv, MiniWoBEnv, local_miniwob_url

ROOT = Path(__file__).resolve().parents[1]


class EnvTest(unittest.TestCase):
    def test_only_local_asset_urls_are_accepted(self):
        self.assertEqual(local_miniwob_url("http://127.0.0.1:8000"), "http://127.0.0.1:8000/")
        self.assertEqual(local_miniwob_url("file:///tmp/miniwob/"), "file:///tmp/miniwob/")
        for url in ["https://example.com/", "", "file://remote/share/", "http://localhost:8000/?x=y"]:
            with patch.dict(os.environ, {"MINIWOB_URL": ""}):
                with self.assertRaises(ValueError):
                    local_miniwob_url(url)

    def test_finish_calls_evaluator_without_trusting_status(self):
        class GymEnv:
            def step(self, command):
                self.command = command
                return {"last_action_error": ""}, 0, False, False, {}
        env = MiniWoBEnv("unused", base_url="file:///tmp/miniwob/")
        env.env = GymEnv()
        env.last = {"width": 500, "height": 320}
        with patch.object(env, "_observation", return_value={"feedback": "ready"}):
            result = env.step({"action": "finish", "status": "success"})
        self.assertEqual(env.env.command, "noop(0)")
        self.assertFalse(result["success"])


class LaunchTest(unittest.TestCase):
    def test_training_never_starts_when_fresh_gate_fails(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            output = root / "new-run"
            plan = {"environment": {}, "checkpoint_manifest_sha256": "digest",
                    "gate_command": ["gate"], "training_command": ["training"]}
            args = ["agentrl", "--mode", "train", "--checkpoint", str(root / "model"),
                    "--data-dir", str(root / "data"), "--output", str(output)]
            stack.enter_context(patch.object(sys, "argv", args))
            stack.enter_context(patch("toolvista.agentrl.launch.launch_plan", return_value=plan))
            for target in ("check_versions", "load_verl_api", "check_chromium", "local_miniwob_url", "validate_dataset", "file_info"):
                stack.enter_context(patch(f"toolvista.agentrl.launch.{target}", return_value={}))
            run = stack.enter_context(patch("toolvista.agentrl.launch.subprocess.run",
                                            side_effect=subprocess.CalledProcessError(1, ["gate"])))
            with self.assertRaisesRegex(SystemExit, "launch refused/failed"):
                main()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0], ["gate"])

    def test_dependency_error_happens_before_creating_run(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            output = Path(temp) / "new-run"
            args = ["agentrl", "--mode", "train", "--checkpoint", temp, "--data-dir", temp, "--output", str(output)]
            stack.enter_context(patch.object(sys, "argv", args))
            stack.enter_context(patch("toolvista.agentrl.launch.launch_plan", return_value={}))
            stack.enter_context(patch("toolvista.agentrl.launch.check_versions", side_effect=RuntimeError("Missing verl")))
            run = stack.enter_context(patch("toolvista.agentrl.launch.subprocess.run"))
            with self.assertRaisesRegex(SystemExit, "Missing verl"):
                main()
            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_dry_run_builds_separate_verl_command_and_rejects_reuse(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, adapter, merged, data = [root / n for n in ("Qwen3-VL", "adapter", "merged", "data")]
            write_model(base)
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "base_model_name_or_path": str(base)}))
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            def exporter(command, **kwargs):
                write_model(Path(command[-1]))
            export_checkpoint(base, adapter, merged, runner=exporter)
            data.mkdir()
            (data / "train.parquet").write_bytes(b"fixture")
            (data / "val.parquet").write_bytes(b"fixture")
            output = root / "run"
            plan = launch_plan(merged, data, output, mode="dry-run")
            self.assertIn("verl.trainer.main_ppo", plan["training_command"])
            second = launch_plan(merged, data, root / "gigpo", mode="dry-run", algorithm="gigpo")
            self.assertIn("toolvista.agentrl.train_gigpo", second["training_command"])
            self.assertIn("miniwob_gigpo", second["training_command"])
            self.assertEqual(second["reward_mode"], plan["reward_mode"])
            self.assertEqual(second["environment"]["AGENTRL_TRAIN_DATA"], plan["environment"]["AGENTRL_TRAIN_DATA"])
            self.assertIn("--gate", plan["gate_command"])
            self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaises(FileExistsError):
                launch_plan(merged, data, output, mode="train")
            (data / "train.parquet").unlink()
            with self.assertRaisesRegex(ValueError, "missing train.parquet"):
                launch_plan(merged, data, root / "new-run", mode="dry-run")

    def test_new_shell_scripts_parse(self):
        from tests.test_autodl_launchers import find_bash
        bash = find_bash()
        if bash is None:
            self.skipTest("Bash unavailable")
        for name in ("train_agent_rl.sh", "train_agent_gigpo.sh", "train_verl_agent.sh", "setup_verl_agent.sh",
                     "download_verl_agent.sh", "setup_env.sh"):
            result = subprocess.run([str(bash), "-n", str(ROOT / "scripts/autodl" / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(os.environ.get("TOOLVISTA_RUN_MINIWOB_TESTS") == "1",
                     "optional BrowserGym/Chromium smoke: set TOOLVISTA_RUN_MINIWOB_TESTS=1 and MINIWOB_URL")
class RealMiniWoBSmoke(unittest.IsolatedAsyncioTestCase):
    async def test_reset_step_close(self):
        with tempfile.TemporaryDirectory() as temp:
            env = AsyncMiniWoBEnv(temp)
            try:
                observation = await env.reset(task_id="browsergym/miniwob.click-test", seed=0)
                self.assertTrue(Path(observation["image_path"]).is_file())
                self.assertNotIn("dom_object", observation)
                result = await env.step({"action": "press", "key": "Tab"})
                self.assertIn("environment_reward", result)
            finally:
                await env.close()


if __name__ == "__main__":
    unittest.main()
