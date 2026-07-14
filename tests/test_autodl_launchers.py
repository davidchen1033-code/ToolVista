from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTODL = REPO_ROOT / "scripts" / "autodl"


class AutoDLLauncherStaticContractTest(unittest.TestCase):
    def test_serial_sft_wrappers_delegate_to_common_runner(self) -> None:
        expected = {
            "05_train_xlam_sft.sh": ("c1-xlam", "xlam_train.jsonl"),
            "06_build_and_train_toolvqa_replay_sft.sh": ("c2-toolvqa-replay", "toolvqa_replay_train.jsonl"),
            "07_build_and_train_mllm_router_replay_sft.sh": ("c3-router-replay", "mllm_router_replay_train.jsonl"),
        }
        for filename, fragments in expected.items():
            with self.subTest(filename=filename):
                text = (AUTODL / filename).read_text(encoding="utf-8")
                self.assertIn("sft_common.sh", text)
                for fragment in fragments:
                    self.assertIn(fragment, text)

        c2 = (AUTODL / "06_build_and_train_toolvqa_replay_sft.sh").read_text(encoding="utf-8")
        self.assertIn("toolvqa=0.8", c2)
        self.assertIn("xlam=0.2", c2)
        self.assertIn("C1_CHECKPOINT", c2)

        c3 = (AUTODL / "07_build_and_train_mllm_router_replay_sft.sh").read_text(encoding="utf-8")
        self.assertIn("mllm=0.6", c3)
        self.assertIn("toolvqa=0.2", c3)
        self.assertIn("xlam=0.2", c3)
        self.assertIn("C2_CHECKPOINT", c3)

    def test_sft_common_loads_only_previous_model_weights(self) -> None:
        text = (AUTODL / "sft_common.sh").read_text(encoding="utf-8")
        for required in (
            '"${SWIFT_BIN}" sft',
            '--resume_from_checkpoint',
            '--resume_only_model true',
            '--load_args false',
            '--report_to "${REPORT_TO}"',
            'WANDB_PROJECT',
            'MAX_STEPS',
        ):
            self.assertIn(required, text)

    def test_rl_wrappers_only_select_branch_configuration(self) -> None:
        expected = {
            "08_train_visual_rft_grpo.sh": "grpo",
            "09_train_visual_rft_bnpo.sh": "bnpo",
            "10_train_visual_rft_dapo.sh": "dapo",
            "11_train_visual_rft_drgrpo.sh": "dr_grpo",
        }
        for filename, loss_type in expected.items():
            with self.subTest(filename=filename):
                text = (AUTODL / filename).read_text(encoding="utf-8")
                self.assertIn(f'LOSS_TYPE="{loss_type}"', text)
                self.assertIn("rl_common.sh", text)

        dapo = (AUTODL / "10_train_visual_rft_dapo.sh").read_text(encoding="utf-8")
        for required in ("--epsilon_high 0.28", "--dynamic_sample true", "--max_resample_times 3", "--overlong_filter true"):
            self.assertIn(required, dapo)
        self.assertNotIn("soft_overlong", dapo)

        drgrpo = (AUTODL / "11_train_visual_rft_drgrpo.sh").read_text(encoding="utf-8")
        self.assertIn("--scale_rewards none", drgrpo)

    def test_rl_common_uses_same_c3_policy_reference_data_and_reward(self) -> None:
        text = (AUTODL / "rl_common.sh").read_text(encoding="utf-8")
        for required in (
            '"${SWIFT_BIN}" rlhf',
            '--rlhf_type grpo',
            '--model "${BASE_MODEL}"',
            '--adapters "${C3_CHECKPOINT}"',
            '--ref_adapters "${C3_CHECKPOINT}"',
            '--external_plugins "${REWARD_PLUGIN}"',
            'REWARD_FUNCS="${REWARD_FUNCS:-visual_rft format}"',
            '--reward_funcs "${reward_func_args[@]}"',
            '--loss_type "${LOSS_TYPE}"',
            '--seed "${SEED}"',
            '--report_to "${REPORT_TO}"',
            '--log_entropy true',
            '--log_rollout_offpolicy_metrics true',
            '--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"',
            '--offload_model "${OFFLOAD_MODEL}"',
            '--offload_optimizer "${OFFLOAD_OPTIMIZER}"',
        ):
            self.assertIn(required, text)

        self.assertIn("validate_visual_rft_rl_dataset.py", text)
        self.assertIn('"${PYTHON_BIN}" -c "import vllm"', text)

    def test_light_eval_wrapper_delegates_to_unified_evaluator(self) -> None:
        text = (AUTODL / "12_run_light_eval.sh").read_text(encoding="utf-8")
        for required in (
            'scripts/evaluate.py',
            '--suite "${SUITE}"',
            '--checkpoint "${CHECKPOINT}"',
            '--run-id "${RUN_ID}"',
            '--stage "${STAGE}"',
            '--api-base "${API_BASE}"',
            '--api-model "${API_MODEL}"',
            '--wandb-group "${WANDB_GROUP}"',
        ):
            self.assertIn(required, text)


def find_bash() -> Path | None:
    discovered = shutil.which("bash")
    candidates = [
        Path(discovered) if discovered else None,
        Path("F:/software/Git/bin/bash.exe"),
        Path("F:/software/Git/usr/bin/bash.exe"),
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


@unittest.skipUnless(os.environ.get("TOOLVISTA_RUN_BASH_TESTS") == "1", "set TOOLVISTA_RUN_BASH_TESTS=1")
class AutoDLLauncherRuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bash = find_bash()
        if self.bash is None:
            self.skipTest("bash is unavailable")
        self.root = REPO_ROOT / "tests" / f"tmp-launcher-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)
        self.ms_swift = self.root / "ms-swift"
        self.model = self.root / "model"
        self.c1 = self.root / "c1-checkpoint"
        self.c2 = self.root / "c2-checkpoint"
        self.c3 = self.root / "c3-checkpoint"
        for directory in (self.ms_swift, self.model, self.c1, self.c2, self.c3):
            directory.mkdir()
        self.log = self.root / "command.jsonl"
        self.fake_python = self.make_recorder("fake-python.sh", "python")
        self.fake_swift = self.make_recorder("fake-swift.sh", "swift")

    def make_recorder(self, name: str, command: str) -> Path:
        path = self.root / name
        path.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"python -c 'import json,os,sys; print(json.dumps({{\"command\":\"{command}\",\"args\":sys.argv[1:],\"wandb_group\":os.environ.get(\"WANDB_RUN_GROUP\")}}))' \"$@\" >> \"$COMMAND_LOG\"\n",
            encoding="utf-8",
            newline="\n",
        )
        path.chmod(0o755)
        return path

    @staticmethod
    def posix(path: Path) -> str:
        resolved = path.resolve()
        if os.name == "nt":
            return f"/{resolved.drive[0].lower()}{resolved.as_posix()[2:]}"
        return resolved.as_posix()

    def write_rows(self, name: str, count: int, *, rl: bool = False) -> Path:
        path = self.root / name
        rows = []
        for index in range(count):
            row = {"messages": [{"role": "user", "content": f"prompt-{index}"}]}
            if rl:
                row.update(
                    {
                        "images": ["image.jpg"],
                        "solution": '<answer>[{"Position":[0,0,10,10],"Confidence":1}]</answer>',
                        "task": "grounding",
                    }
                )
            rows.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def run_script(self, filename: str, extra_env: dict[str, str]) -> list[dict]:
        env = os.environ.copy()
        env.update(
            {
                "TOOLVISTA_ROOT": self.posix(REPO_ROOT),
                "MS_SWIFT_ROOT": self.posix(self.ms_swift),
                "BASE_MODEL": self.posix(self.model),
                "PYTHON_BIN": self.posix(self.fake_python),
                "SWIFT_BIN": self.posix(self.fake_swift),
                "COMMAND_LOG": self.posix(self.log),
                "OUTPUT_DIR": self.posix(self.root / "output"),
                "REPORT_TO": "wandb",
            }
        )
        env.update(extra_env)
        process = subprocess.run(
            [str(self.bash), self.posix(AUTODL / filename)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        commands = [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]
        self.log.unlink()
        return commands

    @staticmethod
    def option(args: list[str], name: str) -> str:
        return args[args.index(name) + 1]

    def test_c1_c2_c3_expand_expected_resume_and_replay_arguments(self) -> None:
        xlam = self.write_rows("xlam.jsonl", 3)
        toolvqa = self.write_rows("toolvqa.jsonl", 4)
        mllm = self.write_rows("mllm.jsonl", 6)

        c1_commands = self.run_script("05_train_xlam_sft.sh", {"XLAM_DATASET": self.posix(xlam)})
        c1_args = next(item["args"] for item in c1_commands if item["command"] == "swift")
        self.assertEqual(c1_args[0], "sft")
        self.assertNotIn("--resume_from_checkpoint", c1_args)

        c2_output = self.root / "c2-replay.jsonl"
        c2_commands = self.run_script(
            "06_build_and_train_toolvqa_replay_sft.sh",
            {
                "C1_CHECKPOINT": self.posix(self.c1),
                "XLAM_DATASET": self.posix(xlam),
                "TOOLVQA_DATASET": self.posix(toolvqa),
                "REPLAY_DATASET": self.posix(c2_output),
                "REPLAY_REPORT": self.posix(self.root / "c2-report.json"),
            },
        )
        c2_args = next(item["args"] for item in c2_commands if item["command"] == "swift")
        self.assertEqual(self.option(c2_args, "--resume_from_checkpoint"), self.posix(self.c1))
        self.assertEqual(self.option(c2_args, "--resume_only_model"), "true")
        self.assertEqual(len(c2_output.read_text(encoding="utf-8").splitlines()), 5)

        c3_output = self.root / "c3-replay.jsonl"
        c3_commands = self.run_script(
            "07_build_and_train_mllm_router_replay_sft.sh",
            {
                "C2_CHECKPOINT": self.posix(self.c2),
                "XLAM_DATASET": self.posix(xlam),
                "TOOLVQA_DATASET": self.posix(toolvqa),
                "MLLM_DATASET": self.posix(mllm),
                "REPLAY_DATASET": self.posix(c3_output),
                "REPLAY_REPORT": self.posix(self.root / "c3-report.json"),
            },
        )
        c3_args = next(item["args"] for item in c3_commands if item["command"] == "swift")
        self.assertEqual(self.option(c3_args, "--resume_from_checkpoint"), self.posix(self.c2))
        self.assertEqual(len(c3_output.read_text(encoding="utf-8").splitlines()), 10)

    def test_all_rl_branches_share_c3_and_only_change_algorithm_arguments(self) -> None:
        dataset = self.write_rows("visual-rft.jsonl", 2, rl=True)
        branches = {
            "08_train_visual_rft_grpo.sh": "grpo",
            "09_train_visual_rft_bnpo.sh": "bnpo",
            "10_train_visual_rft_dapo.sh": "dapo",
            "11_train_visual_rft_drgrpo.sh": "dr_grpo",
        }
        common_values = []
        for filename, loss_type in branches.items():
            commands = self.run_script(
                filename,
                {
                    "C3_CHECKPOINT": self.posix(self.c3),
                    "DATASET": self.posix(dataset),
                },
            )
            args = next(item["args"] for item in commands if item["command"] == "swift")
            self.assertEqual(args[:2], ["rlhf", "--rlhf_type"])
            self.assertEqual(self.option(args, "--loss_type"), loss_type)
            self.assertEqual(self.option(args, "--adapters"), self.posix(self.c3))
            self.assertEqual(self.option(args, "--ref_adapters"), self.posix(self.c3))
            reward_index = args.index("--reward_funcs")
            self.assertEqual(args[reward_index + 1:reward_index + 3], ["visual_rft", "format"])
            common_values.append(
                (
                    self.option(args, "--dataset"),
                    self.option(args, "--seed"),
                    self.option(args, "--num_generations"),
                    self.option(args, "--max_completion_length"),
                )
            )
            if loss_type == "dapo":
                self.assertEqual(self.option(args, "--epsilon_high"), "0.28")
                self.assertEqual(self.option(args, "--dynamic_sample"), "true")
            if loss_type == "dr_grpo":
                self.assertEqual(self.option(args, "--scale_rewards"), "none")
        self.assertEqual(len(set(common_values)), 1)


if __name__ == "__main__":
    unittest.main()
