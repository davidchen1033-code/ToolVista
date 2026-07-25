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
    def test_repository_forces_shell_scripts_to_lf(self) -> None:
        attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attributes)

    def test_setup_checks_a_supported_swift_route(self) -> None:
        text = (AUTODL / "setup_env.sh").read_text(encoding="utf-8")
        self.assertIn('"${SWIFT_BIN}" sft --help', text)
        self.assertNotIn("swift --help", text)

    def test_autodl_defaults_use_verified_venv_and_hf_mirror(self) -> None:
        for filename in ("setup_env.sh", "lib/sft_common.sh", "lib/rl_common.sh"):
            with self.subTest(filename=filename):
                text = (AUTODL / filename).read_text(encoding="utf-8")
                self.assertIn("/root/autodl-tmp/venvs/toolvista/bin/python", text)
                self.assertIn("https://hf-mirror.com", text)
        for filename in ("lib/sft_common.sh", "lib/rl_common.sh"):
            text = (AUTODL / filename).read_text(encoding="utf-8")
            self.assertIn("/root/autodl-tmp/venvs/toolvista/bin/swift", text)

    def test_serial_sft_wrappers_delegate_to_common_runner(self) -> None:
        expected = {
            "lib/train_c1_raw.sh": ("c1-xlam", "xlam_train.jsonl"),
            "lib/train_c2_raw.sh": ("c2-toolvqa-replay", "toolvqa_replay_train.jsonl"),
            "lib/train_c3_raw.sh": ("c3-router-v2-retention", "mllm_router_v2_replay_train.jsonl"),
        }
        for filename, fragments in expected.items():
            with self.subTest(filename=filename):
                text = (AUTODL / filename).read_text(encoding="utf-8")
                self.assertIn("sft_common.sh", text)
                for fragment in fragments:
                    self.assertIn(fragment, text)

        c2 = (AUTODL / "lib/train_c2_raw.sh").read_text(encoding="utf-8")
        self.assertIn("toolvqa=0.8", c2)
        self.assertIn("xlam=0.2", c2)
        self.assertIn("C1_CHECKPOINT", c2)

        c3 = (AUTODL / "lib/train_c3_raw.sh").read_text(encoding="utf-8")
        self.assertIn('MLLM_WEIGHT="${MLLM_WEIGHT:-0.4}"', c3)
        self.assertIn('DIRECT_ANSWER_WEIGHT="${DIRECT_ANSWER_WEIGHT:-0.3}"', c3)
        self.assertIn('TOOLVQA_WEIGHT="${TOOLVQA_WEIGHT:-0.15}"', c3)
        self.assertIn('XLAM_WEIGHT="${XLAM_WEIGHT:-0.15}"', c3)
        self.assertIn("C2_CHECKPOINT", c3)
        self.assertIn("mllm_tool_router_v2_train.jsonl", c3)
        self.assertIn("toolvqa_direct_answer_retention_train.jsonl", c3)
        self.assertIn('LEARNING_RATE="${LEARNING_RATE:-2e-5}"', c3)
        self.assertIn('MAX_STEPS="${MAX_STEPS:-800}"', c3)
        self.assertIn('SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"', c3)

    def test_sft_common_loads_only_previous_model_weights(self) -> None:
        text = (AUTODL / "lib/sft_common.sh").read_text(encoding="utf-8")
        for required in (
            '"${SWIFT_BIN}" sft',
            '--adapters "${PREVIOUS_CHECKPOINT}"',
            '--load_args false',
            'adapter_model.safetensors',
            'REQUIRE_NEW_ADAPTER',
            'cmp -s',
            '--report_to "${REPORT_TO}"',
            'WANDB_PROJECT',
            'MAX_STEPS',
        ):
            self.assertIn(required, text)

    def test_unified_rl_launcher_selects_only_algorithm_configuration(self) -> None:
        text = (AUTODL / "train_rl.sh").read_text(encoding="utf-8")
        for required in (
            "grpo) loss_type=grpo",
            "bnpo) loss_type=bnpo",
            "dapo) loss_type=dapo",
            "drgrpo) loss_type=dr_grpo",
            "--epsilon_high 0.28",
            "--dynamic_sample true",
            "--max_resample_times 3",
            "--overlong_filter true",
            "--scale_rewards none",
            "scripts/autodl/lib/rl_common.sh",
        ):
            self.assertIn(required, text)
        self.assertNotIn("soft_overlong", text)

    def test_rl_common_uses_same_c3_policy_reference_data_and_reward(self) -> None:
        text = (AUTODL / "lib/rl_common.sh").read_text(encoding="utf-8")
        for required in (
            '"${SWIFT_BIN}" rlhf',
            '--rlhf_type grpo',
            '--model "${BASE_MODEL}"',
            '--adapters "${C3_CHECKPOINT}"',
            '--ref_adapters "${C3_CHECKPOINT}"',
            'visual_rft_coco_base65_train.jsonl',
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
        self.assertIn("verify_c3_retention_gate.py", text)
        self.assertIn("C3_RETENTION_GATE", text)

    def test_light_eval_wrapper_delegates_to_unified_evaluator(self) -> None:
        text = (AUTODL / "lib/evaluate_raw.sh").read_text(encoding="utf-8")
        for required in (
            'SUITE="${SUITE:?',
            'ALGORITHM="${ALGORITHM:-}"',
            'TOOLVQA_PYTHON="${TOOLVQA_PYTHON:-/root/autodl-tmp/venvs/toolvqa/bin/python}"',
            'BFCL_PYTHON="${BFCL_PYTHON:-/root/autodl-tmp/venvs/bfcl/bin/python}"',
            'from opencompass.datasets.gta_bench import GTABenchEvaluator',
            'export TOOLVQA_PYTHON',
            'export BFCL_PYTHON',
            'algorithm_args+=(--algorithm "$ALGORITHM")',
            'scripts/evaluate.py',
            '--suite "${SUITE}"',
            '--checkpoint "${CHECKPOINT}"',
            '--run-id "${RUN_ID}"',
            '--stage "${STAGE}"',
            '--api-base "${API_BASE}"',
            '--api-model "${API_MODEL}"',
            'JUDGE_API_BASE="${JUDGE_API_BASE:?',
            'JUDGE_MODEL="${JUDGE_MODEL:?',
            '--judge-api-base "${JUDGE_API_BASE}"',
            '--judge-model "${JUDGE_MODEL}"',
            'patch_msswift_vlmeval.sh',
            '--wandb-group "${WANDB_GROUP}"',
        ):
            self.assertIn(required, text)
        self.assertIn("LMUData", text)
        self.assertIn("/root/autodl-tmp/cache/vlmevalkit", text)
        self.assertIn("check_c3_retention.py", text)
        self.assertIn("C3_ADAPTER_CHECKPOINT", text)
        self.assertIn('CHECKPOINT must equal C3_ADAPTER_CHECKPOINT', text)
        self.assertIn('rm -f "${C3_RETENTION_GATE}"', text)

    def test_unified_benchmark_selects_raw_and_datafixed_configs(self) -> None:
        text = (AUTODL / "benchmark.sh").read_text(encoding="utf-8")
        self.assertIn("configs/evaluation/benchmarks.yaml", text)
        self.assertIn("configs/evaluation/benchmarks_datafixed.yaml", text)
        self.assertIn("scripts/autodl/lib/evaluate_raw.sh", text)
        self.assertIn("scripts/autodl/lib/evaluate_c4_datafixed.sh", text)
        self.assertIn("c4_full", text)
        self.assertIn("c4_datafixed_full", text)
        self.assertIn("c4_datafixed_agentretention_full", text)
        self.assertIn('C3 benchmark supports --scope light only', text)

    def test_c0_preparation_wrapper_runs_split_then_readiness(self) -> None:
        text = (AUTODL / "lib/prepare_c0_raw.sh").read_text(encoding="utf-8")
        self.assertIn("prepare_training_and_eval_data.py", text)
        self.assertIn("prepare_c3_direct_answer_retention.py", text)
        self.assertIn("prepare_visual_rft_coco_eval.py", text)
        self.assertIn("prepare_bfcl_eval.py", text)
        self.assertIn("prefetch_vlmevalkit_data.py", text)
        self.assertIn("check_c0_readiness.py", text)
        self.assertIn("--mllm-file-keys text image video", text)
        self.assertIn("--overwrite", text)

    def test_datafixed_launchers_validate_lexical_and_resolved_paths(self) -> None:
        for filename in (
            "lib/prepare_c3_datafixed.sh",
            "lib/train_c3_datafixed.sh",
            "lib/evaluate_c3_datafixed.sh",
        ):
            with self.subTest(filename=filename):
                text = (AUTODL / filename).read_text(encoding="utf-8")
                self.assertIn("require_datafixed_path", text)
                self.assertIn('realpath -m -- "${value}"', text)
                self.assertIn("resolves outside the datafixed namespace", text)
                self.assertIn("has_datafixed_marker", text)

        train = (AUTODL / "lib/train_c3_datafixed.sh").read_text(encoding="utf-8")
        self.assertIn('MAX_STEPS="$((10#${max_steps_decimal}))"', train)
        self.assertNotIn("(( MAX_STEPS > 800 ))", train)

        evaluate = (AUTODL / "lib/evaluate_c3_datafixed.sh").read_text(encoding="utf-8")
        self.assertIn("require_datafixed_run_id", evaluate)
        self.assertIn('run_root_resolved="$(realpath -m -- "${TOOLVISTA_ROOT}/runs")"', evaluate)
        self.assertIn("C3_DATAFIXED_RETENTION_GATE must be inside", evaluate)

    def test_datafixed_usage_assigns_artifacts_to_the_creating_step(self) -> None:
        text = (REPO_ROOT / "docs" / "c3_datafixed_usage.md").read_text(encoding="utf-8")
        prepare, train = text.split("## 2. 从 C2 训练", 1)
        self.assertNotIn("mllm_router_datafixed_replay_train.jsonl", prepare)
        self.assertIn("mllm_router_datafixed_replay_train.jsonl", train)
        self.assertIn("PYTHON_BIN=/root/autodl-tmp/venvs/toolvista/bin/python", text)
        self.assertIn('"${PYTHON_BIN}" scripts/evaluate.py', text)


def find_bash() -> Path | None:
    discovered = shutil.which("bash")
    candidates = [
        Path(discovered) if discovered else None,
        Path("F:/software/Git/bin/bash.exe"),
        Path("F:/software/Git/usr/bin/bash.exe"),
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


@unittest.skipUnless(
    os.name != "nt" and find_bash() is not None and shutil.which("realpath") is not None,
    "requires POSIX bash and realpath",
)
class AutoDLDatafixedPathRuntimeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bash = find_bash()
        assert self.bash is not None
        self.root = REPO_ROOT / "tests" / f"tmp-launcher-security-{uuid.uuid4().hex}"
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root, True)

    def run_rejected(self, filename: str, extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(extra_env)
        process = subprocess.run(
            [str(self.bash), str(AUTODL / filename)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.assertEqual(process.returncode, 2, process.stderr)
        return process

    @staticmethod
    def write_nonempty(path: Path, content: str = "{}\n") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_prepare_rejects_parent_traversal_after_datafixed_component(self) -> None:
        project = self.root / "project"
        project.mkdir()
        config = self.write_nonempty(project / "autodl.yaml", "seed: 42\n")
        direct = self.write_nonempty(project / "toolvqa_train.jsonl")
        control_output = project / "mllm_tool_router_train.jsonl"
        lexical_bypass = project / "datafixed" / ".." / control_output.name

        process = self.run_rejected(
            "lib/prepare_c3_datafixed.sh",
            {
                "TOOLVISTA_ROOT": str(project),
                "PYTHON_BIN": str(self.bash),
                "CONFIG": str(config),
                "DIRECT_SOURCE": str(direct),
                "ROUTER_TRAIN": str(lexical_bypass),
            },
        )

        self.assertIn("ROUTER_TRAIN resolves outside the datafixed namespace", process.stderr)
        self.assertFalse(control_output.exists())

    def test_train_rejects_symlinked_control_output_and_parses_leading_zero_as_decimal(self) -> None:
        project = self.root / "project"
        project.mkdir()
        c2_checkpoint = project / "c2-checkpoint"
        c2_checkpoint.mkdir()
        router = self.write_nonempty(project / "router_datafixed.jsonl")
        direct = self.write_nonempty(project / "direct_datafixed.jsonl")
        prep_report = self.write_nonempty(project / "prep_datafixed.report.json")
        toolvqa = self.write_nonempty(project / "toolvqa.jsonl")
        xlam = self.write_nonempty(project / "xlam.jsonl")
        control_output = self.write_nonempty(project / "mllm_router_replay_train.jsonl", "sentinel\n")
        replay_link = project / "mllm_router_datafixed_replay_train.jsonl"
        replay_link.symlink_to(control_output)

        process = self.run_rejected(
            "lib/train_c3_datafixed.sh",
            {
                "TOOLVISTA_ROOT": str(project),
                "C2_CHECKPOINT": str(c2_checkpoint),
                "ROUTER_DATASET": str(router),
                "DIRECT_ANSWER_DATASET": str(direct),
                "TOOLVQA_DATASET": str(toolvqa),
                "XLAM_DATASET": str(xlam),
                "DATAFIXED_PREP_REPORT": str(prep_report),
                "REPLAY_DATASET": str(replay_link),
                "REPLAY_REPORT": str(project / "replay_datafixed.report.json"),
                "OUTPUT_DIR": str(project / "output_datafixed"),
                "MAX_STEPS": "008",
            },
        )

        self.assertIn("REPLAY_DATASET resolves outside the datafixed namespace", process.stderr)
        self.assertEqual(control_output.read_text(encoding="utf-8"), "sentinel\n")

    def test_eval_rejects_run_id_parent_traversal(self) -> None:
        checkpoint = self.root / "checkpoint_datafixed"
        checkpoint.mkdir()
        history_dir = self.root / "c2_datafixed_run"
        history_summary = self.write_nonempty(history_dir / "summary.json")
        self.write_nonempty(history_dir / "manifest.json")
        control_run = f"control_run_{uuid.uuid4().hex}"

        process = self.run_rejected(
            "lib/evaluate_c3_datafixed.sh",
            {
                "TOOLVISTA_ROOT": str(REPO_ROOT),
                "CHECKPOINT": str(checkpoint),
                "C3_DATAFIXED_ADAPTER_CHECKPOINT": str(checkpoint),
                "RUN_ID": f"candidate_datafixed/../{control_run}",
                "API_MODEL": "candidate-datafixed",
                "HISTORY_SUMMARY": str(history_summary),
                "JUDGE_API_BASE": "http://127.0.0.1:8011/v1",
                "JUDGE_MODEL": "judge-datafixed",
            },
        )

        self.assertIn("RUN_ID must be one datafixed path component", process.stderr)
        self.assertFalse((REPO_ROOT / "runs" / control_run).exists())

    def test_eval_rejects_datafixed_symlink_to_control_checkpoint(self) -> None:
        control_checkpoint = self.root / "control-checkpoint"
        control_checkpoint.mkdir()
        checkpoint_link = self.root / "checkpoint_datafixed"
        checkpoint_link.symlink_to(control_checkpoint, target_is_directory=True)
        history_dir = self.root / "c2_datafixed_run"
        history_summary = self.write_nonempty(history_dir / "summary.json")
        self.write_nonempty(history_dir / "manifest.json")

        process = self.run_rejected(
            "lib/evaluate_c3_datafixed.sh",
            {
                "TOOLVISTA_ROOT": str(REPO_ROOT),
                "CHECKPOINT": str(checkpoint_link),
                "C3_DATAFIXED_ADAPTER_CHECKPOINT": str(checkpoint_link),
                "RUN_ID": "candidate_datafixed",
                "API_MODEL": "candidate-datafixed",
                "HISTORY_SUMMARY": str(history_summary),
                "JUDGE_API_BASE": "http://127.0.0.1:8011/v1",
                "JUDGE_MODEL": "judge-datafixed",
            },
        )

        self.assertIn("CHECKPOINT resolves outside the datafixed namespace", process.stderr)


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
        (self.c3 / "toolvista_c3_retention_gate.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "authorized_checkpoint": str(self.c3.resolve()),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (self.c3 / "adapter_model.safetensors").write_bytes(b"fixture-adapter\n")
        self.log = self.root / "command.jsonl"
        self.fake_python = self.make_recorder("fake-python.sh", "python")
        self.fake_swift = self.make_recorder("fake-swift.sh", "swift")

    def make_recorder(self, name: str, command: str) -> Path:
        delegate = (
            "if [[ \"${1:-}\" == */build_sft_replay.py ]]; then\n"
            "  python \"$@\"\n"
            "fi\n"
            if command == "python"
            else ""
        )
        script = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"python -c 'import json,os,sys; print(json.dumps({{\"command\":\"{command}\",\"args\":sys.argv[1:],\"wandb_group\":os.environ.get(\"WANDB_RUN_GROUP\")}}))' \"$@\" >> \"$COMMAND_LOG\"\n"
            + delegate
        )
        path = self.root / name
        path.write_text(
            script,
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

    @staticmethod
    def normalized_bash_path(value: str) -> str:
        value = value.replace("\\", "/")
        if os.name == "nt" and len(value) >= 3 and value[1:3] == ":/":
            return f"/{value[0].lower()}{value[2:]}"
        return value

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

    def run_script(
        self,
        filename: str,
        extra_env: dict[str, str],
        args: tuple[str, ...] = (),
    ) -> list[dict]:
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
                "REQUIRE_NEW_ADAPTER": "false",
                "REPORT_TO": "wandb",
            }
        )
        env.update(extra_env)
        process = subprocess.run(
            [str(self.bash), self.posix(AUTODL / filename), *args],
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
        direct = self.write_rows("direct.jsonl", 3)

        c1_commands = self.run_script("lib/train_c1_raw.sh", {"XLAM_DATASET": self.posix(xlam)})
        c1_args = next(item["args"] for item in c1_commands if item["command"] == "swift")
        self.assertEqual(c1_args[0], "sft")
        self.assertNotIn("--resume_from_checkpoint", c1_args)

        c2_output = self.root / "c2-replay.jsonl"
        c2_commands = self.run_script(
            "lib/train_c2_raw.sh",
            {
                "C1_CHECKPOINT": self.posix(self.c1),
                "XLAM_DATASET": self.posix(xlam),
                "TOOLVQA_DATASET": self.posix(toolvqa),
                "REPLAY_DATASET": self.posix(c2_output),
                "REPLAY_REPORT": self.posix(self.root / "c2-report.json"),
            },
        )
        c2_args = next(item["args"] for item in c2_commands if item["command"] == "swift")
        self.assertEqual(
            self.normalized_bash_path(self.option(c2_args, "--adapters")),
            self.posix(self.c1),
        )
        self.assertNotIn("--resume_from_checkpoint", c2_args)
        self.assertNotIn("--resume_only_model", c2_args)
        self.assertEqual(len(c2_output.read_text(encoding="utf-8").splitlines()), 5)

        c3_output = self.root / "c3-replay.jsonl"
        c3_commands = self.run_script(
            "lib/train_c3_raw.sh",
            {
                "C2_CHECKPOINT": self.posix(self.c2),
                "XLAM_DATASET": self.posix(xlam),
                "TOOLVQA_DATASET": self.posix(toolvqa),
                "MLLM_DATASET": self.posix(mllm),
                "DIRECT_ANSWER_DATASET": self.posix(direct),
                "REPLAY_DATASET": self.posix(c3_output),
                "REPLAY_REPORT": self.posix(self.root / "c3-report.json"),
            },
        )
        c3_args = next(item["args"] for item in c3_commands if item["command"] == "swift")
        self.assertEqual(
            self.normalized_bash_path(self.option(c3_args, "--adapters")),
            self.posix(self.c2),
        )
        self.assertNotIn("--resume_from_checkpoint", c3_args)
        self.assertEqual(len(c3_output.read_text(encoding="utf-8").splitlines()), 14)

    def test_all_rl_branches_share_c3_and_only_change_algorithm_arguments(self) -> None:
        dataset = self.write_rows("visual-rft.jsonl", 2, rl=True)
        branches = {
            "grpo": "grpo",
            "bnpo": "bnpo",
            "dapo": "dapo",
            "drgrpo": "dr_grpo",
        }
        common_values = []
        for algorithm, loss_type in branches.items():
            commands = self.run_script(
                "train_rl.sh",
                {
                    "C3_CHECKPOINT": self.posix(self.c3),
                    "DATASET": self.posix(dataset),
                    "OUTPUT_DIR": self.posix(self.root / f"output-{algorithm}-raw"),
                    "TRAIN_LOG": self.posix(self.root / f"train-{algorithm}-raw.log"),
                    "PROVENANCE_MANIFEST": self.posix(
                        self.root / f"manifest-{algorithm}-raw.json"
                    ),
                },
                ("--variant", "raw", "--algorithm", algorithm, "--foreground"),
            )
            args = next(item["args"] for item in commands if item["command"] == "swift")
            self.assertEqual(args[:2], ["rlhf", "--rlhf_type"])
            self.assertEqual(self.option(args, "--loss_type"), loss_type)
            self.assertEqual(
                self.normalized_bash_path(self.option(args, "--adapters")),
                self.posix(self.c3),
            )
            self.assertEqual(
                self.normalized_bash_path(self.option(args, "--ref_adapters")),
                self.posix(self.c3),
            )
            reward_index = args.index("--reward_funcs")
            self.assertEqual(self.option(args, "--sleep_level"), "0")
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
