from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "autodl"
    / "lib"
    / "evaluate_c4_datafixed.sh"
)
BENCHMARKS = [
    "xlam_function_calling",
    "toolvqa_step_light",
    "bfcl_non_live_light",
    "mllm_router_datafixed",
    "qwen3vl_general_light",
    "visual_rft_coco_light",
]


class C4DatafixedEvalLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tv-c4-datafixed-eval-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "project"
        self.control_checkpoint = (
            self.root / "outputs" / "c4-dapo-datafixed" / "checkpoint-6000"
        )
        self.negative_checkpoint = (
            self.root
            / "outputs"
            / "c4-dapo-datafixed-withnegative"
            / "checkpoint-6000"
        )
        self.history = self.root / "runs" / "c3-datafixed-history"
        for path in (
            self.project / "configs" / "evaluation",
            self.project / "scripts" / "autodl",
            self.control_checkpoint,
            self.negative_checkpoint,
            self.history,
            self.root / "bin",
            self.root / "ms-swift",
        ):
            path.mkdir(parents=True, exist_ok=True)

        for name in ("benchmarks_datafixed.yaml", "suites_datafixed.yaml"):
            (self.project / "configs" / "evaluation" / name).write_text(
                "placeholder: true\n", encoding="utf-8"
            )
        patch_script = self.project / "scripts" / "autodl" / "patch_msswift_vlmeval.sh"
        patch_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
        patch_script.chmod(0o755)

        self.history_summary = self.history / "summary.json"
        self.history_manifest = self.history / "manifest.json"
        self.write_history()

        self.gate = self.root / "checkpoints" / "c3-datafixed" / "toolvista_c3_datafixed_retention_gate.json"
        self.gate.parent.mkdir(parents=True)
        self.gate.write_text("C3-GATE-MUST-STAY-UNCHANGED\n", encoding="utf-8")

        self.command_log = self.root / "commands.jsonl"
        self.fake_python = self.root / "bin" / "fake-python"
        real_python = shlex.quote(str(Path(sys.executable).resolve()))
        self.fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"${1:-}\" == \"-\" || \"${1:-}\" == \"-c\" ]]; then\n"
            f"  exec {real_python} \"$@\"\n"
            "fi\n"
            f"{real_python} -c 'import json,sys; "
            "open(sys.argv[1], \"a\", encoding=\"utf-8\").write("
            "json.dumps({\"cwd\": sys.argv[2], \"args\": sys.argv[3:]}) + \"\\n\")' "
            "\"$COMMAND_LOG\" \"$PWD\" \"$@\"\n",
            encoding="utf-8",
        )
        self.fake_python.chmod(0o755)

        self.noop = self.root / "bin" / "noop"
        self.noop.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
        self.noop.chmod(0o755)
        self.fake_swift = self.root / "bin" / "swift"
        self.fake_swift.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
        self.fake_swift.chmod(0o755)

    def write_history(self, **manifest_overrides: object) -> None:
        checkpoint = str(self.root / "checkpoints" / "c3-datafixed" / "checkpoint-1063")
        manifest: dict[str, object] = {
            "run_id": "c3-datafixed-history",
            "checkpoint": checkpoint,
            "stage": "c3-router-datafixed-retention",
            "seed": 42,
            "benchmarks": BENCHMARKS,
            "evaluator": {
                "judge_api_base": "http://127.0.0.1:8011/v1",
                "judge_model": "fixed-judge",
                "policy": "fixed_across_c0_c4",
            },
        }
        manifest.update(manifest_overrides)
        self.history_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        summary = {
            "run_id": manifest["run_id"],
            "checkpoint": manifest["checkpoint"],
            "status": "passed",
            "benchmarks": {name: {"status": "passed"} for name in BENCHMARKS},
            "metrics": {},
        }
        self.history_summary.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    def environment(self, **overrides: str) -> dict[str, str]:
        env = os.environ.copy()
        for name in ("SUITE", "EXPERIMENT_VARIANT", "BASELINE_SUMMARY"):
            env.pop(name, None)
        env.update(
            {
                "TOOLVISTA_ROOT": "project",
                "PYTHON_BIN": "bin/fake-python",
                "SWIFT_BIN": "bin/swift",
                "CHECKPOINT": "outputs/c4-dapo-datafixed/checkpoint-6000",
                "RUN_ID": "c4-dapo-datafixed-light",
                "STAGE": "c4-visualrft-dapo-datafixed",
                "ALGORITHM": "dapo",
                "API_MODEL": "c4-dapo-datafixed",
                "HISTORY_SUMMARY": "runs/c3-datafixed-history/summary.json",
                "JUDGE_API_BASE": "http://user:secret@127.0.0.1:8011/v1/?token=secret",
                "JUDGE_MODEL": "fixed-judge",
                "LMUData": "cache/vlmevalkit",
                "HF_HOME": "cache/huggingface",
                "TOOLVQA_PYTHON": "bin/noop",
                "BFCL_PYTHON": "bin/noop",
                "MS_SWIFT_ROOT": "ms-swift",
                "COMMAND_LOG": str(self.command_log),
            }
        )
        env.update(overrides)
        return env

    def run_launcher(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=self.root,
            env=self.environment(**overrides),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def recorded_commands(self) -> list[dict[str, object]]:
        if not self.command_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.command_log.read_text(encoding="utf-8").splitlines()
        ]

    @staticmethod
    def flag(args: list[str], name: str) -> str:
        return args[args.index(name) + 1]

    def test_control_invocation_is_c4_datafixed_and_keeps_c3_gate_read_only(self) -> None:
        result = self.run_launcher()

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.recorded_commands()
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command["cwd"], str(self.project.resolve()))
        args = command["args"]
        assert isinstance(args, list)
        self.assertEqual(self.flag(args, "--suite"), "c4_datafixed_light")
        self.assertEqual(
            self.flag(args, "--config"),
            str((self.project / "configs/evaluation/benchmarks_datafixed.yaml").resolve()),
        )
        self.assertEqual(
            self.flag(args, "--suite-config"),
            str((self.project / "configs/evaluation/suites_datafixed.yaml").resolve()),
        )
        self.assertEqual(
            self.flag(args, "--history-summary"), str(self.history_summary.resolve())
        )
        self.assertEqual(
            self.flag(args, "--checkpoint"), str(self.control_checkpoint.resolve())
        )
        self.assertEqual(self.flag(args, "--judge-api-base"), "http://127.0.0.1:8011/v1")
        self.assertEqual(self.flag(args, "--algorithm"), "dapo")
        self.assertEqual(self.gate.read_text(encoding="utf-8"), "C3-GATE-MUST-STAY-UNCHANGED\n")

    def test_withnegative_variant_is_detected_and_kept_distinct(self) -> None:
        result = self.run_launcher(
            CHECKPOINT="outputs/c4-dapo-datafixed-withnegative/checkpoint-6000",
            RUN_ID="c4-dapo-datafixed-withnegative-light",
            STAGE="c4-visualrft-dapo-datafixed-withnegative",
            API_MODEL="c4-dapo-datafixed-withnegative",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.recorded_commands()[0]["args"]
        assert isinstance(args, list)
        self.assertEqual(
            self.flag(args, "--checkpoint"), str(self.negative_checkpoint.resolve())
        )
        self.assertEqual(
            self.flag(args, "--wandb-group"),
            "c4-visualrft-datafixed-withnegative",
        )

    def test_fixed_judge_mismatch_is_rejected_before_evaluation(self) -> None:
        result = self.run_launcher(JUDGE_MODEL="different-judge")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JUDGE_MODEL does not match", result.stderr)
        self.assertEqual(self.recorded_commands(), [])

    def test_non_c3_history_is_rejected_before_evaluation(self) -> None:
        self.write_history(stage="c4-visualrft-datafixed")

        result = self.run_launcher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stage must start with c3", result.stderr)
        self.assertEqual(self.recorded_commands(), [])

    def test_unsupported_suite_is_rejected(self) -> None:
        result = self.run_launcher(SUITE="c4_light")

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Unsupported C4 datafixed suite", result.stderr)
        self.assertEqual(self.recorded_commands(), [])

    def test_variant_mismatch_is_rejected(self) -> None:
        result = self.run_launcher(
            CHECKPOINT="outputs/c4-dapo-datafixed-withnegative/checkpoint-6000",
            EXPERIMENT_VARIANT="datafixed",
        )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("belongs to datafixed-withnegative", result.stderr)
        self.assertEqual(self.recorded_commands(), [])

    def test_existing_run_requires_explicit_resume(self) -> None:
        run_dir = self.project / "runs" / "c4-dapo-datafixed-light"
        run_dir.mkdir(parents=True)

        rejected = self.run_launcher()
        self.assertEqual(rejected.returncode, 2, rejected.stderr)
        self.assertIn("set RESUME=true", rejected.stderr)
        self.assertEqual(self.recorded_commands(), [])

        resumed = self.run_launcher(RESUME="true")
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        args = self.recorded_commands()[0]["args"]
        assert isinstance(args, list)
        self.assertIn("--resume", args)

    def test_launcher_has_valid_bash_syntax_and_no_gate_writer(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run(
            [bash, "-n", str(LAUNCHER)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("check_c3_datafixed_retention.py", text)
        self.assertNotIn("toolvista_c3_datafixed_retention_gate.json", text)


if __name__ == "__main__":
    unittest.main()
