from __future__ import annotations

import hashlib
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
    / "train_c3_datafixed.sh"
)
EVAL_LAUNCHER = (
    REPO_ROOT
    / "scripts"
    / "autodl"
    / "lib"
    / "evaluate_c3_datafixed.sh"
)


class C3DatafixedTrainingLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tv-c3-launcher-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        self.datafixed = self.root / "experiment-datafixed"
        self.control = self.root / "control"
        self.datafixed.mkdir()
        self.control.mkdir()
        self.c2 = self.root / "c2-checkpoint"
        self.ms_swift = self.root / "ms-swift"
        self.model = self.root / "model"
        for path in (self.c2, self.ms_swift, self.model):
            path.mkdir()

        self.router = self._write_rows(self.datafixed / "router_datafixed.jsonl", 3)
        self.direct = self._write_rows(self.datafixed / "direct_datafixed.jsonl", 2)
        self.toolvqa = self._write_rows(self.root / "toolvqa.jsonl", 2)
        self.xlam = self._write_rows(self.root / "xlam.jsonl", 2)
        self.report = self.datafixed / "router_datafixed.report.json"
        self.replay = self.datafixed / "replay_datafixed.jsonl"
        self.replay_report = self.datafixed / "replay_datafixed.report.json"
        self.output_dir = self.datafixed / "output"
        self.command_log = self.root / "commands.log"
        self.fake_python = self.root / "fake-python.sh"
        self.fake_swift = self.root / "fake-swift.sh"
        self._write_recorders()
        self._write_report()

    @staticmethod
    def _write_rows(path: Path, count: int) -> Path:
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"prompt-{index}"},
                    {"role": "assistant", "content": f"answer-{index}"},
                ]
            }
            for index in range(count)
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_report(
        self,
        *,
        router_path: Path | None = None,
        direct_path: Path | None = None,
        router_sha: str | None = None,
        direct_sha: str | None = None,
        destination: Path | None = None,
    ) -> Path:
        destination = destination or self.report
        payload = {
            "schema_version": "1.0",
            "variant": "datafixed",
            "artifacts": {
                "router_train": str(router_path or self.router),
                "direct_answer_train": str(direct_path or self.direct),
            },
            "audit": {
                "router_train": {"sha256": router_sha or self._sha256(self.router)},
                "direct_answer": {"sha256": direct_sha or self._sha256(self.direct)},
            },
        }
        destination.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return destination

    def _write_recorders(self) -> None:
        real_python = shlex.quote(str(Path(sys.executable).resolve()))
        self.fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf 'python %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
            "if [[ \"${1:-}\" == \"scripts/validate_ms_swift_dataset.py\" ]]; then\n"
            "  exit 0\n"
            "fi\n"
            f"exec {real_python} \"$@\"\n",
            encoding="utf-8",
        )
        self.fake_swift.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf 'swift %s\\n' \"$*\" >> \"$COMMAND_LOG\"\n",
            encoding="utf-8",
        )
        self.fake_python.chmod(0o755)
        self.fake_swift.chmod(0o755)

    def _environment(self, **overrides: Path | str) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("MAX_STEPS", None)
        env.update(
            {
                "TOOLVISTA_ROOT": str(REPO_ROOT),
                "PYTHON_BIN": str(self.fake_python),
                "SWIFT_BIN": str(self.fake_swift),
                "COMMAND_LOG": str(self.command_log),
                "C2_CHECKPOINT": str(self.c2),
                "ROUTER_DATASET": str(self.router),
                "DIRECT_ANSWER_DATASET": str(self.direct),
                "TOOLVQA_DATASET": str(self.toolvqa),
                "XLAM_DATASET": str(self.xlam),
                "DATAFIXED_PREP_REPORT": str(self.report),
                "REPLAY_DATASET": str(self.replay),
                "REPLAY_REPORT": str(self.replay_report),
                "OUTPUT_DIR": str(self.output_dir),
                "MS_SWIFT_ROOT": str(self.ms_swift),
                "BASE_MODEL": str(self.model),
                "REQUIRE_NEW_ADAPTER": "false",
                "REPORT_TO": "none",
            }
        )
        env.update({key: str(value) for key, value in overrides.items()})
        return env

    def _run(self, **overrides: Path | str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=self._environment(**overrides),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def _assert_no_build_or_training(self) -> None:
        commands = (
            self.command_log.read_text(encoding="utf-8")
            if self.command_log.exists()
            else ""
        )
        self.assertNotIn("build_sft_replay.py", commands)
        self.assertNotIn("swift sft", commands)
        self.assertFalse(self.replay.exists())
        self.assertFalse(self.output_dir.exists())

    def test_default_1063_reaches_replay_build_and_training_command(self) -> None:
        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.command_log.read_text(encoding="utf-8")
        self.assertIn("build_sft_replay.py", commands)
        self.assertIn("swift sft", commands)
        self.assertIn("--max_steps 1063", commands)
        self.assertTrue(self.replay.is_file())

    def test_1064_is_rejected_before_replay_build_or_training(self) -> None:
        result = self._run(MAX_STEPS="1064")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "MAX_STEPS must be an integer between 1 and 1063",
            result.stderr,
        )
        self._assert_no_build_or_training()

    def test_report_artifact_path_mismatch_fails_before_build(self) -> None:
        cases = (
            ("router", "router_path", "router_train"),
            ("direct", "direct_path", "direct_answer_train"),
        )
        for name, argument, artifact_key in cases:
            with self.subTest(name=name):
                self.command_log.unlink(missing_ok=True)
                other = self._write_rows(
                    self.datafixed / f"other_{name}_datafixed.jsonl", 1
                )
                self._write_report(**{argument: other})

                result = self._run()

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"artifacts.{artifact_key} does not authorize",
                    result.stderr,
                )
                self._assert_no_build_or_training()

    def test_lexical_path_requires_a_safe_datafixed_marker(self) -> None:
        unsafe = self._write_rows(self.control / "notdatafixed.jsonl", 1)

        result = self._run(ROUTER_DATASET=unsafe)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ROUTER_DATASET lexical path must contain", result.stderr)
        self._assert_no_build_or_training()

    def test_report_hash_mismatch_fails_before_build(self) -> None:
        cases = (
            ("router", {"router_sha": "0" * 64}, "audit.router_train.sha256"),
            ("direct", {"direct_sha": "f" * 64}, "audit.direct_answer.sha256"),
        )
        for name, report_args, expected in cases:
            with self.subTest(name=name):
                self.command_log.unlink(missing_ok=True)
                self._write_report(**report_args)
                result = self._run()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                self._assert_no_build_or_training()

    def test_datafixed_paths_reject_symlink_escape_to_control(self) -> None:
        cases = {
            "ROUTER_DATASET": (self.control / "router.jsonl", False),
            "DIRECT_ANSWER_DATASET": (self.control / "direct.jsonl", False),
            "DATAFIXED_PREP_REPORT": (self.control / "prep-report.json", False),
            "REPLAY_DATASET": (self.control / "replay.jsonl", False),
            "REPLAY_REPORT": (self.control / "replay-report.json", False),
            "OUTPUT_DIR": (self.control / "output", True),
        }
        for variable, (target, is_directory) in cases.items():
            with self.subTest(variable=variable):
                if is_directory:
                    target.mkdir()
                elif variable == "DATAFIXED_PREP_REPORT":
                    self._write_report(destination=target)
                else:
                    target.write_text("{}\n", encoding="utf-8")
                alias = self.datafixed / f"escaped_{variable.lower()}"
                alias.symlink_to(target, target_is_directory=is_directory)

                result = self._run(**{variable: alias})

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"{variable} resolves outside the datafixed namespace",
                    result.stderr,
                )
                self._assert_no_build_or_training()
                alias.unlink()
                if is_directory:
                    target.rmdir()
                else:
                    target.unlink()


class C3DatafixedEvalLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tv-c3-eval-launcher-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "project"
        (self.project / "scripts" / "autodl").mkdir(parents=True)

        patch_helper = self.project / "scripts" / "autodl" / "patch_msswift_vlmeval.sh"
        patch_helper.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        patch_helper.chmod(0o755)

        self.command_log = self.root / "commands.log"
        self.fake_python = self.root / "bin" / "fake-python"
        self.fake_swift = self.root / "bin" / "swift"
        self.fake_python.parent.mkdir()
        real_python = shlex.quote(str(Path(sys.executable).resolve()))
        self.fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "{ printf 'PATH=<%s>' \"$PATH\"; printf '<%s>' \"$@\"; printf '\\n'; } >> \"$COMMAND_LOG\"\n"
            "if [[ \"${1:-}\" == \"-c\" && \"$#\" -ge 5 ]]; then\n"
            f"  exec {real_python} \"$@\"\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        self.fake_swift.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        self.fake_python.chmod(0o755)
        self.fake_swift.chmod(0o755)

    @staticmethod
    def _write(path: Path, content: str = "{}\n") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_relative_swift_is_frozen_and_passed_when_initial_path_has_no_swift(self) -> None:
        checkpoint = self.root / "checkpoint_datafixed"
        checkpoint.mkdir()
        history_dir = self.root / "history_datafixed"
        history_summary = self._write(history_dir / "summary.json")
        self._write(
            history_dir / "manifest.json",
            json.dumps(
                {
                    "evaluator": {
                        "policy": "fixed_across_c0_c4",
                        "judge_api_base": "http://127.0.0.1:8011/v1",
                        "judge_model": "fixed-judge",
                    }
                }
            )
            + "\n",
        )
        benchmark_config = self._write(self.root / "benchmarks_datafixed.yaml")
        suite_config = self._write(self.root / "suites_datafixed.yaml")
        ms_swift = self.root / "ms-swift"
        ms_swift.mkdir()

        initial_path = "/usr/bin:/bin"
        self.assertIsNone(shutil.which("swift", path=initial_path))
        env = os.environ.copy()
        env.update(
            {
                "PATH": initial_path,
                "COMMAND_LOG": str(self.command_log),
                "TOOLVISTA_ROOT": str(self.project),
                "PYTHON_BIN": str(self.fake_python),
                "SWIFT_BIN": "bin/swift",
                "CHECKPOINT": str(checkpoint),
                "C3_DATAFIXED_ADAPTER_CHECKPOINT": str(checkpoint),
                "RUN_ID": "c3_datafixed_test",
                "API_MODEL": "c3-datafixed-test",
                "HISTORY_SUMMARY": str(history_summary),
                "JUDGE_API_BASE": "http://127.0.0.1:8011/v1",
                "JUDGE_MODEL": "fixed-judge",
                "TOOLVQA_PYTHON": str(self.fake_python),
                "BFCL_PYTHON": str(self.fake_python),
                "MS_SWIFT_ROOT": str(ms_swift),
                "DATAFIXED_BENCHMARK_CONFIG": str(benchmark_config),
                "DATAFIXED_SUITE_CONFIG": str(suite_config),
            }
        )

        result = subprocess.run(
            ["bash", str(EVAL_LAUNCHER)],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.command_log.read_text(encoding="utf-8")
        swift = str(self.fake_swift.resolve())
        self.assertIn(f"<--swift-executable><{swift}>", commands)
        self.assertIn(f"PATH=<{self.fake_swift.parent.resolve()}:{initial_path}>", commands)


if __name__ == "__main__":
    unittest.main()
