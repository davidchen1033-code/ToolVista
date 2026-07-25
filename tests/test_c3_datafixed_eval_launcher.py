from __future__ import annotations

import json
import os
import shlex
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
    / "evaluate_c3_datafixed.sh"
)


class C3DatafixedEvalLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="tv-eval-launcher-")
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "project"
        self.checkpoint = self.root / "checkpoints" / "c3-datafixed"
        self.history = self.root / "runs" / "c2-datafixed"
        self.baseline = self.root / "runs" / "c0-baseline" / "summary.json"
        for path in (
            self.project / "configs" / "evaluation",
            self.project / "scripts" / "autodl",
            self.checkpoint,
            self.history,
            self.baseline.parent,
            self.root / "bin",
            self.root / "ms-swift",
        ):
            path.mkdir(parents=True, exist_ok=True)

        for name in ("benchmarks_datafixed.yaml", "suites_datafixed.yaml"):
            (self.project / "configs" / "evaluation" / name).write_text(
                "placeholder: true\n", encoding="utf-8"
            )
        (self.history / "summary.json").write_text("{}\n", encoding="utf-8")
        (self.history / "manifest.json").write_text(
            json.dumps(
                {
                    "evaluator": {
                        "judge_api_base": "http://127.0.0.1:8011/v1",
                        "judge_model": "fixed-judge",
                        "policy": "fixed_across_c0_c4",
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.baseline.write_text("{}\n", encoding="utf-8")

        patch_script = self.project / "scripts" / "autodl" / "patch_msswift_vlmeval.sh"
        patch_script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
        )
        patch_script.chmod(0o755)

        self.command_log = self.root / "commands.jsonl"
        self.fake_python = self.root / "bin" / "fake-python"
        real_python = shlex.quote(str(Path(sys.executable).resolve()))
        self.fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "if [[ \"$#\" -gt 0 && \"$1\" == \"-c\" ]]; then\n"
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
        self.noop.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8"
        )
        self.noop.chmod(0o755)

    def environment(self, **overrides: str) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "TOOLVISTA_ROOT": "project",
                "PYTHON_BIN": "bin/fake-python",
                "CHECKPOINT": "checkpoints/c3-datafixed",
                "C3_DATAFIXED_ADAPTER_CHECKPOINT": "checkpoints/c3-datafixed",
                "RUN_ID": "c3-datafixed-candidate",
                "STAGE": "c3-router-datafixed-retention",
                "API_MODEL": "c3-datafixed-candidate",
                "BASELINE_SUMMARY": "runs/c0-baseline/summary.json",
                "HISTORY_SUMMARY": "runs/c2-datafixed/summary.json",
                "HISTORY_MANIFEST": "control/manifest.json",
                "JUDGE_API_BASE": (
                    "http://user:secret@127.0.0.1:8011/v1/?token=secret"
                ),
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

    def test_relative_paths_are_frozen_before_project_cd(self) -> None:
        result = self.run_launcher()

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.recorded_commands()
        self.assertEqual(len(commands), 2)
        evaluate_args = commands[0]["args"]
        assert isinstance(evaluate_args, list)
        self.assertEqual(commands[0]["cwd"], str(self.project.resolve()))
        self.assertEqual(
            self.flag(evaluate_args, "--checkpoint"), str(self.checkpoint.resolve())
        )
        self.assertEqual(
            self.flag(evaluate_args, "--history-summary"),
            str((self.history / "summary.json").resolve()),
        )
        self.assertEqual(
            self.flag(evaluate_args, "--baseline-summary"),
            str(self.baseline.resolve()),
        )
        self.assertEqual(
            self.flag(evaluate_args, "--judge-api-base"),
            "http://127.0.0.1:8011/v1",
        )

    def test_agent_retention_suite_and_strict_gate_args_are_forwarded(self) -> None:
        result = self.run_launcher(
            EVAL_SUITE="c3_datafixed_agentretention_light",
            RETENTION_VARIANT="datafixed-agentretention",
            ROUTER_BENCHMARK_ID="mllm_router_datafixed_agentretention",
            TOOLVQA_ANSWER_RETENTION="0.80",
            BFCL_IRRELEVANCE_MINIMUM="0.20",
            ROUTER_VIDEO_MINIMUM="0.25",
            ROUTER_INVALID_MAXIMUM="0.05",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.recorded_commands()
        self.assertEqual(len(commands), 2)
        evaluate_args = commands[0]["args"]
        gate_args = commands[1]["args"]
        assert isinstance(evaluate_args, list)
        assert isinstance(gate_args, list)
        self.assertEqual(
            self.flag(evaluate_args, "--suite"),
            "c3_datafixed_agentretention_light",
        )
        expected = {
            "--variant": "datafixed-agentretention",
            "--router-benchmark-id": "mllm_router_datafixed_agentretention",
            "--toolvqa-answer-retention": "0.80",
            "--bfcl-irrelevance-minimum": "0.20",
            "--router-video-minimum": "0.25",
            "--router-invalid-maximum": "0.05",
        }
        for flag, value in expected.items():
            self.assertEqual(self.flag(gate_args, flag), value)

    def test_non_c3_stage_is_rejected_before_benchmark(self) -> None:
        result = self.run_launcher(STAGE="c4-datafixed")

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("STAGE must start with 'c3'", result.stderr)
        self.assertEqual(self.recorded_commands(), [])

    def test_judge_model_mismatch_is_rejected_before_benchmark(self) -> None:
        result = self.run_launcher(JUDGE_MODEL="other-judge")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JUDGE_MODEL does not match", result.stderr)
        self.assertEqual(self.recorded_commands(), [])

    def test_checkpoint_symlink_escape_keeps_resolved_diagnostic(self) -> None:
        control = self.root / "checkpoints" / "control"
        control.mkdir()
        alias = self.root / "checkpoints" / "escaped-datafixed"
        alias.symlink_to(control, target_is_directory=True)

        result = self.run_launcher(
            CHECKPOINT="checkpoints/escaped-datafixed",
            C3_DATAFIXED_ADAPTER_CHECKPOINT="checkpoints/escaped-datafixed",
        )

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("CHECKPOINT resolves outside the datafixed namespace", result.stderr)
        self.assertEqual(self.recorded_commands(), [])


if __name__ == "__main__":
    unittest.main()
