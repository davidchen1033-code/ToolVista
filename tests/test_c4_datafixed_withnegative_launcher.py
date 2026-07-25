from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/autodl/legacy/16_train_visual_rft_dapo_datafixed_withnegative.sh"


class C4DatafixedWithnegativeLauncherTest(unittest.TestCase):
    @staticmethod
    def initialize_git_repository(path: Path) -> str:
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "config", "user.name", "Fixture"], check=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "fixture@example.invalid"],
            check=True,
        )
        subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "fixture"], check=True)
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "ToolVista"
        (self.root / "scripts/autodl").mkdir(parents=True)
        (self.root / "toolvista/rewards").mkdir(parents=True)

        self.checkpoint = self.root / "outputs/c3_router_datafixed/checkpoint-1063"
        self.checkpoint.mkdir(parents=True)
        self.gate = self.checkpoint / "toolvista_c3_datafixed_retention_gate.json"
        self.gate.write_text(json.dumps({"variant": "datafixed"}), encoding="utf-8")
        self.adapter = self.checkpoint / "adapter_model.safetensors"
        self.adapter.write_bytes(b"fixture-c3-adapter\n")

        self.base_model = self.root / "models/base"
        self.base_model.mkdir(parents=True)
        self.base_config = self.base_model / "config.json"
        self.base_config.write_text(json.dumps({"model_type": "fixture"}), encoding="utf-8")

        self.ms_swift_root = Path(self.tempdir.name) / "ms-swift"
        swift_package = self.ms_swift_root / "swift"
        swift_package.mkdir(parents=True)
        self.swift_init = swift_package / "__init__.py"
        self.swift_init.write_text('__version__ = "fixture-4.5.0"\n', encoding="utf-8")

        self.dataset = (
            self.root / "data/exports/datafixed-withnegative_canonical/train.jsonl"
        )
        self.report = (
            self.root / "data/reports/datafixed-withnegative_canonical/report.json"
        )
        self.media_dir = (
            self.root / "data/exports/datafixed-withnegative_canonical_media"
        )
        self.dataset.parent.mkdir(parents=True)
        self.report.parent.mkdir(parents=True)
        self.media_dir.mkdir(parents=True)
        self.dataset.write_text('{"messages": [], "solution": "<answer>No Objects</answer>"}\n', encoding="utf-8")
        media_payload = b"fixture-canonical-image\n"
        media_sha256 = hashlib.sha256(media_payload).hexdigest()
        self.media_file = self.media_dir / f"{media_sha256}.jpg"
        self.media_file.write_bytes(media_payload)
        self.media_manifest = self.media_dir / "manifest.jsonl"
        media_manifest_content = (
            json.dumps(
                {
                    "file": self.media_file.name,
                    "sha256": media_sha256,
                    "bytes": len(media_payload),
                }
            )
            + "\n"
        ).encode("utf-8")
        self.media_manifest.write_bytes(media_manifest_content)
        self.report.write_text(
            json.dumps(
                {
                    "variant": "datafixed-withnegative",
                    "canonical_media": {
                        "path": str(self.media_dir.resolve()),
                        "file_count": 1,
                        "bytes": len(media_payload),
                        "directory_bytes": len(media_payload) + len(media_manifest_content),
                        "manifest": {
                            "path": str(self.media_manifest.resolve()),
                            "entries": 1,
                            "bytes": len(media_manifest_content),
                            "sha256": hashlib.sha256(media_manifest_content).hexdigest(),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        self.builder_capture = self.root / "builder-args.json"
        self.builder = self.root / "scripts/prepare_visual_rft_datafixed_withnegative.py"
        self.builder.write_text(
            "import argparse, json, os\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--output', required=True)\n"
            "parser.add_argument('--report', required=True)\n"
            "parser.add_argument('--media-dir', required=True)\n"
            "parser.add_argument('--verify-only', action='store_true')\n"
            "args = parser.parse_args()\n"
            "assert args.verify_only\n"
            "with open(os.environ['BUILDER_CAPTURE'], 'w', encoding='utf-8') as handle:\n"
            "    json.dump(vars(args), handle)\n"
            "delete_path = os.environ.get('BUILDER_DELETE_PATH')\n"
            "if delete_path:\n"
            "    os.unlink(delete_path)\n",
            encoding="utf-8",
        )
        self.reward_plugin = self.root / "toolvista/rewards/visual_rft_withnegative.py"
        self.reward_plugin.write_text("# fixture\n", encoding="utf-8")

        self.env_capture = self.root / "launcher-env.json"
        self.delegated = self.root / "scripts/autodl/10_train_visual_rft_dapo.sh"
        self.delegated.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "python_bin=\"${PYTHON_BIN:-python}\"\n"
            "\"${python_bin}\" - \"${ENV_CAPTURE}\" <<'PY'\n"
            "import json, os, sys\n"
            "keys = [\n"
            "    'C3_CHECKPOINT', 'C3_RETENTION_GATE', 'TOOLVISTA_ROOT', 'MS_SWIFT_ROOT',\n"
            "    'BASE_MODEL', 'PYTHON_BIN',\n"
            "    'DATASET', 'MEDIA_DIR', 'REWARD_PLUGIN',\n"
            "    'REWARD_FUNCS', 'SEED', 'MAX_STEPS', 'LEARNING_RATE', 'BETA',\n"
            "    'SAVE_STEPS', 'SAVE_TOTAL_LIMIT', 'STAGE', 'WANDB_GROUP',\n"
            "    'WANDB_NAME', 'OUTPUT_DIR', 'TRAIN_LOG', 'PROVENANCE_MANIFEST', 'LOSS_TYPE',\n"
            "]\n"
            "with open(sys.argv[1], 'w', encoding='utf-8') as handle:\n"
            "    json.dump({key: os.environ.get(key) for key in keys}, handle)\n"
            "PY\n"
            "echo delegated\n",
            encoding="utf-8",
        )
        self.rl_common = self.root / "scripts/autodl/rl_common.sh"
        self.rl_common.write_text("# fixture shared RL contract\n", encoding="utf-8")

        self.toolvista_commit = self.initialize_git_repository(self.root)
        self.ms_swift_commit = self.initialize_git_repository(self.ms_swift_root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def environment(self) -> dict[str, str]:
        output = self.root / "outputs/rl/dapo_datafixed-withnegative_seed42_runfixture"
        train_log = self.root / "logs/c4/dapo-datafixed-withnegative-fixture.train.log"
        return {
            **os.environ,
            "TOOLVISTA_ROOT": str(self.root),
            "MS_SWIFT_ROOT": str(self.ms_swift_root),
            "BASE_MODEL": str(self.base_model),
            "PYTHON_BIN": sys.executable,
            "C3_DATAFIXED_CHECKPOINT": str(self.checkpoint),
            "DATASET": str(self.dataset),
            "DATASET_REPORT": str(self.report),
            "MEDIA_DIR": str(self.media_dir),
            "OUTPUT_DIR": str(output),
            "TRAIN_LOG": str(train_log),
            "RUN_TAG": "fixture",
            "BUILDER_CAPTURE": str(self.builder_capture),
            "ENV_CAPTURE": str(self.env_capture),
            "REPORT_TO": "none",
        }

    def run_launcher(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = self.environment()
        env.update(overrides)
        return subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_launcher_has_valid_bash_syntax(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable")
        result = subprocess.run([bash, "-n", str(LAUNCHER)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_launcher_verifies_dataset_and_exports_matched_dapo_contract(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("delegated", result.stdout)

        builder_args = json.loads(self.builder_capture.read_text(encoding="utf-8"))
        self.assertEqual(builder_args, {
            "output": str(self.dataset),
            "report": str(self.report),
            "media_dir": str(self.media_dir),
            "verify_only": True,
        })

        values = json.loads(self.env_capture.read_text(encoding="utf-8"))
        self.assertEqual(values["C3_CHECKPOINT"], str(self.checkpoint.resolve()))
        self.assertEqual(values["C3_RETENTION_GATE"], str(self.gate.resolve()))
        self.assertEqual(values["TOOLVISTA_ROOT"], str(self.root.resolve()))
        self.assertEqual(values["MS_SWIFT_ROOT"], str(self.ms_swift_root.resolve()))
        self.assertEqual(values["BASE_MODEL"], str(self.base_model.resolve()))
        self.assertEqual(values["PYTHON_BIN"], sys.executable)
        self.assertEqual(values["DATASET"], str(self.dataset.resolve()))
        self.assertEqual(values["MEDIA_DIR"], str(self.media_dir.resolve()))
        self.assertEqual(
            values["REWARD_PLUGIN"],
            str(self.root / "toolvista/rewards/visual_rft_withnegative.py"),
        )
        self.assertEqual(
            values["REWARD_FUNCS"],
            "visual_rft_withnegative visual_rft_withnegative_format",
        )
        self.assertNotIn("format", values["REWARD_FUNCS"].split())
        self.assertEqual(values["MAX_STEPS"], "6000")
        self.assertEqual(values["LEARNING_RATE"], "1e-6")
        self.assertEqual(values["BETA"], "0.04")
        self.assertEqual(values["SAVE_STEPS"], "500")
        self.assertEqual(values["SAVE_TOTAL_LIMIT"], "13")
        self.assertEqual(values["STAGE"], "rl-dapo-datafixed-withnegative")
        self.assertEqual(values["WANDB_GROUP"], "rl-datafixed-withnegative")
        self.assertIn("datafixed-withnegative", values["WANDB_NAME"])
        self.assertIn("datafixed-withnegative", values["OUTPUT_DIR"])

        train_log = Path(self.environment()["TRAIN_LOG"])
        self.assertEqual(values["TRAIN_LOG"], str(train_log.resolve()))
        manifest_path = Path(str(train_log).removesuffix(".train.log") + ".manifest.json")
        self.assertEqual(values["PROVENANCE_MANIFEST"], str(manifest_path.resolve()))
        self.assertTrue(manifest_path.is_file())
        self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o444)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["variant"], "datafixed-withnegative")
        self.assertEqual(manifest["run_tag"], "fixture")
        self.assertEqual(manifest["seed"], "42")
        self.assertEqual(manifest["stage"], "rl-dapo-datafixed-withnegative")
        self.assertEqual(manifest["output"]["resolved_path"], values["OUTPUT_DIR"])
        self.assertEqual(manifest["log"]["resolved_path"], str(train_log.resolve()))
        self.assertTrue(manifest["created_at_utc"].endswith("Z"))

        expected_files = {
            "dataset": self.dataset,
            "dataset_report": self.report,
            "canonical_media_manifest": self.media_manifest,
            "dataset_builder": self.builder,
            "reward_plugin": self.reward_plugin,
            "launcher_datafixed_withnegative": LAUNCHER,
            "launcher_dapo": self.delegated,
            "rl_common": self.rl_common,
            "c3_retention_gate": self.gate,
            "c3_adapter_model": self.adapter,
            "base_model_config": self.base_config,
        }
        for key, path in expected_files.items():
            with self.subTest(provenance_file=key):
                entry = manifest["files"][key]
                self.assertEqual(entry["resolved_path"], str(path.resolve()))
                self.assertEqual(entry["sha256"], self.sha256(path))
                self.assertEqual(entry["size_bytes"], path.stat().st_size)

        self.assertEqual(manifest["canonical_media"]["resolved_path"], str(self.media_dir.resolve()))
        self.assertEqual(manifest["canonical_media"]["file_count"], 1)
        self.assertEqual(manifest["canonical_media"]["bytes"], self.media_file.stat().st_size)
        self.assertEqual(manifest["canonical_media"]["directory_bytes"], sum(path.stat().st_size for path in self.media_dir.iterdir()))
        self.assertEqual(manifest["canonical_media"]["manifest_entries"], 1)
        self.assertEqual(manifest["canonical_media"]["manifest_sha256"], self.sha256(self.media_manifest))

        embedded_files = {
            "dataset_builder": self.builder,
            "reward_plugin": self.reward_plugin,
            "launcher_datafixed_withnegative": LAUNCHER,
            "launcher_dapo": self.delegated,
            "rl_common": self.rl_common,
        }
        for key, path in embedded_files.items():
            with self.subTest(embedded_source=key):
                self.assertEqual(
                    manifest["files"][key]["utf8_content"].encode("utf-8"),
                    path.read_bytes(),
                )

        self.assertEqual(manifest["reward_funcs"]["raw"], values["REWARD_FUNCS"])
        self.assertEqual(
            manifest["reward_funcs"]["names"],
            ["visual_rft_withnegative", "visual_rft_withnegative_format"],
        )
        self.assertEqual(manifest["training_parameters"]["loss_type"], "dapo")
        self.assertEqual(manifest["training_parameters"]["max_steps"], "6000")
        self.assertEqual(manifest["training_parameters"]["epsilon_high"], "0.28")
        self.assertEqual(manifest["repositories"]["toolvista"]["git_commit"], self.toolvista_commit)
        self.assertTrue(manifest["repositories"]["toolvista"]["git_dirty"])
        self.assertEqual(
            manifest["repositories"]["ms_swift"]["git_commit"], self.ms_swift_commit
        )
        self.assertFalse(manifest["repositories"]["ms_swift"]["git_dirty"])
        self.assertEqual(
            manifest["repositories"]["ms_swift"]["swift_version"], "fixture-4.5.0"
        )
        self.assertEqual(
            manifest["repositories"]["ms_swift"]["swift_file"]["resolved_path"],
            str(self.swift_init.resolve()),
        )
        self.assertEqual(
            manifest["repositories"]["ms_swift"]["swift_file"]["sha256"],
            self.sha256(self.swift_init),
        )

    def test_launcher_resolves_relative_paths_before_verification_and_delegate(self) -> None:
        output = self.root / "outputs/rl/dapo_datafixed-withnegative_relative"
        train_log = self.root / "logs/c4/dapo-datafixed-withnegative-relative.train.log"
        env = self.environment()
        env.update(
            {
                "DATASET": os.path.relpath(self.dataset, ROOT),
                "DATASET_REPORT": os.path.relpath(self.report, ROOT),
                "MEDIA_DIR": os.path.relpath(self.media_dir, ROOT),
                "OUTPUT_DIR": os.path.relpath(output, ROOT),
                "TRAIN_LOG": os.path.relpath(train_log, ROOT),
            }
        )
        result = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        builder_args = json.loads(self.builder_capture.read_text(encoding="utf-8"))
        self.assertEqual(builder_args["output"], str(self.dataset.resolve()))
        self.assertEqual(builder_args["report"], str(self.report.resolve()))
        self.assertEqual(builder_args["media_dir"], str(self.media_dir.resolve()))
        values = json.loads(self.env_capture.read_text(encoding="utf-8"))
        self.assertEqual(values["DATASET"], str(self.dataset.resolve()))
        self.assertEqual(values["MEDIA_DIR"], str(self.media_dir.resolve()))
        self.assertEqual(values["OUTPUT_DIR"], str(output.resolve()))
        self.assertEqual(values["TRAIN_LOG"], str(train_log.resolve()))
        manifest = Path(str(train_log).removesuffix(".train.log") + ".manifest.json")
        self.assertEqual(values["PROVENANCE_MANIFEST"], str(manifest.resolve()))
        self.assertTrue(manifest.is_file())

    def test_provenance_input_failure_never_delegates_training(self) -> None:
        result = self.run_launcher(BUILDER_DELETE_PATH=str(self.reward_plugin))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.builder_capture.is_file())
        self.assertFalse(self.env_capture.exists())
        manifest = Path(self.environment()["TRAIN_LOG"].removesuffix(".train.log") + ".manifest.json")
        self.assertFalse(manifest.exists())

    def test_launcher_rejects_legacy_dataset_namespace(self) -> None:
        legacy = self.root / "data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text("{}\n", encoding="utf-8")
        result = self.run_launcher(DATASET=str(legacy))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("datafixed-withnegative", result.stderr)
        self.assertFalse(self.env_capture.exists())

    def test_launcher_refuses_existing_output(self) -> None:
        env = self.environment()
        Path(env["OUTPUT_DIR"]).mkdir(parents=True)
        result = subprocess.run(
            ["bash", str(LAUNCHER)], cwd=ROOT, env=env, capture_output=True, text=True, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)
        self.assertFalse(self.env_capture.exists())

    def test_launcher_refuses_existing_provenance_manifest(self) -> None:
        env = self.environment()
        manifest = Path(env["TRAIN_LOG"].removesuffix(".train.log") + ".manifest.json")
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}\n", encoding="utf-8")
        result = subprocess.run(
            ["bash", str(LAUNCHER)], cwd=ROOT, env=env, capture_output=True, text=True, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("provenance manifest", result.stderr)
        self.assertFalse(self.builder_capture.exists())
        self.assertFalse(self.env_capture.exists())

    def test_launcher_rejects_provenance_override_without_exact_marker(self) -> None:
        result = self.run_launcher(
            PROVENANCE_MANIFEST=str(self.root / "logs/c4/provenance.manifest.json")
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("datafixed-withnegative", result.stderr)
        self.assertFalse(self.builder_capture.exists())
        self.assertFalse(self.env_capture.exists())


if __name__ == "__main__":
    unittest.main()
