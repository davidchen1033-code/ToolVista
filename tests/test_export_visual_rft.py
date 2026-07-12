from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.export_visual_rft import write_export
from toolvista.adapters.visual_rft import VisualRFTAdapter
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "data" / "visual_rft"


def make_paths() -> ToolVistaPaths:
    return ToolVistaPaths(
        config_path=Path("test"),
        raw={
            "datasets": {
                "visual_rft": {
                    "root": str(FIXTURE_ROOT),
                    "data_path": "train.jsonl",
                    "default_split": "train",
                    "default_task": "vqa",
                    "materialized_media_dir": "materialized_images",
                }
            }
        },
    )


class VisualRFTExportTest(unittest.TestCase):
    def test_write_export_writes_grpo_reward_columns(self) -> None:
        adapter = VisualRFTAdapter(make_paths())
        output = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-script-test.jsonl"
        errors = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-script-test.errors.jsonl"
        try:
            stats = write_export(adapter.iter_results(), output, errors, include_metadata=False)

            self.assertEqual(stats["total_raw"], 1)
            self.assertEqual(stats["converted"], 1)
            self.assertEqual(stats["skipped"], 0)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["messages"][-1]["role"], "user")
            self.assertEqual(row["messages"][-1]["content"].count("<image>"), 1)
            self.assertEqual(row["solution"], "<think>Look at the shape.</think><answer>aircraft</answer>")
            self.assertEqual(row["ground_truth"], "aircraft")
            self.assertEqual(row["task"], "classification")
            self.assertEqual(errors.read_text(encoding="utf-8"), "")
        finally:
            output.unlink(missing_ok=True)
            errors.unlink(missing_ok=True)

    def test_cli_exports_fixture_and_writes_report(self) -> None:
        output = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-cli-test.jsonl"
        report = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-cli-test.report.json"
        errors = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-cli-test.errors.jsonl"
        config = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-cli-test.config.yaml"
        config.write_text(
            json.dumps(
                {
                    "datasets": {
                        "visual_rft": {
                            "root": str(FIXTURE_ROOT),
                            "data_path": "train.jsonl",
                            "default_split": "train",
                            "default_task": "vqa",
                            "materialized_media_dir": "materialized_images",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/export_visual_rft.py",
                    "--config",
                    str(config),
                    "--output",
                    str(output),
                    "--report",
                    str(report),
                    "--errors",
                    str(errors),
                    "--max-items",
                    "1",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(summary["stats"]["converted"], 1)
            self.assertEqual(summary["data_path"], "train.jsonl")
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)
        finally:
            output.unlink(missing_ok=True)
            report.unlink(missing_ok=True)
            errors.unlink(missing_ok=True)
            config.unlink(missing_ok=True)

    def test_write_export_records_bad_jsonl_line_and_continues(self) -> None:
        source = FIXTURE_ROOT / ".export-invalid-line-test.jsonl"
        output = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-invalid-line-test.jsonl"
        errors = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-invalid-line-test.errors.jsonl"
        record = json.loads((FIXTURE_ROOT / "train.jsonl").read_text(encoding="utf-8"))
        source.write_text(
            json.dumps(record) + "\n{bad json}\n" + json.dumps(record) + "\n",
            encoding="utf-8",
        )
        try:
            adapter = VisualRFTAdapter(make_paths(), data_path=source.name)
            stats = write_export(adapter.iter_results(), output, errors, include_metadata=False)

            self.assertEqual(stats["total_raw"], 3)
            self.assertEqual(stats["converted"], 2)
            self.assertEqual(stats["skipped"], 1)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)
            error_row = json.loads(errors.read_text(encoding="utf-8"))
            self.assertIn("Invalid JSONL", error_row["errors"][0])
            self.assertEqual(error_row["raw_ref"]["source_line"], 2)
        finally:
            source.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            errors.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
