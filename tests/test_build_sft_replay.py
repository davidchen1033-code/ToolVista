from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts.build_sft_replay import write_transaction


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_sft_replay.py"


class BuildSFTReplayTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-replay-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    @staticmethod
    def write_rows(path: Path, source: str, count: int) -> None:
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": f"{source}-{index}"}],
                        "source": source,
                        "index": index,
                    }
                )
                + "\n"
                for index in range(count)
            ),
            encoding="utf-8",
        )

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_anchor_weights_build_expected_mix_and_cycle_small_replay(self) -> None:
        root = self.make_temp_dir()
        primary = root / "primary.jsonl"
        replay = root / "replay.jsonl"
        output = root / "mixed.jsonl"
        report = root / "mixed.report.json"
        self.write_rows(primary, "primary", 6)
        self.write_rows(replay, "replay", 1)

        process = self.run_cli(
            "--source", f"primary={primary}",
            "--source", f"replay={replay}",
            "--weight", "primary=0.6",
            "--weight", "replay=0.4",
            "--anchor", "primary",
            "--seed", "42",
            "--output", str(output),
            "--report", str(report),
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        summary = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(row["source"] == "primary" for row in rows), 6)
        self.assertEqual(sum(row["source"] == "replay" for row in rows), 4)
        self.assertEqual(summary["sources"]["replay"]["duplicate_count"], 3)
        self.assertEqual(summary["sources"]["primary"]["selected_count"], 6)
        self.assertEqual(len(summary["output_sha256"]), 64)

    def test_same_seed_is_deterministic_and_different_seed_changes_order(self) -> None:
        root = self.make_temp_dir()
        first = root / "first.jsonl"
        second = root / "second.jsonl"
        self.write_rows(first, "first", 5)
        self.write_rows(second, "second", 5)

        outputs = [root / f"out-{index}.jsonl" for index in range(3)]
        seeds = (7, 7, 8)
        for output, seed in zip(outputs, seeds):
            process = self.run_cli(
                "--source", f"first={first}",
                "--source", f"second={second}",
                "--weight", "first=1",
                "--weight", "second=1",
                "--anchor", "first",
                "--seed", str(seed),
                "--output", str(output),
            )
            self.assertEqual(process.returncode, 0, process.stderr)

        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertNotEqual(outputs[0].read_bytes(), outputs[2].read_bytes())

    def test_bad_json_does_not_create_outputs(self) -> None:
        root = self.make_temp_dir()
        bad = root / "bad.jsonl"
        output = root / "mixed.jsonl"
        report = root / "mixed.report.json"
        bad.write_text('{"messages": []}\nnot-json\n', encoding="utf-8")

        process = self.run_cli(
            "--source", f"bad={bad}",
            "--weight", "bad=1",
            "--anchor", "bad",
            "--output", str(output),
            "--report", str(report),
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("line 2", process.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(report.exists())

    def test_empty_non_object_and_duplicate_names_are_rejected(self) -> None:
        root = self.make_temp_dir()
        empty = root / "empty.jsonl"
        array = root / "array.jsonl"
        output = root / "mixed.jsonl"
        empty.write_text("\n", encoding="utf-8")
        array.write_text("[]\n", encoding="utf-8")

        cases = (
            (["--source", f"empty={empty}", "--weight", "empty=1", "--anchor", "empty"], "no JSON objects"),
            (["--source", f"array={array}", "--weight", "array=1", "--anchor", "array"], "JSON object"),
            (
                [
                    "--source", f"same={array}", "--source", f"same={empty}",
                    "--weight", "same=1", "--anchor", "same",
                ],
                "duplicate source",
            ),
        )
        for extra_args, expected in cases:
            with self.subTest(expected=expected):
                process = self.run_cli(*extra_args, "--output", str(output))
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(expected, process.stderr)
                self.assertFalse(output.exists())

    def test_existing_outputs_require_overwrite(self) -> None:
        root = self.make_temp_dir()
        source = root / "source.jsonl"
        output = root / "mixed.jsonl"
        report = root / "mixed.report.json"
        self.write_rows(source, "source", 2)
        output.write_text("old output\n", encoding="utf-8")
        report.write_text('{"old": true}\n', encoding="utf-8")
        args = (
            "--source", f"source={source}",
            "--weight", "source=1",
            "--anchor", "source",
            "--output", str(output),
            "--report", str(report),
        )

        rejected = self.run_cli(*args)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("--overwrite", rejected.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "old output\n")
        self.assertEqual(report.read_text(encoding="utf-8"), '{"old": true}\n')

        accepted = self.run_cli(*args, "--overwrite")
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["total_rows"], 2)

    def test_second_promotion_failure_restores_both_existing_outputs(self) -> None:
        root = self.make_temp_dir()
        output = root / "mixed.jsonl"
        report = root / "mixed.report.json"
        output.write_bytes(b"old-jsonl\n")
        report.write_bytes(b"old-report\n")
        real_replace = __import__("os").replace
        promotion_count = 0

        def fail_second_promotion(source: Path, destination: Path) -> None:
            nonlocal promotion_count
            if str(source).endswith(".tmp"):
                promotion_count += 1
                if promotion_count == 2:
                    raise OSError("promotion failed")
            real_replace(source, destination)

        with patch("scripts.build_sft_replay.os.replace", side_effect=fail_second_promotion):
            with self.assertRaisesRegex(OSError, "promotion failed"):
                write_transaction({output: b"new-jsonl\n", report: b"new-report\n"})

        self.assertEqual(output.read_bytes(), b"old-jsonl\n")
        self.assertEqual(report.read_bytes(), b"old-report\n")
        self.assertEqual(sorted(path.name for path in root.iterdir()), ["mixed.jsonl", "mixed.report.json"])


if __name__ == "__main__":
    unittest.main()
