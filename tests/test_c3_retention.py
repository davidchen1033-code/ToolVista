from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

from scripts.check_c3_retention import evaluate_retention
from scripts.prepare_c3_direct_answer_retention import build_rows, direct_answer_row
from toolvista.router_data import (
    ROUTER_MODE_MARKER,
    build_router_catalog,
    with_router_system_prompt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def summary(router: float, realworld: float, mathvision: float, mmmu: float, ocr: float) -> dict:
    return {
        "status": "passed",
        "benchmarks": {
            "mllm_router_v2": {
                "test_summary": {
                    "dataset": "MLLM-Tool Router v2 / router_v2_fixed_seed_dev",
                    "method": "ToolVista MLLM Router v2 per-row-catalog scoring",
                },
                "metrics": {"mllm_router/top1_accuracy": router},
            },
            "qwen3vl_general_light": {
                "metrics": {
                    "vlmevalkit/realworldqa/acc": realworld,
                    "vlmevalkit/mathvision_mini/accuracy": mathvision,
                    "vlmevalkit/mmmu_dev_val/accuracy": mmmu,
                    "vlmevalkit/ocrbench/score": ocr,
                }
            },
        },
    }


class C3RetentionTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-c3-retention-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_router_catalog_is_bounded_deterministic_and_contains_targets(self) -> None:
        candidates = [f"tool-{index}" for index in range(100)]
        first = build_router_catalog(
            ["tool-42"], candidates, identity="sample", seed=42, max_size=8
        )
        second = build_router_catalog(
            ["tool-42"], candidates, identity="sample", seed=42, max_size=8
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertIn("tool-42", first)
        messages = with_router_system_prompt(
            [{"role": "user", "content": "answer this"}], first
        )
        self.assertIn(ROUTER_MODE_MARKER, messages[0]["content"])
        self.assertEqual(messages[1]["content"], "answer this")

    def test_direct_answer_rows_remove_tool_trajectory_and_schema(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "<image>What is shown?"},
                {"role": "tool_call", "content": "call"},
                {"role": "tool_response", "content": "result"},
                {"role": "assistant", "content": "A bridge"},
            ],
            "images": ["image.jpg"],
            "tools": "secret schema",
        }
        converted, reason = direct_answer_row(row, source_row=1)
        self.assertIsNone(reason)
        assert converted is not None
        self.assertEqual(
            [message["role"] for message in converted["messages"]],
            ["user", "assistant"],
        )
        self.assertNotIn("tools", converted)
        self.assertEqual(converted["messages"][-1]["content"], "A bridge")

    def test_direct_answer_selection_is_deterministic(self) -> None:
        root = self.make_temp_dir()
        source = root / "source.jsonl"
        source.write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"<image>question {index}"},
                            {"role": "assistant", "content": f"answer {index}"},
                        ],
                        "images": [f"{index}.jpg"],
                    }
                )
                + "\n"
                for index in range(5)
            ),
            encoding="utf-8",
        )
        first, _ = build_rows(source, seed=42, max_rows=3)
        second, _ = build_rows(source, seed=42, max_rows=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_retention_gate_rejects_observed_c3_collapse(self) -> None:
        baseline = summary(0.0, 0.75, 0.24, 0.43, 98.0)
        collapsed = summary(0.695, 0.06, 0.11, 0.0, 0.0)
        result = evaluate_retention(
            collapsed,
            baseline,
            router_minimum=0.60,
            general_retention=0.80,
            mathvision_retention=0.70,
        )
        self.assertFalse(result["passed"])
        by_metric = {check["metric"]: check for check in result["checks"]}
        self.assertTrue(by_metric["mllm_router/top1_accuracy"]["passed"])
        self.assertFalse(by_metric["vlmevalkit/mmmu_dev_val/accuracy"]["passed"])

    def test_retention_gate_authorizes_only_exact_checkpoint(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint-800"
        other = root / "checkpoint-other"
        checkpoint.mkdir()
        other.mkdir()
        gate = root / "gate.json"
        gate.write_text(
            json.dumps(
                {
                    "passed": True,
                    "authorized_checkpoint": str(checkpoint.resolve()),
                }
            ),
            encoding="utf-8",
        )
        script = REPO_ROOT / "scripts/verify_c3_retention_gate.py"
        accepted = subprocess.run(
            [sys.executable, str(script), "--gate", str(gate), "--checkpoint", str(checkpoint)],
            capture_output=True,
            text=True,
            check=False,
        )
        rejected = subprocess.run(
            [sys.executable, str(script), "--gate", str(gate), "--checkpoint", str(other)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("mismatch", rejected.stderr)

    def test_gate_cli_rejects_summary_from_another_checkpoint(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint-800"
        checkpoint.mkdir()
        candidate = summary(0.7, 0.7, 0.2, 0.4, 90.0)
        candidate["checkpoint"] = str(root / "wrong-checkpoint")
        baseline = summary(0.0, 0.75, 0.24, 0.43, 98.0)
        baseline["checkpoint"] = "c2"
        candidate_path = root / "candidate.json"
        baseline_path = root / "baseline.json"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        process = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/check_c3_retention.py"),
                "--candidate-summary",
                str(candidate_path),
                "--baseline-summary",
                str(baseline_path),
                "--authorized-checkpoint",
                str(checkpoint),
                "--output",
                str(root / "gate.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("does not match", process.stderr)
        self.assertFalse((root / "gate.json").exists())

    def test_gate_cli_rejects_old_router_protocol_baseline(self) -> None:
        root = self.make_temp_dir()
        checkpoint = root / "checkpoint-800"
        checkpoint.mkdir()
        candidate = summary(0.7, 0.7, 0.2, 0.4, 90.0)
        candidate["checkpoint"] = str(checkpoint)
        baseline = summary(0.0, 0.75, 0.24, 0.43, 98.0)
        baseline["checkpoint"] = "c2"
        baseline["benchmarks"]["mllm_router_v2"]["test_summary"]["dataset"] = (
            "MLLM-Tool Router fixed-seed manifest"
        )
        baseline["benchmarks"]["mllm_router_v2"]["test_summary"]["method"] = (
            "legacy router scoring"
        )
        candidate_path = root / "candidate.json"
        baseline_path = root / "baseline.json"
        candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

        process = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/check_c3_retention.py"),
                "--candidate-summary",
                str(candidate_path),
                "--baseline-summary",
                str(baseline_path),
                "--authorized-checkpoint",
                str(checkpoint),
                "--output",
                str(root / "gate.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("does not use the MLLM Router v2 protocol", process.stderr)
        self.assertFalse((root / "gate.json").exists())


if __name__ == "__main__":
    unittest.main()
