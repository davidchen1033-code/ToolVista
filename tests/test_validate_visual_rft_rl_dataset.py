from __future__ import annotations

import json
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_visual_rft_rl_dataset.py"


class VisualRFTRLValidationTest(unittest.TestCase):
    def make_dataset(self, row: dict) -> Path:
        root = REPO_ROOT / "tests" / f"tmp-rl-validation-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        path = root / "train.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def run_validator(self, path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def row(task: str = "grounding") -> dict:
        return {
            "messages": [{"role": "user", "content": "<image>Find the target"}],
            "images": ["image.jpg"],
            "solution": "<answer>[{\"Position\":[0,0,10,10],\"Confidence\":1}]</answer>",
            "task": task,
        }

    def test_grounding_row_passes(self) -> None:
        process = self.run_validator(self.make_dataset(self.row()))

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("reward validation passed", process.stdout)

    def test_classification_row_passes_for_routed_answer_reward(self) -> None:
        process = self.run_validator(self.make_dataset(self.row("classification")))

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_unknown_task_is_rejected_before_training(self) -> None:
        process = self.run_validator(self.make_dataset(self.row("segmentation")))

        self.assertNotEqual(process.returncode, 0)
        self.assertIn("not supported by visual_rft", process.stderr)

    def test_missing_solution_or_user_prompt_is_rejected(self) -> None:
        missing_solution = self.row()
        missing_solution.pop("solution")
        assistant_prompt = self.row()
        assistant_prompt["messages"][-1]["role"] = "assistant"

        for row, expected in ((missing_solution, "solution"), (assistant_prompt, "end with a user")):
            with self.subTest(expected=expected):
                process = self.run_validator(self.make_dataset(row))
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(expected, process.stderr)


if __name__ == "__main__":
    unittest.main()
