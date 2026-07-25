from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/autodl/legacy/15_train_visual_rft_dapo_datafixed.sh"


class C4DatafixedLauncherTest(unittest.TestCase):
    def test_launcher_is_thin_and_keeps_datafixed_lineage_separate(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("C3_DATAFIXED_CHECKPOINT", text)
        self.assertIn("toolvista_c3_datafixed_retention_gate.json", text)
        self.assertIn('gate.get("variant") != "datafixed"', text)
        self.assertIn("dapo_datafixed_seed", text)
        self.assertIn('tee "${TRAIN_LOG}"', text)
        self.assertIn("10_train_visual_rft_dapo.sh", text)
        self.assertNotIn("swift rlhf", text)

    def test_launcher_has_valid_bash_syntax(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
