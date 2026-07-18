from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_bfcl_eval import CATEGORIES, build_manifests
from toolvista.evaluation.bfcl import CSV_METRICS, parse_non_live_csv, run_bfcl_benchmark
from toolvista.evaluation.models import BenchmarkSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
BFCL_PYTHON = Path("/root/autodl-tmp/venvs/bfcl/bin/python")


def bfcl_spec(manifest: Path, bootstrap: Path, python: Path = BFCL_PYTHON) -> BenchmarkSpec:
    return BenchmarkSpec.from_mapping(
        "bfcl_non_live_light",
        {
            "execution": "bfcl",
            "profile": "light",
            "method": "BFCL",
            "source_project": "Berkeley Function-Calling Leaderboard",
            "dataset_name": "BFCL v4 non-live",
            "split": "fixed",
            "evaluator": "bfcl_eval",
            "dataset_path": manifest.as_posix(),
            "settings": {
                "python": python.as_posix(),
                "bootstrap": bootstrap.as_posix(),
                "num_threads": 2,
            },
            "primary_metrics": [
                "bfcl/non_live_overall_accuracy",
                "bfcl/irrelevance_detection",
            ],
            "metric_directions": {
                "bfcl/non_live_overall_accuracy": "maximize",
                "bfcl/irrelevance_detection": "maximize",
            },
        },
    )


class BfclIntegrationTest(unittest.TestCase):
    def test_freezer_is_deterministic_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tests") as directory:
            data = Path(directory)
            for category in CATEGORIES:
                (data / f"BFCL_v4_{category}.json").write_text(
                    "".join(
                        json.dumps({"id": f"{category}_{index}"}) + "\n"
                        for index in range(40)
                    ),
                    encoding="utf-8",
                )
            full, first = build_manifests(data, seed=42, light_limit=200)
            _, second = build_manifests(data, seed=42, light_limit=200)
            self.assertEqual(first, second)
            self.assertEqual(sum(map(len, full.values())), 280)
            self.assertEqual(sum(map(len, first.values())), 200)
            self.assertEqual(tuple(first), CATEGORIES)
            self.assertLessEqual(max(map(len, first.values())) - min(map(len, first.values())), 1)

    def test_csv_parser_maps_official_percentages_and_rejects_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tests") as directory:
            path = Path(directory) / "data_non_live.csv"
            headers = ["Rank", "Model", *CSV_METRICS]
            values = ["1", "ToolVista", *(["50.00%"] * len(CSV_METRICS))]
            path.write_text(",".join(headers) + "\n" + ",".join(values) + "\n", encoding="utf-8")
            row, metrics = parse_non_live_csv(path)
            self.assertEqual(row["Model"], "ToolVista")
            self.assertEqual(metrics["bfcl/non_live_overall_accuracy"], 0.5)
            path.write_text(
                ",".join(headers) + "\n" + ",".join(["1", "ToolVista", "nan%", *(["50.00%"] * (len(CSV_METRICS) - 1))]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "finite"):
                parse_non_live_csv(path)

    @unittest.skipUnless(BFCL_PYTHON.is_file(), "local BFCL environment is unavailable")
    def test_bootstrap_registers_toolvista_handler_without_modifying_package(self) -> None:
        bootstrap = REPO_ROOT / "scripts/bfcl_toolvista_bootstrap.py"
        manifest = REPO_ROOT / "data/eval/bfcl/light.json"
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tests") as directory:
            process = subprocess.run(
                [
                    str(BFCL_PYTHON),
                    str(bootstrap),
                    "--project-root",
                    directory,
                    "--id-manifest",
                    str(manifest),
                    "--api-base",
                    "http://127.0.0.1:8000/v1",
                    "--api-model",
                    "Qwen3-VL-4B-Instruct",
                    "--check-registration",
                ],
                capture_output=True,
                text=True,
                check=False,
                env={**__import__("os").environ, "OMP_NUM_THREADS": "1"},
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout)["registry_name"], "toolvista-qwen3vl-fc")

    def test_runner_parses_official_artifacts_from_subprocess(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT / "tests") as directory:
            root = Path(directory)
            manifest = root / "ids.json"
            manifest.write_text(
                json.dumps({category: [f"{category}_0"] for category in CATEGORIES}),
                encoding="utf-8",
            )
            bootstrap = root / "bootstrap.py"
            bootstrap.write_text("# fixture\n", encoding="utf-8")
            python = root / "python"
            python.write_text("", encoding="utf-8")
            benchmark = root / "benchmark"

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                del kwargs
                workspace = Path(command[command.index("--project-root") + 1])
                result_dir = workspace / "result/toolvista-qwen3vl-fc/non_live"
                result_dir.mkdir(parents=True)
                for category in CATEGORIES:
                    (result_dir / f"BFCL_v4_{category}_result.json").write_text(
                        json.dumps({"id": f"{category}_0", "result": []}) + "\n",
                        encoding="utf-8",
                    )
                score = workspace / "score"
                score.mkdir()
                headers = ["Rank", "Model", *CSV_METRICS]
                values = ["1", "ToolVista", *(["75.00%"] * len(CSV_METRICS))]
                (score / "data_non_live.csv").write_text(
                    ",".join(headers) + "\n" + ",".join(values),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "ok", "")

            result = run_bfcl_benchmark(
                bfcl_spec(manifest, bootstrap, python),
                checkpoint="checkpoint",
                run_id="run",
                benchmark_dir=benchmark,
                api_base="http://127.0.0.1:8000/v1",
                api_model="model",
                timeout=10.0,
                score_only=False,
                resume=False,
                display_command="evaluate",
                command_runner=fake_run,
            )
            self.assertEqual(result.status, "passed")
            self.assertEqual((result.attempted, result.completed, result.failed), (7, 7, 0))
            self.assertEqual(result.metrics["bfcl/non_live_overall_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
