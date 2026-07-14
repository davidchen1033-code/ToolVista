from __future__ import annotations

import json
import shutil
import subprocess
import unittest
import uuid
from pathlib import Path

from toolvista.evaluation.toolvqa import (
    map_metrics,
    normalize_api_base,
    prepare_data,
    render_runtime_config,
    resolve_toolmeta,
    run_toolvqa,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "data" / "toolvqa"


class ToolVQAEvaluationTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-toolvqa-eval-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_normalize_api_base_keeps_exactly_one_v1(self) -> None:
        self.assertEqual(normalize_api_base("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1")
        self.assertEqual(normalize_api_base("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1")

    def test_prepare_data_rewrites_fixture_image_to_absolute_path(self) -> None:
        output = self.make_temp_dir() / "prepared"

        prepared, is_fixture, sample_count = prepare_data(FIXTURE_ROOT, output)

        row = json.loads(prepared.read_text(encoding="utf-8"))["0"]
        image = Path(row["files"][0]["path"])
        self.assertTrue(image.is_absolute())
        self.assertTrue(image.is_file())
        self.assertEqual(
            row["dialogs"][1]["tool_calls"][0]["function"]["arguments"]["image"],
            str(image),
        )
        self.assertTrue(is_fixture)
        self.assertEqual(sample_count, 1)

    def test_prepare_data_rejects_missing_and_ambiguous_media(self) -> None:
        root = self.make_temp_dir()
        source = root / "source"
        source.mkdir()
        manifest = {
            "0": {
                "tools": [{"name": "OCR"}],
                "files": [{"type": "image", "path": "/old/path/a.jpg"}],
                "dialogs": [],
                "gt_answer": [],
            }
        }
        (source / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
        media = root / "media"
        media.mkdir()

        with self.assertRaisesRegex(FileNotFoundError, "a.jpg"):
            prepare_data(source, root / "missing", media_root=media)

        (media / "one").mkdir()
        (media / "two").mkdir()
        (media / "one" / "a.jpg").write_bytes(b"one")
        (media / "two" / "a.jpg").write_bytes(b"two")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            prepare_data(source, root / "ambiguous", media_root=media)

    def test_render_runtime_config_separates_step_and_e2e(self) -> None:
        root = self.make_temp_dir()
        common = dict(
            data_root=root / "data",
            toolmeta=root / "toolmeta.json",
            api_base="http://127.0.0.1:8000/v1",
            api_model="qwen3-vl",
            limit=1,
        )

        step = render_runtime_config(mode="step", tool_url=None, **common)
        self.assertIn("every_with_gt", step)
        self.assertIn("tool_meta=", step)
        self.assertNotIn("tool_server=", step)
        self.assertIn("test_range='[0:1]'", step)

        e2e = render_runtime_config(
            mode="e2e",
            tool_url="http://127.0.0.1:16181",
            **common,
        )
        self.assertIn("mode='every'", e2e)
        self.assertIn("tool_server=", e2e)
        self.assertNotIn("tool_meta=", e2e)

    def test_resolve_toolmeta_falls_back_to_upstream_data(self) -> None:
        root = self.make_temp_dir()
        upstream = root / "ToolVQA-release" / "opencompass" / "data" / "ToolVQA"
        upstream.mkdir(parents=True)
        expected = upstream / "toolmeta.json"
        expected.write_text("{}", encoding="utf-8")
        data_root = root / "fixture"
        data_root.mkdir()

        self.assertEqual(
            resolve_toolmeta(data_root, root / "ToolVQA-release"),
            expected.resolve(),
        )

    def test_map_metrics_uses_only_official_mode_fields(self) -> None:
        step = map_metrics(
            {
                "inst_align": 80.0,
                "tool_acc": 70.0,
                "arg_acc": 60.0,
                "answer_acc": 50.0,
            },
            "step",
        )
        self.assertEqual(step["toolvqa_sbs/tool_accuracy"], 70.0)

        e2e = map_metrics(
            {
                "answer_acc": 40.0,
                "p_f1": 30.0,
                "o_f1": 20.0,
                "l_f1": 10.0,
                "c_f1": 5.0,
            },
            "e2e",
        )
        self.assertEqual(e2e["toolvqa_e2e/creativity_f1"], 5.0)
        with self.assertRaisesRegex(ValueError, "tool_acc"):
            map_metrics({"inst_align": 1.0}, "step")

    def test_run_toolvqa_preserves_logs_and_parses_official_json(self) -> None:
        root = self.make_temp_dir()
        toolvqa_root = root / "ToolVQA-release"
        opencompass = toolvqa_root / "opencompass"
        opencompass.mkdir(parents=True)
        (opencompass / "run.py").write_text("# fake", encoding="utf-8")
        upstream_meta = opencompass / "data" / "ToolVQA"
        upstream_meta.mkdir(parents=True)
        (upstream_meta / "toolmeta.json").write_text("{}", encoding="utf-8")
        output = root / "run"
        captured: dict[str, object] = {}

        def fake_runner(command, **kwargs):
            captured["command"] = command
            captured["cwd"] = kwargs["cwd"]
            result_path = (
                output
                / "upstream"
                / "20260714_120000"
                / "results"
                / "toolvista_qwen3vl"
                / "gta_bench.json"
            )
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(
                    {
                        "inst_align": 80.0,
                        "tool_acc": 70.0,
                        "arg_acc": 60.0,
                        "answer_acc": 50.0,
                        "tool_call": 3,
                        "tool_call_error": 0,
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="official stdout",
                stderr="",
            )

        result = run_toolvqa(
            mode="step",
            toolvqa_root=toolvqa_root,
            data_root=FIXTURE_ROOT,
            output_dir=output,
            api_base="http://127.0.0.1:8000",
            api_model="qwen3-vl",
            python_executable="python",
            workers=1,
            limit=1,
            command_runner=fake_runner,
        )

        self.assertEqual(captured["cwd"], str(opencompass.resolve()))
        self.assertIn("--eval_mode", captured["command"])
        self.assertEqual(result["metrics"]["toolvqa_sbs/tool_accuracy"], 70.0)
        self.assertFalse(result["comparable"])
        self.assertEqual(
            (output / "stdout.log").read_text(encoding="utf-8"),
            "official stdout",
        )
        self.assertTrue((output / "runtime_config.py").is_file())
        self.assertTrue((output / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
