from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.export_mllm_tool import normalize_file_keys, write_export
from toolvista.adapters.mllm_tool import MLLMToolAdapter
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "examples" / "data" / "mllm_tool"


def make_paths() -> ToolVistaPaths:
    return ToolVistaPaths(
        config_path=Path("test"),
        raw={
            "datasets": {
                "mllm_tool": {
                    "root": str(SAMPLE_ROOT),
                    "instruction_dir": "data/IT_data_ins/T+X-T_data",
                    "media_dir": "data/IT_data_ins/T+X-T_data/mm_dataset",
                    "files": {
                        "image": "image_tx2t.json",
                        "video": "video_tx2t.json",
                        "audio": "audio_tx2t.json",
                        "text": "text_t2t.json",
                        "combined": "combined_data.json",
                    },
                }
            }
        },
    )


class MLLMToolExportTest(unittest.TestCase):
    def test_normalize_file_keys_accepts_csv_and_rejects_unknown_values(self) -> None:
        self.assertEqual(normalize_file_keys("image, text,image"), ("image", "text"))
        with self.assertRaisesRegex(ValueError, "Unknown MLLM-Tool file key"):
            normalize_file_keys("image,unknown")

    def test_default_adapter_selection_excludes_combined(self) -> None:
        adapter = MLLMToolAdapter(make_paths(), file_keys=normalize_file_keys(None), validate_media_files=False)

        items = list(adapter.iter_raw())

        self.assertEqual([item["_toolvista_file_key"] for item in items], ["image", "text"])

    def test_combined_can_be_exported_explicitly(self) -> None:
        adapter = MLLMToolAdapter(make_paths(), file_keys=normalize_file_keys("combined"))

        result = next(adapter.iter_results())

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertIn("combined", result.sample.sample_id)

    def test_write_export_writes_text_and_image_rows(self) -> None:
        adapter = MLLMToolAdapter(
            make_paths(),
            file_keys=normalize_file_keys("image,text"),
            validate_media_files=False,
        )
        output = REPO_ROOT / "examples" / "ms_swift" / ".mllm-tool-test.jsonl"
        errors = REPO_ROOT / "examples" / "ms_swift" / ".mllm-tool-test.errors.jsonl"
        try:
            stats = write_export(adapter.iter_results(), output, errors, include_metadata=True)

            self.assertEqual(stats["total_raw"], 2)
            self.assertEqual(stats["converted"], 2)
            self.assertEqual(stats["skipped"], 0)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            image_row = next(row for row in rows if "images" in row)
            text_row = next(row for row in rows if "images" not in row)
            self.assertEqual(image_row["messages"][0]["content"].count("<image>"), 1)
            self.assertTrue(image_row["images"][0].endswith("mm_dataset/image/sample.jpg"))
            self.assertNotIn("images", text_row)
            self.assertEqual(errors.read_text(encoding="utf-8"), "")
        finally:
            output.unlink(missing_ok=True)
            errors.unlink(missing_ok=True)

    def test_strict_media_validation_reports_missing_file(self) -> None:
        adapter = MLLMToolAdapter(make_paths(), file_keys="image", validate_media_files=True)
        output = REPO_ROOT / "examples" / "ms_swift" / ".mllm-tool-strict-test.jsonl"
        errors = REPO_ROOT / "examples" / "ms_swift" / ".mllm-tool-strict-test.errors.jsonl"
        try:
            stats = write_export(adapter.iter_results(), output, errors, include_metadata=False)

            self.assertEqual(stats["converted"], 0)
            self.assertEqual(stats["skipped"], 1)
            error_row = json.loads(errors.read_text(encoding="utf-8"))
            self.assertIn("Media file does not exist", error_row["errors"][0])
        finally:
            output.unlink(missing_ok=True)
            errors.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
