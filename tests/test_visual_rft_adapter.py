from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from toolvista.adapters.visual_rft import VisualRFTAdapter
from toolvista.exporters.ms_swift import canonical_to_ms_swift_row
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "data" / "visual_rft"


def make_paths(
    root: Path,
    *,
    data_path: str = "train.jsonl",
    default_task: str | None = "vqa",
) -> ToolVistaPaths:
    visual_rft = {
        "root": str(root),
        "data_path": data_path,
        "default_split": "train",
        "materialized_media_dir": "materialized_images",
    }
    if default_task is not None:
        visual_rft["default_task"] = default_task
    return ToolVistaPaths(
        config_path=Path("test"),
        raw={
            "datasets": {
                "visual_rft": visual_rft,
            }
        },
    )


class VisualRFTAdapterTest(unittest.TestCase):
    def test_jsonl_visual_prompt_preserves_reward_columns_without_assistant_target(self) -> None:
        record = json.loads((FIXTURE_ROOT / "train.jsonl").read_text(encoding="utf-8").strip())
        adapter = VisualRFTAdapter(make_paths(FIXTURE_ROOT))
        result = next(adapter.iter_results())

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        sample = result.sample
        self.assertEqual([message.role for message in sample.messages], ["system", "user"])
        self.assertEqual([part.type for part in sample.messages[-1].content_parts], ["image", "text"])
        self.assertEqual(sample.messages[-1].content, record["problem"])
        expected_image = FIXTURE_ROOT / ".." / "toolvqa" / "images" / "part3" / "TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg"
        self.assertEqual(sample.media.images, [expected_image.resolve().as_posix()])
        self.assertEqual(sample.labels["solution"], record["solution"])
        self.assertEqual(sample.labels["ground_truth"], record["original_answer"])
        self.assertEqual(sample.labels["task"], "classification")
        self.assertEqual(sample.task_type, "vqa")

        row = canonical_to_ms_swift_row(sample)
        self.assertEqual([message["role"] for message in row["messages"]], ["system", "user"])
        self.assertEqual(row["messages"][-1]["content"], f"<image>{record['problem']}")
        self.assertEqual(row["messages"][-1]["content"].count("<image>"), 1)
        self.assertEqual(row["images"], sample.media.images)
        self.assertNotIn("solution", row)
        self.assertNotIn(record["solution"], json.dumps(row, ensure_ascii=False))

    def test_grounding_task_uses_grounding_canonical_type(self) -> None:
        adapter = VisualRFTAdapter(make_paths(FIXTURE_ROOT), validate_media_files=True)
        raw = {
            "image": "../toolvqa/images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg",
            "problem": "Locate the object.",
            "solution": "<answer>(10,10),(20,20)</answer>",
            "task": "grounding",
            "_toolvista_raw_index": 0,
        }

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual(result.sample.task_type, "grounding")

    def test_missing_solution_is_rejected(self) -> None:
        adapter = VisualRFTAdapter(make_paths(Path(".")), validate_media_files=False)

        result = adapter.adapt(
            {
                "image": "sample.png",
                "problem": "Question",
                "_toolvista_raw_index": 4,
            }
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.errors, ["Missing or empty solution."])

    def test_missing_task_is_rejected_without_explicit_default(self) -> None:
        adapter = VisualRFTAdapter(make_paths(Path("."), default_task=None), validate_media_files=False)

        result = adapter.adapt(
            {
                "image": "sample.png",
                "problem": "Question",
                "solution": "<answer>A</answer>",
                "_toolvista_raw_index": 4,
            }
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.errors, ["Missing task; set a source task or pass an explicit default task."])

    def test_missing_image_file_is_rejected_in_strict_mode(self) -> None:
        adapter = VisualRFTAdapter(make_paths(FIXTURE_ROOT), validate_media_files=True)

        result = adapter.adapt(
            {
                "image": "missing.png",
                "problem": "Question",
                "solution": "<answer>A</answer>",
                "_toolvista_raw_index": 2,
            }
        )

        self.assertFalse(result.ok)
        self.assertIn("Image file does not exist", result.errors[0])

    def test_pil_image_from_huggingface_dataset_is_materialized(self) -> None:
        adapter = VisualRFTAdapter(make_paths(FIXTURE_ROOT), validate_media_files=True)
        raw = {
            "image": Image.new("RGB", (32, 32), "red"),
            "problem": "Question",
            "solution": "<answer>A</answer>",
            "_toolvista_raw_index": 7,
        }
        materialized = None
        try:
            result = adapter.adapt(raw)

            self.assertTrue(result.ok, result.to_report_dict())
            assert result.sample is not None
            materialized = Path(result.sample.media.images[0])
            self.assertTrue(materialized.is_file())
            self.assertEqual(materialized.parent, FIXTURE_ROOT / "materialized_images" / "train")
            self.assertTrue(materialized.name.startswith("7-"))
            self.assertTrue(materialized.name.endswith("-0.png"))
            with Image.open(materialized) as saved_image:
                self.assertEqual(saved_image.size, (32, 32))
        finally:
            if materialized is not None:
                materialized.unlink(missing_ok=True)

    def test_materialized_images_do_not_reuse_stale_or_sanitized_collision_paths(self) -> None:
        adapter = VisualRFTAdapter(make_paths(FIXTURE_ROOT), validate_media_files=True)
        created: list[Path] = []
        try:
            results = []
            for source_id, color in (("a/b", "red"), ("a b", "blue"), ("a/b", "green")):
                result = adapter.adapt(
                    {
                        "id": source_id,
                        "image": Image.new("RGB", (4, 4), color),
                        "problem": "Question",
                        "solution": "<answer>A</answer>",
                        "task": "classification",
                        "_toolvista_raw_index": 0,
                    }
                )
                self.assertTrue(result.ok, result.to_report_dict())
                assert result.sample is not None
                created.append(Path(result.sample.media.images[0]))
                results.append(result.sample.media.images[0])

            self.assertEqual(len(set(results)), 3)
            self.assertEqual(Image.open(created[0]).getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(Image.open(created[1]).getpixel((0, 0)), (0, 0, 255))
            self.assertEqual(Image.open(created[2]).getpixel((0, 0)), (0, 128, 0))
        finally:
            for path in created:
                path.unlink(missing_ok=True)

    def test_invalid_jsonl_line_is_reported_and_later_rows_continue(self) -> None:
        source = FIXTURE_ROOT / ".invalid-line-test.jsonl"
        image = "../toolvqa/images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg"
        valid = {
            "image": image,
            "problem": "Question",
            "solution": "<answer>A</answer>",
            "task": "classification",
        }
        source.write_text(
            json.dumps(valid) + "\n{bad json}\n" + json.dumps(valid) + "\n",
            encoding="utf-8",
        )
        try:
            results = list(VisualRFTAdapter(make_paths(FIXTURE_ROOT, data_path=source.name)).iter_results())

            self.assertEqual(len(results), 3)
            self.assertTrue(results[0].ok)
            self.assertFalse(results[1].ok)
            self.assertIn("Invalid JSONL", results[1].errors[0])
            self.assertEqual(results[1].raw_ref["source_line"], 2)
            self.assertTrue(results[2].ok)
        finally:
            source.unlink(missing_ok=True)

    def test_save_to_disk_directory_reports_missing_datasets_dependency(self) -> None:
        adapter = VisualRFTAdapter(make_paths(FIXTURE_ROOT, data_path="hf_dataset"))
        original_import = __import__

        def fail_datasets_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_datasets_import):
            with self.assertRaisesRegex(RuntimeError, "requires the 'datasets' package"):
                list(adapter.iter_raw())


if __name__ == "__main__":
    unittest.main()
