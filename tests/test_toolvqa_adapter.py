from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from toolvista.adapters.toolvqa import ToolVQAAdapter
from toolvista.exporters.ms_swift import canonical_to_ms_swift_row
from toolvista.utils.paths import ToolVistaPaths
from scripts.export_toolvqa import result_records, write_export


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = REPO_ROOT / "examples" / "data" / "toolvqa"


def make_paths(*, root: Path = SAMPLE_ROOT, train_file: str = "train_sample.jsonl") -> ToolVistaPaths:
    return ToolVistaPaths(
        config_path=Path("test"),
        raw={
            "datasets": {
                "toolvqa": {
                    "root": str(root),
                    "media_dir": "images",
                    "files": {"train": train_file},
                }
            }
        },
    )


class ToolVQAAdapterTest(unittest.TestCase):
    def _streaming_paths(self) -> tuple[Path, Path]:
        token = uuid.uuid4().hex
        root = REPO_ROOT / "examples" / "ms_swift"
        return root / f".toolvqa-test-{token}.jsonl", root / f".toolvqa-test-{token}.errors.jsonl"

    def _remove_streaming_files(self, output: Path, errors: Path) -> None:
        output.unlink(missing_ok=True)
        errors.unlink(missing_ok=True)

    def _temporary_dataset_root(self) -> Path:
        root = SAMPLE_ROOT / f".toolvqa-test-{uuid.uuid4().hex}"
        root.mkdir()
        return root

    def test_real_sample_converts_to_ms_swift_agent_trajectory(self) -> None:
        adapter = ToolVQAAdapter(make_paths())
        result = adapter.adapt(next(adapter.iter_raw()))

        self.assertTrue(result.ok, result.to_report_dict())
        sample = result.sample
        assert sample is not None
        self.assertEqual(sample.task_type, "agent_trajectory")
        self.assertEqual(
            [message.role for message in sample.messages],
            [
                "user",
                "tool_call",
                "tool_response",
                "tool_call",
                "tool_response",
                "tool_call",
                "tool_response",
                "assistant",
            ],
        )
        self.assertEqual([tool.name for tool in sample.tools], ["ImageDescription", "OCR", "GoogleSearch"])

        first_call = json.loads(sample.messages[1].content)
        self.assertEqual(first_call, {"name": "ImageDescription", "arguments": {"image": "images[0]"}})
        self.assertEqual(sample.messages[2].content, next(adapter.iter_raw())["context"][0]["output"])
        self.assertEqual(sample.messages[-1].content, "Men's collection")
        self.assertEqual(sample.metadata["tool_count"], 3)
        self.assertEqual(len(sample.metadata["reasoning_trace"]), 3)

        tools = {tool.name: tool for tool in sample.tools}
        self.assertEqual(tools["OCR"].description, "This tool can recognize all text on the input image.")
        self.assertEqual(tools["OCR"].parameters["required"], ["image"])
        self.assertEqual(tools["OCR"].parameters["properties"]["lang"]["type"], "string")

        row = canonical_to_ms_swift_row(sample)
        self.assertEqual(row["messages"][0]["content"], "<image>Are these leggings likely part of a men's or women's collection?")
        self.assertEqual(row["messages"][1]["role"], "tool_call")
        self.assertEqual(row["messages"][2]["role"], "tool_response")
        self.assertEqual(row["messages"][-1], {"role": "assistant", "content": "Men's collection"})
        self.assertEqual(len(row["images"]), 1)
        self.assertTrue(Path(row["images"][0]).is_file())
        self.assertEqual(sum(message["content"].count("<image>") for message in row["messages"]), 1)
        self.assertEqual([tool["function"]["name"] for tool in json.loads(row["tools"])], ["ImageDescription", "OCR", "GoogleSearch"])

    def test_known_tool_schema_does_not_make_optional_arguments_required(self) -> None:
        adapter = ToolVQAAdapter(make_paths())
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [
            dict(raw["context"][1], input='{"image":"images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg"}'),
            dict(raw["context"][1], input='{"image":"images/part3/TB1tM0jLXXXXXa1XFXXunYpLFXX.jpg","lang":"en"}'),
        ]

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual(result.sample.tools[0].parameters["required"], ["image"])

    def test_absolute_posix_image_path_is_preserved(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)

        self.assertEqual(adapter._resolve_image_path("/data/images/a.jpg"), "/data/images/a.jpg")

    def test_image_argument_with_same_stem_but_different_suffix_is_rejected(self) -> None:
        adapter = ToolVQAAdapter(make_paths())
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [dict(raw["context"][0], input='{"image":"image/TB1tM0jLXXXXXa1XFXXunYpLFXX.png"}')]

        result = adapter.adapt(raw)

        self.assertFalse(result.ok)
        self.assertIn("other than the original input", result.errors[0])

    def test_jsonl_reader_accepts_utf8_bom(self) -> None:
        root = self._temporary_dataset_root()
        annotation = root / "train.jsonl"
        try:
            annotation.write_text('\ufeff{"value": 1}\n', encoding="utf-8")
            items = list(ToolVQAAdapter(make_paths(root=root, train_file=annotation.name)).iter_raw())

            self.assertEqual(items[0]["value"], 1)
        finally:
            annotation.unlink(missing_ok=True)
            root.rmdir()

    def test_jsonl_max_items_counts_records_instead_of_blank_lines(self) -> None:
        root = self._temporary_dataset_root()
        annotation = root / "train.jsonl"
        try:
            annotation.write_text('\n{"value": 1}\n{"value": 2}\n', encoding="utf-8")
            adapter = ToolVQAAdapter(make_paths(root=root, train_file=annotation.name), max_items=1)

            items = list(adapter.iter_raw())

            self.assertEqual([item["value"] for item in items], [1])
        finally:
            annotation.unlink(missing_ok=True)
            root.rmdir()

    def test_missing_image_is_rejected_in_strict_mode(self) -> None:
        adapter = ToolVQAAdapter(make_paths())
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["image_path"] = "images/missing.jpg"

        result = adapter.adapt(raw)

        self.assertFalse(result.ok)
        self.assertIn("Image file does not exist", result.errors[0])

    def test_generated_image_trajectory_is_exported_as_tool_routing_only(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [
            raw["context"][0],
            {
                "name": "TextToImage",
                "input": {"keywords": "boy walking on grass, pink T-shirt"},
                "output": "image/output.png",
                "thought": "Generate the requested image using the color found in the input image.",
            },
        ]
        raw["answer"] = "boy walking on grass, pink T-shirt (image/output.png)"

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual(
            [message.role for message in result.sample.messages],
            ["user", "tool_call", "tool_response", "tool_call"],
        )
        self.assertEqual(json.loads(result.sample.messages[-1].content)["name"], "TextToImage")
        self.assertEqual(result.sample.metadata["conversion_mode"], "tool_routing_only")
        self.assertFalse(result.sample.metadata["generated_media_available"])
        self.assertEqual(result.sample.metadata["source_final_answer"], raw["answer"])
        self.assertNotIn("final_answer", result.sample.labels)

        row = canonical_to_ms_swift_row(result.sample)
        self.assertEqual(row["messages"][-1]["role"], "tool_call")
        self.assertEqual(sum(message["content"].count("<image>") for message in row["messages"]), 1)
        self.assertEqual(len(row["images"]), 1)
        tools = {tool["function"]["name"]: tool["function"] for tool in json.loads(row["tools"])}
        self.assertEqual(tools["TextToImage"]["parameters"]["required"], ["keywords"])

    def test_text_to_image_bare_output_filename_is_exported_as_tool_routing_only(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [
            {
                "name": "TextToImage",
                "input": {"keywords": "a generated image"},
                "output": "output.png",
            }
        ]
        raw["answer"] = "output.png"

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual([message.role for message in result.sample.messages], ["user", "tool_call"])
        self.assertEqual(result.sample.metadata["generated_media_placeholder"], "output.png")

    def test_image_url_in_text_tool_response_does_not_truncate_trajectory(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [
            {
                "name": "GoogleSearch",
                "input": {"query": "example product"},
                "output": "Result thumbnail: https://cdn.example.com/images/product.jpg; product page follows.",
            }
        ]
        raw["answer"] = "This is a normal text answer."

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual(
            [message.role for message in result.sample.messages],
            ["user", "tool_call", "tool_response", "assistant"],
        )
        self.assertEqual(result.sample.metadata["conversion_mode"], "full_trajectory")
        self.assertEqual(result.sample.messages[-1].content, raw["answer"])

    def test_text_answer_mentioning_image_path_is_not_treated_as_media_answer(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["answer"] = "The documentation example is stored at docs/example.png."

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual(result.sample.messages[-1].role, "assistant")
        self.assertEqual(result.sample.messages[-1].content, raw["answer"])
        self.assertFalse(result.sample.metadata["final_output_is_media"])

    def test_media_answer_without_generation_tool_is_rejected(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["answer"] = "image/output.png"

        result = adapter.adapt(raw)

        self.assertFalse(result.ok)
        self.assertIn("no matching image-producing tool call", result.errors[0])

    def test_steps_after_missing_generated_media_are_truncated(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [
            {
                "name": "TextToImage",
                "input": {"keywords": "a chart"},
                "output": "image/output.png",
            },
            {
                "name": "OCR",
                "input": {"image": "image/output.png"},
                "output": "chart text",
            },
        ]
        raw["answer"] = "chart text"

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        assert result.sample is not None
        self.assertEqual([message.role for message in result.sample.messages], ["user", "tool_call"])
        self.assertTrue(result.sample.metadata["trajectory_truncated"])
        self.assertEqual(result.sample.metadata["discarded_tool_steps"], 1)
        self.assertEqual(result.sample.metadata["truncated_after_tool"], "TextToImage")

    def test_google_drive_image_prefix_resolves_into_media_directory(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)

        resolved = adapter._resolve_image_path("datasets/ToolVQA/2412880.jpg")

        self.assertEqual(resolved, (SAMPLE_ROOT / "images" / "2412880.jpg").as_posix())

    def test_invalid_tool_input_is_preserved_with_warning(self) -> None:
        adapter = ToolVQAAdapter(make_paths())
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [dict(raw["context"][0], input="not-json")]

        result = adapter.adapt(raw)

        self.assertTrue(result.ok, result.to_report_dict())
        self.assertEqual(len(result.warnings), 1)
        assert result.sample is not None
        call = json.loads(result.sample.messages[1].content)
        self.assertEqual(call["arguments"], {"raw_input": "not-json"})

    def test_generated_intermediate_image_input_is_rejected(self) -> None:
        adapter = ToolVQAAdapter(make_paths())
        raw = json.loads((SAMPLE_ROOT / "train_sample.jsonl").read_text(encoding="utf-8"))
        raw["context"] = [dict(raw["context"][0], input='{"image":"image/output.png"}')]

        result = adapter.adapt(raw)

        self.assertFalse(result.ok)
        self.assertIn("other than the original input", result.errors[0])

    def test_streaming_export_writes_dataset_report_inputs(self) -> None:
        adapter = ToolVQAAdapter(make_paths())
        output, errors = self._streaming_paths()
        try:
            stats = write_export(result_records(adapter), output, errors, include_metadata=True)

            self.assertEqual(stats["total_raw"], 1)
            self.assertEqual(stats["converted"], 1)
            self.assertEqual(stats["skipped"], 0)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["messages"][-1]["content"], "Men's collection")
            self.assertEqual(errors.read_text(encoding="utf-8"), "")
        finally:
            self._remove_streaming_files(output, errors)

    def test_streaming_export_writes_tool_routing_output_media(self) -> None:
        adapter = ToolVQAAdapter(make_paths(), validate_media_files=False)
        raw = next(adapter.iter_raw())
        raw["context"] = [
            {
                "name": "TextToImage",
                "input": {"keywords": "a generated image"},
                "output": "image/output.png",
            }
        ]
        raw["answer"] = "image/output.png"
        result = adapter.adapt(raw)
        output, errors = self._streaming_paths()
        try:
            stats = write_export([result], output, errors, include_metadata=False)

            self.assertEqual(stats["converted"], 1)
            self.assertEqual(stats["skipped"], 0)
            self.assertEqual(stats["errors"], 0)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["messages"][-1]["role"], "tool_call")
            self.assertEqual(errors.read_text(encoding="utf-8"), "")
        finally:
            self._remove_streaming_files(output, errors)


if __name__ == "__main__":
    unittest.main()
