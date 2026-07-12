from __future__ import annotations

import json
import unittest
from pathlib import Path

from toolvista.adapters.visual_rft import VisualRFTAdapter
from toolvista.exporters.ms_swift_grpo import (
    MsSwiftGRPOExportError,
    canonical_to_ms_swift_grpo_row,
    export_jsonl,
)
from toolvista.schema.canonical import CanonicalMessage, PreferencePair
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "data" / "visual_rft"


def make_sample():
    paths = ToolVistaPaths(
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
    result = next(VisualRFTAdapter(paths).iter_results())
    assert result.ok and result.sample is not None
    return result.sample


class MsSwiftGRPOExporterTest(unittest.TestCase):
    def test_reward_columns_are_promoted_without_leaking_other_labels(self) -> None:
        sample = make_sample()
        sample.labels["private_note"] = "must not reach the ORM"

        row = canonical_to_ms_swift_grpo_row(sample)

        self.assertEqual([message["role"] for message in row["messages"]], ["system", "user"])
        self.assertEqual(row["messages"][-1]["content"].count("<image>"), 1)
        self.assertEqual(row["images"], sample.media.images)
        self.assertEqual(row["solution"], sample.labels["solution"])
        self.assertEqual(row["ground_truth"], sample.labels["ground_truth"])
        self.assertEqual(row["task"], "classification")
        self.assertNotIn("private_note", row)
        self.assertNotIn("metadata", row)

    def test_optional_reward_columns_are_preserved_when_present(self) -> None:
        sample = make_sample()
        sample.labels.update(
            {
                "class_label": "aircraft",
                "bbox": [10, 20, 30, 40],
                "image_width": 640,
                "image_height": 480,
            }
        )

        row = canonical_to_ms_swift_grpo_row(sample)

        self.assertEqual(row["class_label"], "aircraft")
        self.assertEqual(row["bbox"], [10, 20, 30, 40])
        self.assertEqual(row["image_width"], 640)
        self.assertEqual(row["image_height"], 480)

    def test_missing_solution_is_rejected(self) -> None:
        sample = make_sample()
        sample.labels.pop("solution")

        with self.assertRaisesRegex(MsSwiftGRPOExportError, "non-empty labels.solution"):
            canonical_to_ms_swift_grpo_row(sample)

    def test_missing_task_is_rejected(self) -> None:
        sample = make_sample()
        sample.labels.pop("task")

        with self.assertRaisesRegex(MsSwiftGRPOExportError, "non-empty labels.task"):
            canonical_to_ms_swift_grpo_row(sample)

    def test_assistant_target_is_rejected_to_prevent_gold_answer_leakage(self) -> None:
        sample = make_sample()
        sample.messages.append(CanonicalMessage(role="assistant", content=sample.labels["solution"]))

        with self.assertRaisesRegex(MsSwiftGRPOExportError, "must end with a user prompt"):
            canonical_to_ms_swift_grpo_row(sample)

    def test_preference_sample_is_rejected(self) -> None:
        sample = make_sample()
        sample.preference = PreferencePair(chosen="A", rejected="B")

        with self.assertRaisesRegex(MsSwiftGRPOExportError, "must not contain preference data"):
            canonical_to_ms_swift_grpo_row(sample)

    def test_include_metadata_keeps_debug_data_without_nesting_reward_fields(self) -> None:
        sample = make_sample()

        row = canonical_to_ms_swift_grpo_row(sample, include_metadata=True)

        self.assertEqual(row["metadata"]["sample_id"], sample.sample_id)
        self.assertEqual(row["solution"], sample.labels["solution"])

    def test_non_json_reward_value_is_rejected(self) -> None:
        sample = make_sample()
        sample.labels["bbox"] = {1, 2, 3, 4}

        with self.assertRaisesRegex(MsSwiftGRPOExportError, "not JSON serializable"):
            canonical_to_ms_swift_grpo_row(sample)

    def test_export_jsonl_writes_rows_and_returns_count(self) -> None:
        output = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-grpo-test.jsonl"
        try:
            count = export_jsonl([make_sample()], output)

            self.assertEqual(count, 1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertIn("solution", rows[0])
            self.assertEqual(rows[0]["messages"][-1]["role"], "user")
        finally:
            output.unlink(missing_ok=True)

    def test_export_failure_does_not_replace_existing_output(self) -> None:
        output = REPO_ROOT / "examples" / "ms_swift" / ".visual-rft-grpo-atomic-test.jsonl"
        output.write_text("existing\n", encoding="utf-8")
        valid = make_sample()
        invalid = make_sample()
        invalid.labels.pop("solution")
        try:
            with self.assertRaisesRegex(MsSwiftGRPOExportError, "Failed to export row 1"):
                export_jsonl([valid, invalid], output)

            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")
        finally:
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
