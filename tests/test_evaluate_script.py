from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_eval_data import (
    _mllm_router_row,
    _require_available_outputs,
    _training_identities,
    _write_output_transaction,
    _write_manifest,
    check_no_exact_overlap,
    filter_training_rows,
    prepare_mllm_router_rows,
    prepare_xlam_rows,
    validate_manifest,
)
from toolvista.schema.canonical import CanonicalMessage, CanonicalSample
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_paths() -> ToolVistaPaths:
    examples = REPO_ROOT / "examples" / "data"
    return ToolVistaPaths(
        config_path=Path("test"),
        raw={
            "datasets": {
                "xlam": {
                    "root": str(examples / "xlam"),
                    "unified_data_dir": ".",
                    "default_subset": "actionstudio-98k",
                    "default_split": "train",
                },
                "mllm_tool": {
                    "root": str(examples / "mllm_tool"),
                    "instruction_dir": "data/IT_data_ins/T+X-T_data",
                    "media_dir": "data/IT_data_ins/T+X-T_data/mm_dataset",
                    "files": {
                        "image": "image_tx2t.json",
                        "video": "video_tx2t.json",
                        "audio": "audio_tx2t.json",
                        "text": "text_t2t.json",
                        "combined": "combined_data.json",
                    },
                },
            }
        },
    )


class PrepareEvaluationDataTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-evaluate-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_xlam_preparation_keeps_only_one_step_tool_call_targets(self) -> None:
        rows = prepare_xlam_rows(make_paths(), limit=10, seed=42)

        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(all(row["target_tool_calls"] for row in rows))
        self.assertTrue(
            all({message["role"] for message in row["messages"]} <= {"system", "user"} for row in rows)
        )
        self.assertTrue(all(row["tools"] for row in rows))
        self.assertEqual(rows, prepare_xlam_rows(make_paths(), limit=10, seed=42))

    def test_mllm_preparation_builds_a_shared_text_router_catalog(self) -> None:
        rows = prepare_mllm_router_rows(make_paths(), limit=10, seed=42)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_modality"], "text")
        self.assertIn(rows[0]["target_tools"][0], rows[0]["router_catalog"])
        self.assertIn("Return exactly one tool ID", rows[0]["messages"][0]["content"])

    def test_mllm_router_excludes_samples_with_messages_after_the_assistant_target(self) -> None:
        normal = CanonicalSample(
            sample_id="normal",
            source="mllm_tool",
            task_type="tool_selection",
            messages=[
                CanonicalMessage(role="user", content="Route this"),
                CanonicalMessage(role="assistant", content="tool-id"),
            ],
            labels={"target_tools": ["tool-id"]},
        )
        trailing_user = CanonicalSample(
            sample_id="trailing-user",
            source="mllm_tool",
            task_type="tool_selection",
            messages=[
                CanonicalMessage(role="user", content="Route this"),
                CanonicalMessage(role="assistant", content="tool-id"),
                CanonicalMessage(role="user", content="This must not expose the prior gold"),
            ],
            labels={"target_tools": ["tool-id"]},
        )

        self.assertIsNotNone(_mllm_router_row(normal, ["tool-id"], "router prompt"))
        self.assertIsNone(_mllm_router_row(trailing_user, ["tool-id"], "router prompt"))

    def test_exact_training_overlap_is_reported_before_evaluation(self) -> None:
        eval_rows = prepare_xlam_rows(make_paths(), limit=1, seed=42)
        train_path = self.make_temp_dir() / "train.jsonl"
        train_path.write_text(
            json.dumps(
                {
                    "metadata": {"sample_id": eval_rows[0]["sample_id"]},
                    "messages": eval_rows[0]["messages"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "overlap"):
            check_no_exact_overlap([train_path], eval_rows)

    def test_exact_normalized_text_overlap_is_reported_without_matching_id(self) -> None:
        eval_rows = prepare_xlam_rows(make_paths(), limit=1, seed=42)
        train_path = self.make_temp_dir() / "train.jsonl"
        train_path.write_text(
            json.dumps(
                {
                    "metadata": {"sample_id": "different-id"},
                    "messages": eval_rows[0]["messages"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "normalized_text=1"):
            check_no_exact_overlap([train_path], eval_rows)

    def test_training_filter_removes_eval_ids_and_exact_text(self) -> None:
        eval_rows = prepare_xlam_rows(make_paths(), limit=1, seed=42)
        keep_row = {
            "metadata": {"sample_id": "keep"},
            "messages": [{"role": "user", "content": "A different prompt"}],
        }
        overlapping_row = {
            "metadata": {"sample_id": eval_rows[0]["sample_id"]},
            "messages": eval_rows[0]["messages"],
        }
        text_overlap_row = {
            "metadata": {"sample_id": "different-id"},
            "messages": eval_rows[0]["messages"],
        }

        filtered = filter_training_rows([overlapping_row, text_overlap_row, keep_row], eval_rows)

        self.assertEqual(filtered, [keep_row])

    def test_training_identity_uses_all_ids_and_all_available_text_sources(self) -> None:
        row = {
            "sample_id": "top-level-id",
            "metadata": {"sample_id": "metadata-id"},
            "normalized_text": "Supplied text",
            "messages": [{"role": "user", "content": "Message-derived text"}],
        }

        sample_ids, normalized_texts = _training_identities(row)

        self.assertEqual(sample_ids, {"top-level-id", "metadata-id"})
        self.assertEqual(normalized_texts, {"supplied text", "message-derived text"})

    def test_blank_supplied_text_does_not_hide_message_overlap(self) -> None:
        eval_rows = prepare_xlam_rows(make_paths(), limit=1, seed=42)
        training_row = {
            "sample_id": "different-id",
            "normalized_text": "  ",
            "messages": eval_rows[0]["messages"],
        }

        self.assertEqual(filter_training_rows([training_row], eval_rows), [])

    def test_top_level_id_is_used_when_metadata_has_no_sample_id(self) -> None:
        eval_rows = prepare_xlam_rows(make_paths(), limit=1, seed=42)
        training_row = {
            "sample_id": eval_rows[0]["sample_id"],
            "metadata": {},
            "messages": [{"role": "user", "content": "Different prompt"}],
        }

        self.assertEqual(filter_training_rows([training_row], eval_rows), [])

    def test_non_positive_limit_selects_all_eligible_rows(self) -> None:
        all_rows = prepare_xlam_rows(make_paths(), limit=0, seed=42)

        self.assertGreaterEqual(len(all_rows), 2)
        self.assertEqual(all_rows, prepare_xlam_rows(make_paths(), limit=-1, seed=999))

    def test_empty_and_duplicate_manifests_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_manifest([])
        with self.assertRaisesRegex(ValueError, "normalized_text"):
            validate_manifest([{"sample_id": "sample-1", "normalized_text": "  "}])

        base = {
            "sample_id": "sample-1",
            "normalized_text": "same prompt",
        }
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            validate_manifest([base, dict(base)])
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            validate_manifest([base, {"sample_id": " sample-1 ", "normalized_text": "different"}])
        with self.assertRaisesRegex(ValueError, "duplicate normalized text"):
            validate_manifest(
                [
                    base,
                    {"sample_id": "sample-2", "normalized_text": "  SAME   prompt "},
                ]
            )

    def test_manifest_write_requires_overwrite_and_records_selection(self) -> None:
        rows = prepare_mllm_router_rows(make_paths(), limit=1, seed=42)
        output = self.make_temp_dir() / "dev.jsonl"

        _write_manifest(output, rows, seed=42, overwrite=False)
        report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))

        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(report["sample_ids"], [rows[0]["sample_id"]])
        self.assertEqual(len(report["selection_keys"]), 1)
        self.assertTrue(output.exists())
        self.assertTrue(output.with_suffix(".report.json").exists())
        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            _write_manifest(output, rows, seed=42, overwrite=False)

        _write_manifest(output, rows, seed=7, overwrite=True)
        overwritten_report = json.loads(output.with_suffix(".report.json").read_text(encoding="utf-8"))
        self.assertEqual(overwritten_report["seed"], 7)
        self.assertEqual(
            sorted(path.name for path in output.parent.iterdir()),
            ["dev.jsonl", "dev.report.json"],
        )

    def test_output_targets_must_be_distinct(self) -> None:
        output = self.make_temp_dir() / "shared.jsonl"

        with self.assertRaisesRegex(ValueError, "same path"):
            _require_available_outputs([output, output], overwrite=True)

    def test_staging_failure_leaves_no_final_or_temporary_outputs(self) -> None:
        directory = self.make_temp_dir()
        jsonl_output = directory / "manifest.jsonl"
        json_output = directory / "manifest.report.json"

        def fail_after_partial_write(path: Path, value: object) -> None:
            del value
            path.write_text("partial", encoding="utf-8")
            raise RuntimeError("serialization failed")

        with patch("scripts.prepare_eval_data.write_json", side_effect=fail_after_partial_write):
            with self.assertRaisesRegex(RuntimeError, "serialization failed"):
                _write_output_transaction(
                    jsonl_outputs=[(jsonl_output, [{"sample_id": "one"}])],
                    json_outputs=[(json_output, {"sample_count": 1})],
                )

        self.assertFalse(jsonl_output.exists())
        self.assertFalse(json_output.exists())
        self.assertEqual(list(directory.iterdir()), [])

    def test_promotion_failure_removes_new_finals_and_transaction_artifacts(self) -> None:
        directory = self.make_temp_dir()
        first = directory / "first.jsonl"
        second = directory / "second.jsonl"
        real_replace = os.replace
        promotion_count = 0

        def fail_second_promotion(source: Path, destination: Path) -> None:
            nonlocal promotion_count
            if ".tmp" in Path(source).name:
                promotion_count += 1
                if promotion_count == 2:
                    raise OSError("promotion failed")
            real_replace(source, destination)

        with patch("scripts.prepare_eval_data.os.replace", side_effect=fail_second_promotion):
            with self.assertRaisesRegex(OSError, "promotion failed"):
                _write_output_transaction(
                    jsonl_outputs=[
                        (first, [{"sample_id": "new-first"}]),
                        (second, [{"sample_id": "new-second"}]),
                    ],
                    json_outputs=[],
                )

        self.assertFalse(first.exists())
        self.assertFalse(second.exists())
        self.assertEqual(list(directory.iterdir()), [])

    def test_promotion_failure_restores_preexisting_final_contents(self) -> None:
        directory = self.make_temp_dir()
        first = directory / "first.jsonl"
        second = directory / "second.jsonl"
        first.write_text("old-first\n", encoding="utf-8")
        second.write_text("old-second\n", encoding="utf-8")
        real_replace = os.replace
        promotion_count = 0

        def fail_second_promotion(source: Path, destination: Path) -> None:
            nonlocal promotion_count
            if ".tmp" in Path(source).name:
                promotion_count += 1
                if promotion_count == 2:
                    raise OSError("promotion failed")
            real_replace(source, destination)

        with patch("scripts.prepare_eval_data.os.replace", side_effect=fail_second_promotion):
            with self.assertRaisesRegex(OSError, "promotion failed"):
                _write_output_transaction(
                    jsonl_outputs=[
                        (first, [{"sample_id": "new-first"}]),
                        (second, [{"sample_id": "new-second"}]),
                    ],
                    json_outputs=[],
                )

        self.assertEqual(first.read_text(encoding="utf-8"), "old-first\n")
        self.assertEqual(second.read_text(encoding="utf-8"), "old-second\n")
        self.assertEqual(sorted(path.name for path in directory.iterdir()), ["first.jsonl", "second.jsonl"])


if __name__ == "__main__":
    unittest.main()
