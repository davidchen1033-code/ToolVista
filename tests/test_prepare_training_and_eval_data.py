from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from scripts.prepare_training_and_eval_data import (
    _is_mllm_router_sample,
    _write_json,
    belongs_to_eval,
    prepare_mllm_router,
    prepare_xlam,
)
from toolvista.router_data import ROUTER_CONTRACT_VERSION, ROUTER_MODE_MARKER
from toolvista.evaluation.io import load_jsonl
from toolvista.schema.canonical import CanonicalMessage, CanonicalSample
from toolvista.utils.paths import ToolVistaPaths


REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainingAndEvaluationSplitTest(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        path = REPO_ROOT / "tests" / f"tmp-data-split-{uuid.uuid4().hex}"
        path.mkdir()
        self.addCleanup(shutil.rmtree, path, True)
        return path

    @staticmethod
    def xlam_paths() -> ToolVistaPaths:
        root = REPO_ROOT / "examples" / "data" / "xlam"
        return ToolVistaPaths(
            config_path=Path("test"),
            raw={
                "datasets": {
                    "xlam": {
                        "root": str(root),
                        "unified_data_dir": ".",
                        "default_subset": "actionstudio-98k",
                        "default_split": "train",
                    }
                }
            },
        )

    def test_xlam_holds_out_complete_trajectories_and_light_is_full_subset(self) -> None:
        trajectory_ids = [
            "actionstudio-sample---train---1",
            "actionstudio-sample---train---2",
            "actionstudio-sample---train---3",
        ]
        seed = next(
            candidate
            for candidate in range(1000)
            if belongs_to_eval(trajectory_ids[0], seed=candidate, namespace="xlam", fraction=0.5)
            and not belongs_to_eval(trajectory_ids[1], seed=candidate, namespace="xlam", fraction=0.5)
        )
        root = self.make_temp_dir()

        report = prepare_xlam(
            self.xlam_paths(),
            train_output=root / "xlam_train.jsonl",
            full_output=root / "xlam_full.jsonl",
            light_output=root / "xlam_light.jsonl",
            split_output=root / "xlam_split.json",
            seed=seed,
            eval_fraction=0.5,
            light_limit=1,
            overwrite=False,
        )

        train_rows = load_jsonl(root / "xlam_train.jsonl")
        full_rows = load_jsonl(root / "xlam_full.jsonl")
        light_rows = load_jsonl(root / "xlam_light.jsonl")
        train_trajectories = {
            row["metadata"]["metadata"]["trajectory_id"] for row in train_rows
        }
        full_trajectories = {row["trajectory_id"] for row in full_rows}

        self.assertTrue(train_trajectories)
        self.assertTrue(full_trajectories)
        self.assertTrue(train_trajectories.isdisjoint(full_trajectories))
        self.assertEqual(
            {row["sample_id"] for row in light_rows},
            {row["sample_id"] for row in full_rows} & {row["sample_id"] for row in light_rows},
        )
        self.assertEqual(report["overlap"]["trajectory_count"], 0)

    def make_mllm_paths(self, root: Path) -> ToolVistaPaths:
        instruction_dir = root / "data" / "IT_data_ins" / "T+X-T_data"
        media_dir = instruction_dir / "mm_dataset" / "image"
        media_dir.mkdir(parents=True)
        (media_dir / "sample.jpg").write_bytes(b"image")

        def row(raw_id: int, prompt: str, target: str, **extra: object) -> dict[str, object]:
            return {
                "id": raw_id,
                "input_modality": extra.pop("input_modality", "text"),
                "output_modality": "text",
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": target, "ambiguity": -1},
                ],
                **extra,
            }

        text_rows = [
            row(1, "Route   This", "tool-a"),
            row(2, "route this", "tool-b"),
            row(3, "Keep this", "tool-c"),
        ]
        image_rows = [
            row(4, "ROUTE THIS", "tool-d", input_modality="image", image_path="sample.jpg")
        ]
        for filename, rows in (
            ("text_t2t.json", text_rows),
            ("image_tx2t.json", image_rows),
            ("video_tx2t.json", []),
            ("audio_tx2t.json", []),
            ("combined_data.json", []),
        ):
            (instruction_dir / filename).write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )

        return ToolVistaPaths(
            config_path=Path("test"),
            raw={
                "datasets": {
                    "mllm_tool": {
                        "root": str(root),
                        "instruction_dir": "data/IT_data_ins/T+X-T_data",
                        "media_dir": "data/IT_data_ins/T+X-T_data/mm_dataset",
                        "files": {
                            "text": "text_t2t.json",
                            "image": "image_tx2t.json",
                            "video": "video_tx2t.json",
                            "audio": "audio_tx2t.json",
                            "combined": "combined_data.json",
                        },
                    }
                }
            },
        )

    def test_mllm_groups_normalized_prompts_and_merges_targets_per_modality(self) -> None:
        root = self.make_temp_dir()
        paths = self.make_mllm_paths(root / "raw")
        seed = next(
            candidate
            for candidate in range(1000)
            if belongs_to_eval("route this", seed=candidate, namespace="mllm_router", fraction=0.5)
            and not belongs_to_eval("keep this", seed=candidate, namespace="mllm_router", fraction=0.5)
        )

        report = prepare_mllm_router(
            paths,
            train_output=root / "mllm_train.jsonl",
            full_output=root / "mllm_full.jsonl",
            light_output=root / "mllm_light.jsonl",
            split_output=root / "mllm_split.json",
            seed=seed,
            eval_fraction=0.5,
            light_limit=10,
            file_keys=("text", "image", "video"),
            validate_media_files=True,
            overwrite=False,
        )

        train_rows = load_jsonl(root / "mllm_train.jsonl")
        full_rows = load_jsonl(root / "mllm_full.jsonl")
        self.assertEqual(len(train_rows), 1)
        self.assertIn(ROUTER_MODE_MARKER, train_rows[0]["messages"][0]["content"])
        self.assertIn("Keep this", train_rows[0]["messages"][1]["content"])
        self.assertEqual(train_rows[0]["router_contract"], ROUTER_CONTRACT_VERSION)
        self.assertIn("tool-c", train_rows[0]["router_catalog"])
        self.assertEqual({row["input_modality"] for row in full_rows}, {"text", "image"})
        text_row = next(row for row in full_rows if row["input_modality"] == "text")
        image_row = next(row for row in full_rows if row["input_modality"] == "image")
        self.assertEqual(text_row["target_tools"], ["tool-a", "tool-b"])
        self.assertEqual(image_row["target_tools"], ["tool-d"])
        self.assertEqual(text_row["messages"][1], {"role": "user", "content": "Route   This"})
        self.assertIn(ROUTER_MODE_MARKER, text_row["messages"][0]["content"])
        self.assertEqual(set(text_row["router_catalog"]), {"tool-a", "tool-b", "tool-c"})
        self.assertEqual(image_row["router_catalog"], ["tool-d"])
        self.assertEqual(text_row["router_contract"], ROUTER_CONTRACT_VERSION)
        self.assertEqual(image_row["images"], [str(root / "raw" / "data" / "IT_data_ins" / "T+X-T_data" / "mm_dataset" / "image" / "sample.jpg").replace("\\", "/")])
        self.assertEqual(report["overlap"]["normalized_prompt_count"], 0)

    def test_existing_outputs_require_explicit_overwrite(self) -> None:
        root = self.make_temp_dir()
        output = root / "train.jsonl"
        output.write_text("existing\n", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "--overwrite"):
            prepare_xlam(
                self.xlam_paths(),
                train_output=output,
                full_output=root / "full.jsonl",
                light_output=root / "light.jsonl",
                split_output=root / "split.json",
                seed=42,
                eval_fraction=0.5,
                light_limit=1,
                overwrite=False,
            )

    def test_json_writer_creates_parent_directories(self) -> None:
        output = self.make_temp_dir() / "nested/report.json"

        _write_json(output, {"ready": True})

        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"ready": True})

    def test_mllm_router_rejects_targets_with_trailing_user_history(self) -> None:
        sample = CanonicalSample(
            sample_id="bad-router",
            source="mllm_tool",
            task_type="tool_selection",
            messages=[
                CanonicalMessage(role="user", content="route this"),
                CanonicalMessage(role="assistant", content="tool-a"),
                CanonicalMessage(role="user", content="trailing user"),
            ],
            labels={"target_tools": ["tool-a"]},
        )

        self.assertFalse(_is_mllm_router_sample(sample))


if __name__ == "__main__":
    unittest.main()
