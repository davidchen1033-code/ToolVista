from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.prepare_c3_router_datafixed import (
    _promote_datafixed,
    audit_direct_answer_data,
    audit_router_data,
    main,
    parse_args,
    prepare_mllm_router_datafixed,
    with_datafixed_router_system_prompt,
)
from scripts.prepare_training_and_eval_data import belongs_to_eval
from toolvista.evaluation.io import (
    load_benchmark_registry,
    load_jsonl,
    load_suites,
)
from toolvista.router_data import (
    ROUTER_CONTRACT_VERSION,
    ROUTER_MODE_MARKER,
    build_router_catalog,
    router_system_prompt,
    with_router_system_prompt,
)
from toolvista.utils.paths import ToolVistaPaths


def raw_row(
    raw_id: int,
    prompt: str,
    target: str,
    **extra: object,
) -> dict[str, object]:
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


def make_mllm_paths(root: Path) -> ToolVistaPaths:
    instruction_dir = root / "data" / "IT_data_ins" / "T+X-T_data"
    media_dir = instruction_dir / "mm_dataset" / "image"
    media_dir.mkdir(parents=True)
    (media_dir / "sample.jpg").write_bytes(b"image")
    files = {
        "text_t2t.json": [
            raw_row(1, "Route   This", "tool-a"),
            raw_row(2, "route this", "tool-b"),
            raw_row(3, "Keep this", "tool-c"),
        ],
        "image_tx2t.json": [
            raw_row(
                4,
                "ROUTE THIS",
                "tool-d",
                input_modality="image",
                image_path="sample.jpg",
            )
        ],
        "video_tx2t.json": [],
        "audio_tx2t.json": [],
        "combined_data.json": [],
    }
    for filename, rows in files.items():
        (instruction_dir / filename).write_text(
            json.dumps(rows),
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


class C3DatafixedTest(unittest.TestCase):
    def test_defaults_never_select_control_outputs(self) -> None:
        args = parse_args([])
        outputs = (
            args.router_train,
            args.router_full,
            args.router_light,
            args.direct_output,
            args.report,
        )
        self.assertTrue(all("datafixed" in str(path) for path in outputs))
        self.assertEqual(
            args.router_train.name,
            "mllm_tool_router_datafixed_train.jsonl",
        )

    def test_entry_point_refuses_control_output_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-control-") as value:
            control_path = Path(value) / "mllm_tool_router_train.jsonl"

            with self.assertRaisesRegex(SystemExit, "must contain 'datafixed'"):
                main(["--router-train", str(control_path)])

    def test_entry_point_refuses_datafixed_symlink_to_control_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-symlink-") as value:
            root = Path(value)
            control_dir = root / "mllm_router_v2"
            control_dir.mkdir()
            control_full = control_dir / "full.jsonl"
            control_full.write_text("control\n", encoding="utf-8")
            datafixed_alias = root / "mllm_router_datafixed"
            datafixed_alias.symlink_to(control_dir, target_is_directory=True)

            outputs = {
                "router_train": root / "mllm_tool_router_datafixed_train.jsonl",
                "router_full": datafixed_alias / "full.jsonl",
                "router_light": root / "mllm_router_datafixed_light.jsonl",
                "router_split": root / "mllm_router_datafixed_split.json",
                "direct_output": root
                / "toolvqa_direct_answer_datafixed_train.jsonl",
                "direct_report": root
                / "toolvqa_direct_answer_datafixed_train.report.json",
                "report": root / "router_datafixed.report.json",
            }

            with self.assertRaisesRegex(SystemExit, "must contain 'datafixed'"):
                main(
                    [
                        "--config",
                        str(root / "must-not-be-read.yaml"),
                        "--router-train",
                        str(outputs["router_train"]),
                        "--router-full",
                        str(outputs["router_full"]),
                        "--router-light",
                        str(outputs["router_light"]),
                        "--router-split",
                        str(outputs["router_split"]),
                        "--direct-source",
                        str(root / "must-not-be-read.jsonl"),
                        "--direct-output",
                        str(outputs["direct_output"]),
                        "--direct-report",
                        str(outputs["direct_report"]),
                        "--report",
                        str(outputs["report"]),
                        "--overwrite",
                    ]
                )

            self.assertEqual(control_full.read_text(encoding="utf-8"), "control\n")
            for name, path in outputs.items():
                if name != "router_full":
                    self.assertFalse(path.exists(), name)

    def test_router_prompt_replaces_all_source_system_messages(self) -> None:
        messages = with_datafixed_router_system_prompt(
            [
                {"role": "system", "content": "legacy behavior"},
                {
                    "role": "system",
                    "content": f"{ROUTER_MODE_MARKER}\nstale catalog",
                },
                {"role": "user", "content": "route this"},
            ],
            ["tool-b", "tool-a"],
        )

        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertEqual(
            messages[0]["content"],
            router_system_prompt(["tool-a", "tool-b"]),
        )
        self.assertNotIn("legacy behavior", json.dumps(messages))

        control_messages = with_router_system_prompt(
            [
                {"role": "system", "content": "legacy behavior"},
                {"role": "user", "content": "route this"},
            ],
            ["tool-b", "tool-a"],
        )
        self.assertEqual(
            [item["role"] for item in control_messages],
            ["system", "system", "user"],
        )
        self.assertEqual(control_messages[1]["content"], "legacy behavior")

    def test_router_catalog_rejects_target_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "target count exceeds catalog size"):
            build_router_catalog(
                ["tool-a", "tool-b", "tool-c"],
                ["tool-a", "tool-b", "tool-c"],
                identity="sample",
                seed=42,
                max_size=2,
            )

    def test_datafixed_promotion_preserves_backup_when_rollback_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-promotion-") as value:
            root = Path(value)
            final_path = root / "artifact_datafixed.json"
            temporary_path = root / ".artifact_datafixed.tmp"
            final_path.write_text("old", encoding="utf-8")
            temporary_path.write_text("new", encoding="utf-8")
            real_replace = os.replace
            calls = 0

            def fail_publish_and_restore(source: object, destination: object) -> None:
                nonlocal calls
                calls += 1
                if calls in {2, 3}:
                    raise OSError(f"injected replace failure {calls}")
                real_replace(source, destination)

            with mock.patch(
                "scripts.prepare_c3_router_datafixed.os.replace",
                side_effect=fail_publish_and_restore,
            ):
                with self.assertRaisesRegex(RuntimeError, "preserved backups"):
                    _promote_datafixed({final_path: temporary_path})

            backups = list(root.glob(".artifact_datafixed.json.*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(encoding="utf-8"), "old")
            self.assertFalse(temporary_path.exists())

    def test_registry_keeps_all_datafixed_suites_on_isolated_router_paths(self) -> None:
        root = Path(__file__).resolve().parents[1]
        registry = load_benchmark_registry(
            root / "configs/evaluation/benchmarks_datafixed.yaml"
        )
        suites = load_suites(root / "configs/evaluation/suites_datafixed.yaml")
        for stage in range(5):
            suite = suites[f"c{stage}_datafixed_light"]
            self.assertIn("mllm_router_datafixed", suite)
            self.assertNotIn("mllm_router_v2", suite)
            self.assertTrue(set(suite) <= set(registry))
        for suite_name in ("c0_datafixed_full", "c4_datafixed_full"):
            self.assertIn("mllm_router_datafixed_full", suites[suite_name])
            self.assertTrue(set(suites[suite_name]) <= set(registry))
        self.assertEqual(
            str(registry["mllm_router_datafixed"].dataset_path),
            "data/eval/mllm_router_datafixed/light.jsonl",
        )
        self.assertEqual(
            str(registry["mllm_router_datafixed_full"].dataset_path),
            "data/eval/mllm_router_datafixed/full.jsonl",
        )

    def test_builder_merges_labels_and_adds_one_explicit_router_prompt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-datafixed-") as value:
            root = Path(value)
            paths = make_mllm_paths(root / "raw")
            seed = next(
                candidate
                for candidate in range(1000)
                if belongs_to_eval(
                    "route this",
                    seed=candidate,
                    namespace="mllm_router",
                    fraction=0.5,
                )
                and not belongs_to_eval(
                    "keep this",
                    seed=candidate,
                    namespace="mllm_router",
                    fraction=0.5,
                )
            )
            train = root / "mllm_tool_router_datafixed_train.jsonl"
            full = root / "mllm_router_datafixed_full.jsonl"
            light = root / "mllm_router_datafixed_light.jsonl"
            split = root / "mllm_router_datafixed_split.json"

            report = prepare_mllm_router_datafixed(
                paths,
                train_output=train,
                full_output=full,
                light_output=light,
                split_output=split,
                seed=seed,
                eval_fraction=0.5,
                light_limit=10,
                file_keys=("text", "image", "video"),
                validate_media_files=True,
                catalog_size=32,
            )

            train_rows = load_jsonl(train)
            full_rows = load_jsonl(full)
            self.assertEqual(len(train_rows), 1)
            self.assertEqual(report["variant"], "datafixed")
            self.assertEqual(report["namespace"], "mllm_router")
            self.assertIn("control", report["namespace_reason"])
            train_row = train_rows[0]
            systems = [
                message
                for message in train_row["messages"]
                if message["role"] == "system"
            ]
            self.assertEqual(len(systems), 1)
            self.assertIn(ROUTER_MODE_MARKER, systems[0]["content"])
            self.assertEqual(
                train_row["router_contract"],
                ROUTER_CONTRACT_VERSION,
            )
            selected = train_row["messages"][-1]["content"]
            self.assertIn(selected, train_row["router_catalog"])
            self.assertEqual(
                selected,
                train_row["metadata"]["labels"]["selected_sft_target"],
            )
            text_eval = next(
                row for row in full_rows if row["input_modality"] == "text"
            )
            self.assertEqual(text_eval["target_tools"], ["tool-a", "tool-b"])
            self.assertTrue(
                set(text_eval["target_tools"]) <= set(text_eval["router_catalog"])
            )
            self.assertFalse(
                any(message["role"] == "assistant" for message in text_eval["messages"])
            )
            self.assertEqual(
                audit_router_data(train, training=True, max_catalog_size=32)[
                    "rows"
                ],
                1,
            )
            self.assertEqual(
                audit_router_data(full, training=False, max_catalog_size=32)[
                    "rows"
                ],
                len(full_rows),
            )

    def test_main_builds_and_audits_all_datafixed_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-datafixed-main-") as value:
            root = Path(value)
            paths = make_mllm_paths(root / "raw")
            seed = next(
                candidate
                for candidate in range(1000)
                if belongs_to_eval(
                    "route this",
                    seed=candidate,
                    namespace="mllm_router",
                    fraction=0.5,
                )
                and not belongs_to_eval(
                    "keep this",
                    seed=candidate,
                    namespace="mllm_router",
                    fraction=0.5,
                )
            )
            direct_image = root / "direct-image.jpg"
            direct_image.write_bytes(b"direct image")
            direct_source = root / "toolvqa-source.jsonl"
            direct_source.write_text(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "<image>What is visible?",
                            },
                            {"role": "assistant", "content": "A bridge"},
                        ],
                        "images": [str(direct_image)],
                        "tools": [{"name": "irrelevant-source-tool"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            outputs = {
                "router_train": root / "mllm_tool_router_datafixed_train.jsonl",
                "router_full": root / "mllm_router_datafixed_full.jsonl",
                "router_light": root / "mllm_router_datafixed_light.jsonl",
                "router_split": root / "mllm_router_datafixed_split.json",
                "direct_output": root
                / "toolvqa_direct_answer_datafixed_train.jsonl",
                "direct_report": root
                / "toolvqa_direct_answer_datafixed_train.report.json",
                "report": root / "router_datafixed.report.json",
            }
            config = root / "fixture.yaml"

            with mock.patch(
                "scripts.prepare_c3_router_datafixed.ToolVistaPaths.from_yaml",
                return_value=paths,
            ) as from_yaml, mock.patch("builtins.print"):
                main(
                    [
                        "--config",
                        str(config),
                        "--seed",
                        str(seed),
                        "--eval-fraction",
                        "0.5",
                        "--light-limit",
                        "10",
                        "--router-train",
                        str(outputs["router_train"]),
                        "--router-full",
                        str(outputs["router_full"]),
                        "--router-light",
                        str(outputs["router_light"]),
                        "--router-split",
                        str(outputs["router_split"]),
                        "--direct-source",
                        str(direct_source),
                        "--direct-output",
                        str(outputs["direct_output"]),
                        "--direct-report",
                        str(outputs["direct_report"]),
                        "--direct-max-rows",
                        "10",
                        "--report",
                        str(outputs["report"]),
                    ]
                )

            from_yaml.assert_called_once_with(str(config))
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            report = json.loads(outputs["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["variant"], "datafixed")
            self.assertEqual(report["audit"]["router_train"]["rows"], 1)
            self.assertGreater(report["audit"]["router_full"]["rows"], 0)
            self.assertEqual(
                report["audit"]["router_light"]["rows"],
                report["audit"]["router_full"]["rows"],
            )
            self.assertEqual(report["audit"]["direct_answer"]["rows"], 1)
            self.assertEqual(
                set(report["artifacts"]),
                {
                    "router_train",
                    "router_full",
                    "router_light",
                    "router_split",
                    "direct_answer_train",
                    "direct_answer_report",
                },
            )
            self.assertEqual(
                Path(report["artifacts"]["direct_answer_train"]),
                outputs["direct_output"],
            )
            split = json.loads(
                outputs["router_split"].read_text(encoding="utf-8")
            )
            direct_report = json.loads(
                outputs["direct_report"].read_text(encoding="utf-8")
            )
            self.assertEqual(split["variant"], "datafixed")
            self.assertEqual(direct_report["selected_rows"], 1)
            direct_row = load_jsonl(outputs["direct_output"])[0]
            self.assertFalse(
                {"tools", "router_catalog", "router_contract"} & direct_row.keys()
            )
            self.assertFalse(list(root.rglob("*.tmp")))
            self.assertFalse(list(root.rglob("*.bak")))

    def test_direct_answer_audit_requires_media_but_forbids_router_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-datafixed-") as value:
            root = Path(value)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            path = root / "direct_answer_datafixed.jsonl"
            row = {
                "messages": [
                    {"role": "user", "content": "<image>What is visible?"},
                    {"role": "assistant", "content": "A bridge"},
                ],
                "images": [str(image)],
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = audit_direct_answer_data(path)

            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["conflicting_inputs"], 0)
            self.assertNotIn(
                ROUTER_MODE_MARKER,
                path.read_text(encoding="utf-8"),
            )

    def test_direct_answer_audit_rejects_router_and_tool_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-datafixed-") as value:
            root = Path(value)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            for field, leaked_value in (
                ("tools", [{"name": "tool-a"}]),
                ("router_catalog", ["tool-a"]),
                ("router_contract", ROUTER_CONTRACT_VERSION),
            ):
                with self.subTest(field=field):
                    path = root / f"direct_answer_{field}_datafixed.jsonl"
                    row = {
                        "messages": [
                            {
                                "role": "user",
                                "content": "<image>What is visible?",
                            },
                            {"role": "assistant", "content": "A bridge"},
                        ],
                        "images": [str(image)],
                        field: leaked_value,
                    }
                    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

                    with self.assertRaisesRegex(
                        ValueError, "leaked Router/tool fields"
                    ):
                        audit_direct_answer_data(path)

    def test_router_audit_rejects_prompt_catalog_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-datafixed-") as value:
            path = Path(value) / "mismatched_router_datafixed.jsonl"
            row = {
                "messages": [
                    {
                        "role": "system",
                        "content": router_system_prompt(["tool-a", "tool-b"]),
                    },
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "tool-a"},
                ],
                "router_contract": ROUTER_CONTRACT_VERSION,
                "router_catalog": ["tool-a", "tool-c"],
                "metadata": {
                    "labels": {
                        "target_tools": ["tool-a"],
                        "selected_sft_target": "tool-a",
                    }
                },
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prompt/catalog mismatch"):
                audit_router_data(path, training=True, max_catalog_size=32)

    def test_router_audit_rejects_noncanonical_catalog_whitespace_and_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-datafixed-") as value:
            root = Path(value)
            for variant, catalog in (
                ("whitespace", [" tool-a", "tool-b"]),
                ("order", ["tool-b", "tool-a"]),
            ):
                with self.subTest(variant=variant):
                    path = root / f"noncanonical_{variant}_datafixed.jsonl"
                    row = {
                        "messages": [
                            {
                                "role": "system",
                                "content": router_system_prompt(catalog),
                            },
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "tool-a"},
                        ],
                        "router_contract": ROUTER_CONTRACT_VERSION,
                        "router_catalog": catalog,
                        "metadata": {
                            "labels": {
                                "target_tools": ["tool-a"],
                                "selected_sft_target": "tool-a",
                            }
                        },
                    }
                    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

                    with self.assertRaisesRegex(
                        ValueError, "Invalid or oversized router catalog"
                    ):
                        audit_router_data(path, training=True, max_catalog_size=32)

    def test_router_audit_rejects_implicit_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-c3-datafixed-") as value:
            path = Path(value) / "bad_router.jsonl"
            row = {
                "messages": [
                    {"role": "system", "content": "Select a tool"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "tool-a"},
                ],
                "router_contract": ROUTER_CONTRACT_VERSION,
                "router_catalog": ["tool-a"],
                "metadata": {
                    "labels": {
                        "target_tools": ["tool-a"],
                        "selected_sft_target": "tool-a",
                    }
                },
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "mode marker"):
                audit_router_data(path, training=True, max_catalog_size=32)


if __name__ == "__main__":
    unittest.main()
