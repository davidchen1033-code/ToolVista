from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from scripts import prepare_visual_rft_datafixed_withnegative as builder


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROL_SOURCE = (
    REPO_ROOT
    / "data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl"
)


def prompt(category: str) -> str:
    return (
        f"<image>Detect all objects belonging to the category '{category}' "
        "in the image.\n"
        f"If no object belonging to the category '{category}' is in the "
        "image, return 'No Objects'."
    )


def positive_row(root: Path, index: int, category: str) -> dict[str, object]:
    image = root / "ViRFT_COCO_base65/images" / f"{index:06d}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new(
        "RGBA",
        (32, 24),
        ((index * 31) % 256, (index * 47) % 256, (index * 71) % 256, 255),
    ).save(image, format="PNG")
    return {
        "messages": [
            {"role": "system", "content": "reason, then answer"},
            {"role": "user", "content": prompt(category)},
        ],
        "images": [str(image)],
        "solution": "<answer>[{'Position':[1,2,3,4],'Confidence':1}]</answer>",
        "ground_truth": "<answer>[{'Position':[1,2,3,4],'Confidence':1}]</answer>",
        "task": "detection",
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.positive_source = root / "control.jsonl"
        self.annotations = root / "instances_val2017.json"
        self.images_dir = root / "val2017"
        self.eval_manifest = root / "full.jsonl"
        self.output = root / "datafixed-withnegative_canonical_train.jsonl"
        self.report = root / "datafixed-withnegative_canonical_train.report.json"
        self.media_dir = root / "datafixed-withnegative_canonical_media"
        self.images_dir.mkdir()

        categories = ["apple", "chair", "dog"]
        positives = [
            positive_row(root, index, category)
            for index, category in enumerate(
                ["apple", "apple", "chair", "chair", "dog", "dog"]
            )
        ]
        write_jsonl(self.positive_source, positives)

        images = [
            {
                "id": image_id,
                "file_name": f"{image_id:012d}.jpg",
                "width": 32,
                "height": 24,
            }
            for image_id in range(1, 13)
        ]
        for image in images:
            image_id = int(image["id"])
            Image.new(
                "RGB",
                (32, 24),
                (
                    (image_id * 17) % 256,
                    (image_id * 29) % 256,
                    (image_id * 43) % 256,
                ),
            ).save(
                self.images_dir / str(image["file_name"]),
                format="JPEG",
            )
        category_rows = [
            {"id": 1, "name": "apple"},
            {"id": 2, "name": "chair"},
            {"id": 3, "name": "dog"},
            {"id": 4, "name": "cat"},
        ]
        annotations = [
            {"id": image_id, "image_id": image_id, "category_id": category_id}
            for image_id, category_id in {
                1: 4,
                2: 1,
                3: 2,
                4: 3,
                5: 1,
                6: 2,
                7: 3,
                8: 1,
                9: 2,
                10: 3,
            }.items()
        ]
        self.annotations.write_text(
            json.dumps(
                {
                    "images": images,
                    "categories": category_rows,
                    "annotations": annotations,
                }
            ),
            encoding="utf-8",
        )
        write_jsonl(
            self.eval_manifest,
            [
                {
                    "sample_id": "eval-1-cat",
                    "image_id": 1,
                    "category": "cat",
                }
            ],
        )
        self.present = {
            (int(item["image_id"]), int(item["category_id"]))
            for item in annotations
        }
        self.category_ids = {str(item["name"]): int(item["id"]) for item in category_rows}

    def build(self) -> tuple[bytes, dict[str, object]]:
        with tempfile.TemporaryDirectory(
            dir=self.root,
            prefix="media-stage-",
        ) as value:
            return builder.build_dataset(
                positive_source=self.positive_source,
                annotations_path=self.annotations,
                images_dir=self.images_dir,
                eval_manifest=self.eval_manifest,
                output=self.output,
                report_path=self.report,
                media_dir=self.media_dir,
                media_staging_dir=Path(value),
                seed=42,
                total_rows=8,
                positive_cap_per_category=1,
                expected_source_rows=6,
                expected_category_count=3,
                expected_positive_rows=3,
            )


class VisualRFTDatafixedWithNegativeTest(unittest.TestCase):
    def test_defaults_are_isolated_from_control_artifacts(self) -> None:
        args = builder.parse_args([])

        self.assertIn("datafixed-withnegative", str(args.output))
        self.assertIn("datafixed-withnegative", str(args.report))
        self.assertIn("datafixed-withnegative", str(args.media_dir))
        self.assertIn("canonical", str(args.output))
        self.assertIn("canonical", str(args.report))
        self.assertIn("canonical", str(args.media_dir))
        self.assertIn("data/reports/c4", str(args.report))
        self.assertNotEqual(args.output, args.positive_source)

    def test_builder_is_deterministic_balanced_and_leakage_safe(self) -> None:
        with tempfile.TemporaryDirectory(prefix="visual-rft-negative-") as value:
            fixture = Fixture(Path(value))

            first_content, first_report = fixture.build()
            second_content, second_report = fixture.build()

            self.assertEqual(first_content, second_content)
            self.assertEqual(first_report, second_report)
            rows = [json.loads(line) for line in first_content.splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                {row["example_type"] for row in rows},
                {"positive", "negative"},
            )
            for row in rows:
                self.assertIn("target_category", row)
                self.assertIn("source_image_id", row)

                self.assertIsInstance(row["source_image_id"], str)
            positives = [row for row in rows if row["example_type"] == "positive"]
            negatives = [row for row in rows if row["example_type"] == "negative"]
            self.assertEqual(len(positives), 3)
            self.assertEqual(len(negatives), 5)
            self.assertEqual(
                {str(row["target_category"]) for row in positives},
                {"apple", "chair", "dog"},
            )

            negative_image_ids = [int(row["coco_image_id"]) for row in negatives]
            self.assertEqual(len(set(negative_image_ids)), 5)
            self.assertNotIn(1, negative_image_ids)
            for row in negatives:
                image_id = int(row["coco_image_id"])
                self.assertEqual(row["source_image_id"], f"coco-val2017:{image_id}")
                category_id = fixture.category_ids[str(row["target_category"])]
                self.assertNotIn((image_id, category_id), fixture.present)
                self.assertEqual(row["solution"], builder.NO_OBJECTS_SOLUTION)
                self.assertEqual(row["ground_truth"], builder.NO_OBJECTS_SOLUTION)

            negative_counts = first_report["selection"]["negative_by_category"]
            self.assertEqual(sum(negative_counts.values()), 5)
            self.assertEqual(set(negative_counts.values()), {1, 2})
            self.assertEqual(
                first_report["leakage_checks"],
                {
                    "negative_evaluation_image_overlap": 0,
                    "positive_known_near_duplicate_overlap": 0,
                    "positive_known_near_duplicate_candidates_excluded": 0,
                    "positive_evaluation_image_overlap": None,
                    "positive_evaluation_image_overlap_reason": (
                        "the upstream positive artifact removed official COCO "
                        "image IDs, so positive-image overlap cannot be computed"
                    ),
                    "evaluation_category_overlap": [],
                    "annotation_present_negative_pairs": 0,
                    "duplicate_negative_images": 0,
                },
            )
            self.assertEqual(
                first_report["output_sha256"],
                builder._sha256_bytes(first_content),
            )

    def test_cli_requires_overwrite_and_verify_only_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="visual-rft-negative-cli-") as value:
            fixture = Fixture(Path(value))
            argv = [
                "--positive-source",
                str(fixture.positive_source),
                "--annotations",
                str(fixture.annotations),
                "--images-dir",
                str(fixture.images_dir),
                "--eval-manifest",
                str(fixture.eval_manifest),
                "--output",
                str(fixture.output),
                "--report",
                str(fixture.report),
                "--media-dir",
                str(fixture.media_dir),
            ]
            constants = {
                "SOURCE_ROWS": 6,
                "TOTAL_ROWS": 8,
                "POSITIVE_CAP_PER_CATEGORY": 1,
                "EXPECTED_CATEGORY_COUNT": 3,
                "EXPECTED_POSITIVE_ROWS": 3,
                "POSITIVE_MEDIA_DENYLIST": {},
            }
            with mock.patch.multiple(builder, **constants):
                builder.main(argv)
                with self.assertRaisesRegex(SystemExit, "immutable"):
                    builder.main(argv)
                builder.main([*argv, "--verify-only"])

                media_path = next(path for path in fixture.media_dir.glob("*.jpg"))
                original_media = media_path.read_bytes()
                media_path.write_bytes(original_media + b"tampered")
                with self.assertRaisesRegex(SystemExit, "canonical media"):
                    builder.main([*argv, "--verify-only"])
                media_path.write_bytes(original_media)

                fixture.output.write_bytes(fixture.output.read_bytes() + b"\n")
                with self.assertRaisesRegex(SystemExit, "deterministic rebuild"):
                    builder.main([*argv, "--verify-only"])

    def test_control_output_path_is_rejected_before_inputs_are_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="visual-rft-negative-path-") as value:
            root = Path(value)
            with self.assertRaisesRegex(SystemExit, "must contain"):
                builder.main(
                    [
                        "--positive-source",
                        str(root / "missing-control.jsonl"),
                        "--annotations",
                        str(root / "missing-annotations.json"),
                        "--eval-manifest",
                        str(root / "missing-eval.jsonl"),
                        "--output",
                        str(root / "control.jsonl"),
                        "--report",
                        str(root / "datafixed_withnegative.report.json"),
                    ]
                )

    def test_missing_non_eval_coco_image_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="visual-rft-negative-missing-") as value:
            fixture = Fixture(Path(value))
            (fixture.images_dir / "000000000012.jpg").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
                fixture.build()

    @unittest.skipUnless(CONTROL_SOURCE.is_file(), "formal control source unavailable")
    def test_formal_control_cap_selects_expected_3310_unique_positives(self) -> None:
        source_rows = [
            json.loads(line)
            for line in CONTROL_SOURCE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        selected, source_counts, selected_counts, denylist_audit = builder.select_positive_rows(
            source_rows,
            seed=builder.FIXED_SEED,
            cap_per_category=builder.POSITIVE_CAP_PER_CATEGORY,
        )

        self.assertEqual(len(source_rows), 6000)
        self.assertEqual(len(source_counts), 65)
        self.assertEqual(len(selected), 3310)
        self.assertEqual(sum(selected_counts.values()), 3310)
        self.assertTrue(all(count <= 60 for count in selected_counts.values()))
        self.assertEqual(
            len({str(row["source_image_id"]) for row in selected}),
            3310,
        )
        self.assertEqual(denylist_audit["configured_count"], 2)
        self.assertEqual(denylist_audit["excluded_candidate_count"], 2)
        self.assertEqual(denylist_audit["selected_overlap_count"], 0)




    def test_large_negative_assignment_is_iterative_and_unique(self) -> None:
        categories = [f"category-{index:02d}" for index in range(65)]
        quotas = {
            category: 42 if index < 25 else 41
            for index, category in enumerate(categories)
        }
        candidates = {
            category: list(range(index, 4000)) + list(range(index))
            for index, category in enumerate(categories)
        }

        selected = builder._match_negative_images(quotas, candidates)

        flattened = [
            image_id
            for values in selected.values()
            for image_id in values
        ]
        self.assertEqual(len(flattened), 2690)
        self.assertEqual(len(set(flattened)), 2690)
        self.assertEqual(
            {category: len(values) for category, values in selected.items()},
            quotas,
        )
if __name__ == "__main__":
    unittest.main()
