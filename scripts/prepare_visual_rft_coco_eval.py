"""Freeze Visual-RFT COCO open-vocabulary evaluation queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NOVEL15 = (
    "mouse",
    "fork",
    "hot dog",
    "cat",
    "airplane",
    "suitcase",
    "parking meter",
    "sandwich",
    "train",
    "hair drier",
    "toilet",
    "toaster",
    "snowboard",
    "frisbee",
    "bear",
)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _prompt(category: str) -> str:
    return (
        f"Detect all objects belonging to the category '{category}' in the image, and provide the "
        "bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).\n"
        f"If no object belonging to the category '{category}' is in the image, return 'No Objects'.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. "
        "The output answer format should be:\n"
        "<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], "
        "'Confidence': number}, ...]</answer>\nPlease strictly follow the format."
    )


def build_queries(
    annotations: Mapping[str, Any],
    candidates: Mapping[str, Any],
    *,
    images_dir: Path,
    categories: Sequence[str],
) -> list[dict[str, Any]]:
    raw_images = annotations.get("images")
    raw_categories = annotations.get("categories")
    if not isinstance(raw_images, list) or not isinstance(raw_categories, list):
        raise ValueError("COCO annotations require images and categories lists.")
    category_ids = {
        str(item["name"]): int(item["id"])
        for item in raw_categories
        if isinstance(item, Mapping) and "name" in item and "id" in item
    }
    unknown = sorted(set(categories) - set(category_ids))
    if unknown:
        raise ValueError(f"Unknown COCO categories: {unknown}")
    raw_annotations = annotations.get("annotations")
    if not isinstance(raw_annotations, list):
        raise ValueError("COCO annotations require an annotations list.")
    ground_truth_pairs = {
        (int(item["image_id"]), int(item["category_id"]))
        for item in raw_annotations
        if isinstance(item, Mapping) and "image_id" in item and "category_id" in item
    }

    selected = set(categories)
    rows: list[dict[str, Any]] = []
    for image in sorted(raw_images, key=lambda item: int(item["id"])):
        if not isinstance(image, Mapping):
            raise ValueError("COCO image entries must be objects.")
        image_id = int(image["id"])
        image_path = (images_dir / str(image["file_name"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {image_path}")
        raw_candidates = candidates.get(str(image_id), [])
        if not isinstance(raw_candidates, list) or not all(isinstance(item, str) for item in raw_candidates):
            raise ValueError(f"Candidate categories for image {image_id} must be a string list.")
        for category in raw_candidates:
            if category not in selected:
                continue
            rows.append(
                {
                    "sample_id": f"coco-val2017-{image_id}-{category_ids[category]}",
                    "source_project": "Visual-RFT",
                    "split": "COCO val2017 novel15",
                    "image_id": image_id,
                    "image_width": int(image["width"]),
                    "image_height": int(image["height"]),
                    "category_id": category_ids[category],
                    "category": category,
                    "has_ground_truth": (image_id, category_ids[category]) in ground_truth_pairs,
                    "messages": [{"role": "user", "content": f"<image>\n{_prompt(category)}"}],
                    "images": [image_path.as_posix()],
                }
            )
    if not rows:
        raise ValueError("Visual-RFT COCO query manifest would be empty.")
    return rows


def select_light(rows: Sequence[dict[str, Any]], *, seed: int, query_limit: int) -> list[dict[str, Any]]:
    if query_limit <= 0:
        raise ValueError("light query limit must be positive")
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    category_images: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        image_id = int(row["image_id"])
        by_image[image_id].append(row)
        category_images[str(row["category"])].append(image_id)

    def key(image_id: int) -> str:
        return hashlib.sha256(f"visual-rft-coco-light:{seed}:{image_id}".encode()).hexdigest()

    selected_images: set[int] = set()
    for category in NOVEL15:
        choices = sorted(
            {
                image_id
                for image_id in category_images.get(category, [])
                if any(
                    row["category"] == category and row["has_ground_truth"]
                    for row in by_image[image_id]
                )
            },
            key=key,
        )
        if not choices:
            raise ValueError(f"No Visual-RFT COCO query covers required category {category!r}.")
        selected_images.add(choices[0])

    ordered_images = sorted(by_image, key=key)
    query_count = sum(len(by_image[image_id]) for image_id in selected_images)
    for image_id in ordered_images:
        if query_count >= query_limit:
            break
        if image_id in selected_images:
            continue
        selected_images.add(image_id)
        query_count += len(by_image[image_id])

    return [
        row
        for image_id in sorted(selected_images)
        for row in sorted(by_image[image_id], key=lambda item: int(item["category_id"]))
    ]


def _encode_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/COCO2017/annotations/instances_val2017.json"),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/COCO2017/val2017"),
    )
    parser.add_argument(
        "--candidate-map",
        type=Path,
        default=Path(
            "/root/autodl-tmp/Visual-RFT/coco_evaluation/"
            "exist_map_coco_Qwen2_vl_2B_baseline.json"
        ),
    )
    parser.add_argument("--full-output", type=Path, default=Path("data/eval/visual_rft_coco/full.jsonl"))
    parser.add_argument("--light-output", type=Path, default=Path("data/eval/visual_rft_coco/light.jsonl"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/c0/visual_rft_coco_eval.report.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--light-query-limit", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = [args.full_output, args.light_output, args.report]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs already exist; pass --overwrite: {existing}")

    full = build_queries(
        _load_object(args.annotations),
        _load_object(args.candidate_map),
        images_dir=args.images_dir,
        categories=NOVEL15,
    )
    light = select_light(full, seed=args.seed, query_limit=args.light_query_limit)
    report = {
        "source": "Visual-RFT COCO open-vocabulary evaluation",
        "annotations": str(args.annotations.resolve()),
        "candidate_map": str(args.candidate_map.resolve()),
        "categories": list(NOVEL15),
        "seed": args.seed,
        "full_queries": len(full),
        "full_images": len({row["image_id"] for row in full}),
        "light_queries": len(light),
        "light_images": len({row["image_id"] for row in light}),
        "full_output": str(args.full_output),
        "light_output": str(args.light_output),
    }
    _replace(args.full_output, _encode_jsonl(full))
    _replace(args.light_output, _encode_jsonl(light))
    _replace(args.report, (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
