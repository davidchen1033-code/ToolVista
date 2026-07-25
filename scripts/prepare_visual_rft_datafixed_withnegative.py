"""Build the isolated Visual-RFT datafixed-withnegative RL dataset.

The control dataset contains 6,000 positive COCO base-65 queries.  This
builder caps each positive class deterministically, then fills the remaining
rows with queries whose target category is absent according to official
annotations. Negative source images are absent from the frozen Visual-RFT
evaluation manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import PIL
from PIL import Image, features


VARIANT = "datafixed-withnegative"
CONTRACT_VERSION = "visual-rft-datafixed-withnegative-v2-canonical-media"
FIXED_SEED = 42
SOURCE_ROWS = 6000
TOTAL_ROWS = 6000
POSITIVE_CAP_PER_CATEGORY = 60
EXPECTED_CATEGORY_COUNT = 65
EXPECTED_POSITIVE_ROWS = 3310
NO_OBJECTS_SOLUTION = "<answer>No Objects</answer>"
CANONICAL_FORMAT = "JPEG"
CANONICAL_EXTENSION = ".jpg"
CANONICAL_QUALITY = 95
CANONICAL_SUBSAMPLING = 0
CANONICAL_OPTIMIZE = False
CANONICAL_PROGRESSIVE = False
CANONICAL_MODE = "RGB"
ALPHA_BACKGROUND_RGB = (255, 255, 255)
MEDIA_MANIFEST_NAME = "manifest.jsonl"
POSITIVE_MEDIA_PREFIX = ("ViRFT_COCO_base65", "images")
POSITIVE_MEDIA_DENYLIST: dict[str, dict[str, str]] = {
    "ViRFT_COCO_base65/images/000414.png": {
        "reason": "confirmed same-scene near-duplicate of frozen evaluation media",
        "evidence": "manual local cross-split image audit, 2026-07-22",
    },
    "ViRFT_COCO_base65/images/003442.png": {
        "reason": "confirmed same-scene near-duplicate of frozen evaluation media",
        "evidence": "manual local cross-split image audit, 2026-07-22",
    },
}

CATEGORY_PATTERN = re.compile(r"\bcategory\s+(['\"])(.+?)\1", re.IGNORECASE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--positive-source",
        type=Path,
        default=Path(
            "data/exports/msswift_grpo/"
            "visual_rft_coco_base65_train.jsonl"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            "/root/autodl-tmp/datasets/COCO2017/annotations/"
            "instances_val2017.json"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/COCO2017/val2017"),
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=Path("data/eval/visual_rft_coco/full.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/exports/msswift_grpo/"
            "visual_rft_coco_base65_datafixed-withnegative_canonical_train.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/reports/c4/"
            "visual_rft_coco_base65_datafixed-withnegative_canonical_train.report.json"
        ),
    )
    parser.add_argument(
        "--media-dir",
        type=Path,
        default=Path(
            "data/exports/msswift_grpo/"
            "visual_rft_coco_base65_datafixed-withnegative_canonical_media"
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Rebuild JSON, report, and canonical media in a temporary directory "
            "and verify every published artifact."
        ),
    )
    return parser.parse_args(argv)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_bytes(path: Path) -> tuple[bytes, str]:
    content = path.read_bytes()
    return content, _sha256_bytes(content)


def _jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    content, digest = _read_bytes(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(content.splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL contains no rows: {path}")
    return rows, digest


def _json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    content, digest = _read_bytes(path)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value, digest


def _hash_key(*parts: object) -> str:
    payload = ":".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _user_content(row: Mapping[str, Any], *, row_number: int) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"positive row {row_number}: messages must be a list")
    users = [
        message.get("content")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if not users or not isinstance(users[-1], str) or not users[-1].strip():
        raise ValueError(f"positive row {row_number}: missing user prompt")
    return users[-1]


def _target_category(row: Mapping[str, Any], *, row_number: int) -> str:
    prompt = _user_content(row, row_number=row_number)
    prompt_categories = {
        match.group(2).strip() for match in CATEGORY_PATTERN.finditer(prompt)
    }
    if len(prompt_categories) != 1:
        raise ValueError(
            f"positive row {row_number}: expected one repeated target category, "
            f"found {sorted(prompt_categories)}"
        )
    category = next(iter(prompt_categories))
    declared = row.get("target_category")
    if declared is not None and str(declared).strip() != category:
        raise ValueError(
            f"positive row {row_number}: target_category disagrees with prompt"
        )
    return category


def _positive_image_path(
    row: Mapping[str, Any],
    *,
    row_number: int,
) -> Path:
    images = row.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(
            f"positive row {row_number}: exactly one image path is required"
        )
    image = images[0]
    if not isinstance(image, str) or not image.strip():
        raise ValueError(f"positive row {row_number}: invalid image path")
    return Path(image)


def _positive_media_identifier(
    row: Mapping[str, Any],
    *,
    row_number: int,
) -> str:
    path = _positive_image_path(row, row_number=row_number)
    parts = path.parts
    if len(parts) < 3 or tuple(parts[-3:-1]) != POSITIVE_MEDIA_PREFIX:
        raise ValueError(
            f"positive row {row_number}: image is outside the exact "
            f"{'/'.join(POSITIVE_MEDIA_PREFIX)!r} namespace: {path}"
        )
    return Path(*parts[-3:]).as_posix()


def _positive_image_id(row: Mapping[str, Any], *, row_number: int) -> str:
    media_identifier = _positive_media_identifier(row, row_number=row_number)
    stem = Path(media_identifier).stem
    if not stem:
        raise ValueError(f"positive row {row_number}: empty image identifier")
    # The upstream Hugging Face artifact removed official COCO image IDs.
    return f"visual-rft-base65:{stem}"


def _validate_positive_row(row: Mapping[str, Any], *, row_number: int) -> None:
    solution = row.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError(f"positive row {row_number}: missing solution")
    if "No Objects" in solution:
        raise ValueError(f"positive row {row_number}: control source is not positive")
    task = str(row.get("task", "")).strip().lower().replace("-", "_")
    if task not in {"detection", "object_detection", "grounding"}:
        raise ValueError(f"positive row {row_number}: invalid task {task!r}")


def select_positive_rows(
    source_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    cap_per_category: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    dict[str, int],
    dict[str, Any],
]:
    if cap_per_category <= 0:
        raise ValueError("positive cap per category must be positive")
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    source_image_ids: set[str] = set()
    excluded_candidates: list[dict[str, Any]] = []

    for row_number, source_row in enumerate(source_rows, 1):
        _validate_positive_row(source_row, row_number=row_number)
        category = _target_category(source_row, row_number=row_number)
        source_image_id = _positive_image_id(source_row, row_number=row_number)
        media_identifier = _positive_media_identifier(
            source_row,
            row_number=row_number,
        )
        if source_image_id in source_image_ids:
            raise ValueError(f"duplicate positive source image: {source_image_id}")
        source_image_ids.add(source_image_id)
        source_counts[category] += 1

        denylist_entry = POSITIVE_MEDIA_DENYLIST.get(media_identifier)
        if denylist_entry is not None:
            excluded_candidates.append(
                {
                    "media_identifier": media_identifier,
                    "source_image_id": source_image_id,
                    "category": category,
                    **denylist_entry,
                }
            )
            continue

        row = copy.deepcopy(dict(source_row))
        row["example_type"] = "positive"
        row["target_category"] = category
        row["source_image_id"] = source_image_id
        identity = _hash_key(seed, "positive", category, source_image_id)
        grouped[category].append((identity, row))

    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for category in sorted(grouped):
        candidates = sorted(
            grouped[category],
            key=lambda item: (item[0], str(item[1]["source_image_id"])),
        )
        for _, row in candidates[:cap_per_category]:
            selected.append(row)
            selected_counts[category] += 1
    if not selected:
        raise ValueError("positive selection produced no rows")

    selected_media_identifiers = {
        _positive_media_identifier(row, row_number=index)
        for index, row in enumerate(selected, 1)
    }
    selected_overlap = sorted(
        selected_media_identifiers & set(POSITIVE_MEDIA_DENYLIST)
    )
    if selected_overlap:
        raise AssertionError(
            f"positive denylist leaked into selected rows: {selected_overlap}"
        )
    configured = [
        {
            "media_identifier": media_identifier,
            **POSITIVE_MEDIA_DENYLIST[media_identifier],
        }
        for media_identifier in sorted(POSITIVE_MEDIA_DENYLIST)
    ]
    denylist_audit = {
        "matching": (
            "exact normalized ViRFT_COCO_base65/images/<filename> identifier"
        ),
        "application_stage": (
            "before deterministic per-category cap; excluded candidates may be "
            "replaced by the next eligible candidate"
        ),
        "configured_count": len(configured),
        "configured": configured,
        "excluded_candidate_count": len(excluded_candidates),
        "excluded_candidates": sorted(
            excluded_candidates,
            key=lambda item: str(item["media_identifier"]),
        ),
        "selected_overlap_count": len(selected_overlap),
        "selected_overlap": selected_overlap,
    }
    return (
        selected,
        dict(sorted(source_counts.items())),
        dict(sorted(selected_counts.items())),
        denylist_audit,
    )


def _coco_index(
    annotations: Mapping[str, Any],
) -> tuple[
    dict[int, dict[str, Any]],
    dict[str, int],
    dict[int, set[int]],
    int,
]:
    raw_images = annotations.get("images")
    raw_categories = annotations.get("categories")
    raw_annotations = annotations.get("annotations")
    if not isinstance(raw_images, list):
        raise ValueError("COCO annotations require an images list")
    if not isinstance(raw_categories, list):
        raise ValueError("COCO annotations require a categories list")
    if not isinstance(raw_annotations, list):
        raise ValueError("COCO annotations require an annotations list")

    images: dict[int, dict[str, Any]] = {}
    for raw_image in raw_images:
        if not isinstance(raw_image, Mapping):
            raise ValueError("COCO image entries must be objects")
        image_id = int(raw_image["id"])
        if image_id in images:
            raise ValueError(f"duplicate COCO image ID: {image_id}")
        file_name = str(raw_image.get("file_name", "")).strip()
        if not file_name:
            raise ValueError(f"COCO image {image_id} has no file_name")
        images[image_id] = dict(raw_image)

    category_ids: dict[str, int] = {}
    for raw_category in raw_categories:
        if not isinstance(raw_category, Mapping):
            raise ValueError("COCO category entries must be objects")
        name = str(raw_category.get("name", "")).strip()
        category_id = int(raw_category["id"])
        if not name or name in category_ids:
            raise ValueError(f"invalid or duplicate COCO category: {name!r}")
        category_ids[name] = category_id

    present: dict[int, set[int]] = defaultdict(set)
    for raw_annotation in raw_annotations:
        if not isinstance(raw_annotation, Mapping):
            raise ValueError("COCO annotation entries must be objects")
        image_id = int(raw_annotation["image_id"])
        category_id = int(raw_annotation["category_id"])
        if image_id not in images:
            raise ValueError(f"annotation references unknown image ID: {image_id}")
        present[image_id].add(category_id)
    return images, category_ids, present, len(raw_annotations)


def _evaluation_ids(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[int], set[str]]:
    image_ids: set[int] = set()
    categories: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        try:
            image_ids.add(int(row["image_id"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"evaluation row {row_number}: invalid image_id"
            ) from exc
        category = str(row.get("category", "")).strip()
        if category:
            categories.add(category)
    return image_ids, categories


def balanced_negative_quotas(
    categories: Sequence[str],
    *,
    total: int,
    seed: int,
) -> dict[str, int]:
    ordered = sorted(set(categories))
    if not ordered:
        raise ValueError("cannot allocate negatives without categories")
    if total < 0:
        raise ValueError("negative row count cannot be negative")
    quotient, remainder = divmod(total, len(ordered))
    remainder_order = sorted(
        ordered,
        key=lambda category: (
            _hash_key(seed, "negative-quota", category),
            category,
        ),
    )
    extra = set(remainder_order[:remainder])
    return {
        category: quotient + int(category in extra)
        for category in ordered
    }


def _match_negative_images(
    quotas: Mapping[str, int],
    candidates: Mapping[str, Sequence[int]],
) -> dict[str, list[int]]:
    category_order = sorted(
        quotas,
        key=lambda value: (len(candidates[value]), value),
    )
    used_images: set[int] = set()
    result: dict[str, list[int]] = {category: [] for category in category_order}
    offsets = {category: 0 for category in category_order}

    # Candidate lists are already hash-ordered. Allocate one image per category
    # per round so all 2,690 formal negatives are assigned without recursive
    # augmenting paths or recursion-depth failures.
    for slot_index in range(max(quotas.values(), default=0)):
        for category in category_order:
            if slot_index >= quotas[category]:
                continue
            options = candidates[category]
            offset = offsets[category]
            while offset < len(options) and options[offset] in used_images:
                offset += 1
            if offset >= len(options):
                raise ValueError(
                    "unable to assign globally unique negative images for "
                    f"category {category!r}; required quotas are infeasible"
                )
            image_id = options[offset]
            offsets[category] = offset + 1
            used_images.add(image_id)
            result[category].append(image_id)

    result = {
        category: sorted(image_ids)
        for category, image_ids in sorted(result.items())
    }
    if {category: len(result.get(category, [])) for category in quotas} != dict(quotas):
        raise AssertionError("negative matching did not satisfy category quotas")
    return result


def build_negative_rows(
    *,
    positive_templates: Sequence[Mapping[str, Any]],
    annotations: Mapping[str, Any],
    images_dir: Path,
    excluded_eval_image_ids: set[int],
    negative_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    templates: dict[str, Mapping[str, Any]] = {}
    for row in positive_templates:
        category = str(row["target_category"])
        templates.setdefault(category, row)
    categories = sorted(templates)
    images, category_ids, present, annotation_count = _coco_index(annotations)
    unknown = sorted(set(categories) - set(category_ids))
    if unknown:
        raise ValueError(f"positive source contains unknown COCO categories: {unknown}")
    unknown_eval_ids = sorted(excluded_eval_image_ids - set(images))
    if unknown_eval_ids:
        raise ValueError(
            f"evaluation manifest references unknown COCO image IDs: {unknown_eval_ids[:10]}"
        )

    available: dict[int, Path] = {}
    missing: list[str] = []
    for image_id, image in sorted(images.items()):
        if image_id in excluded_eval_image_ids:
            continue
        image_path = (images_dir / str(image["file_name"])).resolve()
        if not image_path.is_file():
            missing.append(str(image_path))
        else:
            available[image_id] = image_path
    if missing:
        raise FileNotFoundError(
            f"COCO images outside the evaluation manifest are incomplete: "
            f"missing={len(missing)}, first={missing[0]}"
        )

    quotas = balanced_negative_quotas(categories, total=negative_count, seed=seed)
    candidate_ids: dict[str, list[int]] = {}
    for category in categories:
        category_id = category_ids[category]
        candidate_ids[category] = sorted(
            (
                image_id
                for image_id in available
                if category_id not in present.get(image_id, set())
            ),
            key=lambda image_id: (
                _hash_key(seed, "negative-image", category, image_id),
                image_id,
            ),
        )
        if len(candidate_ids[category]) < quotas[category]:
            raise ValueError(
                f"not enough annotation-proven negatives for {category!r}: "
                f"eligible={len(candidate_ids[category])}, required={quotas[category]}"
            )

    assignments = _match_negative_images(quotas, candidate_ids)
    rows: list[dict[str, Any]] = []
    selected_image_ids: set[int] = set()
    for category in categories:
        category_id = category_ids[category]
        template = templates[category]
        messages = template.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"positive template for {category!r} has invalid messages")
        for image_id in assignments.get(category, []):
            if image_id in selected_image_ids:
                raise AssertionError(f"duplicate negative image assignment: {image_id}")
            if image_id in excluded_eval_image_ids:
                raise AssertionError(f"evaluation image selected as negative: {image_id}")
            if category_id in present.get(image_id, set()):
                raise AssertionError(
                    f"false negative pair selected: image={image_id}, category={category}"
                )
            selected_image_ids.add(image_id)
            rows.append(
                {
                    "messages": copy.deepcopy(messages),
                    "images": [available[image_id].as_posix()],
                    "solution": NO_OBJECTS_SOLUTION,
                    "ground_truth": NO_OBJECTS_SOLUTION,
                    "task": "detection",
                    "example_type": "negative",
                    "target_category": category,
                    "source_image_id": f"coco-val2017:{image_id}",
                    "coco_image_id": image_id,
                }
            )

    if len(rows) != negative_count or len(selected_image_ids) != negative_count:
        raise AssertionError("negative row or unique-image count mismatch")
    return rows, {
        "annotation_images": len(images),
        "annotation_categories": len(category_ids),
        "annotation_instances": annotation_count,
        "excluded_eval_images": len(excluded_eval_image_ids),
        "available_non_eval_images": len(available),
        "selected_unique_images": len(selected_image_ids),
        "quota_by_category": dict(sorted(quotas.items())),
        "eligible_images_by_category": {
            category: len(candidate_ids[category]) for category in categories
        },
        "selected_image_ids_sha256": _sha256_bytes(
            json.dumps(sorted(selected_image_ids), separators=(",", ":")).encode()
        ),
        "eval_image_overlap": len(selected_image_ids & excluded_eval_image_ids),
        "present_target_pairs": 0,
        "duplicate_images": len(rows) - len(selected_image_ids),
    }


def _encode_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _libjpeg_version() -> str | None:
    try:
        return features.version_codec("jpg")
    except (TypeError, ValueError):
        return None


def _canonical_jpeg(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"canonical media source does not exist: {path}")
    try:
        with Image.open(path) as image:
            source_format = str(image.format or "UNKNOWN")
            source_mode = str(image.mode)
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            nonopaque_alpha = False
            if has_alpha:
                rgba = image.convert("RGBA")
                alpha_min, alpha_max = rgba.getchannel("A").getextrema()
                nonopaque_alpha = alpha_min < 255 or alpha_max < 255
                if nonopaque_alpha:
                    background = Image.new(
                        "RGBA",
                        rgba.size,
                        (*ALPHA_BACKGROUND_RGB, 255),
                    )
                    rgb = Image.alpha_composite(background, rgba).convert(
                        CANONICAL_MODE
                    )
                else:
                    rgb = rgba.convert(CANONICAL_MODE)
            else:
                rgb = image.convert(CANONICAL_MODE)
            rgb.load()
            encoded = BytesIO()
            rgb.save(
                encoded,
                format=CANONICAL_FORMAT,
                quality=CANONICAL_QUALITY,
                subsampling=CANONICAL_SUBSAMPLING,
                optimize=CANONICAL_OPTIMIZE,
                progressive=CANONICAL_PROGRESSIVE,
            )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot canonicalize image {path}: {exc}") from exc
    return encoded.getvalue(), {
        "format": source_format,
        "mode": source_mode,
        "has_alpha": has_alpha,
        "nonopaque_alpha": nonopaque_alpha,
        "source_bytes": path.stat().st_size,
    }


def _canonicalize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    media_dir: Path,
    staging_dir: Path,
) -> dict[str, Any]:
    if not staging_dir.is_dir() or any(staging_dir.iterdir()):
        raise ValueError(f"canonical media staging directory must be empty: {staging_dir}")

    references: Counter[str] = Counter()
    canonical_sizes: dict[str, int] = {}
    source_formats: dict[str, Counter[str]] = defaultdict(Counter)
    source_modes: dict[str, Counter[str]] = defaultdict(Counter)
    source_bytes: Counter[str] = Counter()
    alpha_images: Counter[str] = Counter()
    nonopaque_alpha_images: Counter[str] = Counter()
    resolved_media_dir = media_dir.resolve(strict=False)

    for row_number, row in enumerate(rows, 1):
        images = row.get("images")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(
                f"output row {row_number}: exactly one image is required"
            )
        source = images[0]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"output row {row_number}: invalid image path")
        example_type = str(row.get("example_type", "")).strip()
        if example_type not in {"positive", "negative"}:
            raise ValueError(
                f"output row {row_number}: invalid example_type {example_type!r}"
            )

        content, audit = _canonical_jpeg(Path(source))
        digest = _sha256_bytes(content)
        filename = f"{digest}{CANONICAL_EXTENSION}"
        staged_path = staging_dir / filename
        if staged_path.exists():
            if staged_path.read_bytes() != content:
                raise AssertionError(f"canonical SHA-256 collision: {digest}")
        else:
            staged_path.write_bytes(content)
            canonical_sizes[digest] = len(content)
        references[digest] += 1
        row["images"] = [(resolved_media_dir / filename).as_posix()]

        source_formats[example_type][audit["format"]] += 1
        source_modes[example_type][audit["mode"]] += 1
        source_bytes[example_type] += int(audit["source_bytes"])
        alpha_images[example_type] += int(bool(audit["has_alpha"]))
        nonopaque_alpha_images[example_type] += int(
            bool(audit["nonopaque_alpha"])
        )

    manifest_rows = [
        {
            "file": f"{digest}{CANONICAL_EXTENSION}",
            "sha256": digest,
            "bytes": canonical_sizes[digest],
            "references": references[digest],
        }
        for digest in sorted(canonical_sizes)
    ]
    manifest_content = _encode_jsonl(manifest_rows)
    (staging_dir / MEDIA_MANIFEST_NAME).write_bytes(manifest_content)
    canonical_bytes = sum(canonical_sizes.values())
    source_types = sorted(source_formats)
    return {
        "path": str(resolved_media_dir),
        "logical_images": len(rows),
        "file_count": len(canonical_sizes),
        "deduplicated_images": len(rows) - len(canonical_sizes),
        "bytes": canonical_bytes,
        "directory_bytes": canonical_bytes + len(manifest_content),
        "manifest": {
            "path": str(resolved_media_dir / MEDIA_MANIFEST_NAME),
            "entries": len(manifest_rows),
            "bytes": len(manifest_content),
            "sha256": _sha256_bytes(manifest_content),
        },
        "encoder": {
            "format": CANONICAL_FORMAT,
            "mode": CANONICAL_MODE,
            "quality": CANONICAL_QUALITY,
            "subsampling": CANONICAL_SUBSAMPLING,
            "optimize": CANONICAL_OPTIMIZE,
            "progressive": CANONICAL_PROGRESSIVE,
            "pillow_version": PIL.__version__,
            "libjpeg_version": _libjpeg_version(),
            "metadata_policy": "do not copy source EXIF, ICC, DPI, or comments",
            "alpha_policy": {
                "check": "inspect decoded alpha extrema for every source image",
                "nonopaque_background_rgb": list(ALPHA_BACKGROUND_RGB),
            },
        },
        "source_by_example_type": {
            example_type: {
                "format_counts": dict(sorted(source_formats[example_type].items())),
                "mode_counts": dict(sorted(source_modes[example_type].items())),
                "source_bytes": source_bytes[example_type],
                "alpha_images": alpha_images[example_type],
                "nonopaque_alpha_images": nonopaque_alpha_images[example_type],
            }
            for example_type in source_types
        },
    }


def build_dataset(
    *,
    positive_source: Path,
    annotations_path: Path,
    images_dir: Path,
    eval_manifest: Path,
    output: Path,
    report_path: Path,
    media_dir: Path,
    media_staging_dir: Path,
    seed: int = FIXED_SEED,
    total_rows: int = TOTAL_ROWS,
    positive_cap_per_category: int = POSITIVE_CAP_PER_CATEGORY,
    expected_source_rows: int | None = None,
    expected_category_count: int | None = None,
    expected_positive_rows: int | None = None,
    expected_positive_denylist_matches: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    source_rows, source_sha256 = _jsonl_snapshot(positive_source)
    eval_rows, eval_sha256 = _jsonl_snapshot(eval_manifest)
    annotations, annotations_sha256 = _json_snapshot(annotations_path)
    if expected_source_rows is not None and len(source_rows) != expected_source_rows:
        raise ValueError(
            f"positive source row count changed: {len(source_rows)} != {expected_source_rows}"
        )

    (
        positives,
        source_counts,
        positive_counts,
        positive_denylist_audit,
    ) = select_positive_rows(
        source_rows,
        seed=seed,
        cap_per_category=positive_cap_per_category,
    )
    categories = sorted(positive_counts)
    if expected_category_count is not None and len(categories) != expected_category_count:
        raise ValueError(
            f"positive category count changed: {len(categories)} != "
            f"{expected_category_count}"
        )
    if expected_positive_rows is not None and len(positives) != expected_positive_rows:
        raise ValueError(
            f"positive selection count changed: {len(positives)} != "
            f"{expected_positive_rows}"
        )
    denylist_match_count = positive_denylist_audit["excluded_candidate_count"]
    if (
        expected_positive_denylist_matches is not None
        and denylist_match_count != expected_positive_denylist_matches
    ):
        raise ValueError(
            "positive denylist match count changed: "
            f"{denylist_match_count} != {expected_positive_denylist_matches}"
        )
    negative_count = total_rows - len(positives)
    if negative_count <= 0:
        raise ValueError(
            f"total rows must exceed selected positives: {total_rows} <= {len(positives)}"
        )

    excluded_eval_ids, eval_categories = _evaluation_ids(eval_rows)
    negatives, negative_audit = build_negative_rows(
        positive_templates=positives,
        annotations=annotations,
        images_dir=images_dir,
        excluded_eval_image_ids=excluded_eval_ids,
        negative_count=negative_count,
        seed=seed,
    )
    overlap_categories = sorted(set(categories) & eval_categories)
    if overlap_categories:
        raise ValueError(
            f"base training categories overlap evaluation categories: {overlap_categories}"
        )

    rows = positives + negatives
    rows.sort(
        key=lambda row: (
            _hash_key(
                seed,
                "output-order",
                row["example_type"],
                row["target_category"],
                row["source_image_id"],
            ),
            str(row["source_image_id"]),
        )
    )
    if len(rows) != total_rows:
        raise AssertionError(f"output row count mismatch: {len(rows)} != {total_rows}")
    canonical_media = _canonicalize_rows(
        rows,
        media_dir=media_dir,
        staging_dir=media_staging_dir,
    )
    type_counts = Counter(str(row["example_type"]) for row in rows)
    negative_counts = Counter(
        str(row["target_category"])
        for row in rows
        if row["example_type"] == "negative"
    )
    combined_counts = Counter(str(row["target_category"]) for row in rows)
    content = _encode_jsonl(rows)
    output_sha256 = _sha256_bytes(content)

    report: dict[str, Any] = {
        "variant": VARIANT,
        "contract_version": CONTRACT_VERSION,
        "seed": seed,
        "inputs": {
            "positive_source": {
                "path": str(positive_source.resolve()),
                "sha256": source_sha256,
                "rows": len(source_rows),
            },
            "annotations": {
                "path": str(annotations_path.resolve()),
                "sha256": annotations_sha256,
            },
            "images_dir": str(images_dir.resolve()),
            "evaluation_manifest": {
                "path": str(eval_manifest.resolve()),
                "sha256": eval_sha256,
                "rows": len(eval_rows),
                "unique_images": len(excluded_eval_ids),
                "categories": sorted(eval_categories),
            },
        },
        "selection": {
            "total_rows": len(rows),
            "positive_cap_per_category": positive_cap_per_category,
            "positive_rows": len(positives),
            "negative_rows": len(negatives),
            "category_count": len(categories),
            "categories": categories,
            "example_type_counts": dict(sorted(type_counts.items())),
            "source_positive_by_category": source_counts,
            "positive_by_category": positive_counts,
            "negative_by_category": dict(sorted(negative_counts.items())),
            "combined_by_category": dict(sorted(combined_counts.items())),
            "negative_quota_min": min(negative_counts.values()),
            "negative_quota_max": max(negative_counts.values()),
        },
        "positive_denylist": positive_denylist_audit,
        "canonical_media": canonical_media,
        "negative_assignment": negative_audit,
        "leakage_checks": {
            "negative_evaluation_image_overlap": negative_audit[
                "eval_image_overlap"
            ],
            "positive_known_near_duplicate_overlap": positive_denylist_audit[
                "selected_overlap_count"
            ],
            "positive_known_near_duplicate_candidates_excluded": (
                positive_denylist_audit["excluded_candidate_count"]
            ),
            "positive_evaluation_image_overlap": None,
            "positive_evaluation_image_overlap_reason": (
                "the upstream positive artifact removed official COCO image IDs, "
                "so positive-image overlap cannot be computed"
            ),
            "evaluation_category_overlap": overlap_categories,
            "annotation_present_negative_pairs": negative_audit[
                "present_target_pairs"
            ],
            "duplicate_negative_images": negative_audit["duplicate_images"],
        },
        "audit_fields": {
            "required": [
                "example_type",
                "target_category",
                "source_image_id",
            ],
            "positive_source_image_id": (
                "visual-rft-base65:<materialized-image-stem>; the upstream "
                "artifact does not expose official COCO image IDs"
            ),
            "negative_source_image_id": "coco-val2017:<official integer image_id>",
            "negative_coco_image_id": "official COCO val2017 integer image_id",
        },
        "algorithm": {
            "positive_selection": (
                "per-category ascending sha256(seed:positive:category:source_image_id), "
                "capped at 60"
            ),
            "negative_quota": (
                "floor/remainder balanced across 65 base categories; remainder "
                "assigned by sha256(seed:negative-quota:category)"
            ),
            "negative_pairing": (
                "deterministic round-robin assignment over hash-ordered, globally "
                "unique non-evaluation images for which official COCO annotations "
                "omit the target category"
            ),
            "output_order": (
                "ascending sha256(seed:output-order:example_type:category:source_image_id)"
            ),
        },
        "output": {
            "path": str(output.resolve()),
            "report": str(report_path.resolve()),
            "rows": len(rows),
            "bytes": len(content),
            "sha256": output_sha256,
        },
        "output_sha256": output_sha256,
    }
    return content, report


def _contains_variant_marker(path: Path) -> bool:
    return VARIANT in str(path).casefold().replace("_", "-")


def _contains_canonical_marker(path: Path) -> bool:
    return "canonical" in str(path).casefold()


def _validate_output_paths(
    output: Path,
    report: Path,
    media_dir: Path,
    inputs: Sequence[Path],
) -> None:
    targets = (output, report, media_dir)
    for path in targets:
        resolved = path.resolve(strict=False)
        if (
            not _contains_variant_marker(path)
            or not _contains_variant_marker(resolved)
            or not _contains_canonical_marker(path)
            or not _contains_canonical_marker(resolved)
        ):
            raise ValueError(
                "canonical datafixed-withnegative output paths must contain both "
                f"{VARIANT!r} and 'canonical': {path}"
            )
    resolved_targets = {path.resolve(strict=False) for path in targets}
    if len(resolved_targets) != len(targets):
        raise ValueError("output, report, and media directory must be distinct")
    resolved_inputs = {path.resolve(strict=False) for path in inputs}
    if resolved_targets & resolved_inputs:
        raise ValueError("canonical outputs must not overwrite an input")
    resolved_media = media_dir.resolve(strict=False)
    for artifact in (output.resolve(strict=False), report.resolve(strict=False)):
        if resolved_media in artifact.parents or artifact in resolved_media.parents:
            raise ValueError(
                "output and report must not contain or be inside the canonical "
                f"media directory: media={resolved_media}, artifact={artifact}"
            )
    for source in resolved_inputs:
        if resolved_media in source.parents or source in resolved_media.parents:
            raise ValueError(
                "canonical media directory must not contain or be inside an input: "
                f"media={resolved_media}, input={source}"
            )


def _stage(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _refuse_existing(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            f"canonical outputs are immutable; refusing to overwrite: {existing}"
        )


def _publish_new(
    contents: Mapping[Path, bytes],
    *,
    media_staging_dir: Path,
    media_dir: Path,
) -> None:
    staged: dict[Path, Path] = {}
    promoted: list[Path] = []
    targets = [media_dir, *contents]
    _refuse_existing(targets)
    try:
        for final_path, content in contents.items():
            staged[final_path] = _stage(final_path, content)
        _refuse_existing(targets)
        media_dir.parent.mkdir(parents=True, exist_ok=True)
        os.rename(media_staging_dir, media_dir)
        promoted.append(media_dir)
        for final_path in contents:
            os.rename(staged[final_path], final_path)
            promoted.append(final_path)
    except BaseException:
        for final_path in reversed(promoted):
            if final_path.is_dir() and not final_path.is_symlink():
                shutil.rmtree(final_path)
            else:
                final_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _report_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")


def _media_files(root: Path, *, label: str) -> dict[str, Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"{label} canonical media directory is missing or unsafe: {root}")
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"{label} canonical media contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"{label} canonical media contains a non-file: {path}")
        files[path.relative_to(root).as_posix()] = path
    if not files:
        raise ValueError(f"{label} canonical media directory is empty: {root}")
    return files


def _verify(
    *,
    output: Path,
    report_path: Path,
    media_dir: Path,
    expected_media_dir: Path,
    expected_output: bytes,
    expected_report: Mapping[str, Any],
) -> None:
    actual_output = output.read_bytes()
    if actual_output != expected_output:
        raise ValueError(
            f"output does not match deterministic rebuild: expected "
            f"{_sha256_bytes(expected_output)}, got {_sha256_bytes(actual_output)}"
        )
    actual_report, _ = _json_snapshot(report_path)
    if actual_report != expected_report:
        raise ValueError("report does not match deterministic rebuild")

    actual_files = _media_files(media_dir, label="published")
    expected_files = _media_files(expected_media_dir, label="rebuilt")
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        raise ValueError(
            "canonical media file set does not match deterministic rebuild: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    for relative_path in sorted(expected_files):
        expected_content, expected_sha256 = _read_bytes(expected_files[relative_path])
        actual_content, actual_sha256 = _read_bytes(actual_files[relative_path])
        if actual_content != expected_content:
            raise ValueError(
                "canonical media does not match deterministic rebuild: "
                f"{relative_path}, expected={expected_sha256}, got={actual_sha256}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    inputs = [
        args.positive_source,
        args.annotations,
        args.images_dir,
        args.eval_manifest,
    ]
    media_staging_dir: Path | None = None
    try:
        _validate_output_paths(
            args.output,
            args.report,
            args.media_dir,
            inputs,
        )
        targets = (args.output, args.report, args.media_dir)
        if not args.verify_only:
            _refuse_existing(targets)

        args.media_dir.parent.mkdir(parents=True, exist_ok=True)
        media_staging_dir = Path(
            tempfile.mkdtemp(
                dir=args.media_dir.parent,
                prefix=f".{args.media_dir.name}.",
                suffix=".tmp",
            )
        )
        output_content, report = build_dataset(
            positive_source=args.positive_source,
            annotations_path=args.annotations,
            images_dir=args.images_dir,
            eval_manifest=args.eval_manifest,
            output=args.output,
            report_path=args.report,
            media_dir=args.media_dir,
            media_staging_dir=media_staging_dir,
            seed=FIXED_SEED,
            total_rows=TOTAL_ROWS,
            positive_cap_per_category=POSITIVE_CAP_PER_CATEGORY,
            expected_source_rows=SOURCE_ROWS,
            expected_category_count=EXPECTED_CATEGORY_COUNT,
            expected_positive_rows=EXPECTED_POSITIVE_ROWS,
            expected_positive_denylist_matches=len(POSITIVE_MEDIA_DENYLIST),
        )
        if args.verify_only:
            _verify(
                output=args.output,
                report_path=args.report,
                media_dir=args.media_dir,
                expected_media_dir=media_staging_dir,
                expected_output=output_content,
                expected_report=report,
            )
            print(
                json.dumps(
                    {
                        "variant": VARIANT,
                        "contract_version": CONTRACT_VERSION,
                        "verification": "passed",
                        "output": str(args.output.resolve()),
                        "sha256": report["output_sha256"],
                        "media_manifest_sha256": report["canonical_media"][
                            "manifest"
                        ]["sha256"],
                        "rows": report["selection"]["total_rows"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        _publish_new(
            {
                args.output: output_content,
                args.report: _report_bytes(report),
            },
            media_staging_dir=media_staging_dir,
            media_dir=args.media_dir,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc
    finally:
        if media_staging_dir is not None and media_staging_dir.exists():
            shutil.rmtree(media_staging_dir)


if __name__ == "__main__":
    main()
