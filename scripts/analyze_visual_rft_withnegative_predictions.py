"""Audit Visual-RFT datafixed-withnegative predictions without changing run artifacts.

The input may be a ToolVista summary JSON or a prediction JSONL. The analyzer
uses the exact parser and scorer registered by visual_rft_withnegative.py.
Positive gold boxes come from an inline solution/ground-truth field when
present, or from a COCO annotation file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.io import load_json, load_jsonl


SCHEMA_VERSION = "1.0"
ANALYSIS_ID = "visual_rft_datafixed-withnegative_predictions"
REWARD_REGISTRY_NAME = "visual_rft_withnegative"
REWARD_PLUGIN_PATH = REPO_ROOT / "toolvista" / "rewards" / "visual_rft_withnegative.py"
PREDICTION_FIELD_CANDIDATES = ("prediction", "response", "completion", "output")
GOLD_FIELD_CANDIDATES = ("solution", "ground_truth")
LABEL_FIELD_CANDIDATES = ("example_type", "has_ground_truth")
POSITIVE_LABELS = frozenset({"1", "true", "yes", "positive", "present", "has_ground_truth"})
NEGATIVE_LABELS = frozenset({"0", "false", "no", "negative", "absent", "no_ground_truth"})
_MISSING = object()


@dataclass(frozen=True)
class ResolvedSource:
    input_path: Path
    predictions_path: Path
    summary_path: Path | None
    benchmark_id: str | None


@dataclass(frozen=True)
class RewardContract:
    parse: Callable[[Any], Any]
    score: Callable[[Any, Any], float]


class CocoGold:
    """Resolve query-specific normalized boxes from a COCO annotation file."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        payload = load_json(self.path)
        raw_images = payload.get("images")
        raw_annotations = payload.get("annotations")
        if not isinstance(raw_images, list) or not isinstance(raw_annotations, list):
            raise ValueError(
                f"COCO annotations require images and annotations lists: {self.path}"
            )

        self.images: dict[int, tuple[float, float]] = {}
        for index, item in enumerate(raw_images):
            if not isinstance(item, Mapping):
                raise ValueError(f"COCO images[{index}] is not an object: {self.path}")
            image_id = _integer(item.get("id"), source=f"COCO images[{index}].id")
            width = _positive_number(item.get("width"), source=f"COCO image {image_id} width")
            height = _positive_number(item.get("height"), source=f"COCO image {image_id} height")
            if image_id in self.images:
                raise ValueError(f"Duplicate COCO image ID {image_id}: {self.path}")
            self.images[image_id] = (width, height)

        self.annotations: dict[tuple[int, int], list[tuple[list[float], bool]]] = {}
        for index, item in enumerate(raw_annotations):
            if not isinstance(item, Mapping):
                raise ValueError(f"COCO annotations[{index}] is not an object: {self.path}")
            image_id = _integer(
                item.get("image_id"), source=f"COCO annotations[{index}].image_id"
            )
            category_id = _integer(
                item.get("category_id"), source=f"COCO annotations[{index}].category_id"
            )
            bbox = item.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(_finite_number(value) for value in bbox)
            ):
                raise ValueError(f"COCO annotations[{index}].bbox must be four finite numbers")
            iscrowd = bool(item.get("iscrowd", 0))
            self.annotations.setdefault((image_id, category_id), []).append(
                ([float(value) for value in bbox], iscrowd)
            )

    def boxes_for(
        self,
        row: Mapping[str, Any],
        *,
        line_number: int,
        image_id_field: str,
        category_id_field: str,
        image_width_field: str,
        image_height_field: str,
    ) -> tuple[list[dict[str, Any]], int]:
        image_id = _integer(
            _required_field(row, image_id_field, line_number=line_number),
            source=f"line {line_number} field {image_id_field!r}",
        )
        category_id = _integer(
            _required_field(row, category_id_field, line_number=line_number),
            source=f"line {line_number} field {category_id_field!r}",
        )
        if image_id not in self.images:
            raise ValueError(
                f"line {line_number}: image ID {image_id} is absent from {self.path}"
            )
        annotation_width, annotation_height = self.images[image_id]
        width = _optional_positive_field(
            row, image_width_field, annotation_width, line_number=line_number
        )
        height = _optional_positive_field(
            row, image_height_field, annotation_height, line_number=line_number
        )
        if not math.isclose(width, annotation_width) or not math.isclose(
            height, annotation_height
        ):
            raise ValueError(
                f"line {line_number}: image dimensions {width}x{height} disagree with "
                f"COCO {annotation_width}x{annotation_height} for image {image_id}"
            )

        boxes: list[dict[str, Any]] = []
        crowd_count = 0
        for bbox, iscrowd in self.annotations.get((image_id, category_id), []):
            x, y, box_width, box_height = bbox
            x1 = min(max(x, 0.0), width)
            y1 = min(max(y, 0.0), height)
            x2 = min(max(x + box_width, 0.0), width)
            y2 = min(max(y + box_height, 0.0), height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"line {line_number}: COCO target pair ({image_id}, {category_id}) "
                    "contains a degenerate bbox after clipping"
                )
            boxes.append(
                {
                    "Position": [
                        x1 / width * 1000.0,
                        y1 / height * 1000.0,
                        x2 / width * 1000.0,
                        y2 / height * 1000.0,
                    ],
                    "Confidence": 1.0,
                }
            )
            crowd_count += int(iscrowd)
        return boxes, crowd_count


def _non_empty_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must be non-empty")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ToolVista summary.json or prediction JSONL")
    parser.add_argument("--source-kind", choices=("auto", "summary", "jsonl"), default="auto")
    parser.add_argument("--benchmark-id", type=_non_empty_text)
    parser.add_argument("--prediction-field", type=_non_empty_text)
    parser.add_argument("--gold-field", type=_non_empty_text)
    parser.add_argument("--label-field", type=_non_empty_text)
    parser.add_argument("--positive-label", type=_non_empty_text)
    parser.add_argument("--negative-label", type=_non_empty_text)
    parser.add_argument("--error-field", type=_non_empty_text, default="error")
    parser.add_argument("--image-id-field", type=_non_empty_text, default="image_id")
    parser.add_argument("--category-id-field", type=_non_empty_text, default="category_id")
    parser.add_argument("--image-width-field", type=_non_empty_text, default="image_width")
    parser.add_argument("--image-height-field", type=_non_empty_text, default="image_height")
    parser.add_argument("--coco-annotations", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Optional JSON report; stdout is always emitted"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (args.positive_label is None) != (args.negative_label is None):
        parser.error("--positive-label and --negative-label must be provided together")
    if (
        args.positive_label is not None
        and _normalized_label(args.positive_label)
        == _normalized_label(args.negative_label)
    ):
        parser.error("--positive-label and --negative-label must differ")
    return args


def _load_reward_contract() -> RewardContract:
    try:
        from toolvista.rewards.visual_rft_withnegative import (
            _parse_detection,
            _score_detection,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "swift" or (exc.name and exc.name.startswith("swift.")):
            raise RuntimeError(
                "ms-swift is required to load the exact visual_rft_withnegative reward; "
                "run this script with /root/autodl-tmp/venvs/toolvista/bin/python"
            ) from exc
        raise
    return RewardContract(parse=_parse_detection, score=_score_detection)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _positive_number(value: Any, *, source: str) -> float:
    if not _finite_number(value) or float(value) <= 0.0:
        raise ValueError(f"{source} must be a positive finite number: {value!r}")
    return float(value)


def _integer(value: Any, *, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source} must be an integer: {value!r}")
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be an integer: {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{source} must be an integer: {value!r}")
    if isinstance(value, str) and value.strip() != str(numeric):
        raise ValueError(f"{source} must be an integer: {value!r}")
    return numeric


def _field(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return _MISSING
        value = value[component]
    return value


def _required_field(row: Mapping[str, Any], path: str, *, line_number: int) -> Any:
    value = _field(row, path)
    if value is _MISSING:
        raise ValueError(f"line {line_number}: required field {path!r} is missing")
    return value


def _optional_positive_field(
    row: Mapping[str, Any], path: str, fallback: float, *, line_number: int
) -> float:
    value = _field(row, path)
    if value is _MISSING:
        return fallback
    return _positive_number(value, source=f"line {line_number} field {path!r}")


def _normalized_label(value: Any) -> str:
    return str(value).strip().casefold()


def _parse_label(
    value: Any,
    *,
    line_number: int,
    field: str,
    positive_label: str | None,
    negative_label: str | None,
) -> bool:
    if isinstance(value, bool) and positive_label is None:
        return value
    if _finite_number(value) and positive_label is None and float(value) in {0.0, 1.0}:
        return bool(value)
    normalized = _normalized_label(value)
    positive = (
        {_normalized_label(positive_label)} if positive_label is not None else POSITIVE_LABELS
    )
    negative = (
        {_normalized_label(negative_label)} if negative_label is not None else NEGATIVE_LABELS
    )
    if normalized in positive:
        return True
    if normalized in negative:
        return False
    raise ValueError(
        f"line {line_number}: field {field!r} has unrecognized label {value!r}; "
        "use --positive-label and --negative-label for custom values"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(raw: str, *, summary_path: Path) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = [
            Path.cwd() / candidate,
            REPO_ROOT / candidate,
            summary_path.parent / candidate,
            summary_path.parent / "eval" / candidate,
        ]
    seen: set[Path] = set()
    for value in candidates:
        resolved = value.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(f"prediction artifact {raw!r} not found; tried: {attempted}")


def _prediction_artifact(
    benchmark: Mapping[str, Any], *, summary_path: Path
) -> Path | None:
    artifacts = benchmark.get("artifacts")
    raw = artifacts.get("predictions") if isinstance(artifacts, Mapping) else None
    if isinstance(raw, str) and raw.strip():
        return _resolve_artifact_path(raw, summary_path=summary_path)

    result_path = benchmark.get("result_path")
    if not isinstance(result_path, str) or not result_path.strip():
        return None
    resolved_result = _resolve_artifact_path(result_path, summary_path=summary_path)
    result = load_json(resolved_result)
    result_artifacts = result.get("artifacts")
    raw = result_artifacts.get("predictions") if isinstance(result_artifacts, Mapping) else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    return _resolve_artifact_path(raw, summary_path=summary_path)


def resolve_source(path: Path, *, source_kind: str, benchmark_id: str | None) -> ResolvedSource:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"analysis source does not exist: {source}")
    if source_kind == "auto":
        source_kind = "jsonl" if source.suffix.casefold() == ".jsonl" else "summary"
    if source_kind == "jsonl":
        if benchmark_id is not None:
            raise ValueError("--benchmark-id is only valid when source is a summary")
        return ResolvedSource(source, source, None, None)

    summary = load_json(source)
    raw_benchmarks = summary.get("benchmarks")
    if not isinstance(raw_benchmarks, Mapping) or not raw_benchmarks:
        raise ValueError(f"summary has no non-empty benchmarks mapping: {source}")
    if benchmark_id is not None:
        if benchmark_id not in raw_benchmarks:
            raise ValueError(
                f"summary has no benchmark {benchmark_id!r}; available: "
                f"{sorted(str(key) for key in raw_benchmarks)}"
            )
        selected_ids = [benchmark_id]
    else:
        selected_ids = [
            str(name)
            for name, value in raw_benchmarks.items()
            if "visual_rft" in str(name).casefold() and isinstance(value, Mapping)
        ]
        if len(selected_ids) != 1:
            raise ValueError(
                "summary must contain exactly one Visual-RFT benchmark for automatic "
                f"selection, found {selected_ids}; use --benchmark-id"
            )
    selected_id = selected_ids[0]
    selected = raw_benchmarks[selected_id]
    if not isinstance(selected, Mapping):
        raise ValueError(f"summary benchmark {selected_id!r} is not an object")
    predictions = _prediction_artifact(selected, summary_path=source)
    if predictions is None:
        raise ValueError(
            f"summary benchmark {selected_id!r} has no predictions artifact or usable result_path"
        )
    return ResolvedSource(source, predictions, source, selected_id)


def _select_field(
    rows: Sequence[Mapping[str, Any]],
    explicit: str | None,
    candidates: Sequence[str],
    *,
    role: str,
    required: bool,
    allow_absent: bool = False,
) -> str | None:
    if explicit is not None:
        if not any(_field(row, explicit) is not _MISSING for row in rows) and not allow_absent:
            raise ValueError(
                f"configured {role} field {explicit!r} is absent from every row"
            )
        return explicit
    for candidate in candidates:
        if any(_field(row, candidate) is not _MISSING for row in rows):
            return candidate
    if required:
        if allow_absent:
            return candidates[0]
        observed = sorted({str(key) for row in rows for key in row})
        raise ValueError(
            f"unable to find a {role} field; tried {list(candidates)}; "
            f"observed top-level fields: {observed}. Configure --{role}-field."
        )
    return None


def _label_from_gold(expected: Any, *, line_number: int, gold_field: str) -> bool:
    if expected.kind == "malformed":
        raise ValueError(
            f"line {line_number}: field {gold_field!r} is malformed under "
            "visual_rft_withnegative"
        )
    return expected.kind == "boxes"


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def analyze_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    contract: RewardContract,
    prediction_field: str | None = None,
    gold_field: str | None = None,
    label_field: str | None = None,
    positive_label: str | None = None,
    negative_label: str | None = None,
    error_field: str = "error",
    coco_gold: CocoGold | None = None,
    image_id_field: str = "image_id",
    category_id_field: str = "category_id",
    image_width_field: str = "image_width",
    image_height_field: str = "image_height",
) -> dict[str, Any]:
    if not rows:
        raise ValueError("prediction JSONL must contain at least one row")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("all prediction rows must be objects")

    all_transport_errors = all(
        isinstance(_field(row, error_field), str)
        and bool(_field(row, error_field).strip())
        for row in rows
    )
    resolved_prediction_field = _select_field(
        rows,
        prediction_field,
        PREDICTION_FIELD_CANDIDATES,
        role="prediction",
        required=True,
        allow_absent=all_transport_errors,
    )
    assert resolved_prediction_field is not None
    resolved_gold_field = _select_field(
        rows, gold_field, GOLD_FIELD_CANDIDATES, role="gold", required=False
    )
    resolved_label_field = _select_field(
        rows, label_field, LABEL_FIELD_CANDIDATES, role="label", required=False
    )
    if resolved_label_field is None and resolved_gold_field is None:
        raise ValueError(
            "unable to classify rows as positive/negative: no label or gold field was found; "
            "configure --label-field or --gold-field"
        )

    counts: Counter[str] = Counter()
    parse_kinds: Counter[str] = Counter()
    box_counts: list[float] = []
    confidences: list[float] = []
    positive_rewards: list[float] = []
    all_rewards: list[float] = []
    resolution: Counter[str] = Counter()

    for line_number, row in enumerate(rows, 1):
        inline_expected = None
        if resolved_gold_field is not None:
            raw_gold = _field(row, resolved_gold_field)
            if raw_gold is not _MISSING:
                inline_expected = contract.parse(raw_gold)
                inferred_positive = _label_from_gold(
                    inline_expected,
                    line_number=line_number,
                    gold_field=resolved_gold_field,
                )
            else:
                inferred_positive = None
        else:
            inferred_positive = None

        if resolved_label_field is not None:
            raw_label = _required_field(row, resolved_label_field, line_number=line_number)
            is_positive = _parse_label(
                raw_label,
                line_number=line_number,
                field=resolved_label_field,
                positive_label=positive_label,
                negative_label=negative_label,
            )
            if inferred_positive is not None and inferred_positive != is_positive:
                raise ValueError(
                    f"line {line_number}: label field {resolved_label_field!r} conflicts "
                    f"with gold field {resolved_gold_field!r}"
                )
        elif inferred_positive is not None:
            is_positive = inferred_positive
        else:
            raise ValueError(
                f"line {line_number}: cannot infer positive/negative label because "
                f"gold field {resolved_gold_field!r} is missing"
            )

        split = "positive" if is_positive else "negative"
        counts[f"{split}_rows"] += 1

        coco_boxes: list[dict[str, Any]] | None = None
        crowd_count = 0
        if coco_gold is not None:
            coco_boxes, crowd_count = coco_gold.boxes_for(
                row,
                line_number=line_number,
                image_id_field=image_id_field,
                category_id_field=category_id_field,
                image_width_field=image_width_field,
                image_height_field=image_height_field,
            )
            if bool(coco_boxes) != is_positive:
                raise ValueError(
                    f"line {line_number}: label says {split}, but COCO target pair has "
                    f"{len(coco_boxes)} box(es)"
                )
            counts["coco_target_boxes"] += len(coco_boxes)
            counts["coco_crowd_target_boxes"] += crowd_count

        if inline_expected is not None:
            expected = inline_expected
            resolution["inline_gold_rows"] += 1
        elif not is_positive:
            expected = contract.parse("No Objects")
            resolution["implicit_negative_gold_rows"] += 1
        elif coco_boxes is not None:
            expected = contract.parse(json.dumps(coco_boxes, separators=(",", ":")))
            if expected.kind != "boxes":
                raise ValueError(
                    f"line {line_number}: normalized COCO boxes failed the reward parser"
                )
            resolution["coco_positive_gold_rows"] += 1
        else:
            expected = None
            resolution["unresolved_positive_gold_rows"] += 1

        error = _field(row, error_field)
        has_error = error is not _MISSING and isinstance(error, str) and bool(error.strip())
        if has_error:
            counts["transport_error_rows"] += 1

        raw_prediction = _field(row, resolved_prediction_field)
        if raw_prediction is _MISSING:
            if not has_error:
                raise ValueError(
                    f"line {line_number}: prediction field {resolved_prediction_field!r} is "
                    f"missing and error field {error_field!r} is empty"
                )
            counts["missing_prediction_rows"] += 1
            parse_kinds["missing"] += 1
            continue

        counts["prediction_rows"] += 1
        predicted = contract.parse(raw_prediction)
        parse_kinds[predicted.kind] += 1
        counts[f"{split}_{predicted.kind}_rows"] += 1
        if predicted.kind == "malformed":
            counts["malformed_prediction_rows"] += 1
        else:
            boxes = list(predicted.boxes) if predicted.kind == "boxes" else []
            box_counts.append(float(len(boxes)))
            counts["predicted_boxes"] += len(boxes)
            confidences.extend(float(box.confidence) for box in boxes)

        if expected is not None:
            reward = float(contract.score(predicted, expected))
            if not math.isfinite(reward) or not -1.0 <= reward <= 1.0:
                raise ValueError(f"line {line_number}: reward is invalid: {reward!r}")
            all_rewards.append(reward)
            counts["reward_rows"] += 1
            if is_positive:
                positive_rewards.append(reward)
                counts["positive_reward_rows"] += 1

    positive_rows = counts["positive_rows"]
    negative_rows = counts["negative_rows"]
    total_rows = len(rows)
    positive_reward_complete = counts["positive_reward_rows"] == positive_rows
    all_reward_complete = counts["reward_rows"] == total_rows
    positive_proxy_subset = _mean(positive_rewards)
    overall_reward_subset = _mean(all_rewards)

    return {
        "fields": {
            "prediction": resolved_prediction_field,
            "gold": resolved_gold_field,
            "label": resolved_label_field,
            "error": error_field,
            "image_id": image_id_field,
            "category_id": category_id_field,
            "image_width": image_width_field,
            "image_height": image_height_field,
        },
        "counts": {
            "rows_total": total_rows,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "prediction_rows": counts["prediction_rows"],
            "missing_prediction_rows": counts["missing_prediction_rows"],
            "transport_error_rows": counts["transport_error_rows"],
            "malformed_prediction_rows": counts["malformed_prediction_rows"],
            "predicted_boxes": counts["predicted_boxes"],
            "positive_reward_rows": counts["positive_reward_rows"],
            "reward_rows": counts["reward_rows"],
            "coco_target_boxes": counts["coco_target_boxes"],
            "coco_crowd_target_boxes": counts["coco_crowd_target_boxes"],
        },
        "parse_kind_counts": {
            name: parse_kinds[name]
            for name in ("no_objects", "empty_list", "boxes", "malformed", "missing")
        },
        "gold_resolution": {
            name: resolution[name]
            for name in (
                "inline_gold_rows",
                "implicit_negative_gold_rows",
                "coco_positive_gold_rows",
                "unresolved_positive_gold_rows",
            )
        },
        "coverage": {
            "prediction_rate": _ratio(counts["prediction_rows"], total_rows),
            "positive_reward_rate": _ratio(counts["positive_reward_rows"], positive_rows),
            "reward_rate": _ratio(counts["reward_rows"], total_rows),
            "positive_reward_complete": positive_reward_complete,
            "reward_complete": all_reward_complete,
        },
        "metrics": {
            "negative_abstention_accuracy": _ratio(
                counts["negative_no_objects_rows"]
                + counts["negative_empty_list_rows"],
                negative_rows,
            ),
            "negative_false_positive_rate": _ratio(
                counts["negative_boxes_rows"], negative_rows
            ),
            "negative_explicit_no_objects_accuracy": _ratio(
                counts["negative_no_objects_rows"], negative_rows
            ),
            "negative_any_box_false_positive_rate": _ratio(
                counts["negative_boxes_rows"], negative_rows
            ),
            "positive_any_box_rate": _ratio(counts["positive_boxes_rows"], positive_rows),
            "positive_reward_ap_proxy_mean": (
                positive_proxy_subset if positive_reward_complete else None
            ),
            "positive_reward_ap_proxy_mean_on_resolved_gold": positive_proxy_subset,
            "overall_reward_mean": overall_reward_subset if all_reward_complete else None,
            "overall_reward_mean_on_resolved_gold": overall_reward_subset,
            "malformed_rate": _ratio(counts["malformed_prediction_rows"], total_rows),
            "missing_or_malformed_rate": _ratio(
                counts["missing_prediction_rows"] + counts["malformed_prediction_rows"],
                total_rows,
            ),
            "predicted_box_count_per_row_mean": counts["predicted_boxes"] / total_rows,
            "predicted_box_count_per_valid_prediction_mean": _mean(box_counts),
            "predicted_confidence_mean": _mean(confidences),
            "predicted_confidence_p50": _percentile(confidences, 0.50),
            "predicted_confidence_min": min(confidences) if confidences else None,
            "predicted_confidence_max": max(confidences) if confidences else None,
        },
        "definitions": {
            "negative_abstention_accuracy": (
                "negative rows parsed as explicit No Objects or an empty list "
                "/ all negative rows"
            ),
            "negative_false_positive_rate": (
                "negative rows parsed with one or more boxes / all negative rows"
            ),
            "negative_explicit_no_objects_accuracy": (
                "negative rows parsed as the explicit 'No Objects' form / all negative rows"
            ),
            "negative_any_box_false_positive_rate": (
                "negative rows parsed with one or more boxes / all negative rows"
            ),
            "positive_any_box_rate": (
                "positive rows parsed with one or more boxes / all positive rows"
            ),
            "positive_reward_ap_proxy_mean": (
                "exact visual_rft_withnegative positive reward averaged only when every "
                "positive row has resolved gold; this F1+Brier multi-IoU proxy is not COCO AP"
            ),
            "malformed_rate": "malformed model responses / all rows; transport errors are separate",
            "box_statistics": (
                "box count per row includes missing/malformed rows as zero; the valid-prediction "
                "mean includes explicit No Objects and [] as zero; confidence statistics "
                "are per box"
            ),
        },
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], set[Path]]:
    resolved = resolve_source(
        args.source, source_kind=args.source_kind, benchmark_id=args.benchmark_id
    )
    rows = load_jsonl(resolved.predictions_path)
    annotation_path: Path | None
    if args.coco_annotations is not None:
        annotation_path = args.coco_annotations.expanduser().resolve()
        if not annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotations do not exist: {annotation_path}")
    else:
        sibling = resolved.predictions_path.with_name("coco_annotations.json")
        annotation_path = sibling if sibling.is_file() else None
    coco_gold = CocoGold(annotation_path) if annotation_path is not None else None
    analysis = analyze_predictions(
        rows,
        contract=_load_reward_contract(),
        prediction_field=args.prediction_field,
        gold_field=args.gold_field,
        label_field=args.label_field,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        error_field=args.error_field,
        coco_gold=coco_gold,
        image_id_field=args.image_id_field,
        category_id_field=args.category_id_field,
        image_width_field=args.image_width_field,
        image_height_field=args.image_height_field,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis": ANALYSIS_ID,
        "reward_contract": {
            "registry_name": REWARD_REGISTRY_NAME,
            "plugin_path": str(REWARD_PLUGIN_PATH.resolve()),
            "plugin_sha256": _sha256(REWARD_PLUGIN_PATH),
            "scorer": "_score_detection",
        },
        "source": {
            "input_path": str(resolved.input_path),
            "input_kind": "summary" if resolved.summary_path is not None else "jsonl",
            "summary_path": str(resolved.summary_path) if resolved.summary_path else None,
            "benchmark_id": resolved.benchmark_id,
            "predictions_path": str(resolved.predictions_path),
            "predictions_sha256": _sha256(resolved.predictions_path),
            "coco_annotations_path": str(annotation_path) if annotation_path else None,
            "coco_annotations_sha256": _sha256(annotation_path) if annotation_path else None,
        },
        **analysis,
    }
    protected = {resolved.input_path, resolved.predictions_path, REWARD_PLUGIN_PATH}
    protected.add(resolved.predictions_path.parent)
    if resolved.summary_path is not None:
        protected.add(resolved.summary_path)
        protected.add(resolved.summary_path.parent)
    if annotation_path is not None:
        protected.add(annotation_path)
    return report, {path.resolve() for path in protected}


def _write_report(
    path: Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool,
    protected_paths: set[Path],
) -> None:
    target = path.expanduser().resolve()
    if target in protected_paths:
        raise ValueError(f"refusing to overwrite an input or evaluation artifact: {target}")
    if target.exists():
        protected_root = next(
            (
                protected
                for protected in protected_paths
                if protected.is_dir() and target.is_relative_to(protected)
            ),
            None,
        )
        if protected_root is not None:
            raise ValueError(
                f"refusing to overwrite existing run artifact under {protected_root}: "
                f"{target}"
            )
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report, protected_paths = build_report(args)
    if args.output is not None:
        _write_report(
            args.output,
            report,
            overwrite=args.overwrite,
            protected_paths=protected_paths,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
