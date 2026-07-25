"""Negative-aware ms-swift reward for Visual-RFT detection experiments.

This plugin intentionally lives beside, rather than replacing, ``visual_rft.py``.
It distinguishes explicit abstention, an empty list, valid detections, and malformed
output. Positive queries use a per-query proxy for COCO's IoU 0.50:0.95 metric:
predictions are greedily matched in confidence order at every IoU threshold, then
box F1 is combined with a bounded proper confidence score.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Sequence

from swift.rewards import ORM, orms


GROUNDING_TASKS = {"grounding", "detection", "object_detection", "gui_grounding"}
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ANSWER_ONLY_FORMAT_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
THINK_ANSWER_FORMAT_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>",
    re.DOTALL,
)
FORMAT_CONTROL_TAG_PATTERN = re.compile(r"</?(?:think|answer)>")
IOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

MALFORMED_REWARD = -1.0
POSITIVE_EMPTY_REWARD = -0.75
EXPLICIT_NO_OBJECTS_REWARD = 1.0
EMPTY_LIST_REWARD = 0.75
BOX_F1_WEIGHT = 0.70
CONFIDENCE_WEIGHT = 0.30
ANSWER_ONLY_FORMAT_REWARD = 0.20
THINK_ANSWER_FORMAT_REWARD = 0.25

ParseKind = Literal["no_objects", "empty_list", "boxes", "malformed"]


@dataclass(frozen=True)
class _Box:
    position: tuple[float, float, float, float]
    confidence: float


@dataclass(frozen=True)
class _ParsedDetection:
    kind: ParseKind
    boxes: tuple[_Box, ...] = ()


class VisualRFTWithNegativeORM(ORM):
    """Score detection rows while teaching calibrated abstention and detections."""

    def __call__(
        self,
        completions: Sequence[str],
        solution: Sequence[str],
        task: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        tasks = ["detection"] * len(completions) if task is None else task
        _validate_batch_lengths(completions, solution, tasks)

        rewards: List[Optional[float]] = []
        for index, (completion, gold, name) in enumerate(zip(completions, solution, tasks)):
            if _normalize_task(name) not in GROUNDING_TASKS:
                rewards.append(None)
                continue

            expected = _parse_detection(gold)
            if expected.kind == "malformed":
                raise ValueError(
                    f"visual_rft_withnegative received malformed gold solution at batch index {index}."
                )
            predicted = _parse_detection(completion)
            rewards.append(_score_detection(predicted, expected))
        return rewards


class VisualRFTWithNegativeFormatORM(ORM):
    """Reward strict answer tags without requiring Qwen3-VL thinking output."""

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> List[float]:
        return [_score_completion_format(completion) for completion in completions]


def _score_completion_format(completion: Any) -> float:
    if not isinstance(completion, str):
        return 0.0

    think_answer_match = THINK_ANSWER_FORMAT_PATTERN.fullmatch(completion)
    if think_answer_match is not None:
        think = think_answer_match.group("think")
        answer = think_answer_match.group("answer")
        if not answer.strip() or _contains_format_control_tag(think) or _contains_format_control_tag(answer):
            return 0.0
        return THINK_ANSWER_FORMAT_REWARD if think.strip() else ANSWER_ONLY_FORMAT_REWARD

    answer_match = ANSWER_ONLY_FORMAT_PATTERN.fullmatch(completion)
    if answer_match is not None:
        answer = answer_match.group("answer")
        if not answer.strip() or _contains_format_control_tag(answer):
            return 0.0
        return ANSWER_ONLY_FORMAT_REWARD

    # ms-swift may pass only the generated payload to the ORM during GRPO.
    if _contains_format_control_tag(completion):
        return 0.0
    parsed = _parse_detection(completion)
    if parsed.kind == "malformed":
        return 0.0
    return ANSWER_ONLY_FORMAT_REWARD


def _contains_format_control_tag(value: str) -> bool:
    return FORMAT_CONTROL_TAG_PATTERN.search(value) is not None


def _validate_batch_lengths(*columns: Sequence[Any]) -> None:
    names = ("completions", "solution", "task")
    lengths = {name: len(column) for name, column in zip(names, columns)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"visual_rft_withnegative batch length mismatch: {lengths}.")


def _score_detection(predicted: _ParsedDetection, expected: _ParsedDetection) -> float:
    if predicted.kind == "malformed":
        return MALFORMED_REWARD

    gold_is_empty = expected.kind in {"no_objects", "empty_list"}
    if gold_is_empty:
        if predicted.kind == "no_objects":
            return EXPLICIT_NO_OBJECTS_REWARD
        if predicted.kind == "empty_list":
            return EMPTY_LIST_REWARD
        return _score_false_positive_only(predicted.boxes)

    if predicted.kind in {"no_objects", "empty_list"}:
        return POSITIVE_EMPTY_REWARD
    return _score_positive_boxes(predicted.boxes, expected.boxes)


def _score_false_positive_only(predicted: Sequence[_Box]) -> float:
    # Every hallucinated box has a fixed cost, while high confidence raises it.
    penalty = sum(0.25 + 0.75 * box.confidence for box in predicted)
    return -min(penalty, 1.0)


def _score_positive_boxes(predicted: Sequence[_Box], expected: Sequence[_Box]) -> float:
    if not predicted or not expected:
        return POSITIVE_EMPTY_REWARD

    scores = []
    for threshold in IOU_THRESHOLDS:
        matched_predictions = _greedy_matches(predicted, expected, threshold)
        true_positives = len(matched_predictions)
        false_positives = len(predicted) - true_positives
        false_negatives = len(expected) - true_positives
        f1_denominator = 2 * true_positives + false_positives + false_negatives
        box_f1 = 0.0 if f1_denominator == 0 else 2 * true_positives / f1_denominator

        # This centered Brier score is zero for confidence=0, positive for a
        # matched box, and negative for an unmatched box. Averaging it over the
        # IoU thresholds makes the optimal confidence track localization quality.
        confidence_score = 0.0
        for index, box in enumerate(predicted):
            confidence = box.confidence
            if index in matched_predictions:
                confidence_score += 2.0 * confidence - confidence * confidence
            else:
                confidence_score -= confidence * confidence
        confidence_score /= max(len(expected), len(predicted), 1)
        scores.append(BOX_F1_WEIGHT * box_f1 + CONFIDENCE_WEIGHT * confidence_score)

    reward = sum(scores) / len(scores)
    if not math.isfinite(reward):
        return MALFORMED_REWARD
    return min(max(reward, -1.0), 1.0)


def _greedy_matches(
    predicted: Sequence[_Box],
    expected: Sequence[_Box],
    threshold: float,
) -> set[int]:
    prediction_order = sorted(range(len(predicted)), key=lambda index: (-predicted[index].confidence, index))
    unmatched_gold = set(range(len(expected)))
    matched_predictions: set[int] = set()

    for prediction_index in prediction_order:
        best_gold_index: Optional[int] = None
        best_iou = -1.0
        for gold_index in sorted(unmatched_gold):
            overlap = _iou(expected[gold_index].position, predicted[prediction_index].position)
            if overlap > best_iou:
                best_iou = overlap
                best_gold_index = gold_index
        if best_gold_index is not None and best_iou + 1e-12 >= threshold:
            unmatched_gold.remove(best_gold_index)
            matched_predictions.add(prediction_index)
    return matched_predictions


def _parse_detection(text: Any) -> _ParsedDetection:
    if not isinstance(text, str):
        return _ParsedDetection("malformed")

    match = ANSWER_PATTERN.search(text)
    if match:
        payload = match.group(1).strip()
    elif "<answer>" in text or "</answer>" in text:
        return _ParsedDetection("malformed")
    else:
        payload = text.strip()

    normalized = re.sub(r"\s+", " ", payload).strip().casefold()
    if normalized == "no objects":
        return _ParsedDetection("no_objects")
    if not payload:
        return _ParsedDetection("malformed")

    try:
        value = json.loads(payload)
    except (TypeError, ValueError):
        try:
            value = ast.literal_eval(payload)
        except (SyntaxError, TypeError, ValueError):
            return _ParsedDetection("malformed")

    if not isinstance(value, list):
        return _ParsedDetection("malformed")
    if not value:
        return _ParsedDetection("empty_list")

    boxes = []
    for item in value:
        if not isinstance(item, dict):
            return _ParsedDetection("malformed")
        position = item.get("Position")
        confidence = item.get("Confidence")
        if not _valid_position(position) or not _valid_confidence(confidence):
            return _ParsedDetection("malformed")
        boxes.append(
            _Box(
                position=tuple(float(number) for number in position),
                confidence=float(confidence),
            )
        )
    return _ParsedDetection("boxes", tuple(boxes))


def _valid_position(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_finite_number(number) and 0.0 <= float(number) <= 1000.0 for number in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _valid_confidence(value: Any) -> bool:
    return _finite_number(value) and 0.0 <= float(value) <= 1.0


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _normalize_task(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


orms["visual_rft_withnegative"] = VisualRFTWithNegativeORM
orms["visual_rft_withnegative_format"] = VisualRFTWithNegativeFormatORM
