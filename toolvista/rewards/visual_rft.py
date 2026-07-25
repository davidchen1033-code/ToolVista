"""Minimal ms-swift reward plugin for Visual-RFT answer and grounding tasks."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, List, Optional, Sequence

from swift.rewards import ORM, orms


GROUNDING_TASKS = {"grounding", "detection", "object_detection", "gui_grounding"}
ANSWER_TASKS = {"classification", "vqa", "visual_question_answering", "question_answering", "counting"}
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class VisualRFTORM(ORM):
    """Route each Visual-RFT row to its upstream-style task reward."""

    def __call__(
        self,
        completions: Sequence[str],
        solution: Sequence[str],
        task: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        tasks = task or ["vqa"] * len(completions)
        _validate_batch_lengths(completions, solution, tasks)
        rewards: List[Optional[float]] = []
        for completion, gold, name in zip(completions, solution, tasks):
            normalized_task = _normalize_task(name)
            if normalized_task in GROUNDING_TASKS:
                rewards.append(_score_boxes(completion, gold))
            elif normalized_task in ANSWER_TASKS:
                rewards.append(_score_answer(completion, gold))
            else:
                rewards.append(None)
        return rewards


class VisualRFTIoUORM(ORM):
    """Score Visual-RFT boxes using its confidence-ordered greedy IoU rule."""

    def __call__(
        self,
        completions: Sequence[str],
        solution: Sequence[str],
        task: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[Optional[float]]:
        tasks = task or ["grounding"] * len(completions)
        _validate_batch_lengths(completions, solution, tasks)
        return [
            None if _normalize_task(name) not in GROUNDING_TASKS else _score_boxes(completion, gold)
            for completion, gold, name in zip(completions, solution, tasks)
        ]


def _validate_batch_lengths(*columns: Sequence[Any]) -> None:
    names = ("completions", "solution", "task")
    lengths = {name: len(column) for name, column in zip(names, columns)}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Visual-RFT reward batch length mismatch: {lengths}.")


def _score_answer(completion: Any, solution: Any) -> float:
    student = _normalize_answer(_extract_answer(completion))
    expected = _normalize_answer(_extract_answer(solution))
    if not student or not expected:
        return 0.0
    return 1.0 if expected in student or student in expected else 0.0


def _extract_answer(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("_", "").lower()


def _score_boxes(completion: str, solution: str) -> float:
    predicted = _parse_boxes(completion)
    expected = _parse_boxes(solution)
    if not predicted or not expected:
        return 0.0

    predicted = _deduplicate(predicted)
    predicted.sort(key=lambda box: box["Confidence"], reverse=True)
    unmatched = set(range(len(expected)))
    matched_iou = 0.0

    for prediction in predicted:
        best_index = None
        best_iou = 0.0
        for index in unmatched:
            iou = _iou(expected[index]["Position"], prediction["Position"])
            if iou > best_iou:
                best_index = index
                best_iou = iou
        if best_index is not None and best_iou > 0.5:
            unmatched.remove(best_index)
            matched_iou += best_iou

    return min(matched_iou / max(len(expected), len(predicted)), 1.0)


def _parse_boxes(text: Any) -> Optional[List[dict[str, Any]]]:
    if not isinstance(text, str):
        return None
    match = ANSWER_PATTERN.search(text)
    if match:
        payload = match.group(1)
    elif "<answer>" in text:
        payload = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
    else:
        payload = text
    payload = payload.strip().replace("[[", "[").replace("]]", "]")
    try:
        value = json.loads(payload)
    except ValueError:
        try:
            value = ast.literal_eval(payload)
        except (SyntaxError, TypeError, ValueError):
            value = _recover_truncated_boxes(payload)

    if not isinstance(value, list) or not value:
        return None
    boxes: List[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        position = item.get("Position")
        confidence = item.get("Confidence")
        if not _valid_position(position) or not _finite_number(confidence):
            return None
        boxes.append({"Position": [float(number) for number in position], "Confidence": float(confidence)})
    return boxes


def _recover_truncated_boxes(payload: str) -> Any:
    last_complete = payload.rfind("},")
    if last_complete < 0:
        return None
    recovered = payload[:last_complete + 1].strip()
    if not recovered.startswith("["):
        return None
    recovered += "]"
    try:
        return json.loads(recovered)
    except ValueError:
        try:
            return ast.literal_eval(recovered)
        except (SyntaxError, TypeError, ValueError):
            return None


def _valid_position(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_finite_number(number) for number in value)
        and value[2] > value[0]
        and value[3] > value[1]
    )


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _deduplicate(boxes: Sequence[dict[str, Any]]) -> List[dict[str, Any]]:
    unique = []
    seen = set()
    for box in boxes:
        position = tuple(box["Position"])
        if position not in seen:
            seen.add(position)
            unique.append(box)
    return unique


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
    return intersection / (first_area + second_area - intersection)


def _normalize_task(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


orms["visual_rft"] = VisualRFTORM
orms["visual_rft_iou"] = VisualRFTIoUORM
