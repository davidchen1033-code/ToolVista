"""Fail fast when a GRPO JSONL is incompatible with the Visual-RFT reward plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORTED_TASKS = {
    "classification",
    "counting",
    "detection",
    "grounding",
    "gui_grounding",
    "object_detection",
    "question_answering",
    "visual_question_answering",
    "vqa",
}


def normalize_task(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-rows", type=int, default=100)
    return parser.parse_args()


def validate_row(row: Any, row_number: int) -> str:
    if not isinstance(row, dict):
        raise ValueError(f"row {row_number}: expected a JSON object")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {row_number}: messages must be a non-empty list")
    if not isinstance(messages[-1], dict) or messages[-1].get("role") != "user":
        raise ValueError(f"row {row_number}: GRPO prompt must end with a user message")
    solution = row.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError(f"row {row_number}: solution must be a non-empty string")
    task = normalize_task(row.get("task"))
    if task not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ValueError(
            f"row {row_number}: task {task!r} is not supported by visual_rft; supported: {supported}"
        )
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError(f"row {row_number}: images must contain at least one image path")
    return task


def main() -> None:
    args = parse_args()
    checked = 0
    task_counts: dict[str, int] = {}
    with args.dataset.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, 1):
            if args.max_rows > 0 and checked >= args.max_rows:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"row {row_number}: invalid JSON: {exc.msg}") from exc
            try:
                task = validate_row(row, row_number)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            task_counts[task] = task_counts.get(task, 0) + 1
            checked += 1
    if checked == 0:
        raise SystemExit(f"Dataset contains no rows: {args.dataset}")
    print(f"Visual-RFT reward validation passed: rows={checked}, tasks={task_counts}, dataset={args.dataset}")


if __name__ == "__main__":
    main()
