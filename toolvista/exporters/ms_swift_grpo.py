"""Export canonical reinforcement-learning prompts to ms-swift GRPO JSONL."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from toolvista.exporters.ms_swift import MsSwiftExportError, canonical_to_ms_swift_row
from toolvista.schema.canonical import CanonicalSample


REWARD_FIELDS = (
    "solution",
    "ground_truth",
    "task",
    "class_label",
    "bbox",
    "image_width",
    "image_height",
)


class MsSwiftGRPOExportError(MsSwiftExportError):
    """Raised when a canonical sample is not a valid ms-swift GRPO prompt."""


def canonical_to_ms_swift_grpo_row(
    sample: CanonicalSample,
    *,
    include_metadata: bool = False,
) -> Dict[str, Any]:
    """Lower one canonical prompt and promote approved reward inputs."""

    _validate_grpo_sample(sample)
    try:
        row = canonical_to_ms_swift_row(sample, include_metadata=include_metadata)
    except (MsSwiftExportError, TypeError, ValueError) as exc:
        raise MsSwiftGRPOExportError(str(exc)) from exc

    for field in REWARD_FIELDS:
        if field in sample.labels and sample.labels[field] is not None:
            row[field] = sample.labels[field]
    _assert_json_row(row)
    return row


def export_jsonl(
    samples: Iterable[CanonicalSample],
    output_path: str | Path,
    *,
    include_metadata: bool = False,
) -> int:
    """Atomically write ms-swift GRPO rows and return the row count."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Optional[Path] = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for row_index, sample in enumerate(samples):
                try:
                    row = canonical_to_ms_swift_grpo_row(
                        sample,
                        include_metadata=include_metadata,
                    )
                except (MsSwiftGRPOExportError, TypeError, ValueError) as exc:
                    raise MsSwiftGRPOExportError(
                        f"Failed to export row {row_index}, sample_id={sample.sample_id!r}: {exc}"
                    ) from exc
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return count


def _validate_grpo_sample(sample: CanonicalSample) -> None:
    if sample.preference is not None:
        raise MsSwiftGRPOExportError("GRPO samples must not contain preference data.")
    if not sample.messages or sample.messages[-1].role != "user":
        raise MsSwiftGRPOExportError("GRPO samples must end with a user prompt.")

    solution = sample.labels.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise MsSwiftGRPOExportError("GRPO samples require a non-empty labels.solution string.")
    task = sample.labels.get("task")
    if not isinstance(task, str) or not task.strip():
        raise MsSwiftGRPOExportError("GRPO samples require a non-empty labels.task string.")


def _assert_json_row(row: Dict[str, Any]) -> None:
    try:
        json.dumps(row, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MsSwiftGRPOExportError(f"GRPO row is not JSON serializable: {exc}") from exc
