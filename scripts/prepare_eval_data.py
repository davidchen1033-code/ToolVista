"""Materialize frozen xLAM and MLLM-Tool text evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.adapters.mllm_tool import MLLMToolAdapter
from toolvista.adapters.xlam import XLAMAdapter
from toolvista.evaluation.io import load_jsonl, write_json, write_jsonl
from toolvista.schema.canonical import CanonicalMessage, CanonicalSample
from toolvista.utils.paths import ToolVistaPaths


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _message_dict(message: CanonicalMessage) -> dict[str, Any]:
    return {key: value for key, value in message.to_dict().items() if key in {"role", "content"}}


def _last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if message.get("role") == "user" and isinstance(content, str):
            return content
    return ""


def _selection_key(sample_id: str, *, seed: int, namespace: str) -> str:
    digest = hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode("utf-8")).hexdigest()
    return f"{digest}:{sample_id}"


def _fixed_seed_select(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return sorted(rows, key=lambda row: str(row["sample_id"]))

    selected: list[tuple[int, str, int, dict[str, Any]]] = []
    for source_index, row in enumerate(rows):
        sample_id = str(row["sample_id"])
        digest = hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode("utf-8")).digest()
        score = int.from_bytes(digest, "big")
        item = (-score, sample_id, source_index, row)
        if len(selected) < limit:
            heapq.heappush(selected, item)
        elif item > selected[0]:
            heapq.heapreplace(selected, item)
    return [item[3] for item in sorted(selected, key=lambda item: (-item[0], item[1], item[2]))]


def _xlam_row(sample: CanonicalSample) -> dict[str, Any] | None:
    target_calls = sample.labels.get("target_tool_calls")
    if sample.labels.get("target_kind") != "tool_calls" or not isinstance(target_calls, list) or not target_calls:
        return None

    prompt_messages = sample.messages[: -len(target_calls)]
    if not prompt_messages or any(message.role not in {"system", "user"} for message in prompt_messages):
        return None
    messages = [_message_dict(message) for message in prompt_messages]
    return {
        "sample_id": sample.sample_id,
        "source_project": "xLAM ActionStudio",
        "split": "fixed_seed_dev",
        "messages": messages,
        "tools": [tool.to_openai_tool() for tool in sample.tools],
        "target_tool_calls": target_calls,
        "normalized_text": normalize_text(_last_user_text(messages)),
    }


def prepare_xlam_rows(paths: ToolVistaPaths, *, limit: int, seed: int) -> list[dict[str, Any]]:
    adapter = XLAMAdapter(paths)
    rows = (
        row
        for result in adapter.iter_results()
        if result.ok and result.sample is not None
        for row in [_xlam_row(result.sample)]
        if row is not None
    )
    selected = _fixed_seed_select(rows, limit=limit, seed=seed, namespace="xlam")
    validate_manifest(selected)
    return selected


def prepare_mllm_router_rows(paths: ToolVistaPaths, *, limit: int, seed: int) -> list[dict[str, Any]]:
    adapter = MLLMToolAdapter(paths, file_keys=("text",), validate_media_files=False)
    samples = [result.sample for result in adapter.iter_results() if result.ok and result.sample is not None]
    catalog = sorted(
        {
            tool
            for sample in samples
            for tool in sample.labels.get("target_tools", [])
            if isinstance(tool, str) and tool.strip()
        }
    )
    system_prompt = (
        "You are a tool router. Choose the best tool ID from this catalog: "
        + json.dumps(catalog, ensure_ascii=False)
        + ". Return exactly one tool ID and no explanation."
    )

    rows = [
        row
        for sample in samples
        for row in [_mllm_router_row(sample, catalog, system_prompt)]
        if row is not None
    ]

    merged_rows: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
        text = str(row["normalized_text"])
        if text not in merged_rows:
            merged_rows[text] = dict(row)
            merged_rows[text]["target_tools"] = sorted(set(row["target_tools"]))
        else:
            merged_rows[text]["target_tools"] = sorted(
                set(merged_rows[text]["target_tools"]) | set(row["target_tools"])
            )

    selected = _fixed_seed_select(merged_rows.values(), limit=limit, seed=seed, namespace="mllm_router")
    validate_manifest(selected)
    return selected


def _mllm_router_row(
    sample: CanonicalSample,
    catalog: Sequence[str],
    system_prompt: str,
) -> dict[str, Any] | None:
    target_tools = sample.labels.get("target_tools")
    if (
        not isinstance(target_tools, list)
        or not target_tools
        or not sample.messages
        or sample.messages[-1].role != "assistant"
        or any(message.role not in {"system", "user"} for message in sample.messages[:-1])
    ):
        return None
    if any(not isinstance(tool, str) or tool not in catalog for tool in target_tools):
        return None

    messages = [{"role": "system", "content": system_prompt}] + [
        _message_dict(message) for message in sample.messages[:-1]
    ]
    return {
        "sample_id": sample.sample_id,
        "source_project": "MLLM-Tool",
        "split": "fixed_seed_dev",
        "input_modality": "text",
        "messages": messages,
        "router_catalog": list(catalog),
        "target_tools": target_tools,
        "normalized_text": normalize_text(_last_user_text(messages)),
    }


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Evaluation manifest is empty.")

    raw_sample_ids = [row.get("sample_id") for row in rows]
    if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in raw_sample_ids):
        raise ValueError("Every evaluation row requires a non-empty sample_id.")
    sample_ids = [str(sample_id).strip() for sample_id in raw_sample_ids]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Evaluation manifest contains duplicate sample IDs.")

    raw_text = [row.get("normalized_text") for row in rows]
    if any(not isinstance(value, str) or not normalize_text(value) for value in raw_text):
        raise ValueError("Every evaluation row requires non-empty normalized_text.")
    normalized = [normalize_text(str(value)) for value in raw_text]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Evaluation manifest contains duplicate normalized text.")


def _training_identities(row: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    sample_ids: set[str] = set()
    top_level_id = row.get("sample_id")
    if isinstance(top_level_id, str) and top_level_id.strip():
        sample_ids.add(top_level_id.strip())

    metadata = row.get("metadata")
    metadata_id = metadata.get("sample_id") if isinstance(metadata, Mapping) else None
    if isinstance(metadata_id, str) and metadata_id.strip():
        sample_ids.add(metadata_id.strip())

    normalized_texts: set[str] = set()
    supplied_text = row.get("normalized_text")
    if isinstance(supplied_text, str) and normalize_text(supplied_text):
        normalized_texts.add(normalize_text(supplied_text))
    messages = row.get("messages")
    if isinstance(messages, list):
        message_text = normalize_text(_last_user_text(messages))
        if message_text:
            normalized_texts.add(message_text)
    return sample_ids, normalized_texts


def _eval_identities(eval_rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    eval_ids: set[str] = set()
    eval_text: set[str] = set()
    for row in eval_rows:
        sample_ids, normalized_texts = _training_identities(row)
        eval_ids.update(sample_ids)
        eval_text.update(normalized_texts)
    return eval_ids, eval_text


def _check_no_exact_row_overlap(
    training_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> None:
    validate_manifest(eval_rows)
    eval_ids, eval_text = _eval_identities(eval_rows)
    overlapping_ids: set[str] = set()
    overlapping_text: set[str] = set()
    for row in training_rows:
        sample_ids, normalized_texts = _training_identities(row)
        overlapping_ids.update(sample_ids & eval_ids)
        overlapping_text.update(normalized_texts & eval_text)
    if overlapping_ids or overlapping_text:
        raise ValueError(
            f"Training/evaluation overlap detected: sample_ids={len(overlapping_ids)}, "
            f"normalized_text={len(overlapping_text)}"
        )


def check_no_exact_overlap(training_files: Sequence[Path], eval_rows: Sequence[Mapping[str, Any]]) -> None:
    training_rows = [row for path in training_files for row in load_jsonl(path)]
    _check_no_exact_row_overlap(training_rows, eval_rows)


def filter_training_rows(
    training_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_manifest(eval_rows)
    eval_ids, eval_text = _eval_identities(eval_rows)
    filtered: list[dict[str, Any]] = []
    for row in training_rows:
        sample_ids, normalized_texts = _training_identities(row)
        if sample_ids & eval_ids or normalized_texts & eval_text:
            continue
        filtered.append(dict(row))
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl.yaml")
    parser.add_argument("--xlam-output", type=Path, default=Path("data/eval/xlam/dev.jsonl"))
    parser.add_argument("--mllm-output", type=Path, default=Path("data/eval/mllm_router/dev.jsonl"))
    parser.add_argument("--xlam-limit", type=int, default=200)
    parser.add_argument("--mllm-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-file", type=Path, action="append", default=[])
    parser.add_argument(
        "--filter-training",
        type=Path,
        nargs=2,
        action="append",
        default=[],
        metavar=("INPUT", "OUTPUT"),
        help="Filter one ms-swift training JSONL into OUTPUT using the frozen evaluation IDs/text.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_manifest(path: Path, rows: list[dict[str, Any]], *, seed: int, overwrite: bool) -> None:
    validate_manifest(rows)
    report_path = path.with_suffix(".report.json")
    _require_available_outputs([path, report_path], overwrite=overwrite)
    _write_output_transaction(
        jsonl_outputs=[(path, rows)],
        json_outputs=[(report_path, _manifest_report(path, rows, seed=seed))],
    )


def _manifest_report(path: Path, rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, Any]:
    return {
        "output": path.as_posix(),
        "sample_count": len(rows),
        "seed": seed,
        "sample_ids": [row["sample_id"] for row in rows],
        "selection_keys": [
            _selection_key(
                str(row["sample_id"]),
                seed=seed,
                namespace="xlam" if "target_tool_calls" in row else "mllm_router",
            )
            for row in rows
        ],
    }


def _require_available_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    materialized = list(paths)
    normalized = [str(path.resolve()).casefold() for path in materialized]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Multiple outputs resolve to the same path.")
    if overwrite:
        return
    existing = [path for path in materialized if path.exists()]
    if existing:
        joined = ", ".join(path.as_posix() for path in existing)
        raise FileExistsError(f"Output already exists: {joined}. Use --overwrite explicitly.")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _backup_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.bak")


def _write_output_transaction(
    *,
    jsonl_outputs: Sequence[tuple[Path, Iterable[Mapping[str, Any]]]],
    json_outputs: Sequence[tuple[Path, Mapping[str, Any]]],
) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for final_path, rows in jsonl_outputs:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = _temporary_sibling(final_path)
            staged.append((temporary_path, final_path))
            write_jsonl(temporary_path, rows)
        for final_path, value in json_outputs:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = _temporary_sibling(final_path)
            staged.append((temporary_path, final_path))
            write_json(temporary_path, value)
    except BaseException:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        raise

    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for _, final_path in staged:
            if final_path.exists():
                backup_path = _backup_sibling(final_path)
                os.replace(final_path, backup_path)
                backups.append((backup_path, final_path))
        for temporary_path, final_path in staged:
            os.replace(temporary_path, final_path)
            promoted.append(final_path)
    except BaseException:
        for final_path in reversed(promoted):
            final_path.unlink(missing_ok=True)
        for backup_path, final_path in reversed(backups):
            if backup_path.exists():
                os.replace(backup_path, final_path)
        raise
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        for backup_path, _ in backups:
            backup_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    paths = ToolVistaPaths.from_yaml(args.config)
    xlam_rows = prepare_xlam_rows(paths, limit=args.xlam_limit, seed=args.seed)
    mllm_rows = prepare_mllm_router_rows(paths, limit=args.mllm_limit, seed=args.seed)
    eval_rows = [*xlam_rows, *mllm_rows]
    validate_manifest(eval_rows)

    output_paths = [
        args.xlam_output,
        args.xlam_output.with_suffix(".report.json"),
        args.mllm_output,
        args.mllm_output.with_suffix(".report.json"),
        *(output_path for _, output_path in args.filter_training),
    ]
    _require_available_outputs(output_paths, overwrite=args.overwrite)

    filtered_reports: list[dict[str, Any]] = []
    filtered_outputs: list[tuple[Path, list[dict[str, Any]]]] = []
    for input_path, output_path in args.filter_training:
        source_rows = load_jsonl(input_path)
        filtered_rows = filter_training_rows(source_rows, eval_rows)
        _check_no_exact_row_overlap(filtered_rows, eval_rows)
        filtered_outputs.append((output_path, filtered_rows))
        filtered_reports.append(
            {
                "input": input_path.as_posix(),
                "output": output_path.as_posix(),
                "input_rows": len(source_rows),
                "output_rows": len(filtered_rows),
                "removed_rows": len(source_rows) - len(filtered_rows),
            }
        )
    if args.training_file:
        check_no_exact_overlap(args.training_file, eval_rows)

    _write_output_transaction(
        jsonl_outputs=[
            *filtered_outputs,
            (args.xlam_output, xlam_rows),
            (args.mllm_output, mllm_rows),
        ],
        json_outputs=[
            (args.xlam_output.with_suffix(".report.json"), _manifest_report(args.xlam_output, xlam_rows, seed=args.seed)),
            (
                args.mllm_output.with_suffix(".report.json"),
                _manifest_report(args.mllm_output, mllm_rows, seed=args.seed),
            ),
        ],
    )
    print(
        json.dumps(
            {
                "xlam": len(xlam_rows),
                "mllm_router": len(mllm_rows),
                "filtered_training": filtered_reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
