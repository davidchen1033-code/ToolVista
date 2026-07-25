"""Build the isolated C3 datafixed-agentretention dataset track.

The existing datafixed artifacts remain immutable controls. This builder derives
an agent-retention track with three targeted fixes:

* a modality-balanced Router light manifest containing all holdout videos;
* complete ToolVQA trajectories plus final-answer-only terminal transitions;
* high-confidence xLAM no-tool examples for function-call abstention.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.build_sft_replay import encode_jsonl, write_transaction
except ModuleNotFoundError:  # Direct execution sets sys.path[0] to scripts/.
    from build_sft_replay import encode_jsonl, write_transaction


VARIANT = "datafixed-agentretention"
ABSTENTION_PATTERN = re.compile(
    r"(?:none of (?:the )?(?:functions|tools)|"
    r"(?:given|provided|available) (?:functions|tools).{0,80}(?:not|cannot|unable)|"
    r"(?:functions|tools).{0,80}(?:do not|does not|don't|doesn't|cannot|can't) (?:support|include|provide)|"
    r"(?:cannot|can't|unable to) (?:help|complete|perform|fulfill|answer)|"
    r"(?:missing|lack(?:s|ing)?|insufficient).{0,80}(?:parameter|information))",
    flags=re.IGNORECASE,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--light-limit", type=int, default=200)
    parser.add_argument("--video-light-min", type=int, default=16)
    parser.add_argument("--terminal-max-rows", type=int, default=12000)
    parser.add_argument("--abstention-max-rows", type=int, default=2000)
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path("data/reports/c3/router_datafixed.report.json"),
    )
    parser.add_argument(
        "--router-train-source",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/mllm_tool_router_datafixed_train.jsonl"
        ),
    )
    parser.add_argument(
        "--router-full-source",
        type=Path,
        default=Path("data/eval/mllm_router_datafixed/full.jsonl"),
    )
    parser.add_argument(
        "--direct-answer-source",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/toolvqa_direct_answer_datafixed_train.jsonl"
        ),
    )
    parser.add_argument(
        "--toolvqa-source",
        type=Path,
        default=Path("data/exports/msswift_sft/toolvqa_train.jsonl"),
    )
    parser.add_argument(
        "--xlam-source",
        type=Path,
        default=Path("data/exports/msswift_sft/xlam_train.jsonl"),
    )
    parser.add_argument(
        "--router-train",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/"
            "mllm_tool_router_datafixed-agentretention_train.jsonl"
        ),
    )
    parser.add_argument(
        "--router-full",
        type=Path,
        default=Path("data/eval/mllm_router_datafixed-agentretention/full.jsonl"),
    )
    parser.add_argument(
        "--router-light",
        type=Path,
        default=Path("data/eval/mllm_router_datafixed-agentretention/light.jsonl"),
    )
    parser.add_argument(
        "--toolvqa-complete",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/"
            "toolvqa_complete_datafixed-agentretention_train.jsonl"
        ),
    )
    parser.add_argument(
        "--toolvqa-terminal",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/"
            "toolvqa_terminal_answer_datafixed-agentretention_train.jsonl"
        ),
    )
    parser.add_argument(
        "--xlam-abstention",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/"
            "xlam_abstention_datafixed-agentretention_train.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/reports/c3/router_datafixed-agentretention.report.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
            yield line_number, value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in _iter_jsonl(path)]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _require_variant_output(path: Path) -> None:
    lexical = str(path).casefold()
    resolved = str(_resolved(path)).casefold()
    if VARIANT not in lexical or VARIANT not in resolved:
        raise ValueError(
            f"output must remain in the {VARIANT!r} namespace: {path}"
        )


def _identity(row: Mapping[str, Any]) -> str:
    material = {
        key: row.get(key)
        for key in ("messages", "images", "videos", "audios", "tools")
        if key in row
    }
    return hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _deterministic_select(
    rows: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    if count < 0 or count > len(rows):
        raise ValueError(
            f"invalid selection count for {namespace}: {count}/{len(rows)}"
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"{seed}:{namespace}:{_identity(row)}".encode("utf-8")
            ).hexdigest(),
            _identity(row),
        ),
    )
    return ordered[:count]


def _validate_base_report(
    report_path: Path,
    *,
    router_train: Path,
    router_full: Path,
    direct_answer: Path,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("variant") != "datafixed":
        raise ValueError("base report must be the audited datafixed preparation report")
    artifacts = report.get("artifacts")
    audit = report.get("audit")
    if not isinstance(artifacts, dict) or not isinstance(audit, dict):
        raise ValueError("base report is missing artifacts/audit mappings")
    bindings = (
        ("router_train", router_train, "router_train"),
        ("router_full", router_full, "router_full"),
        ("direct_answer_train", direct_answer, "direct_answer"),
    )
    for artifact_key, actual, audit_key in bindings:
        reported = _resolved(Path(str(artifacts.get(artifact_key, ""))))
        if reported != actual.resolve(strict=True):
            raise ValueError(
                f"base report artifact mismatch for {artifact_key}: "
                f"{reported} != {actual.resolve(strict=True)}"
            )
        audit_value = audit.get(audit_key)
        if not isinstance(audit_value, dict):
            raise ValueError(f"base report is missing audit.{audit_key}")
        if audit_value.get("sha256") != _sha256_file(actual):
            raise ValueError(f"base report SHA-256 mismatch for {artifact_key}")
    return report


def _balanced_router_light(
    full_rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    video_minimum: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if limit <= 0 or limit > len(full_rows):
        raise ValueError(f"light limit must be in [1, {len(full_rows)}]")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        modality = str(row.get("input_modality", "")).strip().lower()
        if not modality:
            raise ValueError("router full row is missing input_modality")
        grouped[modality].append(row)
    video_count = len(grouped.get("video", []))
    if video_minimum <= 0 or video_minimum > video_count:
        raise ValueError(
            f"video-light-min must be in [1, {video_count}], got {video_minimum}"
        )
    if video_minimum >= limit:
        raise ValueError("video-light-min must be smaller than light-limit")

    quotas = {"video": video_minimum}
    remaining = limit - video_minimum
    other_modalities = sorted(name for name in grouped if name != "video")
    other_total = sum(len(grouped[name]) for name in other_modalities)
    if not other_modalities or other_total < remaining:
        raise ValueError("router full data cannot satisfy balanced light quotas")
    raw = {
        name: remaining * len(grouped[name]) / other_total
        for name in other_modalities
    }
    for name in other_modalities:
        quotas[name] = min(len(grouped[name]), math.floor(raw[name]))
    unassigned = limit - sum(quotas.values())
    for name in sorted(
        other_modalities,
        key=lambda item: (-(raw[item] - math.floor(raw[item])), item),
    ):
        if not unassigned:
            break
        if quotas[name] < len(grouped[name]):
            quotas[name] += 1
            unassigned -= 1
    if unassigned:
        raise ValueError(f"failed to assign {unassigned} balanced light rows")

    selected: list[dict[str, Any]] = []
    for modality in sorted(quotas):
        selected.extend(
            _deterministic_select(
                grouped[modality],
                count=quotas[modality],
                seed=seed,
                namespace=f"router-light:{modality}",
            )
        )
    selected = _deterministic_select(
        selected,
        count=len(selected),
        seed=seed,
        namespace="router-light:output-order",
    )
    sample_ids = [str(row.get("sample_id")) for row in selected]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("balanced router light contains duplicate sample IDs")
    return selected, dict(sorted(quotas.items()))


def _terminal_rows(
    source: Path,
    *,
    seed: int,
    max_rows: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    complete_by_id: dict[str, dict[str, Any]] = {}
    terminal_by_id: dict[str, dict[str, Any]] = {}
    skipped = Counter()
    input_rows = 0
    for source_row, row in _iter_jsonl(source):
        input_rows += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            skipped["invalid_messages"] += 1
            continue
        last = messages[-1]
        if (
            not isinstance(last, Mapping)
            or last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
            or not last["content"].strip()
        ):
            skipped["missing_final_answer"] += 1
            continue
        roles = [
            message.get("role")
            for message in messages
            if isinstance(message, Mapping)
        ]
        if "tool_call" not in roles or "tool_response" not in roles:
            skipped["missing_tool_history"] += 1
            continue
        if not isinstance(row.get("tools"), str) or not row["tools"].strip():
            skipped["missing_tools"] += 1
            continue

        complete = copy.deepcopy(row)
        metadata = complete.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"upstream_metadata": metadata}
            complete["metadata"] = metadata
        metadata.update(
            {
                "agent_retention_source": "toolvqa_complete_trajectory",
                "agent_retention_source_row": source_row,
                "agent_retention_contract": "tool-history-to-final-answer-v1",
            }
        )
        identity = _identity(complete)
        complete_by_id.setdefault(identity, complete)

        terminal = copy.deepcopy(complete)
        terminal_messages = terminal["messages"]
        for index, message in enumerate(terminal_messages):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role in {"assistant", "tool_call"}:
                message["loss_scale"] = 1.0 if index == len(terminal_messages) - 1 else 0.0
        terminal["metadata"]["agent_retention_source"] = (
            "toolvqa_terminal_answer"
        )
        terminal["metadata"]["supervision"] = "final_assistant_only"
        terminal_by_id.setdefault(identity, terminal)

    complete_rows = _deterministic_select(
        list(complete_by_id.values()),
        count=min(max_rows, len(complete_by_id)),
        seed=seed,
        namespace="toolvqa-complete",
    )
    selected_ids = {_identity(row) for row in complete_rows}
    terminal_rows = [
        terminal_by_id[identity]
        for identity in sorted(
            selected_ids,
            key=lambda value: hashlib.sha256(
                f"{seed}:toolvqa-terminal:{value}".encode("utf-8")
            ).hexdigest(),
        )
    ]
    if not terminal_rows:
        raise ValueError("ToolVQA terminal conversion produced no rows")
    return complete_rows, terminal_rows, {
        "input_rows": input_rows,
        "eligible_unique_rows": len(complete_by_id),
        "selected_rows": len(complete_rows),
        "max_rows": max_rows,
        "skipped": dict(sorted(skipped.items())),
        "contract": "complete tool history with final assistant; terminal copy supervises final assistant only",
    }


def _valid_tools(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, list) and bool(parsed)


def _abstention_rows(
    source: Path,
    *,
    seed: int,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    skipped = Counter()
    input_rows = 0
    for source_row, row in _iter_jsonl(source):
        input_rows += 1
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            skipped["invalid_messages"] += 1
            continue
        if any(
            isinstance(message, Mapping) and message.get("role") == "tool_call"
            for message in messages
        ):
            skipped["contains_tool_call"] += 1
            continue
        last = messages[-1]
        if (
            not isinstance(last, Mapping)
            or last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
        ):
            skipped["missing_final_answer"] += 1
            continue
        content = last["content"].strip()
        words = content.split()
        if (
            len(words) < 6
            or len(words) > 180
            or not ABSTENTION_PATTERN.search(content)
            or not re.search(r"\b(?:function|tool)s?\b", content, re.IGNORECASE)
            or content[0] in ",;:)]}"
            or not _valid_tools(row.get("tools"))
        ):
            skipped["not_high_confidence_abstention"] += 1
            continue
        converted = copy.deepcopy(row)
        for index, message in enumerate(converted["messages"]):
            if isinstance(message, dict) and message.get("role") == "assistant":
                message["loss_scale"] = 1.0 if index == len(messages) - 1 else 0.0
        metadata = converted.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"upstream_metadata": metadata}
            converted["metadata"] = metadata
        metadata.update(
            {
                "agent_retention_source": "xlam_no_tool_abstention",
                "agent_retention_source_row": source_row,
                "agent_retention_contract": "catalog-present-no-tool-call-v1",
            }
        )
        by_id.setdefault(_identity(converted), converted)

    selected = _deterministic_select(
        list(by_id.values()),
        count=min(max_rows, len(by_id)),
        seed=seed,
        namespace="xlam-abstention",
    )
    if not selected:
        raise ValueError("xLAM abstention conversion produced no rows")
    return selected, {
        "input_rows": input_rows,
        "eligible_unique_rows": len(by_id),
        "selected_rows": len(selected),
        "max_rows": max_rows,
        "skipped": dict(sorted(skipped.items())),
        "contract": "tool catalog present, no tool call, explicit high-confidence abstention",
    }


def build_outputs(args: argparse.Namespace) -> tuple[dict[Path, bytes], dict[str, Any]]:
    if args.light_limit <= 0:
        raise ValueError("--light-limit must be positive")
    if args.terminal_max_rows <= 0 or args.abstention_max_rows <= 0:
        raise ValueError("terminal/abstention max rows must be positive")

    inputs = {
        "base_report": args.base_report.resolve(strict=True),
        "router_train": args.router_train_source.resolve(strict=True),
        "router_full": args.router_full_source.resolve(strict=True),
        "direct_answer": args.direct_answer_source.resolve(strict=True),
        "toolvqa": args.toolvqa_source.resolve(strict=True),
        "xlam": args.xlam_source.resolve(strict=True),
    }
    outputs = {
        "router_train": _resolved(args.router_train),
        "router_full": _resolved(args.router_full),
        "router_light": _resolved(args.router_light),
        "toolvqa_complete": _resolved(args.toolvqa_complete),
        "toolvqa_terminal": _resolved(args.toolvqa_terminal),
        "xlam_abstention": _resolved(args.xlam_abstention),
        "report": _resolved(args.report),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("agent-retention output paths must be unique")
    for path in outputs.values():
        _require_variant_output(path)
    if set(inputs.values()) & set(outputs.values()):
        raise ValueError("agent-retention outputs must not overwrite input artifacts")

    _validate_base_report(
        inputs["base_report"],
        router_train=inputs["router_train"],
        router_full=inputs["router_full"],
        direct_answer=inputs["direct_answer"],
    )
    router_train_rows = _read_jsonl(inputs["router_train"])
    router_full_rows = _read_jsonl(inputs["router_full"])
    router_light_rows, light_quotas = _balanced_router_light(
        router_full_rows,
        limit=args.light_limit,
        video_minimum=args.video_light_min,
        seed=args.seed,
    )
    complete_rows, terminal_rows, terminal_report = _terminal_rows(
        inputs["toolvqa"],
        seed=args.seed,
        max_rows=args.terminal_max_rows,
    )
    abstention_rows, abstention_report = _abstention_rows(
        inputs["xlam"],
        seed=args.seed,
        max_rows=args.abstention_max_rows,
    )

    encoded = {
        outputs["router_train"]: encode_jsonl(router_train_rows),
        outputs["router_full"]: encode_jsonl(router_full_rows),
        outputs["router_light"]: encode_jsonl(router_light_rows),
        outputs["toolvqa_complete"]: encode_jsonl(complete_rows),
        outputs["toolvqa_terminal"]: encode_jsonl(terminal_rows),
        outputs["xlam_abstention"]: encode_jsonl(abstention_rows),
    }
    report = {
        "schema_version": "1.0",
        "variant": VARIANT,
        "seed": args.seed,
        "contract": "c3-datafixed-agentretention-v1",
        "inputs": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for name, path in inputs.items()
        },
        "router": {
            "train_rows": len(router_train_rows),
            "full_rows": len(router_full_rows),
            "light_rows": len(router_light_rows),
            "light_by_modality": light_quotas,
            "video_light_minimum": args.video_light_min,
            "derivation": "byte-equivalent rows from audited datafixed Router; light is deterministically modality-balanced",
        },
        "toolvqa_complete": terminal_report,
        "toolvqa_terminal": {
            **terminal_report,
            "supervision": "loss_scale=0 for earlier assistant/tool calls; loss_scale=1 for final assistant",
        },
        "xlam_abstention": abstention_report,
        "artifacts": {
            name: str(path) for name, path in outputs.items()
        },
        "audit": {
            name: {
                "rows": sum(1 for line in payload.splitlines() if line.strip()),
                "sha256": _sha256_bytes(payload),
            }
            for name, path in outputs.items()
            if name != "report"
            for payload in (encoded[path],)
        },
    }
    report_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    encoded[outputs["report"]] = report_payload
    return encoded, report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.overwrite and args.verify_only:
        raise SystemExit("--overwrite and --verify-only are mutually exclusive")
    try:
        encoded, report = build_outputs(args)
        if args.verify_only:
            mismatches = [
                str(path)
                for path, expected in encoded.items()
                if not path.is_file() or path.read_bytes() != expected
            ]
            if mismatches:
                raise ValueError(
                    f"agent-retention deterministic verification failed: {mismatches}"
                )
            print(
                json.dumps(
                    {
                        "variant": VARIANT,
                        "verification": "passed",
                        "artifacts": len(encoded),
                    },
                    indent=2,
                )
            )
            return
        existing = [path for path in encoded if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"outputs already exist; pass --overwrite: {existing}"
            )
        write_transaction(encoded)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
