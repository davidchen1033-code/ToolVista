"""Build an isolated, audited C3 Router-v2 datafixed experiment track.

This script deliberately writes to datafixed-specific paths so the legacy and
existing Router-v2 artifacts remain available as control groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_c3_direct_answer_retention import build_rows
from scripts.prepare_training_and_eval_data import (
    _is_mllm_router_sample,
    _mllm_eval_identity,
    _mllm_prompt,
    _require_outputs,
    _temporary_sibling,
    _write_json,
    _write_jsonl,
    belongs_to_eval,
    select_light,
)
from toolvista.adapters.mllm_tool import MLLMToolAdapter
from toolvista.exporters.ms_swift import MsSwiftExportError, canonical_to_ms_swift_row
from toolvista.router_data import (
    DEFAULT_ROUTER_CATALOG_SIZE,
    ROUTER_CONTRACT_VERSION,
    ROUTER_MODE_MARKER,
    build_router_catalog,
    normalize_tool_ids,
    router_system_prompt,
    with_router_system_prompt,
)
from toolvista.schema.canonical import CanonicalSample
from toolvista.utils.paths import ToolVistaPaths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.20)
    parser.add_argument("--light-limit", type=int, default=200)
    parser.add_argument("--catalog-size", type=int, default=DEFAULT_ROUTER_CATALOG_SIZE)
    parser.add_argument("--file-keys", nargs="+", default=["text", "image", "video"])
    parser.add_argument("--skip-media-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--router-train",
        type=Path,
        default=Path("data/exports/msswift_sft/mllm_tool_router_datafixed_train.jsonl"),
    )
    parser.add_argument(
        "--router-full",
        type=Path,
        default=Path("data/eval/mllm_router_datafixed/full.jsonl"),
    )
    parser.add_argument(
        "--router-light",
        type=Path,
        default=Path("data/eval/mllm_router_datafixed/light.jsonl"),
    )
    parser.add_argument("--router-split", type=Path)
    parser.add_argument(
        "--direct-source",
        type=Path,
        default=Path("data/exports/msswift_sft/toolvqa_train.jsonl"),
    )
    parser.add_argument(
        "--direct-output",
        type=Path,
        default=Path("data/exports/msswift_sft/toolvqa_direct_answer_datafixed_train.jsonl"),
    )
    parser.add_argument("--direct-report", type=Path)
    parser.add_argument("--direct-max-rows", type=int, default=12000)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/c3/router_datafixed.report.json"),
    )
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _promote_datafixed(temporary: Mapping[Path, Path]) -> None:
    """Publish a datafixed artifact set while preserving recoverable backups."""
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    succeeded = False
    try:
        for final_path in temporary:
            if final_path.exists():
                backup = final_path.with_name(
                    f".{final_path.name}.{uuid.uuid4().hex}.bak"
                )
                os.replace(final_path, backup)
                backups[final_path] = backup
        for final_path, temporary_path in temporary.items():
            os.replace(temporary_path, final_path)
            promoted.append(final_path)
        succeeded = True
    except BaseException as promotion_error:
        rollback_errors: list[str] = []
        for final_path in reversed(promoted):
            if final_path in backups:
                continue
            try:
                final_path.unlink(missing_ok=True)
            except BaseException as exc:
                rollback_errors.append(
                    f"remove {final_path}: {type(exc).__name__}: {exc}"
                )
        for final_path, backup in reversed(list(backups.items())):
            if not backup.exists():
                continue
            try:
                os.replace(backup, final_path)
            except BaseException as exc:
                rollback_errors.append(
                    f"restore {backup} -> {final_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
        if rollback_errors:
            preserved = [str(path) for path in backups.values() if path.exists()]
            raise RuntimeError(
                "Datafixed promotion failed and rollback was incomplete; "
                f"preserved backups={preserved}; errors={rollback_errors}"
            ) from promotion_error
        raise
    finally:
        for temporary_path in temporary.values():
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        if succeeded:
            for backup in backups.values():
                try:
                    backup.unlink(missing_ok=True)
                except OSError:
                    pass


def _messages(row: Mapping[str, Any], *, path: Path, line_number: int) -> list[Mapping[str, Any]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(item, Mapping) for item in messages):
        raise ValueError(f"Invalid messages: {path}:{line_number}")
    return messages
def _is_datafixed_output(path: Path) -> bool:
    return (
        "datafixed" in path.name.casefold()
        or path.parent.name.casefold() == "mllm_router_datafixed"
    )


def _non_datafixed_outputs(paths: Iterable[Path]) -> list[Path]:
    return [
        path
        for path in paths
        if not _is_datafixed_output(path)
        or not _is_datafixed_output(path.resolve(strict=False))
    ]


def with_datafixed_router_system_prompt(
    messages: Sequence[Mapping[str, Any]],
    catalog: Sequence[Any],
) -> list[dict[str, Any]]:
    """Emit one canonical Router system prompt for the isolated track."""
    non_system_messages = [
        dict(message) for message in messages if message.get("role") != "system"
    ]
    return with_router_system_prompt(non_system_messages, catalog)


def _sample_identity(sample: CanonicalSample) -> tuple[Any, ...]:
    exported = canonical_to_ms_swift_row(sample)
    base_row: dict[str, Any] = {
        "normalized_text": _mllm_prompt(sample),
        "input_modality": str(sample.labels.get("input_modality", "text")),
    }
    for key in ("images", "videos", "audios"):
        values = exported.get(key)
        if isinstance(values, list) and values:
            base_row[key] = list(values)
    return _mllm_eval_identity(base_row)


def _datafixed_eval_row(
    sample: CanonicalSample,
    *,
    catalog: Sequence[str],
    target_tools: Sequence[str],
) -> dict[str, Any]:
    exported = canonical_to_ms_swift_row(sample)
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "source_project": "MLLM-Tool",
        "split": "full_holdout",
        "input_modality": str(sample.labels.get("input_modality", "text")),
        "router_contract": ROUTER_CONTRACT_VERSION,
        "messages": with_datafixed_router_system_prompt(
            exported.get("messages", [])[:-1],
            catalog,
        ),
        "router_catalog": list(catalog),
        "target_tools": list(target_tools),
        "normalized_text": _mllm_prompt(sample),
    }
    for key in ("images", "videos", "audios"):
        values = exported.get(key)
        if isinstance(values, list) and values:
            row[key] = list(values)
    return row


def prepare_mllm_router_datafixed(
    paths: ToolVistaPaths,
    *,
    train_output: Path,
    full_output: Path,
    light_output: Path,
    split_output: Path,
    seed: int,
    eval_fraction: float,
    light_limit: int,
    file_keys: Sequence[str],
    validate_media_files: bool,
    catalog_size: int,
) -> dict[str, Any]:
    """Build the cleaned Router track without invoking the control generator."""
    outputs = [train_output, full_output, light_output, split_output]
    non_datafixed = _non_datafixed_outputs(outputs)
    if non_datafixed:
        raise ValueError(
            f"all Router builder outputs must contain 'datafixed': {non_datafixed}"
        )
    _require_outputs(outputs, overwrite=False)
    temporary = {path: _temporary_sibling(path) for path in outputs}
    stats = Counter()
    modality_counts = Counter()
    modality_catalogs: dict[str, set[str]] = defaultdict(set)
    train_samples: list[CanonicalSample] = []
    eval_samples: list[CanonicalSample] = []
    train_prompts: set[str] = set()
    eval_prompts: set[str] = set()

    try:
        adapter = MLLMToolAdapter(
            paths,
            file_keys=file_keys,
            validate_media_files=validate_media_files,
        )
        for result in adapter.iter_results():
            stats["adapter_results"] += 1
            if not result.ok or result.sample is None:
                stats["skipped"] += 1
                continue
            sample = result.sample
            if not _is_mllm_router_sample(sample):
                stats["skipped"] += 1
                continue
            normalized_text = _mllm_prompt(sample)
            targets = {
                value.strip()
                for value in sample.labels.get("target_tools", [])
                if isinstance(value, str) and value.strip()
            }
            if not normalized_text or not targets:
                stats["skipped"] += 1
                continue
            modality = str(sample.labels.get("input_modality", "text"))
            modality_catalogs[modality].update(targets)
            if belongs_to_eval(
                normalized_text,
                seed=seed,
                # Match the control split so data cleaning is the only variable.
                namespace="mllm_router",
                fraction=eval_fraction,
            ):
                eval_prompts.add(normalized_text)
                eval_samples.append(sample)
            else:
                train_prompts.add(normalized_text)
                train_samples.append(sample)

        if not modality_catalogs:
            raise ValueError("MLLM-Tool produced an empty datafixed router catalog.")

        grouped_train: dict[tuple[Any, ...], dict[str, Any]] = {}
        for sample in train_samples:
            try:
                identity = _sample_identity(sample)
            except (MsSwiftExportError, TypeError, ValueError):
                stats["skipped"] += 1
                continue
            targets = {
                str(item).strip()
                for item in sample.labels.get("target_tools", [])
                if isinstance(item, str) and item.strip()
            }
            if identity not in grouped_train:
                grouped_train[identity] = {"sample": sample, "targets": targets}
            else:
                grouped_train[identity]["targets"].update(targets)
                if sample.sample_id < grouped_train[identity]["sample"].sample_id:
                    grouped_train[identity]["sample"] = sample

        train_rows: list[dict[str, Any]] = []
        for identity, grouped in sorted(
            grouped_train.items(),
            key=lambda item: json.dumps(item[0], ensure_ascii=False),
        ):
            sample = grouped["sample"]
            all_targets = sorted(grouped["targets"])
            if not all_targets:
                stats["skipped"] += 1
                continue
            chosen_target = min(
                all_targets,
                key=lambda target: (
                    hashlib.sha256(
                        f"{seed}:{identity}:{target}".encode("utf-8")
                    ).hexdigest(),
                    target,
                ),
            )
            modality = str(sample.labels.get("input_modality", "text"))
            catalog = build_router_catalog(
                [chosen_target],
                [
                    chosen_target,
                    *sorted(modality_catalogs[modality] - set(all_targets)),
                ],
                identity=json.dumps(identity, ensure_ascii=False),
                seed=seed,
                max_size=catalog_size,
            )
            try:
                row = canonical_to_ms_swift_row(sample, include_metadata=True)
            except (MsSwiftExportError, TypeError, ValueError):
                stats["skipped"] += 1
                continue
            row["messages"][-1]["content"] = chosen_target
            row["messages"] = with_datafixed_router_system_prompt(
                row["messages"], catalog
            )
            row["router_catalog"] = catalog
            row["router_contract"] = ROUTER_CONTRACT_VERSION
            row["metadata"]["router_contract"] = ROUTER_CONTRACT_VERSION
            row["metadata"]["router_catalog_size"] = len(catalog)
            row["metadata"]["labels"]["target_tools"] = all_targets
            row["metadata"]["labels"]["selected_sft_target"] = chosen_target
            row["metadata"]["labels"].pop("target_tool_calls", None)
            train_rows.append(row)
            modality_counts[modality] += 1

        grouped_eval: dict[tuple[Any, ...], dict[str, Any]] = {}
        for sample in eval_samples:
            try:
                identity = _sample_identity(sample)
            except (MsSwiftExportError, TypeError, ValueError):
                stats["skipped"] += 1
                continue
            targets = {
                str(item).strip()
                for item in sample.labels.get("target_tools", [])
                if isinstance(item, str) and item.strip()
            }
            if identity not in grouped_eval:
                grouped_eval[identity] = {"sample": sample, "targets": targets}
            else:
                grouped_eval[identity]["targets"].update(targets)
                if sample.sample_id < grouped_eval[identity]["sample"].sample_id:
                    grouped_eval[identity]["sample"] = sample

        full_rows: list[dict[str, Any]] = []
        for identity, grouped in grouped_eval.items():
            sample = grouped["sample"]
            targets = sorted(grouped["targets"])
            if len(targets) > catalog_size:
                raise ValueError(
                    f"Router target set exceeds catalog size for {sample.sample_id}: "
                    f"{len(targets)} > {catalog_size}"
                )
            modality = str(sample.labels.get("input_modality", "text"))
            catalog = build_router_catalog(
                targets,
                modality_catalogs[modality],
                identity=json.dumps(identity, ensure_ascii=False),
                seed=seed,
                max_size=catalog_size,
            )
            try:
                full_rows.append(
                    _datafixed_eval_row(
                        sample,
                        catalog=catalog,
                        target_tools=targets,
                    )
                )
            except (MsSwiftExportError, TypeError, ValueError):
                stats["skipped"] += 1

        full_rows.sort(key=lambda row: str(row["sample_id"]))
        light_rows = select_light(
            full_rows,
            limit=light_limit,
            seed=seed,
            namespace="mllm_router_light",
        )
        overlap = train_prompts & eval_prompts
        if overlap:
            raise ValueError(
                f"MLLM-Tool prompt leakage detected: {len(overlap)} normalized prompts"
            )
        if not train_rows or not full_rows:
            raise ValueError(
                "MLLM-Tool datafixed split produced an empty train or eval set."
            )

        split_manifest = {
            "source": "MLLM-Tool",
            "variant": "datafixed",
            "file_keys": list(file_keys),
            "seed": seed,
            "namespace": "mllm_router",
            "namespace_reason": "kept equal to control for an apples-to-apples split",
            "algorithm": "sha256-first-64-bits",
            "eval_fraction": eval_fraction,
            "candidate_catalog_size": catalog_size,
            "router_contract": ROUTER_CONTRACT_VERSION,
            "catalog_size_by_modality": {
                modality: len(values)
                for modality, values in sorted(modality_catalogs.items())
            },
            "train_normalized_prompt_count": len(train_prompts),
            "eval_normalized_prompt_count": len(eval_prompts),
            "train_rows": len(train_rows),
            "collapsed_duplicate_train_rows": len(train_samples)
            - len(grouped_train),
            "train_rows_by_modality": dict(sorted(modality_counts.items())),
            "full_rows": len(full_rows),
            "light_rows": len(light_rows),
            "skipped": stats["skipped"],
            "adapter_results": stats["adapter_results"],
            "overlap": {"normalized_prompt_count": 0},
        }
        _write_jsonl(temporary[train_output], train_rows)
        _write_jsonl(temporary[full_output], full_rows)
        _write_jsonl(temporary[light_output], light_rows)
        _write_json(temporary[split_output], split_manifest)
        _promote_datafixed(temporary)
        return split_manifest
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def audit_router_data(path: Path, *, training: bool, max_catalog_size: int) -> dict[str, Any]:
    stats = Counter()
    catalog_sizes = Counter()
    for line_number, row in _iter_jsonl(path):
        messages = _messages(row, path=path, line_number=line_number)
        system_messages = [message for message in messages if message.get("role") == "system"]
        if len(system_messages) != 1 or messages[0].get("role") != "system":
            raise ValueError(f"Router row must have exactly one leading system message: {path}:{line_number}")
        system_content = system_messages[0].get("content")
        if not isinstance(system_content, str) or ROUTER_MODE_MARKER not in system_content:
            raise ValueError(f"Router row is missing the mode marker: {path}:{line_number}")
        if row.get("router_contract") != ROUTER_CONTRACT_VERSION:
            raise ValueError(f"Router contract mismatch: {path}:{line_number}")

        catalog = row.get("router_catalog")
        if (
            not isinstance(catalog, list)
            or not catalog
            or not all(isinstance(item, str) and item.strip() for item in catalog)
            or catalog != normalize_tool_ids(catalog)
            or len(catalog) != len(set(catalog))
            or len(catalog) > max_catalog_size
        ):
            raise ValueError(f"Invalid or oversized router catalog: {path}:{line_number}")
        catalog_set = set(catalog)
        catalog_sizes[len(catalog)] += 1

        if system_content != router_system_prompt(catalog):
            raise ValueError(
                f"Router system prompt/catalog mismatch: {path}:{line_number}"
            )
        if training:
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"Router SFT row must end with assistant: {path}:{line_number}")
            selected = str(messages[-1].get("content", "")).strip()
            labels = row.get("metadata", {}).get("labels", {})
            targets = labels.get("target_tools") if isinstance(labels, Mapping) else None
            if selected not in catalog_set or not isinstance(targets, list) or selected not in targets:
                raise ValueError(f"Selected SFT target is not authorized by row labels/catalog: {path}:{line_number}")
            if labels.get("selected_sft_target") != selected:
                raise ValueError(f"Selected SFT target metadata mismatch: {path}:{line_number}")
        else:
            if any(message.get("role") == "assistant" for message in messages):
                raise ValueError(f"Router evaluation prompt must not contain an assistant answer: {path}:{line_number}")
            targets = row.get("target_tools")
            if not isinstance(targets, list) or not targets or not set(targets) <= catalog_set:
                raise ValueError(f"Evaluation targets are not contained in the row catalog: {path}:{line_number}")
        stats["rows"] += 1

    if not stats["rows"]:
        raise ValueError(f"Router data is empty: {path}")
    return {
        "rows": stats["rows"],
        "catalog_size_distribution": {str(key): value for key, value in sorted(catalog_sizes.items())},
        "sha256": _sha256(path),
    }


def audit_direct_answer_data(path: Path) -> dict[str, Any]:
    stats = Counter()
    answer_lengths: list[int] = []
    answers_by_input: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    for line_number, row in _iter_jsonl(path):
        messages = _messages(row, path=path, line_number=line_number)
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise ValueError(f"Direct-answer row must be user -> assistant: {path}:{line_number}")
        leaked_fields = sorted(
            key
            for key in ("tools", "router_catalog", "router_contract")
            if key in row
        )
        if leaked_fields:
            raise ValueError(
                f"Direct-answer row leaked Router/tool fields: "
                f"{path}:{line_number}: {leaked_fields}"
            )
        if any(ROUTER_MODE_MARKER in str(message.get("content", "")) for message in messages):
            raise ValueError(f"Direct-answer row leaked the router marker: {path}:{line_number}")

        user_content = str(messages[0].get("content", "")).strip()
        answer = str(messages[1].get("content", "")).strip()
        media = tuple(
            str(value)
            for key in ("images", "videos", "audios")
            for value in row.get(key, [])
        )
        if not answer or not media or not any(tag in user_content for tag in ("<image>", "<video>", "<audio>")):
            raise ValueError(f"Direct-answer media/answer contract failed: {path}:{line_number}")
        missing = [value for value in media if not Path(value).is_file()]
        if missing:
            raise ValueError(f"Direct-answer row references missing media: {path}:{line_number}: {missing[:3]}")

        answers_by_input[(user_content, media)].add(answer)
        answer_lengths.append(len(answer.split()))
        stats["rows"] += 1

    conflicts = sum(len(values) > 1 for values in answers_by_input.values())
    if not stats["rows"] or conflicts:
        raise ValueError(f"Direct-answer data is empty or conflicting: rows={stats['rows']}, conflicts={conflicts}")
    answer_lengths.sort()
    return {
        "rows": stats["rows"],
        "unique_inputs": len(answers_by_input),
        "conflicting_inputs": conflicts,
        "answer_words": {
            "min": answer_lengths[0],
            "median": answer_lengths[len(answer_lengths) // 2],
            "p90": answer_lengths[min(int(len(answer_lengths) * 0.9), len(answer_lengths) - 1)],
            "max": answer_lengths[-1],
        },
        "sha256": _sha256(path),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.direct_max_rows <= 0:
        raise SystemExit("--direct-max-rows must be positive")
    if args.catalog_size <= 1:
        raise SystemExit("--catalog-size must be greater than one")

    router_split = args.router_split or Path(f"data/splits/mllm_router_datafixed/seed{args.seed}.json")
    direct_report = args.direct_report or args.direct_output.with_suffix(".report.json")
    final_paths = [
        args.router_train,
        args.router_full,
        args.router_light,
        router_split,
        args.direct_output,
        direct_report,
        args.report,
    ]

    non_datafixed = _non_datafixed_outputs(final_paths)
    if non_datafixed:
        raise SystemExit(
            f"all outputs from this entry point must contain 'datafixed': "
            f"{non_datafixed}"
        )
    if args.direct_source.resolve() in {path.resolve() for path in final_paths}:
        raise SystemExit("--direct-source must not also be an output path")
    _require_outputs(final_paths, overwrite=args.overwrite)
    for path in final_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    staged = {path: _temporary_sibling(path) for path in final_paths}

    try:
        paths = ToolVistaPaths.from_yaml(args.config)
        router = prepare_mllm_router_datafixed(
            paths,
            train_output=staged[args.router_train],
            full_output=staged[args.router_full],
            light_output=staged[args.router_light],
            split_output=staged[router_split],
            seed=args.seed,
            eval_fraction=args.eval_fraction,
            light_limit=args.light_limit,
            file_keys=tuple(args.file_keys),
            validate_media_files=not args.skip_media_validation,
            catalog_size=args.catalog_size,
        )
        direct_rows, direct = build_rows(
            args.direct_source,
            seed=args.seed,
            max_rows=args.direct_max_rows,
        )
        _write_jsonl(staged[args.direct_output], direct_rows)
        direct["output"] = str(args.direct_output.resolve())
        _write_json(staged[direct_report], direct)

        audit = {
            "router_train": audit_router_data(
                staged[args.router_train], training=True, max_catalog_size=args.catalog_size
            ),
            "router_full": audit_router_data(
                staged[args.router_full], training=False, max_catalog_size=args.catalog_size
            ),
            "router_light": audit_router_data(
                staged[args.router_light], training=False, max_catalog_size=args.catalog_size
            ),
            "direct_answer": audit_direct_answer_data(staged[args.direct_output]),
        }
        report = {
            "schema_version": "1.0",
            "variant": "datafixed",
            "seed": args.seed,
            "router": router,
            "direct_answer_retention": direct,
            "audit": audit,
            "artifacts": {
                "router_train": str(args.router_train),
                "router_full": str(args.router_full),
                "router_light": str(args.router_light),
                "router_split": str(router_split),
                "direct_answer_train": str(args.direct_output),
                "direct_answer_report": str(direct_report),
            },
        }
        _write_json(staged[args.report], report)
        _promote_datafixed(staged)
    except BaseException:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
