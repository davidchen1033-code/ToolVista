"""Freeze leakage-safe C0 manifests and rebuild the formal SFT inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_eval_data import _last_user_text, _xlam_row, normalize_text
from toolvista.adapters.mllm_tool import MLLMToolAdapter
from toolvista.adapters.xlam import XLAMAdapter
from toolvista.exporters.ms_swift import MsSwiftExportError, canonical_to_ms_swift_row
from toolvista.router_data import (
    DEFAULT_ROUTER_CATALOG_SIZE,
    ROUTER_CONTRACT_VERSION,
    build_router_catalog,
    with_router_system_prompt,
)
from toolvista.schema.canonical import CanonicalSample
from toolvista.utils.paths import ToolVistaPaths


def belongs_to_eval(identity: str, *, seed: int, namespace: str, fraction: float) -> bool:
    if not 0 < fraction < 1:
        raise ValueError("Evaluation fraction must be greater than 0 and less than 1.")
    digest = hashlib.sha256(f"{seed}:{namespace}:{identity}".encode("utf-8")).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return score < fraction


def _selection_key(row: Mapping[str, Any], *, seed: int, namespace: str) -> str:
    sample_id = str(row["sample_id"])
    return hashlib.sha256(f"{seed}:{namespace}:{sample_id}".encode("utf-8")).hexdigest()


def select_light(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
    seed: int,
    namespace: str,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (_selection_key(row, seed=seed, namespace=namespace), row["sample_id"]))
    return ordered if limit <= 0 else ordered[:limit]


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _require_outputs(paths: Iterable[Path], *, overwrite: bool) -> None:
    materialized = list(paths)
    normalized = [str(path.resolve()).casefold() for path in materialized]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Multiple outputs resolve to the same path.")
    if not overwrite:
        existing = [path for path in materialized if path.exists()]
        if existing:
            raise FileExistsError(f"Outputs already exist; pass --overwrite: {existing}")


def _promote(temporary: Mapping[Path, Path]) -> None:
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for final_path in temporary:
            if final_path.exists():
                backup = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.bak")
                os.replace(final_path, backup)
                backups[final_path] = backup
        for final_path, temporary_path in temporary.items():
            os.replace(temporary_path, final_path)
            promoted.append(final_path)
    except BaseException:
        for final_path in reversed(promoted):
            final_path.unlink(missing_ok=True)
        for final_path, backup in reversed(list(backups.items())):
            if backup.exists():
                os.replace(backup, final_path)
        raise
    finally:
        for temporary_path in temporary.values():
            temporary_path.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _xlam_trajectory_id(sample: CanonicalSample) -> str:
    value = sample.metadata.get("trajectory_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    sample_id = sample.sample_id
    head, separator, tail = sample_id.rpartition("---")
    if separator and tail.isdigit() and head:
        return head
    raise ValueError(f"Cannot recover xLAM trajectory ID from sample {sample.sample_id!r}.")


def prepare_xlam(
    paths: ToolVistaPaths,
    *,
    train_output: Path,
    full_output: Path,
    light_output: Path,
    split_output: Path,
    seed: int,
    eval_fraction: float,
    light_limit: int,
    overwrite: bool,
) -> dict[str, Any]:
    outputs = [train_output, full_output, light_output, split_output]
    _require_outputs(outputs, overwrite=overwrite)
    temporary = {path: _temporary_sibling(path) for path in outputs}
    full_rows: list[dict[str, Any]] = []
    train_trajectories: set[str] = set()
    eval_trajectories: set[str] = set()
    stats = Counter()

    try:
        with temporary[train_output].open("w", encoding="utf-8", newline="\n") as train_handle:
            for result in XLAMAdapter(paths).iter_results():
                stats["adapter_results"] += 1
                if not result.ok or result.sample is None:
                    stats["skipped"] += 1
                    continue
                sample = result.sample
                trajectory_id = _xlam_trajectory_id(sample)
                if belongs_to_eval(
                    trajectory_id,
                    seed=seed,
                    namespace="xlam",
                    fraction=eval_fraction,
                ):
                    eval_trajectories.add(trajectory_id)
                    eval_row = _xlam_row(sample)
                    if eval_row is not None:
                        eval_row["trajectory_id"] = trajectory_id
                        eval_row["split"] = "full_holdout"
                        full_rows.append(eval_row)
                        stats["full_rows"] += 1
                    else:
                        stats["heldout_non_eval_targets"] += 1
                    continue

                train_trajectories.add(trajectory_id)
                try:
                    row = canonical_to_ms_swift_row(sample, include_metadata=True)
                except (MsSwiftExportError, TypeError, ValueError):
                    stats["skipped"] += 1
                    continue
                train_handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stats["train_rows"] += 1

        full_rows.sort(key=lambda row: str(row["sample_id"]))
        light_rows = select_light(full_rows, limit=light_limit, seed=seed, namespace="xlam_light")
        overlap = train_trajectories & eval_trajectories
        if overlap:
            raise ValueError(f"xLAM trajectory leakage detected: {len(overlap)} trajectories")
        if not full_rows or not stats["train_rows"]:
            raise ValueError("xLAM split produced an empty train or full manifest.")

        split_manifest = {
            "source": "xLAM ActionStudio",
            "seed": seed,
            "namespace": "xlam",
            "algorithm": "sha256-first-64-bits",
            "eval_fraction": eval_fraction,
            "train_trajectory_count": len(train_trajectories),
            "eval_trajectory_count": len(eval_trajectories),
            "train_rows": stats["train_rows"],
            "full_rows": len(full_rows),
            "light_rows": len(light_rows),
            "skipped": stats["skipped"],
            "heldout_non_eval_targets": stats["heldout_non_eval_targets"],
            "overlap": {"trajectory_count": 0},
        }
        _write_jsonl(temporary[full_output], full_rows)
        _write_jsonl(temporary[light_output], light_rows)
        _write_json(temporary[split_output], split_manifest)
        _promote(temporary)
        return split_manifest
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def _mllm_prompt(sample: CanonicalSample) -> str:
    messages = [
        {"role": message.role, "content": message.content}
        for message in sample.messages[:-1]
    ]
    return normalize_text(_last_user_text(messages))


def _is_mllm_router_sample(sample: CanonicalSample) -> bool:
    return bool(
        sample.messages
        and sample.messages[-1].role == "assistant"
        and all(message.role in {"system", "user"} for message in sample.messages[:-1])
    )


def _mllm_eval_row(
    sample: CanonicalSample,
    *,
    catalog: Sequence[str],
    target_tools: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    if not _is_mllm_router_sample(sample):
        return None
    selected_targets = (
        list(target_tools)
        if target_tools is not None
        else sample.labels.get("target_tools")
    )
    if not isinstance(selected_targets, list) or not selected_targets:
        return None
    normalized_text = _mllm_prompt(sample)
    if not normalized_text:
        return None
    exported = canonical_to_ms_swift_row(sample)
    prompt_messages = with_router_system_prompt(
        exported.get("messages", [])[:-1],
        catalog,
    )
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "source_project": "MLLM-Tool",
        "split": "full_holdout",
        "input_modality": str(sample.labels.get("input_modality", "text")),
        "router_contract": ROUTER_CONTRACT_VERSION,
        "messages": prompt_messages,
        "router_catalog": list(catalog),
        "target_tools": sorted(
            set(str(item).strip() for item in selected_targets if str(item).strip())
        ),
        "normalized_text": normalized_text,
    }
    for key in ("images", "videos", "audios"):
        values = exported.get(key)
        if isinstance(values, list) and values:
            row[key] = list(values)
    return row


def _mllm_eval_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    modality = str(row["input_modality"])
    if modality == "text":
        return (row["normalized_text"], modality)
    media_key = {"image": "images", "video": "videos", "audio": "audios"}.get(modality)
    media = tuple(str(item) for item in row.get(media_key, [])) if media_key else ()
    return (row["normalized_text"], modality, media)


def prepare_mllm_router(
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
    overwrite: bool,
    catalog_size: int = DEFAULT_ROUTER_CATALOG_SIZE,
) -> dict[str, Any]:
    outputs = [train_output, full_output, light_output, split_output]
    _require_outputs(outputs, overwrite=overwrite)
    temporary = {path: _temporary_sibling(path) for path in outputs}
    grouped_eval: dict[tuple[Any, ...], dict[str, Any]] = {}
    train_prompts: set[str] = set()
    eval_prompts: set[str] = set()
    modality_counts = Counter()
    stats = Counter()
    catalog_values: set[str] = set()
    modality_catalog_values: dict[str, set[str]] = {}
    train_samples: list[CanonicalSample] = []
    eval_samples: list[CanonicalSample] = []

    try:
        adapter = MLLMToolAdapter(
            paths,
            file_keys=file_keys,
            validate_media_files=validate_media_files,
        )
        with temporary[train_output].open("w", encoding="utf-8", newline="\n") as train_handle:
            for result in adapter.iter_results():
                stats["adapter_results"] += 1
                if not result.ok or result.sample is None:
                    stats["skipped"] += 1
                    continue
                sample = result.sample
                if not _is_mllm_router_sample(sample):
                    stats["skipped"] += 1
                    continue
                target_tools = sample.labels.get("target_tools", [])
                normalized_targets = {
                    value.strip()
                    for value in target_tools
                    if isinstance(value, str) and value.strip()
                }
                catalog_values.update(normalized_targets)
                modality = str(sample.labels.get("input_modality", "text"))
                modality_catalog_values.setdefault(modality, set()).update(
                    normalized_targets
                )
                normalized_text = _mllm_prompt(sample)
                if not normalized_text:
                    stats["skipped"] += 1
                    continue
                is_eval = belongs_to_eval(
                    normalized_text,
                    seed=seed,
                    namespace="mllm_router",
                    fraction=eval_fraction,
                )
                if is_eval:
                    eval_prompts.add(normalized_text)
                    eval_samples.append(sample)
                    continue

                train_prompts.add(normalized_text)
                train_samples.append(sample)

            if not catalog_values:
                raise ValueError("MLLM-Tool produced an empty router catalog.")
            catalog = sorted(catalog_values)

            grouped_train: dict[tuple[Any, ...], dict[str, Any]] = {}
            for sample in train_samples:
                try:
                    exported = canonical_to_ms_swift_row(sample)
                    base_row: dict[str, Any] = {
                        "normalized_text": _mllm_prompt(sample),
                        "input_modality": str(
                            sample.labels.get("input_modality", "text")
                        ),
                    }
                    for key in ("images", "videos", "audios"):
                        values = exported.get(key)
                        if isinstance(values, list) and values:
                            base_row[key] = list(values)
                    identity = _mllm_eval_identity(base_row)
                except (MsSwiftExportError, TypeError, ValueError):
                    stats["skipped"] += 1
                    continue
                targets = {
                    str(item).strip()
                    for item in sample.labels.get("target_tools", [])
                    if isinstance(item, str) and item.strip()
                }
                if identity not in grouped_train:
                    grouped_train[identity] = {
                        "sample": sample,
                        "targets": targets,
                    }
                else:
                    current = grouped_train[identity]
                    current["targets"].update(targets)
                    if sample.sample_id < current["sample"].sample_id:
                        current["sample"] = sample

            for identity, grouped in sorted(
                grouped_train.items(), key=lambda item: json.dumps(item[0])
            ):
                sample = grouped["sample"]
                all_targets = sorted(grouped["targets"])
                if not all_targets:
                    stats["skipped"] += 1
                    continue
                chosen_target = min(
                    all_targets,
                    key=lambda target: hashlib.sha256(
                        f"{seed}:{identity}:{target}".encode("utf-8")
                    ).hexdigest(),
                )
                try:
                    row = canonical_to_ms_swift_row(sample, include_metadata=True)
                    modality = str(sample.labels.get("input_modality", "text"))
                    candidate_pool = [
                        chosen_target,
                        *sorted(modality_catalog_values[modality] - set(all_targets)),
                    ]
                    sample_catalog = build_router_catalog(
                        [chosen_target],
                        candidate_pool,
                        identity=json.dumps(identity, ensure_ascii=False),
                        seed=seed,
                        max_size=catalog_size,
                    )
                    row["messages"][-1]["content"] = chosen_target
                    row["messages"] = with_router_system_prompt(
                        row["messages"],
                        sample_catalog,
                    )
                    row["router_catalog"] = sample_catalog
                    row["router_contract"] = ROUTER_CONTRACT_VERSION
                    row["metadata"]["router_contract"] = ROUTER_CONTRACT_VERSION
                    row["metadata"]["router_catalog_size"] = len(sample_catalog)
                    row["metadata"]["labels"]["target_tools"] = all_targets
                    row["metadata"]["labels"]["selected_sft_target"] = chosen_target
                    row["metadata"]["labels"].pop("target_tool_calls", None)
                except (MsSwiftExportError, TypeError, ValueError):
                    stats["skipped"] += 1
                    continue
                train_handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stats["train_rows"] += 1
                modality_counts[modality] += 1
            stats["collapsed_duplicate_train_rows"] = len(train_samples) - len(
                grouped_train
            )

        grouped_samples: dict[tuple[Any, ...], dict[str, Any]] = {}
        for sample in eval_samples:
            try:
                exported = canonical_to_ms_swift_row(sample)
                base_row: dict[str, Any] = {
                    "normalized_text": _mllm_prompt(sample),
                    "input_modality": str(
                        sample.labels.get("input_modality", "text")
                    ),
                }
                for key in ("images", "videos", "audios"):
                    values = exported.get(key)
                    if isinstance(values, list) and values:
                        base_row[key] = list(values)
                identity = _mllm_eval_identity(base_row)
            except (MsSwiftExportError, TypeError, ValueError):
                stats["skipped"] += 1
                continue
            targets = {
                str(item).strip()
                for item in sample.labels.get("target_tools", [])
                if isinstance(item, str) and item.strip()
            }
            if identity not in grouped_samples:
                grouped_samples[identity] = {
                    "sample": sample,
                    "targets": targets,
                }
            else:
                current = grouped_samples[identity]
                current["targets"].update(targets)
                if sample.sample_id < current["sample"].sample_id:
                    current["sample"] = sample

        for identity, grouped in grouped_samples.items():
            sample = grouped["sample"]
            targets = sorted(grouped["targets"])
            modality = str(sample.labels.get("input_modality", "text"))
            try:
                sample_catalog = build_router_catalog(
                    targets,
                    modality_catalog_values[modality],
                    identity=json.dumps(identity, ensure_ascii=False),
                    seed=seed,
                    max_size=catalog_size,
                )
                eval_row = _mllm_eval_row(
                    sample,
                    catalog=sample_catalog,
                    target_tools=targets,
                )
            except (MsSwiftExportError, TypeError, ValueError):
                stats["skipped"] += 1
                continue
            if eval_row is not None:
                grouped_eval[identity] = eval_row

        full_rows = sorted(grouped_eval.values(), key=lambda row: str(row["sample_id"]))
        light_rows = select_light(full_rows, limit=light_limit, seed=seed, namespace="mllm_router_light")
        overlap = train_prompts & eval_prompts
        if overlap:
            raise ValueError(f"MLLM-Tool prompt leakage detected: {len(overlap)} normalized prompts")
        if not full_rows or not stats["train_rows"]:
            raise ValueError("MLLM-Tool split produced an empty train or full manifest.")

        split_manifest = {
            "source": "MLLM-Tool",
            "file_keys": list(file_keys),
            "seed": seed,
            "namespace": "mllm_router",
            "algorithm": "sha256-first-64-bits",
            "eval_fraction": eval_fraction,
            "catalog_size": len(catalog),
            "candidate_catalog_size": catalog_size,
            "router_contract": ROUTER_CONTRACT_VERSION,
            "catalog_size_by_modality": {
                modality: len(values)
                for modality, values in sorted(modality_catalog_values.items())
            },
            "train_normalized_prompt_count": len(train_prompts),
            "eval_normalized_prompt_count": len(eval_prompts),
            "train_rows": stats["train_rows"],
            "collapsed_duplicate_train_rows": stats[
                "collapsed_duplicate_train_rows"
            ],
            "train_rows_by_modality": dict(sorted(modality_counts.items())),
            "full_rows": len(full_rows),
            "light_rows": len(light_rows),
            "skipped": stats["skipped"],
            "overlap": {"normalized_prompt_count": 0},
        }
        _write_jsonl(temporary[full_output], full_rows)
        _write_jsonl(temporary[light_output], light_rows)
        _write_json(temporary[split_output], split_manifest)
        _promote(temporary)
        return split_manifest
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xlam-eval-fraction", type=float, default=0.05)
    parser.add_argument("--mllm-eval-fraction", type=float, default=0.20)
    parser.add_argument("--light-limit", type=int, default=200)
    parser.add_argument("--mllm-file-keys", nargs="+", default=["text", "image", "video"])
    parser.add_argument(
        "--mllm-catalog-size",
        type=int,
        default=DEFAULT_ROUTER_CATALOG_SIZE,
    )
    parser.add_argument("--skip-media-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--xlam-train", type=Path, default=Path("data/exports/msswift_sft/xlam_train.jsonl"))
    parser.add_argument("--xlam-full", type=Path, default=Path("data/eval/xlam/full.jsonl"))
    parser.add_argument("--xlam-light", type=Path, default=Path("data/eval/xlam/light.jsonl"))
    parser.add_argument("--xlam-split", type=Path)
    parser.add_argument(
        "--mllm-train",
        type=Path,
        default=Path("data/exports/msswift_sft/mllm_tool_router_v2_train.jsonl"),
    )
    parser.add_argument("--mllm-full", type=Path, default=Path("data/eval/mllm_router_v2/full.jsonl"))
    parser.add_argument("--mllm-light", type=Path, default=Path("data/eval/mllm_router_v2/light.jsonl"))
    parser.add_argument("--mllm-split", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/c0/training_and_eval_data.report.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = ToolVistaPaths.from_yaml(args.config)
    xlam_split = args.xlam_split or Path(f"data/splits/xlam/seed{args.seed}.json")
    mllm_split = args.mllm_split or Path(f"data/splits/mllm_router/seed{args.seed}.json")
    xlam = prepare_xlam(
        paths,
        train_output=args.xlam_train,
        full_output=args.xlam_full,
        light_output=args.xlam_light,
        split_output=xlam_split,
        seed=args.seed,
        eval_fraction=args.xlam_eval_fraction,
        light_limit=args.light_limit,
        overwrite=args.overwrite,
    )
    mllm = prepare_mllm_router(
        paths,
        train_output=args.mllm_train,
        full_output=args.mllm_full,
        light_output=args.mllm_light,
        split_output=mllm_split,
        seed=args.seed,
        eval_fraction=args.mllm_eval_fraction,
        light_limit=args.light_limit,
        file_keys=tuple(args.mllm_file_keys),
        validate_media_files=not args.skip_media_validation,
        overwrite=args.overwrite,
        catalog_size=args.mllm_catalog_size,
    )
    report = {"seed": args.seed, "xlam": xlam, "mllm_router": mllm}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
