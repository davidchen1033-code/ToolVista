"""Convert MLLM-Tool annotations to ms-swift JSONL with reports and media validation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.adapters.mllm_tool import MLLMToolAdapter
from toolvista.exporters.ms_swift import MsSwiftExportError, canonical_to_ms_swift_row
from toolvista.schema.canonical import AdaptResult
from toolvista.utils.paths import ToolVistaPaths


VALID_FILE_KEYS = (*MLLMToolAdapter.default_file_keys, "combined")


def normalize_file_keys(value: Optional[str | Sequence[str]]) -> tuple[str, ...]:
    if value is None:
        candidates = list(MLLMToolAdapter.default_file_keys)
    elif isinstance(value, str):
        candidates = [item.strip() for item in value.split(",") if item.strip()]
    else:
        candidates = [str(item).strip() for item in value if str(item).strip()]

    if not candidates:
        raise ValueError("At least one MLLM-Tool file key is required.")

    normalized: list[str] = []
    for key in candidates:
        if key not in VALID_FILE_KEYS:
            raise ValueError(
                f"Unknown MLLM-Tool file key: {key!r}. Expected one of: {', '.join(VALID_FILE_KEYS)}."
            )
        if key not in normalized:
            normalized.append(key)
    return tuple(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl.yaml", help="ToolVista path configuration YAML.")
    parser.add_argument(
        "--file-keys",
        help="Comma-separated source keys. Defaults to image,video,audio,text; use combined explicitly if needed.",
    )
    parser.add_argument("--output", help="Output JSONL path. Defaults to exports.msswift_sft/mllm_tool_<keys>.jsonl.")
    parser.add_argument("--report", help="Summary JSON path. Defaults beside the output JSONL.")
    parser.add_argument("--errors", help="Per-record warning/error JSONL path. Defaults beside the output JSONL.")
    parser.add_argument("--max-items", type=int, default=None, help="Read at most this many records across all files; 0 means all.")
    parser.add_argument("--include-metadata", action="store_true", help="Keep canonical metadata in each output row.")
    parser.add_argument(
        "--skip-media-validation",
        action="store_true",
        help="Do not require referenced image, video, or audio files to exist on the conversion machine.",
    )
    return parser.parse_args()


def compact_counter(counter: Counter[str]) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def write_export(
    results: Iterable[AdaptResult],
    output_path: Path,
    error_path: Path,
    *,
    include_metadata: bool,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    output_temp: Optional[Path] = None
    error_temp: Optional[Path] = None
    total_raw = converted = skipped = warning_count = error_count = 0
    warning_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()

    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle, tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=error_path.parent,
            prefix=f".{error_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as error_handle:
            output_temp = Path(output_handle.name)
            error_temp = Path(error_handle.name)
            for result in results:
                total_raw += 1
                warning_count += len(result.warnings)
                error_count += len(result.errors)
                warning_counts.update(result.warnings)
                error_counts.update(result.errors)

                if not result.ok or result.sample is None:
                    skipped += 1
                    error_handle.write(json.dumps(result.to_report_dict(), ensure_ascii=False) + "\n")
                    continue

                try:
                    row = canonical_to_ms_swift_row(result.sample, include_metadata=include_metadata)
                except (MsSwiftExportError, TypeError, ValueError) as exc:
                    skipped += 1
                    error_count += 1
                    error_message = f"{type(exc).__name__}: {exc}"
                    error_counts[error_message] += 1
                    error_handle.write(
                        json.dumps(
                            {
                                "ok": False,
                                "sample_id": result.sample.sample_id,
                                "warnings": result.warnings,
                                "errors": [error_message],
                                "raw_ref": result.raw_ref,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                converted += 1
                if result.warnings:
                    error_handle.write(json.dumps(result.to_report_dict(), ensure_ascii=False) + "\n")

        output_temp.replace(output_path)
        error_temp.replace(error_path)
    except Exception:
        if output_temp is not None:
            output_temp.unlink(missing_ok=True)
        if error_temp is not None:
            error_temp.unlink(missing_ok=True)
        raise

    return {
        "total_raw": total_raw,
        "converted": converted,
        "skipped": skipped,
        "warnings": warning_count,
        "errors": error_count,
        "warning_counts": compact_counter(warning_counts),
        "error_counts": compact_counter(error_counts),
    }


def main() -> None:
    args = parse_args()
    paths = ToolVistaPaths.from_yaml(args.config)
    file_keys = normalize_file_keys(args.file_keys)
    key_label = "-".join(file_keys)
    output_path = Path(args.output) if args.output else paths.export_path("msswift_sft") / f"mllm_tool_{key_label}.jsonl"
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")
    error_path = Path(args.errors) if args.errors else output_path.with_suffix(".errors.jsonl")
    max_items = None if args.max_items in (None, 0) else args.max_items

    adapter = MLLMToolAdapter(
        paths,
        file_keys=file_keys,
        max_items=max_items,
        validate_media_files=not args.skip_media_validation,
    )
    stats = write_export(
        adapter.iter_results(),
        output_path,
        error_path,
        include_metadata=args.include_metadata,
    )
    report = {
        "source": adapter.source_name,
        "file_keys": list(file_keys),
        "output": output_path.as_posix(),
        "errors_file": error_path.as_posix(),
        "max_items": max_items,
        "validate_media_files": adapter.validate_media_files,
        "include_metadata": args.include_metadata,
        "stats": stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if stats["converted"] == 0:
        raise SystemExit("No MLLM-Tool samples were exported. Inspect the report and errors JSONL.")


if __name__ == "__main__":
    main()
