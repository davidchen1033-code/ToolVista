"""Convert Visual-RFT data to ms-swift GRPO JSONL with conversion reports."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.adapters.visual_rft import VisualRFTAdapter
from toolvista.exporters.ms_swift_grpo import MsSwiftGRPOExportError, canonical_to_ms_swift_grpo_row
from toolvista.schema.canonical import AdaptResult
from toolvista.utils.paths import ToolVistaPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/autodl.yaml", help="ToolVista path configuration YAML.")
    parser.add_argument("--data-path", help="Dataset-root-relative JSON/JSONL/save_to_disk path override.")
    parser.add_argument("--split", help="Dataset split. Defaults to datasets.visual_rft.default_split.")
    parser.add_argument("--task", help="Fallback task when a source row has no task field.")
    parser.add_argument("--output", help="Output JSONL path. Defaults to exports.msswift_grpo/visual_rft_<split>.jsonl.")
    parser.add_argument("--report", help="Summary JSON path. Defaults beside the output JSONL.")
    parser.add_argument("--errors", help="Per-record warning/error JSONL path. Defaults beside the output JSONL.")
    parser.add_argument("--max-items", type=int, default=None, help="Read at most this many source rows; 0 means all.")
    parser.add_argument("--include-metadata", action="store_true", help="Keep canonical debug metadata in each row.")
    parser.add_argument(
        "--skip-media-validation",
        action="store_true",
        help="Do not require referenced images to exist on the conversion machine.",
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
                    row = canonical_to_ms_swift_grpo_row(
                        result.sample,
                        include_metadata=include_metadata,
                    )
                except (MsSwiftGRPOExportError, TypeError, ValueError) as exc:
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

                output_handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
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
    dataset_config = paths.dataset("visual_rft")
    split = args.split or str(dataset_config.setting("default_split", "train"))
    data_path = args.data_path or str(dataset_config.setting("data_path", "train"))
    configured_task = dataset_config.setting("default_task")
    task = args.task if args.task is not None else configured_task
    output_path = (
        Path(args.output)
        if args.output
        else paths.export_path("msswift_grpo") / f"visual_rft_{split}.jsonl"
    )
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")
    error_path = Path(args.errors) if args.errors else output_path.with_suffix(".errors.jsonl")
    max_items = None if args.max_items in (None, 0) else args.max_items

    adapter = VisualRFTAdapter(
        paths,
        data_path=data_path,
        split=split,
        task=task,
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
        "split": split,
        "task": task,
        "data_path": data_path,
        "output": output_path.as_posix(),
        "errors_file": error_path.as_posix(),
        "validate_media_files": adapter.validate_media_files,
        "max_items": max_items,
        "stats": stats,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if stats["converted"] == 0:
        raise SystemExit("No Visual-RFT samples were exported. Inspect the report and errors JSONL.")


if __name__ == "__main__":
    main()
