"""Validate an exported JSONL with the installed ms-swift preprocessor before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--max-rows", type=int, default=100)
    return parser.parse_args()


def validate_media(row: Dict[str, Any], row_index: int) -> None:
    for key in ("images", "videos", "audios"):
        values = row.get(key, [])
        if isinstance(values, str):
            values = [values]
        for value in values:
            path = value.get("path") if isinstance(value, dict) else value
            if isinstance(path, str) and not Path(path).is_file():
                raise FileNotFoundError(f"row {row_index}: {key} file does not exist: {path}")


def main() -> None:
    args = parse_args()
    try:
        from swift.dataset import MessagesPreprocessor
    except ImportError as exc:
        raise SystemExit(
            "Cannot import ms-swift. Activate the training environment and run "
            "`pip install -e /root/autodl-tmp/ms-swift` first."
        ) from exc

    preprocessor = MessagesPreprocessor()
    checked = 0
    with args.dataset.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if args.max_rows > 0 and checked >= args.max_rows:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            validate_media(row, row_index)
            processed = preprocessor.preprocess(row)
            if processed is None:
                raise ValueError(f"row {row_index}: ms-swift preprocessor rejected the row.")
            checked += 1

    if checked == 0:
        raise SystemExit(f"Dataset contains no rows: {args.dataset}")
    print(f"ms-swift dataset validation passed: rows={checked}, dataset={args.dataset}")


if __name__ == "__main__":
    main()
