"""Derive leakage-safe direct-answer rehearsal rows from ToolVQA train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


MEDIA_TAGS = ("<image>", "<video>", "<audio>")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _identity(row: Mapping[str, Any], user_content: str, answer: str) -> str:
    media = [
        str(value)
        for key in ("images", "videos", "audios")
        for value in row.get(key, [])
    ]
    return hashlib.sha256(
        json.dumps(
            [user_content, answer, media],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def direct_answer_row(
    row: Mapping[str, Any],
    *,
    source_row: int,
) -> tuple[dict[str, Any] | None, str | None]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, "messages_not_list"
    users = [
        message
        for message in messages
        if isinstance(message, Mapping)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    assistants = [
        message
        for message in messages
        if isinstance(message, Mapping)
        and message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    if len(users) != 1:
        return None, "requires_one_user"
    if not assistants:
        return None, "missing_final_answer"
    user_content = str(users[0]["content"]).strip()
    answer = str(assistants[-1]["content"]).strip()
    if not any(tag in user_content for tag in MEDIA_TAGS):
        return None, "missing_media_tag"

    output: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "source": "toolvqa_direct_answer_retention",
            "task_type": "direct_answer",
            "source_row": source_row,
            "tools_removed": True,
        },
    }
    for key in ("images", "videos", "audios"):
        values = row.get(key)
        if isinstance(values, list) and values:
            output[key] = list(values)
    return output, None


def build_rows(
    source: Path,
    *,
    seed: int,
    max_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    skipped = Counter()
    input_rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle, 1):
            if not line.strip():
                continue
            input_rows += 1
            value = json.loads(line)
            if not isinstance(value, Mapping):
                skipped["row_not_object"] += 1
                continue
            converted, reason = direct_answer_row(value, source_row=source_row)
            if converted is None:
                skipped[str(reason)] += 1
                continue
            messages = converted["messages"]
            identity = _identity(value, messages[0]["content"], messages[1]["content"])
            rows.setdefault(identity, converted)

    ordered = sorted(
        rows.items(),
        key=lambda item: (
            hashlib.sha256(f"{seed}:{item[0]}".encode("utf-8")).hexdigest(),
            item[0],
        ),
    )
    if max_rows > 0:
        ordered = ordered[:max_rows]
    selected = [row for _, row in ordered]
    if not selected:
        raise ValueError("Direct-answer retention conversion produced no rows.")
    return selected, {
        "source": str(source.resolve()),
        "seed": seed,
        "input_rows": input_rows,
        "eligible_unique_rows": len(rows),
        "selected_rows": len(selected),
        "max_rows": max_rows,
        "skipped": dict(sorted(skipped.items())),
        "contract": "one user with media -> final natural-language assistant answer; no tools",
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = args.report or args.output.with_suffix(".report.json")
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be zero or positive")
    existing = [path for path in (args.output, report) if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs already exist; pass --overwrite: {existing}")
    try:
        rows, summary = build_rows(
            args.source,
            seed=args.seed,
            max_rows=args.max_rows,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    summary["output"] = str(args.output.resolve())
    _atomic_write(
        args.output,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )
    _atomic_write(report, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
