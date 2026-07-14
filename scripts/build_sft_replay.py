"""Build a deterministic replay mixture from exported ms-swift JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named input JSONL. Repeat for every dataset source.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        required=True,
        metavar="NAME=NUMBER",
        help="Positive target mixture weight for a named source.",
    )
    parser.add_argument(
        "--anchor",
        required=True,
        help="Source whose complete input is retained; other counts are derived from its weight.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_named_values(values: Sequence[str], *, kind: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must use NAME=VALUE syntax: {value!r}")
        name, item = value.split("=", 1)
        if not NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid {kind} name: {name!r}")
        if not item:
            raise ValueError(f"empty value for {kind} {name!r}")
        if name in parsed:
            raise ValueError(f"duplicate {kind} name: {name!r}")
        parsed[name] = item
    return parsed


def parse_weights(values: Sequence[str]) -> Dict[str, float]:
    raw = parse_named_values(values, kind="weight")
    weights: Dict[str, float] = {}
    for name, value in raw.items():
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"weight {name!r} is not a number: {value!r}") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"weight {name!r} must be a positive finite number")
        weights[name] = number
    return weights


def read_jsonl(path: Path, *, source_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"source {source_name!r} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"source {source_name!r}, line {line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"source {source_name!r}, line {line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"source {source_name!r} contains no JSON objects: {path}")
    return rows


def target_counts(
    sources: Mapping[str, Sequence[dict[str, Any]]],
    weights: Mapping[str, float],
    anchor: str,
) -> Dict[str, int]:
    anchor_count = len(sources[anchor])
    anchor_weight = weights[anchor]
    counts: Dict[str, int] = {}
    for name, weight in weights.items():
        if name == anchor:
            counts[name] = anchor_count
            continue
        proportional_count = math.floor(anchor_count * weight / anchor_weight + 0.5)
        counts[name] = max(1, proportional_count)
    return counts


def select_rows(rows: Sequence[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        selected.extend(cycle[: count - len(selected)])
    return selected


def build_mixture(
    sources: Mapping[str, Sequence[dict[str, Any]]],
    weights: Mapping[str, float],
    anchor: str,
    seed: int,
) -> tuple[list[dict[str, Any]], Dict[str, Dict[str, int | float]]]:
    rng = random.Random(seed)
    counts = target_counts(sources, weights, anchor)
    mixed: list[dict[str, Any]] = []
    stats: Dict[str, Dict[str, int | float]] = {}
    for name in sources:
        rows = sources[name]
        count = counts[name]
        chosen = list(rows) if name == anchor else select_rows(rows, count, rng)
        mixed.extend(chosen)
        stats[name] = {
            "weight": weights[name],
            "input_count": len(rows),
            "selected_count": len(chosen),
            "duplicate_count": max(0, len(chosen) - len(rows)),
        }
    rng.shuffle(mixed)
    return mixed, stats


def encode_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for row in rows
    ).encode("utf-8")


def _write_temp(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)


def write_transaction(outputs: Mapping[Path, bytes]) -> None:
    temporary: Dict[Path, Path] = {}
    backups: Dict[Path, Path] = {}
    promoted: list[Path] = []
    token = uuid.uuid4().hex
    try:
        for destination, content in outputs.items():
            temporary[destination] = _write_temp(destination, content)
        for destination in outputs:
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.{token}.bak")
                os.replace(destination, backup)
                backups[destination] = backup
        for destination, temp_path in temporary.items():
            os.replace(temp_path, destination)
            promoted.append(destination)
    except BaseException:
        for destination in promoted:
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(list(backups.items())):
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for temp_path in temporary.values():
            temp_path.unlink(missing_ok=True)

    # Backup cleanup happens only after every new output is committed. A cleanup
    # failure must not turn a successful transaction into destructive rollback.
    for backup in backups.values():
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        source_values = parse_named_values(args.source, kind="source")
        source_paths = {name: Path(value).expanduser().resolve() for name, value in source_values.items()}
        weights = parse_weights(args.weight)
        if set(source_paths) != set(weights):
            missing_weights = sorted(set(source_paths) - set(weights))
            unknown_weights = sorted(set(weights) - set(source_paths))
            raise ValueError(
                f"source/weight names differ: missing_weights={missing_weights}, unknown_weights={unknown_weights}"
            )
        if args.anchor not in source_paths:
            raise ValueError(f"anchor is not a configured source: {args.anchor!r}")

        output = args.output.expanduser().resolve()
        report = (args.report or output.with_suffix(".report.json")).expanduser().resolve()
        if output == report:
            raise ValueError("output and report must be different paths")
        if output in source_paths.values() or report in source_paths.values():
            raise ValueError("output and report must not overwrite an input source")
        existing = [path for path in (output, report) if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"outputs already exist; pass --overwrite: {existing}")

        sources = {name: read_jsonl(path, source_name=name) for name, path in source_paths.items()}
        mixed, stats = build_mixture(sources, weights, args.anchor, args.seed)
        jsonl_content = encode_jsonl(mixed)
        summary = {
            "seed": args.seed,
            "anchor": args.anchor,
            "total_rows": len(mixed),
            "output": str(output),
            "output_sha256": hashlib.sha256(jsonl_content).hexdigest(),
            "sources": {
                name: {"path": str(source_paths[name]), **stats[name]}
                for name in source_paths
            },
        }
        report_content = (json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        write_transaction({output: jsonl_content, report: report_content})
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Replay dataset: {output}")
    print(f"Rows: {len(mixed)}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
