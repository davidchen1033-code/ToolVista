"""Freeze deterministic BFCL v4 non-live ID manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)


def _load_ids(path: Path, category: str) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping) or not isinstance(value.get("id"), str):
                raise ValueError(f"Malformed BFCL entry at {path}:{line_number}")
            ids.append(value["id"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"BFCL category {category!r} is empty or contains duplicate IDs.")
    return ids


def build_manifests(
    data_dir: Path,
    *,
    seed: int,
    light_limit: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if light_limit < len(CATEGORIES):
        raise ValueError(f"light limit must be at least {len(CATEGORIES)}")
    full = {
        category: _load_ids(data_dir / f"BFCL_v4_{category}.json", category)
        for category in CATEGORIES
    }
    if light_limit > sum(len(ids) for ids in full.values()):
        raise ValueError("light limit exceeds the BFCL non-live dataset size.")

    base, remainder = divmod(light_limit, len(CATEGORIES))
    light: dict[str, list[str]] = {}
    for index, category in enumerate(CATEGORIES):
        quota = base + (1 if index < remainder else 0)
        if quota > len(full[category]):
            raise ValueError(f"BFCL category {category!r} cannot satisfy light quota {quota}.")
        light[category] = sorted(
            full[category],
            key=lambda test_id: hashlib.sha256(
                f"toolvista-bfcl-v4-light:{seed}:{category}:{test_id}".encode()
            ).hexdigest(),
        )[:quota]
    return full, light


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/venvs/bfcl/lib/python3.10/site-packages/bfcl_eval/data"
        ),
    )
    parser.add_argument("--full-output", type=Path, default=Path("data/eval/bfcl/full.json"))
    parser.add_argument("--light-output", type=Path, default=Path("data/eval/bfcl/light.json"))
    parser.add_argument("--report", type=Path, default=Path("data/reports/c0/bfcl_eval.report.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--light-limit", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = (args.full_output, args.light_output, args.report)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs already exist; pass --overwrite: {existing}")

    full, light = build_manifests(args.data_dir, seed=args.seed, light_limit=args.light_limit)
    report = {
        "source": "Berkeley Function-Calling Leaderboard BFCL v4",
        "suite": "non_live",
        "categories": list(CATEGORIES),
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "full_count": sum(map(len, full.values())),
        "light_count": sum(map(len, light.values())),
        "full_counts": {name: len(ids) for name, ids in full.items()},
        "light_counts": {name: len(ids) for name, ids in light.items()},
        "full_output": str(args.full_output),
        "light_output": str(args.light_output),
    }
    _replace_json(args.full_output, full)
    _replace_json(args.light_output, light)
    _replace_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
