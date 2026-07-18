#!/usr/bin/env python3
"""Prefetch and verify the four frozen VLMEvalKit TSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.vlmevalkit_cache import (
    VLMEVALKIT_DATASETS,
    audit_vlmevalkit_cache,
    prefetch_vlmevalkit_cache,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/cache/vlmevalkit"),
    )
    parser.add_argument(
        "--base-url",
        default="http://opencompass.openxlab.space/utils/VLMEval",
        help="Public VLMEvalKit object prefix. HTTP avoids the currently expired HTTPS certificate.",
    )
    parser.add_argument("--dataset", action="append", choices=sorted(VLMEVALKIT_DATASETS))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    results = prefetch_vlmevalkit_cache(
        args.cache_dir,
        base_url=args.base_url,
        datasets=args.dataset,
        retries=args.retries,
        timeout=args.timeout,
    )
    report = {
        "cache_dir": args.cache_dir.as_posix(),
        "results": dict(results),
        "audit": audit_vlmevalkit_cache(args.cache_dir),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
