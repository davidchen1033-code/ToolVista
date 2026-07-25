"""Build the versioned C3 router contract and direct-answer retention data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_c3_direct_answer_retention import build_rows
from scripts.prepare_training_and_eval_data import (
    _require_outputs,
    _write_json,
    _write_jsonl,
    prepare_mllm_router,
)
from toolvista.router_data import DEFAULT_ROUTER_CATALOG_SIZE
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
        default=Path("data/exports/msswift_sft/mllm_tool_router_v2_train.jsonl"),
    )
    parser.add_argument(
        "--router-full",
        type=Path,
        default=Path("data/eval/mllm_router_v2/full.jsonl"),
    )
    parser.add_argument(
        "--router-light",
        type=Path,
        default=Path("data/eval/mllm_router_v2/light.jsonl"),
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
        default=Path(
            "data/exports/msswift_sft/toolvqa_direct_answer_retention_train.jsonl"
        ),
    )
    parser.add_argument("--direct-report", type=Path)
    parser.add_argument("--direct-max-rows", type=int, default=12000)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/reports/c3/router_v2_data.report.json"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.direct_max_rows <= 0:
        raise SystemExit("--direct-max-rows must be positive")
    paths = ToolVistaPaths.from_yaml(args.config)
    router_split = args.router_split or Path(
        f"data/splits/mllm_router_v2/seed{args.seed}.json"
    )
    direct_report = args.direct_report or args.direct_output.with_suffix(
        ".report.json"
    )
    _require_outputs(
        [args.direct_output, direct_report, args.report],
        overwrite=args.overwrite,
    )

    router = prepare_mllm_router(
        paths,
        train_output=args.router_train,
        full_output=args.router_full,
        light_output=args.router_light,
        split_output=router_split,
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        light_limit=args.light_limit,
        file_keys=tuple(args.file_keys),
        validate_media_files=not args.skip_media_validation,
        overwrite=args.overwrite,
        catalog_size=args.catalog_size,
    )
    direct_rows, direct = build_rows(
        args.direct_source,
        seed=args.seed,
        max_rows=args.direct_max_rows,
    )
    direct["output"] = str(args.direct_output.resolve())
    _write_jsonl(args.direct_output, direct_rows)
    _write_json(direct_report, direct)

    report = {
        "seed": args.seed,
        "router": router,
        "direct_answer_retention": direct,
        "artifacts": {
            "router_train": str(args.router_train),
            "router_full": str(args.router_full),
            "router_light": str(args.router_light),
            "router_split": str(router_split),
            "direct_answer_train": str(args.direct_output),
            "direct_answer_report": str(direct_report),
        },
    }
    _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
