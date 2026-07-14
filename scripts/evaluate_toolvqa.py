"""Run ToolVQA's official OpenCompass evaluator through a generated config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.toolvqa import run_toolvqa


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("step", "e2e"), required=True)
    parser.add_argument("--toolvqa-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--tool-url")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--limit", type=_positive_int)
    args = parser.parse_args(argv)
    if args.mode == "e2e" and not args.tool_url:
        parser.error("--tool-url is required when --mode e2e")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_toolvqa(
        mode=args.mode,
        toolvqa_root=args.toolvqa_root,
        data_root=args.data_root,
        media_root=args.media_root,
        output_dir=args.output_dir,
        api_base=args.api_base,
        api_model=args.api_model,
        tool_url=args.tool_url,
        python_executable=args.python,
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
