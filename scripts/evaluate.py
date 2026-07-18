"""Run one lightweight ToolVista evaluation benchmark or suite."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import urllib.parse
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.io import load_benchmark_registry, load_json, load_suites
from toolvista.evaluation.runner import evaluate_suite, upload_summary_to_wandb


def _non_empty_text(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("value must be non-empty")
    if value != value.strip():
        raise argparse.ArgumentTypeError("value must not have leading or trailing whitespace")
    return value


def _safe_run_id(value: str) -> str:
    value = _non_empty_text(value)
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise argparse.ArgumentTypeError(
            "run ID must be one path-safe component using letters, digits, '.', '_' or '-'"
        )
    return value


def _positive_finite_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if not isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation/benchmarks.yaml", type=_non_empty_text)
    parser.add_argument("--suite-config", default="configs/evaluation/suites.yaml", type=_non_empty_text)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--suite", type=_non_empty_text)
    selection.add_argument("--benchmark", type=_non_empty_text)
    parser.add_argument("--checkpoint", required=True, type=_non_empty_text)
    parser.add_argument("--run-id", required=True, type=_safe_run_id)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--api-base", type=_non_empty_text)
    parser.add_argument("--api-model", type=_non_empty_text)
    parser.add_argument("--infer-backend", default="vllm", type=_non_empty_text)
    parser.add_argument("--eval-url", type=_non_empty_text)
    parser.add_argument("--judge-api-base", type=_non_empty_text)
    parser.add_argument("--judge-model", type=_non_empty_text)
    parser.add_argument("--swift-executable", default="swift", type=_non_empty_text)
    parser.add_argument("--timeout", type=_positive_finite_float, default=120.0)
    parser.add_argument("--stage", type=_non_empty_text)
    parser.add_argument("--algorithm", type=_non_empty_text)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--history-summary", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="toolvista-qwen3vl", type=_non_empty_text)
    parser.add_argument("--wandb-group", type=_non_empty_text)
    return parser.parse_args(argv)


def _metrics_from_summary(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    metrics = load_json(path).get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Summary metrics must be a mapping: {path}")

    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError(f"Summary metric names must be non-empty strings: {path}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Summary metric {name!r} must be numeric: {path}")
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"Summary metric {name!r} must be finite: {path}")
        normalized[name] = numeric
    return normalized


def _redact_url(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return "<redacted-url>"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        netloc = host if parsed.port is None else f"{host}:{parsed.port}"
        return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except ValueError:
        return "<redacted-url>"


def _display_command(argv: Sequence[str]) -> str:
    """Build a reproducible command without persisting URL credentials or tokens."""
    sanitized = [sys.executable, str(Path(__file__).resolve())]
    redact_next = False
    for token in argv:
        if redact_next:
            sanitized.append(_redact_url(token))
            redact_next = False
            continue
        inline_flag = next(
            (
                flag
                for flag in ("--api-base", "--eval-url", "--judge-api-base")
                if token.startswith(f"{flag}=")
            ),
            None,
        )
        if inline_flag is not None:
            sanitized.append(f"{inline_flag}={_redact_url(token.split('=', 1)[1])}")
            continue
        sanitized.append(token)
        if token in {"--api-base", "--eval-url", "--judge-api-base"}:
            redact_next = True
    return shlex.join(sanitized)


def _print_test_handoff(summary: Mapping[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    benchmarks = summary.get("benchmarks", {})
    if not isinstance(benchmarks, Mapping):
        return
    for benchmark_id, benchmark in benchmarks.items():
        if not isinstance(benchmark, Mapping):
            continue
        result_path = benchmark.get("result_path")
        if not isinstance(result_path, str):
            continue
        try:
            result = load_json(result_path)
            test = result["test_summary"]
            if not isinstance(test, Mapping):
                raise ValueError("test_summary must be a mapping")
            lines = (
                f"测试方式: {test['method']}",
                f"测评集及 split: {test['dataset']}",
                f"来源项目: {test['source_project']}",
                f"使用的上游 evaluator 或 ToolVista scorer: {test['evaluator']}",
                f"样本数或限制: {test['sample_count']}",
                f"实际启动命令: {test['command']}",
                f"主要指标: {', '.join(test['main_metrics'])}",
                "结果与 artifact 路径: "
                f"{test['result_path']} | "
                f"{json.dumps(test['artifact_paths'], ensure_ascii=False)}",
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(
                f"Warning: unable to print test handoff for {benchmark_id}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        print(f"\n[{benchmark_id}]")
        print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    registry = load_benchmark_registry(args.config)
    suites = load_suites(args.suite_config)
    if args.benchmark:
        if args.benchmark not in registry:
            raise SystemExit(f"Unknown benchmark: {args.benchmark}")
        selected_ids = [args.benchmark]
    else:
        if args.suite not in suites:
            raise SystemExit(f"Unknown suite: {args.suite}")
        selected_ids = suites[args.suite]

    missing = [benchmark_id for benchmark_id in selected_ids if benchmark_id not in registry]
    if missing:
        raise SystemExit(f"Suite references unknown benchmarks: {', '.join(missing)}")
    specs = [registry[benchmark_id] for benchmark_id in selected_ids]
    if args.score_only and (
        args.benchmark is None
        or len(specs) != 1
        or specs[0].execution not in {"native_xlam", "native_mllm_router", "visual_rft_coco"}
    ):
        raise SystemExit("--score-only requires one native --benchmark (xLAM, MLLM Router, or Visual-RFT COCO).")

    command_argv = list(sys.argv[1:] if argv is None else argv)
    summary = evaluate_suite(
        specs,
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        runs_root=args.runs_root,
        api_base=args.api_base,
        api_model=args.api_model,
        infer_backend=args.infer_backend,
        eval_url=args.eval_url,
        judge_api_base=args.judge_api_base,
        judge_model=args.judge_model,
        timeout=args.timeout,
        score_only=args.score_only,
        resume=args.resume,
        display_command=_display_command(command_argv),
        stage=args.stage,
        algorithm=args.algorithm,
        seed=args.seed,
        baseline_metrics=_metrics_from_summary(args.baseline_summary),
        historical_metrics=[_metrics_from_summary(path) for path in args.history_summary],
        swift_executable=args.swift_executable,
    )
    if args.wandb:
        warning = upload_summary_to_wandb(
            summary,
            project=args.wandb_project,
            group=args.wandb_group or args.stage,
        )
        if warning:
            print(warning, file=sys.stderr)
    _print_test_handoff(summary)
    if summary["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
