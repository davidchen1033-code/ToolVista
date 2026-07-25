"""Gate C3 for RL by comparing router gain and general retention against C2."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_THRESHOLDS = {
    "mllm_router/top1_accuracy": {"minimum": 0.60},
    "vlmevalkit/realworldqa/acc": {"retention": 0.80},
    "vlmevalkit/mathvision_mini/accuracy": {"retention": 0.70},
    "vlmevalkit/mmmu_dev_val/accuracy": {"retention": 0.80},
    "vlmevalkit/ocrbench/score": {"retention": 0.80},
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--authorized-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--router-minimum", type=float, default=0.60)
    parser.add_argument("--general-retention", type=float, default=0.80)
    parser.add_argument("--mathvision-retention", type=float, default=0.70)
    return parser.parse_args(argv)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Summary root must be an object: {path}")
    return dict(value)


def _metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    benchmarks = summary.get("benchmarks")
    if not isinstance(benchmarks, Mapping):
        raise ValueError("Summary is missing benchmarks.")
    for result in benchmarks.values():
        if not isinstance(result, Mapping):
            continue
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for name, value in metrics.items():
            if (
                isinstance(name, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                output[name] = float(value)
    return output


def _require_router_v2(summary: Mapping[str, Any], *, label: str) -> None:
    benchmarks = summary.get("benchmarks")
    router = benchmarks.get("mllm_router_v2") if isinstance(benchmarks, Mapping) else None
    test_summary = router.get("test_summary") if isinstance(router, Mapping) else None
    dataset = test_summary.get("dataset") if isinstance(test_summary, Mapping) else None
    method = test_summary.get("method") if isinstance(test_summary, Mapping) else None
    signature = f"{dataset or ''} {method or ''}".lower()
    if "router v2" not in signature:
        raise ValueError(
            f"{label} summary does not use the MLLM Router v2 protocol. "
            "Re-evaluate that checkpoint with the current v2 manifest before comparison."
        )


def evaluate_retention(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    router_minimum: float,
    general_retention: float,
    mathvision_retention: float,
) -> dict[str, Any]:
    if candidate.get("status") != "passed":
        raise ValueError(f"C3 candidate summary is not passed: {candidate.get('status')!r}")
    if baseline.get("status") != "passed":
        raise ValueError(f"C2 baseline summary is not passed: {baseline.get('status')!r}")
    candidate_metrics = _metrics(candidate)
    baseline_metrics = _metrics(baseline)
    thresholds = {
        **DEFAULT_THRESHOLDS,
        "mllm_router/top1_accuracy": {"minimum": router_minimum},
        "vlmevalkit/realworldqa/acc": {"retention": general_retention},
        "vlmevalkit/mathvision_mini/accuracy": {
            "retention": mathvision_retention
        },
        "vlmevalkit/mmmu_dev_val/accuracy": {"retention": general_retention},
        "vlmevalkit/ocrbench/score": {"retention": general_retention},
    }
    checks: list[dict[str, Any]] = []
    for metric, rule in thresholds.items():
        if metric not in candidate_metrics:
            raise ValueError(f"C3 summary is missing required metric: {metric}")
        candidate_value = candidate_metrics[metric]
        baseline_value = baseline_metrics.get(metric)
        if "minimum" in rule:
            required = float(rule["minimum"])
            retention = None
        else:
            if baseline_value is None or baseline_value <= 0:
                raise ValueError(
                    f"C2 summary requires a positive baseline for retention metric: {metric}"
                )
            retention = float(rule["retention"])
            required = baseline_value * retention
        checks.append(
            {
                "metric": metric,
                "candidate": candidate_value,
                "baseline": baseline_value,
                "required": required,
                "retention_threshold": retention,
                "passed": candidate_value >= required,
            }
        )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
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
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        candidate = _load(args.candidate_summary)
        baseline = _load(args.baseline_summary)
        candidate_checkpoint = candidate.get("checkpoint")
        if not isinstance(candidate_checkpoint, str) or not candidate_checkpoint.strip():
            raise ValueError("C3 candidate summary has no checkpoint.")
        if os.path.realpath(candidate_checkpoint) != os.path.realpath(
            args.authorized_checkpoint
        ):
            raise ValueError(
                "C3 candidate summary checkpoint does not match the authorized adapter: "
                f"{candidate_checkpoint!r} != {str(args.authorized_checkpoint)!r}"
            )
        _require_router_v2(candidate, label="C3 candidate")
        _require_router_v2(baseline, label="C2 baseline")
        result = evaluate_retention(
            candidate,
            baseline,
            router_minimum=args.router_minimum,
            general_retention=args.general_retention,
            mathvision_retention=args.mathvision_retention,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"C3 retention gate error: {exc}") from exc
    gate = {
        "schema_version": "1.0",
        "stage": "c3-router-v2",
        "passed": result["passed"],
        "authorized_checkpoint": str(args.authorized_checkpoint.resolve()),
        "candidate_summary": str(args.candidate_summary.resolve()),
        "baseline_summary": str(args.baseline_summary.resolve()),
        "checks": result["checks"],
    }
    _atomic_write(args.output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit("C3 retention gate failed; do not start C4 from this checkpoint.")


if __name__ == "__main__":
    main()
