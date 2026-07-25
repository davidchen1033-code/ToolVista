"""Gate the C3 terminal-fix track, including ToolVQA stop behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import check_c3_datafixed_retention as core


VARIANT = "datafixed-agentretention-terminalfix"
ROUTER_BENCHMARK_ID = "mllm_router_datafixed_agentretention"
DEFAULT_OUTPUT_NAME = "toolvista_c3_datafixed_agentretention_terminalfix_retention_gate.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--authorized-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--router-minimum", type=float, default=0.60)
    parser.add_argument("--general-retention", type=float, default=0.80)
    parser.add_argument("--mathvision-retention", type=float, default=0.70)
    parser.add_argument("--tool-retention", type=float, default=0.80)
    parser.add_argument("--visual-retention", type=float, default=0.70)
    parser.add_argument("--toolvqa-answer-minimum", type=float, default=60.0)
    parser.add_argument("--bfcl-irrelevance-minimum", type=float, default=0.20)
    parser.add_argument("--router-video-minimum", type=float, default=0.25)
    parser.add_argument("--router-invalid-maximum", type=float, default=0.05)
    parser.add_argument("--terminal-emission-minimum", type=float, default=80.0)
    parser.add_argument("--terminal-format-error-maximum", type=float, default=20.0)
    parser.add_argument("--conditional-answer-minimum", type=float, default=75.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def terminal_metrics(
    predictions: Mapping[str, Any] | list[Any],
    *,
    answer_accuracy: float,
) -> dict[str, float]:
    rows = list(predictions.values()) if isinstance(predictions, Mapping) else predictions
    if not rows:
        raise ValueError("ToolVQA prediction artifact is empty")
    emitted = 0
    format_errors = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("ToolVQA prediction row must be an object")
        turns = row.get("prediction")
        if not isinstance(turns, list) or not turns:
            raise ValueError("ToolVQA prediction row has no turns")
        final = turns[-1]
        if not isinstance(final, Mapping):
            raise ValueError("ToolVQA final prediction must be an object")
        has_error = "error" in final
        content = final.get("content")
        emitted += int(
            not has_error
            and "tool_calls" not in final
            and isinstance(content, str)
            and bool(content.strip())
        )
        format_errors += int(has_error)
    total = len(rows)
    correct_equivalent = float(answer_accuracy) * total / 100.0
    conditional = min(100.0, correct_equivalent / emitted * 100.0) if emitted else 0.0
    return {
        "toolvqa_sbs/terminal_answer_emission_rate": emitted / total * 100.0,
        "toolvqa_sbs/terminal_format_error_rate": format_errors / total * 100.0,
        "toolvqa_sbs/conditional_answer_accuracy": conditional,
    }


def _prediction_artifact(candidate: Mapping[str, Any]) -> Path:
    results = candidate["summary"].get("benchmarks")
    toolvqa = results.get("toolvqa_step_light") if isinstance(results, Mapping) else None
    artifacts = toolvqa.get("artifacts") if isinstance(toolvqa, Mapping) else None
    upstream_result = artifacts.get("upstream_result") if isinstance(artifacts, Mapping) else None
    if not isinstance(upstream_result, str) or not upstream_result:
        raise ValueError("C3 ToolVQA result does not identify its upstream artifact")
    result_path = Path(upstream_result).expanduser().resolve(strict=True)
    prediction = result_path.parents[2] / "predictions" / result_path.parent.name / result_path.name
    if not prediction.is_file():
        raise FileNotFoundError(f"ToolVQA prediction artifact does not exist: {prediction}")
    return prediction


def _minimum_check(metric: str, candidate: float, required: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "candidate": candidate,
        "required": required,
        "comparison": "minimum",
        "passed": candidate >= required,
    }


def _maximum_check(metric: str, candidate: float, required: float) -> dict[str, Any]:
    return {
        "metric": metric,
        "candidate": candidate,
        "required": required,
        "comparison": "maximum",
        "passed": candidate <= required,
    }


def _validate_thresholds(args: argparse.Namespace) -> None:
    for name in (
        "router_minimum",
        "general_retention",
        "mathvision_retention",
        "tool_retention",
        "visual_retention",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0 or value > 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    for name in (
        "toolvqa_answer_minimum",
        "terminal_emission_minimum",
        "terminal_format_error_maximum",
        "conditional_answer_minimum",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or value > 100:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 100]")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        _validate_thresholds(args)
        authorized = Path(
            core._checkpoint_identity(args.authorized_checkpoint, label="authorized")
        )
        if not authorized.is_dir():
            raise ValueError(f"Authorized checkpoint is not a directory: {authorized}")
        output = (
            args.output.resolve(strict=False)
            if args.output is not None
            else authorized / DEFAULT_OUTPUT_NAME
        )
        if output.parent != authorized or output.name != DEFAULT_OUTPUT_NAME:
            raise ValueError("Terminal-fix gate must be directly inside its checkpoint")
        if output.exists() and not args.overwrite:
            raise ValueError(f"Gate output exists; pass --overwrite: {output}")

        candidate = core._load_run(args.candidate_summary, label="C3 candidate")
        baseline = core._load_run(args.baseline_summary, label="C2 baseline")
        if candidate["checkpoint"] != str(authorized):
            raise ValueError("C3 candidate checkpoint does not match authorized checkpoint")
        stage = str(candidate["manifest"].get("stage", "")).lower()
        if VARIANT not in candidate["run_id"].lower() or VARIANT not in stage:
            raise ValueError("C3 candidate run/stage does not identify terminal-fix")
        core._require_candidate_protocol(
            candidate, router_benchmark_id=ROUTER_BENCHMARK_ID
        )
        core._require_matching_protocols(candidate, baseline)
        retention = core.evaluate_retention(
            candidate["summary"],
            baseline["summary"],
            router_minimum=args.router_minimum,
            general_retention=args.general_retention,
            mathvision_retention=args.mathvision_retention,
            tool_retention=args.tool_retention,
            visual_retention=args.visual_retention,
            toolvqa_answer_minimum=args.toolvqa_answer_minimum,
            bfcl_irrelevance_minimum=args.bfcl_irrelevance_minimum,
            router_video_minimum=args.router_video_minimum,
            router_invalid_maximum=args.router_invalid_maximum,
        )
        prediction_path = _prediction_artifact(candidate)
        prediction_payload = prediction_path.read_bytes()
        predictions = json.loads(prediction_payload)
        answer_accuracy = float(
            candidate["summary"]["metrics"]["toolvqa_sbs/answer_accuracy"]
        )
        terminal = terminal_metrics(
            predictions, answer_accuracy=answer_accuracy
        )
        terminal_checks = [
            _minimum_check(
                "toolvqa_sbs/terminal_answer_emission_rate",
                terminal["toolvqa_sbs/terminal_answer_emission_rate"],
                args.terminal_emission_minimum,
            ),
            _maximum_check(
                "toolvqa_sbs/terminal_format_error_rate",
                terminal["toolvqa_sbs/terminal_format_error_rate"],
                args.terminal_format_error_maximum,
            ),
            _minimum_check(
                "toolvqa_sbs/conditional_answer_accuracy",
                terminal["toolvqa_sbs/conditional_answer_accuracy"],
                args.conditional_answer_minimum,
            ),
        ]
        checks = [*retention["checks"], *terminal_checks]
        gate = {
            "schema_version": core.SCHEMA_VERSION,
            "variant": VARIANT,
            "stage": f"c3-router-{VARIANT}",
            "passed": all(check["passed"] for check in checks),
            "authorized_checkpoint": str(authorized),
            "candidate": core._run_record(candidate),
            "baseline": core._run_record(baseline),
            "toolvqa_prediction": {
                "path": str(prediction_path),
                "sha256": hashlib.sha256(prediction_payload).hexdigest(),
            },
            "terminal_metrics": terminal,
            "thresholds": {
                "router_minimum": args.router_minimum,
                "general_retention": args.general_retention,
                "mathvision_retention": args.mathvision_retention,
                "tool_retention": args.tool_retention,
                "visual_retention": args.visual_retention,
                "toolvqa_answer_minimum": args.toolvqa_answer_minimum,
                "bfcl_irrelevance_minimum": args.bfcl_irrelevance_minimum,
                "router_video_minimum": args.router_video_minimum,
                "router_invalid_maximum": args.router_invalid_maximum,
                "terminal_emission_minimum": args.terminal_emission_minimum,
                "terminal_format_error_maximum": args.terminal_format_error_maximum,
                "conditional_answer_minimum": args.conditional_answer_minimum,
            },
            "router_benchmark_id": ROUTER_BENCHMARK_ID,
            "checks": checks,
        }
        core._atomic_write(output, gate)
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"C3 terminal-fix retention gate error: {exc}") from exc

    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit(
            "C3 terminal-fix retention gate failed; do not start C4 from this checkpoint."
        )


if __name__ == "__main__":
    main()
