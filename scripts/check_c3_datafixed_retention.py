"""Gate C3 datafixed using the runner's real summary and manifest schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_NAME = "toolvista_c3_datafixed_retention_gate.json"
ROUTER_BENCHMARK_ID = "mllm_router_datafixed"
COMMON_BENCHMARKS = frozenset(
    {
        "xlam_function_calling",
        "toolvqa_step_light",
        "bfcl_non_live_light",
        "qwen3vl_general_light",
        "visual_rft_coco_light",
    }
)
DATAFIXED_BENCHMARKS = COMMON_BENCHMARKS | {ROUTER_BENCHMARK_ID}
ROUTER_METRIC = "mllm_router/top1_accuracy"
ROUTER_VIDEO_METRIC = "mllm_router/accuracy_by_modality/video"
ROUTER_INVALID_METRIC = "mllm_router/invalid_route_rate"
TOOLVQA_ANSWER_METRIC = "toolvqa_sbs/answer_accuracy"
BFCL_IRRELEVANCE_METRIC = "bfcl/irrelevance_detection"
GENERAL_METRICS = (
    "vlmevalkit/realworldqa/acc",
    "vlmevalkit/mathvision_mini/accuracy",
    "vlmevalkit/mmmu_dev_val/accuracy",
    "vlmevalkit/ocrbench/score",
)
TOOL_METRICS = (
    "xlam/function_name_accuracy",
    "xlam/argument_exact_match",
    "toolvqa_sbs/tool_accuracy",
    "toolvqa_sbs/argument_accuracy",
    "bfcl/non_live_overall_accuracy",
    "bfcl/irrelevance_detection",
)
VISUAL_RFT_METRIC = "visual_rft_coco/novel15_map"
SPEC_FIELDS = (
    "source_project",
    "dataset",
    "split",
    "evaluator",
    "sample_limit",
    "primary_metrics",
    "metric_directions",
)
INFERENCE_FIELDS = (
    "infer_backend",
    "chat_template",
    "system_prompt",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "image_preprocessing",
    "tool_order",
    "stop_words",
    "reasoning_visibility",
)
EVALUATOR_FIELDS = (
    "judge_api_base",
    "judge_model",
    "policy",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--authorized-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--variant", default="datafixed")
    parser.add_argument("--router-benchmark-id", default=ROUTER_BENCHMARK_ID)
    parser.add_argument("--router-minimum", type=float, default=0.60)
    parser.add_argument("--general-retention", type=float, default=0.80)
    parser.add_argument("--mathvision-retention", type=float, default=0.70)
    parser.add_argument("--tool-retention", type=float, default=0.80)
    parser.add_argument("--visual-retention", type=float, default=0.70)
    parser.add_argument("--toolvqa-answer-minimum", type=float)
    parser.add_argument("--toolvqa-answer-retention", type=float)
    parser.add_argument("--bfcl-irrelevance-minimum", type=float)
    parser.add_argument("--router-video-minimum", type=float)
    parser.add_argument("--router-invalid-maximum", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_json_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object: {path}")
    return dict(value), hashlib.sha256(payload).hexdigest()


def _checkpoint_identity(value: object, *, label: str) -> str:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError(f"{label} checkpoint is missing")
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))


def _benchmark_specs(
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    raw_specs = manifest.get("benchmark_specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(f"{label} manifest is missing benchmark_specs")
    specs: dict[str, dict[str, Any]] = {}
    for index, raw_spec in enumerate(raw_specs):
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"{label} benchmark_specs[{index}] is not an object")
        benchmark = raw_spec.get("benchmark")
        if not isinstance(benchmark, str) or not benchmark.strip():
            raise ValueError(f"{label} benchmark_specs[{index}] has no ID")
        if benchmark in specs:
            raise ValueError(f"{label} has duplicate benchmark spec: {benchmark}")
        specs[benchmark] = dict(raw_spec)
    benchmark_ids = manifest.get("benchmarks")
    if (
        not isinstance(benchmark_ids, list)
        or not all(isinstance(item, str) for item in benchmark_ids)
        or len(benchmark_ids) != len(specs)
        or set(benchmark_ids) != set(specs)
    ):
        raise ValueError(f"{label} manifest benchmarks/specs mismatch")
    return specs


def _load_run(summary_path: Path, *, label: str) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    manifest_path = summary_path.with_name("manifest.json")
    summary, summary_sha256 = _load_json_snapshot(
        summary_path, label=f"{label} summary"
    )
    manifest, manifest_sha256 = _load_json_snapshot(
        manifest_path, label=f"{label} manifest"
    )
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{label} summary schema is not {SCHEMA_VERSION}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{label} manifest schema is not {SCHEMA_VERSION}")
    if summary.get("status") != "passed":
        raise ValueError(
            f"{label} summary status is not passed: {summary.get('status')!r}"
        )

    summary_run_id = summary.get("run_id")
    manifest_run_id = manifest.get("run_id")
    if (
        not isinstance(summary_run_id, str)
        or not summary_run_id.strip()
        or summary_run_id != manifest_run_id
    ):
        raise ValueError(
            f"{label} summary/manifest run_id mismatch: "
            f"{summary_run_id!r} != {manifest_run_id!r}"
        )
    summary_checkpoint = _checkpoint_identity(
        summary.get("checkpoint"),
        label=f"{label} summary",
    )
    manifest_checkpoint = _checkpoint_identity(
        manifest.get("checkpoint"),
        label=f"{label} manifest",
    )
    if summary_checkpoint != manifest_checkpoint:
        raise ValueError(
            f"{label} summary/manifest checkpoint mismatch: "
            f"{summary_checkpoint!r} != {manifest_checkpoint!r}"
        )

    specs = _benchmark_specs(manifest, label=label)
    results = summary.get("benchmarks")
    if not isinstance(results, Mapping) or set(results) != set(specs):
        raise ValueError(f"{label} summary benchmark results/manifest mismatch")
    failed = sorted(
        benchmark
        for benchmark, result in results.items()
        if not isinstance(result, Mapping) or result.get("status") != "passed"
    )
    if failed:
        raise ValueError(f"{label} contains failed benchmarks: {failed}")
    return {
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "summary_sha256": summary_sha256,
        "manifest_sha256": manifest_sha256,
        "summary": summary,
        "manifest": manifest,
        "run_id": summary_run_id,
        "checkpoint": summary_checkpoint,
        "specs": specs,
    }


def _require_stage(
    run: Mapping[str, Any],
    *,
    label: str,
    prefix: str,
    datafixed: bool,
) -> None:
    stage = run["manifest"].get("stage")
    if not isinstance(stage, str) or not stage.lower().startswith(prefix):
        raise ValueError(f"{label} stage must start with {prefix!r}: {stage!r}")
    if datafixed and (
        "datafixed" not in stage.lower()
        or "datafixed" not in run["run_id"].lower()
    ):
        raise ValueError(f"{label} stage/run_id do not identify datafixed")


def _require_candidate_protocol(
    run: Mapping[str, Any], *, router_benchmark_id: str = ROUTER_BENCHMARK_ID
) -> None:
    _require_stage(run, label="C3 candidate", prefix="c3", datafixed=True)
    specs = run["specs"]
    missing = sorted((COMMON_BENCHMARKS | {router_benchmark_id}) - set(specs))
    if missing:
        raise ValueError(f"C3 manifest is missing datafixed benchmarks: {missing}")
    router = specs[router_benchmark_id]
    source = str(router.get("source_project", "")).lower()
    dataset = str(router.get("dataset", "")).lower()
    split = str(router.get("split", "")).lower()
    primary = router.get("primary_metrics")
    if "mllm-tool" not in source and "mllm tool" not in source:
        raise ValueError("C3 datafixed Router source is not MLLM-Tool")
    if "router v2" not in dataset or "datafixed" not in dataset:
        raise ValueError("C3 Router dataset is not datafixed Router v2")
    if "datafixed" not in split:
        raise ValueError("C3 Router split does not identify datafixed")
    if router.get("evaluator") != "toolvista.evaluation.scorers.score_mllm_router":
        raise ValueError("C3 datafixed Router evaluator mismatch")
    if not isinstance(primary, list) or ROUTER_METRIC not in primary:
        raise ValueError(f"C3 Router spec is missing {ROUTER_METRIC}")


def _frozen_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {field: spec.get(field) for field in SPEC_FIELDS}


def _require_matching_protocols(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> None:
    _require_stage(baseline, label="C2 baseline", prefix="c2", datafixed=False)
    candidate_specs = candidate["specs"]
    baseline_specs = baseline["specs"]
    missing = sorted(COMMON_BENCHMARKS - set(baseline_specs))
    if missing:
        raise ValueError(f"C2 baseline is missing common benchmarks: {missing}")
    for benchmark in sorted(COMMON_BENCHMARKS):
        left = _frozen_spec(candidate_specs[benchmark])
        right = _frozen_spec(baseline_specs[benchmark])
        if left != right:
            raise ValueError(
                f"Candidate/baseline protocol mismatch for {benchmark}: "
                f"{left!r} != {right!r}"
            )
    candidate_seed = candidate["manifest"].get("seed")
    baseline_seed = baseline["manifest"].get("seed")
    if candidate_seed != baseline_seed:
        raise ValueError(
            f"Candidate/baseline seed mismatch: "
            f"{candidate_seed!r} != {baseline_seed!r}"
        )
    candidate_inference = candidate["manifest"].get("inference")
    baseline_inference = baseline["manifest"].get("inference")
    if not isinstance(candidate_inference, Mapping) or not isinstance(
        baseline_inference, Mapping
    ):
        raise ValueError("Candidate/baseline manifests are missing inference")
    candidate_missing = [field for field in INFERENCE_FIELDS if field not in candidate_inference]
    baseline_missing = [field for field in INFERENCE_FIELDS if field not in baseline_inference]
    if candidate_missing or baseline_missing:
        raise ValueError(
            "Candidate/baseline inference schema is incomplete: "
            f"candidate={candidate_missing}, baseline={baseline_missing}"
        )
    for field in INFERENCE_FIELDS:
        if candidate_inference.get(field) != baseline_inference.get(field):
            raise ValueError(
                f"Candidate/baseline inference mismatch for {field}: "
                f"{candidate_inference.get(field)!r} != "
                f"{baseline_inference.get(field)!r}"
            )
    candidate_evaluator = candidate["manifest"].get("evaluator")
    baseline_evaluator = baseline["manifest"].get("evaluator")
    if not isinstance(candidate_evaluator, Mapping) or not isinstance(
        baseline_evaluator, Mapping
    ):
        raise ValueError("Candidate/baseline manifests are missing evaluator protocol")
    for label, evaluator in (
        ("candidate", candidate_evaluator),
        ("baseline", baseline_evaluator),
    ):
        missing = [field for field in EVALUATOR_FIELDS if field not in evaluator]
        if missing:
            raise ValueError(
                f"{label} evaluator protocol is incomplete: {missing}"
            )
        for field in EVALUATOR_FIELDS[:2]:
            value = evaluator[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{label} evaluator {field} must be a non-empty string"
                )
        if evaluator["policy"] != "fixed_across_c0_c4":
            raise ValueError(
                f"{label} evaluator policy is not fixed_across_c0_c4"
            )
    for field in EVALUATOR_FIELDS:
        if candidate_evaluator[field] != baseline_evaluator[field]:
            raise ValueError(
                f"Candidate/baseline evaluator mismatch for {field}: "
                f"{candidate_evaluator[field]!r} != {baseline_evaluator[field]!r}"
            )


def _top_level_metrics(
    summary: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, float]:
    raw = summary.get("metrics")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} is missing top-level metrics")
    return {
        name: float(value)
        for name, value in raw.items()
        if isinstance(name, str)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def _threshold_check(
    metric: str,
    *,
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    minimum: float | None = None,
    retention: float | None = None,
) -> dict[str, Any]:
    if metric not in candidate:
        raise ValueError(f"C3 summary is missing required metric: {metric}")
    candidate_value = candidate[metric]
    baseline_value = baseline.get(metric)
    if minimum is not None:
        required = minimum
        observed = None
    else:
        if retention is None:
            raise AssertionError("Retention threshold is missing")
        if baseline_value is None:
            raise ValueError(f"C2 summary is missing required metric: {metric}")
        if baseline_value < 0:
            raise ValueError(f"C2 metric must be non-negative: {metric}")
        required = baseline_value * retention
        observed = (
            candidate_value / baseline_value if baseline_value > 0 else None
        )
    return {
        "metric": metric,
        "candidate": candidate_value,
        "baseline": baseline_value,
        "required": required,
        "retention_threshold": retention,
        "observed_retention": observed,
        "passed": candidate_value >= required,
    }


def _maximum_check(
    metric: str,
    *,
    candidate: Mapping[str, float],
    maximum: float,
) -> dict[str, Any]:
    if metric not in candidate:
        raise ValueError(f"C3 summary is missing required metric: {metric}")
    candidate_value = candidate[metric]
    return {
        "metric": metric,
        "candidate": candidate_value,
        "baseline": None,
        "required": maximum,
        "comparison": "maximum",
        "retention_threshold": None,
        "observed_retention": None,
        "passed": candidate_value <= maximum,
    }


def evaluate_retention(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    router_minimum: float,
    general_retention: float,
    mathvision_retention: float,
    tool_retention: float = 0.80,
    visual_retention: float = 0.70,
    toolvqa_answer_minimum: float | None = None,
    toolvqa_answer_retention: float | None = None,
    bfcl_irrelevance_minimum: float | None = None,
    router_video_minimum: float | None = None,
    router_invalid_maximum: float | None = None,
) -> dict[str, Any]:
    """Read only top-level summary.metrics, matching the actual runner."""
    candidate_metrics = _top_level_metrics(candidate, label="C3 candidate")
    baseline_metrics = _top_level_metrics(baseline, label="C2 baseline")
    checks = [
        _threshold_check(
            ROUTER_METRIC,
            candidate=candidate_metrics,
            baseline=baseline_metrics,
            minimum=router_minimum,
        )
    ]
    for metric in GENERAL_METRICS:
        checks.append(
            _threshold_check(
                metric,
                candidate=candidate_metrics,
                baseline=baseline_metrics,
                retention=(
                    mathvision_retention
                    if metric == "vlmevalkit/mathvision_mini/accuracy"
                    else general_retention
                ),
            )
        )
    for metric in TOOL_METRICS:
        checks.append(
            _threshold_check(
                metric,
                candidate=candidate_metrics,
                baseline=baseline_metrics,
                retention=tool_retention,
            )
        )
    checks.append(
        _threshold_check(
            VISUAL_RFT_METRIC,
            candidate=candidate_metrics,
            baseline=baseline_metrics,
            retention=visual_retention,
        )
    )
    if toolvqa_answer_minimum is not None:
        checks.append(
            _threshold_check(
                TOOLVQA_ANSWER_METRIC,
                candidate=candidate_metrics,
                baseline=baseline_metrics,
                minimum=toolvqa_answer_minimum,
            )
        )
    elif toolvqa_answer_retention is not None:
        checks.append(
            _threshold_check(
                TOOLVQA_ANSWER_METRIC,
                candidate=candidate_metrics,
                baseline=baseline_metrics,
                retention=toolvqa_answer_retention,
            )
        )
    for metric, minimum in (
        (BFCL_IRRELEVANCE_METRIC, bfcl_irrelevance_minimum),
        (ROUTER_VIDEO_METRIC, router_video_minimum),
    ):
        if minimum is not None:
            checks.append(
                _threshold_check(
                    metric,
                    candidate=candidate_metrics,
                    baseline=baseline_metrics,
                    minimum=minimum,
                )
            )
    if router_invalid_maximum is not None:
        checks.append(
            _maximum_check(
                ROUTER_INVALID_METRIC,
                candidate=candidate_metrics,
                maximum=router_invalid_maximum,
            )
        )
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


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
    optional_ranges = {
        "toolvqa_answer_minimum": (0.0, 100.0),
        "bfcl_irrelevance_minimum": (0.0, 1.0),
        "router_video_minimum": (0.0, 1.0),
        "router_invalid_maximum": (0.0, 1.0),
    }
    for name, (lower, upper) in optional_ranges.items():
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(value) or value < lower or value > upper
        ):
            raise ValueError(
                f"--{name.replace('_', '-')} must be in [{lower:g}, {upper:g}]"
            )
    answer_retention = args.toolvqa_answer_retention
    if answer_retention is not None and (
        not math.isfinite(answer_retention)
        or answer_retention <= 0
        or answer_retention > 1
    ):
        raise ValueError("--toolvqa-answer-retention must be in (0, 1]")
    if answer_retention is not None and args.toolvqa_answer_minimum is not None:
        raise ValueError("ToolVQA answer minimum and retention are mutually exclusive")
    for name in ("variant", "router_benchmark_id"):
        value = getattr(args, name)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"--{name.replace('_', '-')} is not a safe identifier")
    if "datafixed" not in args.variant.lower():
        raise ValueError("--variant must identify the datafixed track")


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
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
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _run_record(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "checkpoint": run["checkpoint"],
        "summary": str(run["summary_path"]),
        "summary_sha256": run["summary_sha256"],
        "manifest": str(run["manifest_path"]),
        "manifest_sha256": run["manifest_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        _validate_thresholds(args)
        authorized = Path(
            _checkpoint_identity(args.authorized_checkpoint, label="authorized")
        )
        if not authorized.is_dir():
            raise ValueError(f"Authorized checkpoint is not a directory: {authorized}")
        output = (
            args.output.resolve(strict=False)
            if args.output is not None
            else authorized / DEFAULT_OUTPUT_NAME
        )
        candidate_summary_path = args.candidate_summary.resolve(strict=False)
        baseline_summary_path = args.baseline_summary.resolve(strict=False)
        protected_inputs = {
            candidate_summary_path,
            candidate_summary_path.with_name("manifest.json"),
            baseline_summary_path,
            baseline_summary_path.with_name("manifest.json"),
        }
        if output in protected_inputs:
            raise ValueError(f"Gate output collides with an evaluation input: {output}")
        if "datafixed" not in output.name.lower():
            raise ValueError("Gate output filename must contain 'datafixed'")
        if output.parent != authorized:
            raise ValueError(
                "Gate output must be directly inside the authorized checkpoint: "
                f"{output}"
            )
        if output.exists():
            if not args.overwrite:
                raise ValueError(f"Gate output exists; pass --overwrite: {output}")
            output.unlink()

        candidate = _load_run(args.candidate_summary, label="C3 candidate")
        baseline = _load_run(args.baseline_summary, label="C2 baseline")
        if candidate["checkpoint"] != str(authorized):
            raise ValueError(
                "C3 candidate checkpoint does not match the authorized adapter: "
                f"{candidate['checkpoint']!r} != {str(authorized)!r}"
            )
        if candidate["checkpoint"] == baseline["checkpoint"]:
            raise ValueError("C2 baseline and C3 candidate checkpoints must be different")
        if candidate["run_id"] == baseline["run_id"]:
            raise ValueError("C2 baseline and C3 candidate run IDs must be different")
        candidate_stage = str(candidate["manifest"].get("stage", ""))
        variant = args.variant.lower()
        if (
            variant not in candidate["run_id"].lower()
            or variant not in candidate_stage.lower()
        ):
            raise ValueError(
                "C3 candidate run_id/stage do not identify the requested variant: "
                f"{args.variant!r}"
            )
        _require_candidate_protocol(
            candidate,
            router_benchmark_id=args.router_benchmark_id,
        )
        _require_matching_protocols(candidate, baseline)
        result = evaluate_retention(
            candidate["summary"],
            baseline["summary"],
            router_minimum=args.router_minimum,
            general_retention=args.general_retention,
            mathvision_retention=args.mathvision_retention,
            tool_retention=args.tool_retention,
            visual_retention=args.visual_retention,
            toolvqa_answer_minimum=args.toolvqa_answer_minimum,
            toolvqa_answer_retention=args.toolvqa_answer_retention,
            bfcl_irrelevance_minimum=args.bfcl_irrelevance_minimum,
            router_video_minimum=args.router_video_minimum,
            router_invalid_maximum=args.router_invalid_maximum,
        )
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"C3 datafixed retention gate error: {exc}") from exc

    gate = {
        "schema_version": SCHEMA_VERSION,
        "variant": args.variant,
        "stage": f"c3-router-{args.variant}",
        "passed": result["passed"],
        "authorized_checkpoint": str(authorized),
        "candidate": _run_record(candidate),
        "baseline": _run_record(baseline),
        "thresholds": {
            "router_minimum": args.router_minimum,
            "general_retention": args.general_retention,
            "mathvision_retention": args.mathvision_retention,
            "tool_retention": args.tool_retention,
            "visual_retention": args.visual_retention,
            "toolvqa_answer_minimum": args.toolvqa_answer_minimum,
            "toolvqa_answer_retention": args.toolvqa_answer_retention,
            "bfcl_irrelevance_minimum": args.bfcl_irrelevance_minimum,
            "router_video_minimum": args.router_video_minimum,
            "router_invalid_maximum": args.router_invalid_maximum,
        },
        "router_benchmark_id": args.router_benchmark_id,
        "checks": result["checks"],
    }
    _atomic_write(output, gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit(
            "C3 datafixed retention gate failed; "
            "do not start C4 from this checkpoint."
        )


if __name__ == "__main__":
    main()
