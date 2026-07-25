from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from scripts.check_c3_datafixed_retention import (
    COMMON_BENCHMARKS,
    DATAFIXED_BENCHMARKS,
    DEFAULT_OUTPUT_NAME,
    _atomic_write,
    _load_run,
    _run_record,
    evaluate_retention,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_c3_datafixed_retention.py"


def metrics(*, collapsed: bool = False) -> dict[str, float]:
    return {
        "mllm_router/top1_accuracy": 0.70,
        "vlmevalkit/realworldqa/acc": 0.70,
        "vlmevalkit/mathvision_mini/accuracy": 0.20,
        "vlmevalkit/mmmu_dev_val/accuracy": 0.0 if collapsed else 0.40,
        "vlmevalkit/ocrbench/score": 90.0,
        "xlam/function_name_accuracy": 0.85,
        "xlam/argument_exact_match": 0.72,
        "toolvqa_sbs/tool_accuracy": 0.72,
        "toolvqa_sbs/argument_accuracy": 0.63,
        "bfcl/non_live_overall_accuracy": 0.54,
        "bfcl/irrelevance_detection": 0.45,
        "visual_rft_coco/novel15_map": 0.36,
    }


def baseline_metrics() -> dict[str, float]:
    return {
        "mllm_router/top1_accuracy": 0.10,
        "vlmevalkit/realworldqa/acc": 0.75,
        "vlmevalkit/mathvision_mini/accuracy": 0.24,
        "vlmevalkit/mmmu_dev_val/accuracy": 0.43,
        "vlmevalkit/ocrbench/score": 98.0,
        "xlam/function_name_accuracy": 0.90,
        "xlam/argument_exact_match": 0.80,
        "toolvqa_sbs/tool_accuracy": 0.80,
        "toolvqa_sbs/argument_accuracy": 0.70,
        "bfcl/non_live_overall_accuracy": 0.60,
        "bfcl/irrelevance_detection": 0.50,
        "visual_rft_coco/novel15_map": 0.40,
    }


def spec(benchmark: str) -> dict[str, Any]:
    values: dict[str, tuple[str, str, tuple[str, ...]]] = {
        "xlam_function_calling": (
            "xLAM ActionStudio",
            "xLAM fixed-seed one-step function-calling manifest",
            ("xlam/function_name_accuracy", "xlam/argument_exact_match"),
        ),
        "toolvqa_step_light": (
            "ToolVQA",
            "ToolVQA official OpenCompass GTA benchmark",
            ("toolvqa_sbs/tool_accuracy", "toolvqa_sbs/argument_accuracy"),
        ),
        "bfcl_non_live_light": (
            "Berkeley Function-Calling Leaderboard",
            "BFCL v4 non-live seven-category fixed-seed ID manifest",
            (
                "bfcl/non_live_overall_accuracy",
                "bfcl/irrelevance_detection",
            ),
        ),
        "qwen3vl_general_light": (
            "ms-swift and VLMEvalKit",
            "RealWorldQA, MathVision_MINI, MMMU_DEV_VAL, OCRBench",
            (
                "vlmevalkit/realworldqa/acc",
                "vlmevalkit/mathvision_mini/accuracy",
                "vlmevalkit/mmmu_dev_val/accuracy",
                "vlmevalkit/ocrbench/score",
            ),
        ),
        "visual_rft_coco_light": (
            "Visual-RFT",
            "Visual-RFT COCO open-vocabulary novel-15 fixed-seed query manifest",
            ("visual_rft_coco/novel15_map",),
        ),
        "mllm_router_datafixed": (
            "MLLM-Tool",
            "MLLM-Tool Router v2 datafixed fixed-seed manifest",
            ("mllm_router/top1_accuracy", "mllm_router/macro_f1"),
        ),
        "mllm_router_v2": (
            "MLLM-Tool",
            "MLLM-Tool Router v2 control fixed-seed manifest",
            ("mllm_router/top1_accuracy", "mllm_router/macro_f1"),
        ),
    }
    source, dataset, primary = values[benchmark]
    split = (
        "router_datafixed_fixed_seed_dev"
        if benchmark == "mllm_router_datafixed"
        else "fixed_seed_dev"
    )
    evaluator = (
        "toolvista.evaluation.scorers.score_mllm_router"
        if benchmark.startswith("mllm_router")
        else f"test.{benchmark}"
    )
    return {
        "benchmark": benchmark,
        "source_project": source,
        "dataset": dataset,
        "split": split,
        "evaluator": evaluator,
        "sample_limit": None,
        "primary_metrics": list(primary),
        "metric_directions": {name: "maximize" for name in primary},
    }


def write_run(
    root: Path,
    *,
    name: str,
    stage: str,
    checkpoint: Path,
    metric_values: dict[str, float],
    candidate: bool,
) -> tuple[Path, Path]:
    run_dir = root / name
    run_dir.mkdir()
    run_id = f"{name}-seed42"
    benchmark_ids = (
        sorted(DATAFIXED_BENCHMARKS)
        if candidate
        else sorted(COMMON_BENCHMARKS | {"mllm_router_v2"})
    )
    summary = {
        "schema_version": "1.0",
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "status": "passed",
        "metrics": metric_values,
        "benchmarks": {
            benchmark: {"status": "passed"} for benchmark in benchmark_ids
        },
    }
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "stage": stage,
        "checkpoint": str(checkpoint),
        "seed": 42,
        "benchmarks": benchmark_ids,
        "benchmark_specs": [spec(benchmark) for benchmark in benchmark_ids],
        "inference": {
            "api_base": "http://127.0.0.1:8000/v1",
            "api_model": "c3-datafixed" if candidate else "c2-control",
            "infer_backend": "vllm",
            "eval_url": "http://127.0.0.1:8000/v1",
            "chat_template": None,
            "system_prompt": "from frozen manifest",
            "temperature": 0,
            "top_p": None,
            "top_k": None,
            "max_tokens": 512,
            "image_preprocessing": "upstream default",
            "tool_order": "frozen manifest order",
            "stop_words": [],
            "reasoning_visibility": "upstream default",
        },
        "evaluator": {
            "judge_api_base": "http://127.0.0.1:8011/v1",
            "judge_model": "Qwen3-VL-4B-Instruct",
            "policy": "fixed_across_c0_c4",
        },
    }
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return summary_path, manifest_path


class C3DatafixedRetentionTest(unittest.TestCase):
    def run_gate(
        self,
        root: Path,
        *,
        collapsed: bool = False,
        mutate: Any = None,
        output: Path | None = None,
        overwrite: bool = False,
        stale_gate: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        candidate_checkpoint = root / "c3-datafixed-checkpoint"
        baseline_checkpoint = root / "c2-checkpoint"
        candidate_checkpoint.mkdir()
        baseline_checkpoint.mkdir()
        candidate_summary, candidate_manifest = write_run(
            root,
            name="c3-datafixed-eval",
            stage="c3-router-datafixed-retention",
            checkpoint=candidate_checkpoint,
            metric_values=metrics(collapsed=collapsed),
            candidate=True,
        )
        baseline_summary, baseline_manifest = write_run(
            root,
            name="c2-control-eval",
            stage="c2-toolvqa-replay",
            checkpoint=baseline_checkpoint,
            metric_values=baseline_metrics(),
            candidate=False,
        )
        if mutate is not None:
            mutate(
                candidate_summary,
                candidate_manifest,
                baseline_summary,
                baseline_manifest,
            )
        gate_path = output or candidate_checkpoint / DEFAULT_OUTPUT_NAME
        if stale_gate:
            gate_path.write_text('{"passed": true}\n', encoding="utf-8")
        command = [
            sys.executable,
            str(GATE_SCRIPT),
            "--candidate-summary",
            str(candidate_summary),
            "--baseline-summary",
            str(baseline_summary),
            "--authorized-checkpoint",
            str(candidate_checkpoint),
        ]
        if output is not None:
            command.extend(["--output", str(output)])
        if overwrite:
            command.append("--overwrite")
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return process, gate_path

    def test_gate_accepts_legacy_c2_and_defaults_to_separate_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            process, gate_path = self.run_gate(Path(value))

            self.assertEqual(process.returncode, 0, process.stderr)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["variant"], "datafixed")
            self.assertEqual(gate_path.name, DEFAULT_OUTPUT_NAME)
            self.assertIn("manifest_sha256", gate["candidate"])

    def test_observed_multimodal_collapse_writes_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            process, gate_path = self.run_gate(Path(value), collapsed=True)

            self.assertNotEqual(process.returncode, 0)
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            self.assertFalse(gate["passed"])
            by_metric = {item["metric"]: item for item in gate["checks"]}
            self.assertTrue(by_metric["mllm_router/top1_accuracy"]["passed"])
            self.assertFalse(
                by_metric["vlmevalkit/mmmu_dev_val/accuracy"]["passed"]
            )

    def test_evaluator_uses_top_level_metrics_only(self) -> None:
        candidate = {
            "metrics": metrics(),
            "benchmarks": {
                "fake": {
                    "metrics": {
                        "mllm_router/top1_accuracy": 0.0,
                        "vlmevalkit/mmmu_dev_val/accuracy": 0.0,
                    }
                }
            },
        }
        result = evaluate_retention(
            candidate,
            {"metrics": baseline_metrics()},
            router_minimum=0.60,
            general_retention=0.80,
            mathvision_retention=0.70,
        )
        self.assertTrue(result["passed"])

    def test_strict_agent_retention_checks_cover_known_collapses(self) -> None:
        candidate = metrics()
        baseline = baseline_metrics()
        candidate.update(
            {
                "toolvqa_sbs/answer_accuracy": 10.0,
                "bfcl/irrelevance_detection": 0.45,
                "mllm_router/accuracy_by_modality/video": 0.50,
                "mllm_router/invalid_route_rate": 0.02,
            }
        )
        baseline.update(
            {
                "toolvqa_sbs/answer_accuracy": 12.0,
                "mllm_router/accuracy_by_modality/video": 1.0,
                "mllm_router/invalid_route_rate": 0.10,
            }
        )
        candidate = {"metrics": candidate}
        baseline = {"metrics": baseline}
        result = evaluate_retention(
            candidate,
            baseline,
            router_minimum=0.60,
            general_retention=0.80,
            mathvision_retention=0.70,
            toolvqa_answer_retention=0.80,
            bfcl_irrelevance_minimum=0.20,
            router_video_minimum=0.25,
            router_invalid_maximum=0.05,
        )
        self.assertTrue(result["passed"])
        strict = {
            item["metric"]: item
            for item in result["checks"]
            if item["metric"]
            in {
                "toolvqa_sbs/answer_accuracy",
                "mllm_router/accuracy_by_modality/video",
                "mllm_router/invalid_route_rate",
            }
            or (
                item["metric"] == "bfcl/irrelevance_detection"
                and item.get("required") == 0.20
            )
        }
        self.assertEqual(set(strict), {
            "toolvqa_sbs/answer_accuracy",
            "bfcl/irrelevance_detection",
            "mllm_router/accuracy_by_modality/video",
            "mllm_router/invalid_route_rate",
        })
        candidate["metrics"]["toolvqa_sbs/answer_accuracy"] = 0.0
        candidate["metrics"]["mllm_router/invalid_route_rate"] = 0.10
        failed = evaluate_retention(
            candidate,
            baseline,
            router_minimum=0.60,
            general_retention=0.80,
            mathvision_retention=0.70,
            toolvqa_answer_retention=0.80,
            router_invalid_maximum=0.05,
        )
        self.assertFalse(failed["passed"])

    def test_run_record_hashes_the_validated_json_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            root = Path(value)
            checkpoint = root / "c3-datafixed-checkpoint"
            checkpoint.mkdir()
            summary_path, manifest_path = write_run(
                root,
                name="snapshot-eval",
                stage="c3-router-datafixed-retention",
                checkpoint=checkpoint,
                metric_values=metrics(),
                candidate=True,
            )
            expected_summary_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
            expected_manifest_hash = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            run = _load_run(summary_path, label="snapshot")

            summary_path.write_text("{}\n", encoding="utf-8")
            manifest_path.write_text("{}\n", encoding="utf-8")
            record = _run_record(run)

            self.assertEqual(record["summary_sha256"], expected_summary_hash)
            self.assertEqual(record["manifest_sha256"], expected_manifest_hash)

    def test_summary_manifest_run_id_mismatch_writes_no_gate(self) -> None:
        def mutate(
            _candidate_summary: Path,
            candidate_manifest: Path,
            _baseline_summary: Path,
            _baseline_manifest: Path,
        ) -> None:
            value = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            value["run_id"] = "wrong-run"
            candidate_manifest.write_text(json.dumps(value), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            process, gate_path = self.run_gate(Path(value), mutate=mutate)

            self.assertNotEqual(process.returncode, 0)
            self.assertIn("run_id mismatch", process.stderr)
            self.assertFalse(gate_path.exists())

    def test_non_datafixed_router_protocol_writes_no_gate(self) -> None:
        def mutate(
            _candidate_summary: Path,
            candidate_manifest: Path,
            _baseline_summary: Path,
            _baseline_manifest: Path,
        ) -> None:
            value = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            router = next(
                item
                for item in value["benchmark_specs"]
                if item["benchmark"] == "mllm_router_datafixed"
            )
            router["dataset"] = "MLLM-Tool legacy Router manifest"
            candidate_manifest.write_text(json.dumps(value), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            process, gate_path = self.run_gate(Path(value), mutate=mutate)

            self.assertNotEqual(process.returncode, 0)
            self.assertIn("not datafixed Router v2", process.stderr)
            self.assertFalse(gate_path.exists())

    def test_common_benchmark_protocol_mismatch_writes_no_gate(self) -> None:
        def mutate(
            _candidate_summary: Path,
            _candidate_manifest: Path,
            _baseline_summary: Path,
            baseline_manifest: Path,
        ) -> None:
            value = json.loads(baseline_manifest.read_text(encoding="utf-8"))
            general = next(
                item
                for item in value["benchmark_specs"]
                if item["benchmark"] == "qwen3vl_general_light"
            )
            general["sample_limit"] = 1
            baseline_manifest.write_text(json.dumps(value), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            process, gate_path = self.run_gate(Path(value), mutate=mutate)

            self.assertNotEqual(process.returncode, 0)
            self.assertIn("protocol mismatch", process.stderr)
            self.assertFalse(gate_path.exists())

    def test_visual_rft_protocol_fields_are_frozen(self) -> None:
        cases: dict[str, object] = {
            "dataset": "different Visual-RFT manifest",
            "split": "different_visual_split",
            "sample_limit": 1,
            "evaluator": "different.visual.evaluator",
        }
        for field, replacement in cases.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory(
                    prefix="toolvista-datafixed-visual-"
                ) as value:
                    def mutate(
                        _candidate_summary: Path,
                        _candidate_manifest: Path,
                        _baseline_summary: Path,
                        baseline_manifest: Path,
                    ) -> None:
                        manifest = json.loads(
                            baseline_manifest.read_text(encoding="utf-8")
                        )
                        visual = next(
                            item
                            for item in manifest["benchmark_specs"]
                            if item["benchmark"] == "visual_rft_coco_light"
                        )
                        visual[field] = replacement
                        baseline_manifest.write_text(
                            json.dumps(manifest),
                            encoding="utf-8",
                        )

                    process, gate_path = self.run_gate(
                        Path(value),
                        mutate=mutate,
                    )

                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("visual_rft_coco_light", process.stderr)
                    self.assertIn("protocol mismatch", process.stderr)
                    self.assertFalse(gate_path.exists())

    def test_judge_protocol_fields_are_frozen(self) -> None:
        cases = {
            "judge_api_base": "http://127.0.0.1:8999/v1",
            "judge_model": "different-judge",
            "policy": "different_policy",
        }
        for field, replacement in cases.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory(
                    prefix="toolvista-datafixed-judge-"
                ) as value:
                    def mutate(
                        _candidate_summary: Path,
                        _candidate_manifest: Path,
                        _baseline_summary: Path,
                        baseline_manifest: Path,
                    ) -> None:
                        manifest = json.loads(
                            baseline_manifest.read_text(encoding="utf-8")
                        )
                        manifest["evaluator"][field] = replacement
                        baseline_manifest.write_text(
                            json.dumps(manifest),
                            encoding="utf-8",
                        )

                    process, gate_path = self.run_gate(
                        Path(value),
                        mutate=mutate,
                    )

                    self.assertNotEqual(process.returncode, 0)
                    self.assertIn("evaluator", process.stderr)
                    self.assertFalse(gate_path.exists())

    def test_gate_output_symlink_cannot_overwrite_summary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            root = Path(value)
            output = (
                root
                / "c3-datafixed-checkpoint"
                / "collision_datafixed.json"
            )

            def mutate(
                candidate_summary: Path,
                _candidate_manifest: Path,
                _baseline_summary: Path,
                _baseline_manifest: Path,
            ) -> None:
                output.symlink_to(candidate_summary)

            process, gate_path = self.run_gate(
                root,
                mutate=mutate,
                output=output,
                overwrite=True,
            )

            summary_path = root / "c3-datafixed-eval" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("collides with an evaluation input", process.stderr)
            self.assertEqual(summary["status"], "passed")
            self.assertTrue(gate_path.is_symlink())


    def test_gate_output_must_stay_inside_authorized_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            root = Path(value)
            outside = root / "outside_datafixed_gate.json"

            process, gate_path = self.run_gate(root, output=outside)

            self.assertNotEqual(process.returncode, 0)
            self.assertIn("directly inside the authorized checkpoint", process.stderr)
            self.assertEqual(gate_path, outside)
            self.assertFalse(outside.exists())

    def test_overwrite_removes_stale_gate_before_protocol_validation(self) -> None:
        def mutate(
            _candidate_summary: Path,
            candidate_manifest: Path,
            _baseline_summary: Path,
            _baseline_manifest: Path,
        ) -> None:
            manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
            manifest["run_id"] = "invalid-after-stale-gate"
            candidate_manifest.write_text(json.dumps(manifest), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            process, gate_path = self.run_gate(
                Path(value),
                mutate=mutate,
                overwrite=True,
                stale_gate=True,
            )

            self.assertNotEqual(process.returncode, 0)
            self.assertIn("run_id mismatch", process.stderr)
            self.assertFalse(gate_path.exists())

    def test_atomic_write_failure_removes_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            root = Path(value)
            output = root / "atomic_datafixed_gate.json"

            with mock.patch(
                "scripts.check_c3_datafixed_retention.os.replace",
                side_effect=OSError("injected replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected replace failure"):
                    _atomic_write(output, {"passed": True})

            self.assertFalse(output.exists())
            self.assertEqual(
                list(root.glob(f".{output.name}.*.tmp")),
                [],
            )

    def test_candidate_checkpoint_must_match_authorized_adapter(self) -> None:
        def mutate(
            candidate_summary: Path,
            candidate_manifest: Path,
            _baseline_summary: Path,
            _baseline_manifest: Path,
        ) -> None:
            for path in (candidate_summary, candidate_manifest):
                value = json.loads(path.read_text(encoding="utf-8"))
                value["checkpoint"] = str(path.parent / "wrong-checkpoint")
                path.write_text(json.dumps(value), encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="toolvista-datafixed-") as value:
            process, gate_path = self.run_gate(Path(value), mutate=mutate)

            self.assertNotEqual(process.returncode, 0)
            self.assertIn("does not match the authorized adapter", process.stderr)
            self.assertFalse(gate_path.exists())


if __name__ == "__main__":
    unittest.main()
