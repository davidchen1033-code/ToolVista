from __future__ import annotations

import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from scripts.check_c0_readiness import audit_c0
from toolvista.evaluation.vlmevalkit_cache import VLMEVALKIT_DATASETS


REPO_ROOT = Path(__file__).resolve().parents[1]


class C0ReadinessTest(unittest.TestCase):
    def make_root(self) -> Path:
        root = REPO_ROOT / "tests" / f"tmp-c0-readiness-{uuid.uuid4().hex}"
        root.mkdir()
        self.addCleanup(shutil.rmtree, root, True)
        return root

    @staticmethod
    def write_jsonl(path: Path, count: int = 1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps({"id": index}) + "\n" for index in range(count)), encoding="utf-8")

    def materialize_required_files(self, root: Path) -> tuple[Path, Path, Path, Path]:
        model = root / "model"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}}) + "\n",
            encoding="utf-8",
        )
        for relative in (
            "data/exports/msswift_sft/xlam_train.jsonl",
            "data/exports/msswift_sft/mllm_tool_router_train.jsonl",
            "data/exports/msswift_sft/toolvqa_train.jsonl",
            "data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl",
            "data/eval/xlam/full.jsonl",
            "data/eval/xlam/light.jsonl",
            "data/eval/mllm_router/full.jsonl",
            "data/eval/mllm_router/light.jsonl",
            "data/exports/msswift_sft/toolvqa_test.jsonl",
        ):
            self.write_jsonl(root / relative)
        for relative in (
            "data/splits/xlam/seed42.json",
            "data/splits/mllm_router/seed42.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        config_dir = root / "configs" / "evaluation"
        config_dir.mkdir(parents=True)
        light_suite = [
            "xlam_function_calling", "toolvqa_step_light", "bfcl_non_live_light",
            "mllm_router", "qwen3vl_general_light", "visual_rft_coco_light",
        ]
        full_suite = [
            "xlam_function_calling_full", "toolvqa_step_full", "bfcl_non_live_full",
            "mllm_router_full", "qwen3vl_general_full", "visual_rft_coco_full",
        ]
        suite_payload = {
            "suites": {
                **{name: light_suite for name in ("c0_light", "c1_light", "c2_light", "c3_light", "c4_light")},
                "c0_full": full_suite,
                "c4_full": full_suite,
            }
        }
        import yaml
        (config_dir / "suites.yaml").write_text(yaml.safe_dump(suite_payload), encoding="utf-8")

        toolvqa = root / "ToolVQA-release"
        toolvqa_data = toolvqa / "opencompass" / "data" / "ToolVQA"
        toolvqa_data.mkdir(parents=True)
        (toolvqa / "opencompass" / "run.py").write_text("# fixture\n", encoding="utf-8")
        toolvqa_media = root / "data/raw/ToolVQA/images"
        toolvqa_media.mkdir(parents=True)
        (toolvqa_media / "one.jpg").write_bytes(b"image")
        (toolvqa_data / "dataset.json").write_text(
            json.dumps({"0": {"files": [{"type": "image", "path": "/author/one.jpg"}]}}) + "\n",
            encoding="utf-8",
        )
        (toolvqa_data / "toolmeta.json").write_text("{}\n", encoding="utf-8")
        for profile in ("light", "full"):
            self.write_jsonl(root / f"data/eval/visual_rft_coco/{profile}.jsonl")
        bfcl_manifest = {"simple_python": ["simple_python_0"]}
        for profile in ("light", "full"):
            path = root / f"data/eval/bfcl/{profile}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bfcl_manifest) + "\n", encoding="utf-8")
        bridge = root / "toolvista/evaluation/visual_rft_coco.py"
        bridge.parent.mkdir(parents=True, exist_ok=True)
        bridge.write_text("# fixture\n", encoding="utf-8")
        report = root / "data/reports/c0/visual_rft_coco_eval.report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}\n", encoding="utf-8")
        bootstrap = root / "scripts/bfcl_toolvista_bootstrap.py"
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.write_text("# fixture\n", encoding="utf-8")
        visual_rft = root / "Visual-RFT"
        (visual_rft / "coco_evaluation").mkdir(parents=True)
        (visual_rft / "coco_evaluation" / "coco_evaluation.py").write_text("# fixture\n", encoding="utf-8")
        coco = root / "COCO2017"
        (coco / "val2017").mkdir(parents=True)
        (coco / "val2017" / "one.jpg").write_bytes(b"image")
        (coco / "annotations").mkdir()
        (coco / "annotations" / "instances_val2017.json").write_text("{}\n", encoding="utf-8")
        return model, toolvqa, visual_rft, coco

    def test_missing_frozen_split_blocks_data_readiness(self) -> None:
        root = self.make_root()
        model, toolvqa, visual_rft, coco = self.materialize_required_files(root)
        (root / "data/eval/xlam/full.jsonl").unlink()

        report = audit_c0(
            root=root,
            model=model,
            toolvqa_root=toolvqa,
            visual_rft_root=visual_rft,
            coco_root=coco,
            swift_available=True,
            bfcl_available=False,
            vllm_available=False,
            cuda_available=False,
            wandb_configured=True,
            minimum_rows={key: 1 for key in (
                "xlam_train", "mllm_router_train", "toolvqa_train", "visual_rft_train",
                "xlam_full", "xlam_light", "mllm_router_full", "mllm_router_light",
                "visual_rft_coco_full", "visual_rft_coco_light", "bfcl_full", "bfcl_light",
                "toolvqa_test", "toolvqa_upstream", "coco_val2017",
            )},
        )

        self.assertFalse(report["summary"]["data_ready"])
        self.assertTrue(any(item["id"] == "xlam_full" and item["status"] == "error" for item in report["checks"]))

    def test_missing_vllm_is_runtime_warning_not_data_failure(self) -> None:
        root = self.make_root()
        model, toolvqa, visual_rft, coco = self.materialize_required_files(root)

        report = audit_c0(
            root=root,
            model=model,
            toolvqa_root=toolvqa,
            visual_rft_root=visual_rft,
            coco_root=coco,
            swift_available=True,
            bfcl_available=False,
            vllm_available=False,
            cuda_available=False,
            wandb_configured=True,
            minimum_rows={key: 1 for key in (
                "xlam_train", "mllm_router_train", "toolvqa_train", "visual_rft_train",
                "xlam_full", "xlam_light", "mllm_router_full", "mllm_router_light",
                "visual_rft_coco_full", "visual_rft_coco_light", "bfcl_full", "bfcl_light",
                "toolvqa_test", "toolvqa_upstream", "coco_val2017",
            )},
        )

        self.assertTrue(report["summary"]["data_ready"])
        self.assertTrue(report["summary"]["c0_core_ready"])
        self.assertFalse(report["summary"]["gpu_runtime_ready"])
        self.assertTrue(any(item["id"] == "vllm" and item["status"] == "warning" for item in report["checks"]))
        self.assertTrue(any(item["id"] == "bfcl" and item["status"] == "warning" for item in report["checks"]))
        self.assertFalse(report["summary"]["optional_benchmarks_ready"])
        toolvqa = next(item for item in report["checks"] if item["id"] == "toolvqa_upstream")
        self.assertEqual(toolvqa["detail"]["dataset_rows"], 1)
        self.assertEqual(toolvqa["detail"]["media"]["missing_media_count"], 0)

    def test_six_benchmark_readiness_distinguishes_data_and_runtime(self) -> None:
        root = self.make_root()
        model, toolvqa, visual_rft, coco = self.materialize_required_files(root)
        minimum_rows = {key: 1 for key in (
            "xlam_train", "mllm_router_train", "toolvqa_train", "visual_rft_train",
            "xlam_full", "xlam_light", "mllm_router_full", "mllm_router_light",
            "visual_rft_coco_full", "visual_rft_coco_light", "bfcl_full", "bfcl_light",
            "toolvqa_test", "toolvqa_upstream", "coco_val2017",
        )}

        data_only = audit_c0(
            root=root, model=model, toolvqa_root=toolvqa, visual_rft_root=visual_rft,
            coco_root=coco, swift_available=True, bfcl_available=True,
            vllm_available=False, cuda_available=True, wandb_configured=True,
            toolvqa_runtime_ready=True, coco_runtime_ready=True, minimum_rows=minimum_rows,
        )
        self.assertTrue(data_only["summary"]["five_round_benchmark_data_ready"])
        self.assertFalse(data_only["summary"]["five_round_benchmark_runtime_ready"])

        runnable = audit_c0(
            root=root, model=model, toolvqa_root=toolvqa, visual_rft_root=visual_rft,
            coco_root=coco, swift_available=True, bfcl_available=True,
            vllm_available=True, cuda_available=True, wandb_configured=True,
            toolvqa_runtime_ready=True, coco_runtime_ready=True, minimum_rows=minimum_rows,
        )
        self.assertTrue(runnable["summary"]["five_round_benchmark_runtime_ready"])

    def test_vlmevalkit_cache_requires_all_four_exact_size_files(self) -> None:
        root = self.make_root()
        model, toolvqa, visual_rft, coco = self.materialize_required_files(root)
        cache = root / "vlmevalkit"
        cache.mkdir()
        miniature = {
            dataset: {
                **metadata,
                "size": 1,
                "md5": hashlib.md5(b"x").hexdigest(),
            }
            for dataset, metadata in VLMEVALKIT_DATASETS.items()
        }
        minimum_rows = {key: 1 for key in (
            "xlam_train", "mllm_router_train", "toolvqa_train", "visual_rft_train",
            "xlam_full", "xlam_light", "mllm_router_full", "mllm_router_light",
            "visual_rft_coco_full", "visual_rft_coco_light", "bfcl_full", "bfcl_light",
            "toolvqa_test", "toolvqa_upstream", "coco_val2017",
        )}

        with patch(
            "toolvista.evaluation.vlmevalkit_cache.VLMEVALKIT_DATASETS",
            miniature,
        ):
            incomplete = audit_c0(
                root=root, model=model, toolvqa_root=toolvqa, visual_rft_root=visual_rft,
                coco_root=coco, swift_available=True, bfcl_available=True,
                vllm_available=True, cuda_available=True, wandb_configured=True,
                toolvqa_runtime_ready=True, coco_runtime_ready=True,
                minimum_rows=minimum_rows, lmu_data=cache,
            )
            cache_check = next(item for item in incomplete["checks"] if item["id"] == "vlmevalkit_cache")
            self.assertEqual(cache_check["status"], "warning")
            self.assertEqual(set(cache_check["detail"]["missing"]), set(miniature))
            self.assertFalse(incomplete["summary"]["five_round_benchmark_data_ready"])

            for metadata in miniature.values():
                (cache / metadata["filename"]).write_bytes(b"x")
            complete = audit_c0(
                root=root, model=model, toolvqa_root=toolvqa, visual_rft_root=visual_rft,
                coco_root=coco, swift_available=True, bfcl_available=True,
                vllm_available=True, cuda_available=True, wandb_configured=True,
                toolvqa_runtime_ready=True, coco_runtime_ready=True,
                minimum_rows=minimum_rows, lmu_data=cache,
            )
            cache_check = next(item for item in complete["checks"] if item["id"] == "vlmevalkit_cache")
            self.assertEqual(cache_check["status"], "ok")
            self.assertTrue(complete["summary"]["five_round_benchmark_data_ready"])


    def test_default_thresholds_reject_smoke_sized_training_files(self) -> None:
        root = self.make_root()
        model, toolvqa, visual_rft, coco = self.materialize_required_files(root)

        report = audit_c0(
            root=root,
            model=model,
            toolvqa_root=toolvqa,
            visual_rft_root=visual_rft,
            coco_root=coco,
            swift_available=True,
            bfcl_available=False,
            vllm_available=False,
            cuda_available=False,
            wandb_configured=True,
        )

        xlam_train = next(item for item in report["checks"] if item["id"] == "xlam_train")
        self.assertEqual(xlam_train["status"], "error")
        self.assertEqual(xlam_train["detail"]["rows"], 1)
        self.assertGreater(xlam_train["detail"]["minimum_rows"], 1)

    def test_missing_referenced_media_blocks_data_readiness(self) -> None:
        root = self.make_root()
        model, toolvqa, visual_rft, coco = self.materialize_required_files(root)
        visual_rft_data = root / "data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl"
        visual_rft_data.write_text(
            json.dumps({"images": [str(root / "missing.jpg")]}) + "\n",
            encoding="utf-8",
        )

        report = audit_c0(
            root=root,
            model=model,
            toolvqa_root=toolvqa,
            visual_rft_root=visual_rft,
            coco_root=coco,
            swift_available=True,
            bfcl_available=False,
            vllm_available=False,
            cuda_available=False,
            wandb_configured=True,
            minimum_rows={key: 1 for key in (
                "xlam_train", "mllm_router_train", "toolvqa_train", "visual_rft_train",
                "xlam_full", "xlam_light", "mllm_router_full", "mllm_router_light",
                "visual_rft_coco_full", "visual_rft_coco_light", "bfcl_full", "bfcl_light",
                "toolvqa_test", "toolvqa_upstream", "coco_val2017",
            )},
        )

        visual_rft_check = next(item for item in report["checks"] if item["id"] == "visual_rft_train")
        self.assertEqual(visual_rft_check["status"], "error")
        self.assertEqual(visual_rft_check["detail"]["missing_media_count"], 1)
        self.assertFalse(report["summary"]["training_data_ready"])
        self.assertTrue(report["summary"]["c0_core_ready"])

    def test_missing_model_shard_blocks_c0_core(self) -> None:
        root = self.make_root()
        model, toolvqa, visual_rft, coco = self.materialize_required_files(root)
        (model / "model-00001-of-00001.safetensors").unlink()

        report = audit_c0(
            root=root,
            model=model,
            toolvqa_root=toolvqa,
            visual_rft_root=visual_rft,
            coco_root=coco,
            swift_available=True,
            bfcl_available=False,
            vllm_available=False,
            cuda_available=False,
            wandb_configured=True,
            minimum_rows={key: 1 for key in (
                "xlam_train", "mllm_router_train", "toolvqa_train", "visual_rft_train",
                "xlam_full", "xlam_light", "mllm_router_full", "mllm_router_light",
                "visual_rft_coco_full", "visual_rft_coco_light", "bfcl_full", "bfcl_light",
                "toolvqa_test", "toolvqa_upstream", "coco_val2017",
            )},
        )

        model_check = next(item for item in report["checks"] if item["id"] == "model")
        self.assertEqual(model_check["status"], "error")
        self.assertEqual(model_check["detail"]["missing_shards"], ["model-00001-of-00001.safetensors"])
        self.assertFalse(report["summary"]["c0_core_ready"])


if __name__ == "__main__":
    unittest.main()
