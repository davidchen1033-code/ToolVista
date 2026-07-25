"""Audit C0 data, upstream evaluators, and the current GPU runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolvista.evaluation.io import load_suites
from toolvista.evaluation.vlmevalkit_cache import audit_vlmevalkit_cache


DEFAULT_MINIMUM_ROWS = {
    "xlam_train": 100_000,
    "mllm_router_train": 4_000,
    "c3_direct_answer_train": 10_000,
    "toolvqa_train": 20_000,
    "visual_rft_train": 5_000,
    "xlam_full": 1_000,
    "xlam_light": 100,
    "mllm_router_full": 500,
    "mllm_router_light": 100,
    "visual_rft_coco_full": 2_000,
    "visual_rft_coco_light": 100,
    "bfcl_full": 1_300,
    "bfcl_light": 100,
    "toolvqa_test": 2_500,
    "toolvqa_upstream": 1_500,
    "coco_val2017": 5_000,
}


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _jsonl_stats(path: Path) -> dict[str, Any]:
    if not _nonempty(path):
        return {"rows": None, "valid": False, "error": "missing or empty file", "missing_media": []}
    rows = 0
    media_paths: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not a JSON object")
                rows += 1
                for key in ("images", "videos", "audios"):
                    media = value.get(key, [])
                    if isinstance(media, str):
                        media = [media]
                    if not isinstance(media, list):
                        raise ValueError(f"line {line_number} field {key} is not a list")
                    for item in media:
                        media_path = item.get("path") if isinstance(item, dict) else item
                        if isinstance(media_path, str) and media_path.strip():
                            media_paths.add(media_path.strip())
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as exc:
        return {"rows": rows, "valid": False, "error": str(exc), "missing_media": []}
    missing = sorted(path for path in media_paths if not Path(path).is_file())
    return {
        "rows": rows,
        "valid": not missing,
        "error": None,
        "media_files": len(media_paths),
        "missing_media_count": len(missing),
        "missing_media": missing[:10],
    }


def _json_object_count(path: Path) -> int | None:
    if not _nonempty(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError):
        return None
    return len(value) if isinstance(value, dict) else None


def _json_id_manifest_count(path: Path) -> int | None:
    if not _nonempty(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not value:
            return None
        ids = [item for category_ids in value.values() for item in category_ids]
    except (json.JSONDecodeError, OSError, TypeError, UnicodeError):
        return None
    if not ids or not all(isinstance(item, str) and item.strip() for item in ids):
        return None
    return len(ids) if len(ids) == len(set(ids)) else None


def _five_round_suite_stats(path: Path) -> dict[str, Any]:
    expected_light = [
        "xlam_function_calling",
        "toolvqa_step_light",
        "bfcl_non_live_light",
        "mllm_router_v2",
        "qwen3vl_general_light",
        "visual_rft_coco_light",
    ]
    expected_full = [
        "xlam_function_calling_full",
        "toolvqa_step_full",
        "bfcl_non_live_full",
        "mllm_router_v2_full",
        "qwen3vl_general_full",
        "visual_rft_coco_full",
    ]
    try:
        suites = load_suites(path)
        light_names = ("c0_light", "c1_light", "c2_light", "c3_light", "c4_light")
        light_ready = all(suites[name] == expected_light for name in light_names)
        full_ready = all(suites[name] == expected_full for name in ("c0_full", "c4_full"))
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        return {"valid": False, "error": str(exc)}
    return {
        "valid": light_ready and full_ready,
        "error": None if light_ready and full_ready else "suite contents differ from the frozen contract",
        "light_benchmarks": expected_light,
        "full_benchmarks": expected_full,
    }


def _model_files_ready(model: Path) -> tuple[bool, dict[str, Any]]:
    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    single_file = model / "model.safetensors"
    if not _nonempty(config):
        return False, {"config": str(config), "index": str(index), "missing_shards": []}
    if not _nonempty(index):
        return _nonempty(single_file), {
            "config": str(config),
            "single_file": str(single_file),
            "shard_count": 1 if _nonempty(single_file) else 0,
            "missing_shards": [] if _nonempty(single_file) else [single_file.name],
        }
    try:
        value = json.loads(index.read_text(encoding="utf-8"))
        weight_map = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map is missing or empty")
        shards = sorted({str(filename) for filename in weight_map.values()})
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as exc:
        return False, {"config": str(config), "index": str(index), "error": str(exc), "missing_shards": []}
    missing = [filename for filename in shards if not _nonempty(model / filename)]
    return not missing, {
        "config": str(config),
        "index": str(index),
        "shard_count": len(shards),
        "missing_shards": missing,
    }


def _toolvqa_media_stats(dataset_path: Path, media_root: Path) -> dict[str, Any]:
    if not _nonempty(dataset_path) or not media_root.is_dir():
        return {
            "referenced_media": 0,
            "missing_media_count": 0,
            "ambiguous_media_count": 0,
            "error": "dataset or media directory is missing",
        }
    index: dict[str, int] = {}
    for path in media_root.rglob("*"):
        if path.is_file():
            index[path.name] = index.get(path.name, 0) + 1
    try:
        value = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("dataset root is not an object")
        names = [
            Path(file_item["path"]).name
            for row in value.values()
            if isinstance(row, dict)
            for file_item in row.get("files", [])
            if isinstance(file_item, dict) and isinstance(file_item.get("path"), str)
        ]
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError, KeyError) as exc:
        return {
            "referenced_media": 0,
            "missing_media_count": 0,
            "ambiguous_media_count": 0,
            "error": str(exc),
        }
    missing = sorted({name for name in names if index.get(name, 0) == 0})
    ambiguous = sorted({name for name in names if index.get(name, 0) > 1})
    return {
        "referenced_media": len(names),
        "missing_media_count": len(missing),
        "ambiguous_media_count": len(ambiguous),
        "missing_media": missing[:10],
        "ambiguous_media": ambiguous[:10],
        "error": None,
    }


def _check(checks: list[dict[str, Any]], check_id: str, ok: bool, path: Path | str, *, warning: bool = False, detail: Any = None) -> None:
    checks.append(
        {
            "id": check_id,
            "status": "ok" if ok else ("warning" if warning else "error"),
            "path": str(path),
            "detail": detail,
        }
    )


def audit_c0(
    *,
    root: Path,
    model: Path,
    toolvqa_root: Path,
    visual_rft_root: Path,
    coco_root: Path,
    swift_available: bool,
    bfcl_available: bool,
    vllm_available: bool,
    cuda_available: bool,
    wandb_configured: bool,
    toolvqa_runtime_ready: bool = False,
    toolvqa_python: str | None = None,
    coco_runtime_ready: bool = False,
    minimum_rows: Mapping[str, int] | None = None,
    lmu_data: Path | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    minimums = {**DEFAULT_MINIMUM_ROWS, **dict(minimum_rows or {})}
    data_files = {
        "xlam_train": root / "data/exports/msswift_sft/xlam_train.jsonl",
        "mllm_router_train": root / "data/exports/msswift_sft/mllm_tool_router_v2_train.jsonl",
        "toolvqa_train": root / "data/exports/msswift_sft/toolvqa_train.jsonl",
        "c3_direct_answer_train": root / "data/exports/msswift_sft/toolvqa_direct_answer_retention_train.jsonl",
        "visual_rft_train": root / "data/exports/msswift_grpo/visual_rft_coco_base65_train.jsonl",
        "xlam_full": root / "data/eval/xlam/full.jsonl",
        "xlam_light": root / "data/eval/xlam/light.jsonl",
        "mllm_router_full": root / "data/eval/mllm_router_v2/full.jsonl",
        "mllm_router_light": root / "data/eval/mllm_router_v2/light.jsonl",
        "visual_rft_coco_full": root / "data/eval/visual_rft_coco/full.jsonl",
        "visual_rft_coco_light": root / "data/eval/visual_rft_coco/light.jsonl",
        "toolvqa_test": root / "data/exports/msswift_sft/toolvqa_test.jsonl",
    }
    for check_id, path in data_files.items():
        stats = _jsonl_stats(path)
        count = stats["rows"]
        minimum = minimums[check_id]
        _check(
            checks,
            check_id,
            stats["valid"] and count is not None and count >= minimum,
            path,
            detail={**stats, "minimum_rows": minimum},
        )

    split_files = {
        "xlam_split": root / "data/splits/xlam/seed42.json",
        "mllm_router_split": root / "data/splits/mllm_router_v2/seed42.json",
    }
    for check_id, path in split_files.items():
        _check(checks, check_id, _nonempty(path), path)

    model_ready, model_detail = _model_files_ready(model)
    _check(checks, "model", model_ready, model, detail=model_detail)
    suite_path = root / "configs/evaluation/suites.yaml"
    suite_stats = _five_round_suite_stats(suite_path)
    _check(checks, "c0_full_suite", suite_stats["valid"], suite_path, detail=suite_stats)
    _check(
        checks,
        "five_round_suites",
        suite_stats["valid"],
        suite_path,
        warning=True,
        detail=suite_stats,
    )

    toolvqa_runner = toolvqa_root / "opencompass/run.py"
    toolvqa_data = toolvqa_root / "opencompass/data/ToolVQA"
    toolvqa_dataset = toolvqa_data / "dataset.json"
    toolvqa_toolmeta = toolvqa_data / "toolmeta.json"
    toolvqa_eval_rows = _json_object_count(toolvqa_dataset)
    toolvqa_media = _toolvqa_media_stats(
        toolvqa_dataset,
        root / "data/raw/ToolVQA/images",
    )
    toolvqa_ready = (
        toolvqa_runner.is_file()
        and toolvqa_eval_rows is not None
        and toolvqa_eval_rows >= minimums["toolvqa_upstream"]
        and _nonempty(toolvqa_toolmeta)
        and toolvqa_media["error"] is None
        and toolvqa_media["referenced_media"] >= minimums["toolvqa_upstream"]
        and toolvqa_media["missing_media_count"] == 0
        and toolvqa_media["ambiguous_media_count"] == 0
    )
    _check(
        checks,
        "toolvqa_upstream",
        toolvqa_ready,
        toolvqa_data,
        warning=True,
        detail={
            "run_py": str(toolvqa_runner),
            "dataset": str(toolvqa_dataset),
            "dataset_rows": toolvqa_eval_rows,
            "minimum_rows": minimums["toolvqa_upstream"],
            "toolmeta": str(toolvqa_toolmeta),
            "media": toolvqa_media,
        },
    )
    _check(
        checks,
        "toolvqa_runtime",
        toolvqa_runtime_ready,
        toolvqa_python or os.environ.get("TOOLVQA_PYTHON", sys.executable),
        warning=True,
        detail={
            "required_imports": [
                "lagent.agents.ReAct",
                "opencompass.datasets.gta_bench.GTABenchEvaluator",
                "opencompass.models.OpenAIVisual",
                "opencompass.models.lagent.LagentAgentVisual",
                "opencompass.openicl.icl_inferencer.AgentInferencer",
            ]
        },
    )
    coco_evaluator = visual_rft_root / "coco_evaluation/coco_evaluation.py"
    _check(checks, "visual_rft_coco_evaluator", coco_evaluator.is_file(), coco_evaluator, warning=True)
    visual_rft_bridge = root / "toolvista/evaluation/visual_rft_coco.py"
    _check(
        checks,
        "visual_rft_qwen3vl_bridge",
        visual_rft_bridge.is_file(),
        visual_rft_bridge,
        warning=True,
    )
    visual_rft_query_report = root / "data/reports/c0/visual_rft_coco_eval.report.json"
    _check(
        checks,
        "visual_rft_coco_query_report",
        _nonempty(visual_rft_query_report),
        visual_rft_query_report,
        warning=True,
    )
    coco_annotations = coco_root / "annotations/instances_val2017.json"
    coco_images = coco_root / "val2017"
    image_count = sum(1 for path in coco_images.iterdir() if path.is_file()) if coco_images.is_dir() else 0
    _check(
        checks,
        "coco_val2017",
        _nonempty(coco_annotations) and image_count >= minimums["coco_val2017"],
        coco_root,
        warning=True,
        detail={
            "images": image_count,
            "minimum_images": minimums["coco_val2017"],
            "annotations": str(coco_annotations),
        },
    )

    bfcl_manifests = {
        "bfcl_full": root / "data/eval/bfcl/full.json",
        "bfcl_light": root / "data/eval/bfcl/light.json",
    }
    for check_id, path in bfcl_manifests.items():
        count = _json_id_manifest_count(path)
        _check(
            checks,
            check_id,
            count is not None and count >= minimums[check_id],
            path,
            warning=True,
            detail={"rows": count, "minimum_rows": minimums[check_id]},
        )
    bfcl_bootstrap = root / "scripts/bfcl_toolvista_bootstrap.py"
    _check(checks, "bfcl_bootstrap", bfcl_bootstrap.is_file(), bfcl_bootstrap, warning=True)

    _check(checks, "swift", swift_available, "swift", warning=True)
    _check(checks, "bfcl", bfcl_available, "BFCL_PYTHON import bfcl_eval", warning=True)
    _check(
        checks,
        "visual_rft_coco_runtime",
        coco_runtime_ready,
        sys.executable,
        warning=True,
        detail={"required_import": "pycocotools"},
    )

    _check(checks, "vllm", vllm_available, "python import vllm", warning=True)
    _check(checks, "cuda", cuda_available, "torch.cuda.is_available()", warning=True)
    _check(checks, "wandb", wandb_configured, "WANDB_API_KEY or ~/.netrc", warning=True)
    if lmu_data is not None:
        cache_detail = audit_vlmevalkit_cache(lmu_data)
        _check(
            checks,
            "vlmevalkit_cache",
            bool(cache_detail["ready"]),
            lmu_data,
            warning=True,
            detail=cache_detail,
        )

    by_id = {item["id"]: item for item in checks}
    training_ids = [
        "xlam_train",
        "mllm_router_train",
        "c3_direct_answer_train",
        "toolvqa_train",
        "visual_rft_train",
        "xlam_light",
        "mllm_router_light",
        *split_files,
    ]
    c0_data_ids = ["xlam_full", "mllm_router_full"]
    training_data_ready = all(by_id[check_id]["status"] == "ok" for check_id in training_ids)
    c0_data_ready = all(by_id[check_id]["status"] == "ok" for check_id in c0_data_ids)
    data_ready = training_data_ready and c0_data_ready
    core_ids = ["model", "c0_full_suite", "swift"]
    c0_core_ready = c0_data_ready and all(by_id[check_id]["status"] == "ok" for check_id in core_ids)
    gpu_runtime_ready = swift_available and vllm_available and cuda_available
    five_round_benchmark_data_ids = [
        "five_round_suites",
        "visual_rft_coco_full",
        "visual_rft_coco_light",
        "toolvqa_upstream",
        "visual_rft_coco_evaluator",
        "visual_rft_qwen3vl_bridge",
        "visual_rft_coco_query_report",
        "coco_val2017",
        "bfcl_full",
        "bfcl_light",
        "bfcl_bootstrap",
    ]
    if lmu_data is not None:
        five_round_benchmark_data_ids.append("vlmevalkit_cache")
    five_round_benchmark_data_ready = data_ready and all(
        by_id[check_id]["status"] == "ok" for check_id in five_round_benchmark_data_ids
    )
    five_round_benchmark_runtime_ready = (
        gpu_runtime_ready
        and toolvqa_runtime_ready
        and bfcl_available
        and coco_runtime_ready
        and five_round_benchmark_data_ready
    )
    optional_ids = [
        "toolvqa_upstream",
        "toolvqa_test",
        "visual_rft_coco_evaluator",
        "visual_rft_coco_runtime",
        "visual_rft_qwen3vl_bridge",
        "coco_val2017",
        "toolvqa_runtime",
        "bfcl_full",
        "bfcl_light",
        "bfcl_bootstrap",
        "bfcl",
    ]
    optional_benchmarks_ready = all(by_id[check_id]["status"] == "ok" for check_id in optional_ids)
    return {
        "root": str(root),
        "summary": {
            "data_ready": data_ready,
            "training_data_ready": training_data_ready,
            "c0_data_ready": c0_data_ready,
            "c0_core_ready": c0_core_ready,
            "gpu_runtime_ready": gpu_runtime_ready,
            "five_round_benchmark_data_ready": five_round_benchmark_data_ready,
            "five_round_benchmark_runtime_ready": five_round_benchmark_runtime_ready,
            "optional_benchmarks_ready": optional_benchmarks_ready,
            "bfcl_ready": bfcl_available,
            "wandb_configured": wandb_configured,
        },
        "checks": checks,
    }


def _python_can_import(python_executable: str, module: str) -> bool:
    try:
        process = __import__("subprocess").run(
            [python_executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return process.returncode == 0


def _toolvqa_runtime_available(python_executable: str) -> bool:
    probe = (
        "from lagent.agents import ReAct; "
        "from opencompass.datasets.gta_bench import GTABenchEvaluator; "
        "from opencompass.models import OpenAIVisual; "
        "from opencompass.models.lagent import LagentAgentVisual; "
        "from opencompass.openicl.icl_inferencer import AgentInferencer"
    )
    try:
        process = __import__("subprocess").run(
            [python_executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return process.returncode == 0


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/model/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--toolvqa-root", type=Path, default=Path("/root/autodl-tmp/ToolVQA-release"))
    parser.add_argument("--visual-rft-root", type=Path, default=Path("/root/autodl-tmp/Visual-RFT"))
    parser.add_argument("--coco-root", type=Path, default=Path("/root/autodl-tmp/datasets/COCO2017"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/c0/readiness.json"))
    parser.add_argument("--lmu-data", type=Path, default=Path("/root/autodl-tmp/cache/vlmevalkit"))
    parser.add_argument(
        "--bfcl-python",
        default=os.environ.get("BFCL_PYTHON", "/root/autodl-tmp/venvs/bfcl/bin/python"),
    )
    parser.add_argument("--require-five-round-benchmarks", action="store_true")
    parser.add_argument("--require-gpu-runtime", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    wandb_configured = bool(os.environ.get("WANDB_API_KEY")) or (Path.home() / ".netrc").is_file()
    toolvqa_python = os.environ.get(
        "TOOLVQA_PYTHON", "/root/autodl-tmp/venvs/toolvqa/bin/python"
    )
    report = audit_c0(
        coco_runtime_ready=importlib.util.find_spec("pycocotools") is not None,
        root=args.root.resolve(),
        model=args.model,
        toolvqa_root=args.toolvqa_root,
        visual_rft_root=args.visual_rft_root,
        coco_root=args.coco_root,
        swift_available=shutil.which("swift") is not None,
        bfcl_available=_python_can_import(args.bfcl_python, "bfcl_eval"),
        vllm_available=importlib.util.find_spec("vllm") is not None,
        cuda_available=_cuda_available(),
        wandb_configured=wandb_configured,
        toolvqa_runtime_ready=_toolvqa_runtime_available(toolvqa_python),
        toolvqa_python=toolvqa_python,
        lmu_data=args.lmu_data,
    )
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["summary"]["data_ready"] or not report["summary"]["c0_core_ready"]:
        raise SystemExit(2)
    if args.require_gpu_runtime and not report["summary"]["gpu_runtime_ready"]:
        raise SystemExit(3)
    if args.require_five_round_benchmarks and not report["summary"]["five_round_benchmark_runtime_ready"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
