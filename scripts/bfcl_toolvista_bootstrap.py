"""Run BFCL v4 against a ToolVista OpenAI-compatible endpoint without patching BFCL."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Sequence


REGISTRY_NAME = "toolvista-qwen3vl-fc"
NON_LIVE_CATEGORIES = (
    "simple_python",
    "simple_java",
    "simple_javascript",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--id-manifest", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--api-model", required=True)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--check-registration", action="store_true")
    return parser.parse_args(argv)


def _validate_manifest(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or tuple(value) != NON_LIVE_CATEGORIES:
        raise ValueError(
            "BFCL ID manifest must contain the ordered non-live categories: "
            f"{NON_LIVE_CATEGORIES}"
        )
    for category, ids in value.items():
        if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"BFCL category {category!r} requires a non-empty string ID list.")
        if len(ids) != len(set(ids)):
            raise ValueError(f"BFCL category {category!r} contains duplicate IDs.")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_threads <= 0 or args.timeout <= 0:
        raise ValueError("--num-threads and --timeout must be positive.")
    manifest = _validate_manifest(args.id_manifest)

    project_root = args.project_root.resolve()
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.environ["OPENAI_BASE_URL"] = args.api_base.rstrip("/")
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_TIMEOUT"] = str(args.timeout)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    class ToolVistaOpenAIHandler(OpenAICompletionsHandler):
        def _build_client_kwargs(self):
            kwargs = super()._build_client_kwargs()
            kwargs["timeout"] = args.timeout
            return kwargs

    MODEL_CONFIG_MAPPING[REGISTRY_NAME] = ModelConfig(
        model_name=args.api_model,
        display_name=f"ToolVista Qwen3-VL ({args.api_model})",
        url="local-openai-compatible-endpoint",
        org="ToolVista",
        license="Apache-2.0",
        model_handler=ToolVistaOpenAIHandler,
        is_fc_model=True,
        underscore_to_dot=True,
    )
    if args.check_registration:
        print(json.dumps({"registry_name": REGISTRY_NAME, "api_model": args.api_model}))
        return

    for name in ("result", "score", ".file_locks"):
        path = project_root / name
        if path.exists():
            shutil.rmtree(path)
    project_root.mkdir(parents=True, exist_ok=True)
    for name in ("result", "score", ".file_locks"):
        (project_root / name).mkdir()
    (project_root / "test_case_ids_to_generate.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    from bfcl_eval import _llm_response_generation as generation
    from bfcl_eval.eval_checker import eval_runner

    generation_args = argparse.Namespace(
        model=[REGISTRY_NAME],
        test_category=["non_live"],
        temperature=args.temperature,
        include_input_log=False,
        exclude_state_log=True,
        num_threads=args.num_threads,
        num_gpus=0,
        backend="vllm",
        gpu_memory_utilization=0.0,
        result_dir="result",
        run_ids=True,
        allow_overwrite=True,
        skip_server_setup=True,
        local_model_path=None,
        lora_modules=None,
        enable_lora=False,
        max_lora_rank=None,
    )
    generation.main(generation_args)
    eval_runner.main(
        [REGISTRY_NAME],
        list(NON_LIVE_CATEGORIES),
        "result",
        "score",
        partial_eval=not args.full,
    )


if __name__ == "__main__":
    main()
