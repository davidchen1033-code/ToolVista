"""Minimal bridge helpers for ToolVQA's upstream OpenCompass evaluator."""

from __future__ import annotations

import copy
import os
import shlex
import subprocess
import sys
from math import isfinite
from pathlib import Path
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from toolvista.evaluation.io import load_json, write_json


ToolVQAMode = Literal["step", "e2e"]
MODEL_ABBR = "toolvista_qwen3vl"
STEP_METRICS = {
    "inst_align": "toolvqa_sbs/inst_align",
    "tool_acc": "toolvqa_sbs/tool_accuracy",
    "arg_acc": "toolvqa_sbs/argument_accuracy",
    "answer_acc": "toolvqa_sbs/answer_accuracy",
}
E2E_METRICS = {
    "answer_acc": "toolvqa_e2e/answer_accuracy",
    "p_f1": "toolvqa_e2e/perception_f1",
    "o_f1": "toolvqa_e2e/operation_f1",
    "l_f1": "toolvqa_e2e/logic_f1",
    "c_f1": "toolvqa_e2e/creativity_f1",
}


def normalize_api_base(value: str) -> str:
    """Normalize an OpenAI SDK base URL to exactly one trailing /v1."""
    text = value.strip().rstrip("/")
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid OpenAI-compatible API base: {value!r}")
    path = parsed.path.rstrip("/")
    normalized_path = path if path.endswith("/v1") else f"{path}/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def resolve_toolmeta(data_root: Path, toolvqa_root: Path) -> Path:
    """Use dataset-local tool metadata, then fall back to the upstream copy."""
    candidates = (
        data_root / "toolmeta.json",
        toolvqa_root / "opencompass" / "data" / "ToolVQA" / "toolmeta.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "ToolVQA toolmeta.json was not found in the data or upstream repository."
    )


def _resolve_media(source_root: Path, raw_path: str, media_root: Path | None) -> Path:
    candidate = Path(raw_path)
    if candidate.is_file():
        return candidate.resolve()

    relative = source_root / candidate
    if relative.is_file():
        return relative.resolve()

    if media_root is None:
        raise FileNotFoundError(f"Missing ToolVQA media: {raw_path}")
    matches = [path.resolve() for path in media_root.rglob(candidate.name) if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"Missing ToolVQA media: {raw_path}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous ToolVQA media {candidate.name!r}: {len(matches)} matches")
    return matches[0]


def _rewrite_matching_arguments(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_matching_arguments(item, old, new)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_matching_arguments(item, old, new) for item in value]
    return new if value == old else value


def prepare_data(
    data_root: Path,
    output_root: Path,
    *,
    media_root: Path | None = None,
) -> tuple[Path, bool, int]:
    """Write an upstream-format manifest with locally resolvable media paths."""
    source = data_root / "dataset.json"
    dataset = load_json(source)
    prepared = copy.deepcopy(dataset)
    is_fixture = bool(prepared) and all(
        isinstance(row, Mapping) and row.get("_toolvista_fixture") is True
        for row in prepared.values()
    )

    for sample_id, row in prepared.items():
        if not isinstance(row, dict):
            raise ValueError(f"ToolVQA sample {sample_id!r} must be an object.")
        files = row.get("files", [])
        if not isinstance(files, list):
            raise ValueError(f"ToolVQA sample {sample_id!r} files must be a list.")
        for file_item in files:
            if not isinstance(file_item, dict):
                raise ValueError(f"ToolVQA sample {sample_id!r} has an invalid file entry.")
            old = file_item.get("path")
            if not isinstance(old, str) or not old:
                raise ValueError(f"ToolVQA sample {sample_id!r} has an invalid media path.")
            resolved = str(_resolve_media(data_root, old, media_root))
            file_item["path"] = resolved
            row["dialogs"] = _rewrite_matching_arguments(
                row.get("dialogs", []), old, resolved
            )

    output = output_root / "dataset.json"
    write_json(output, prepared)
    return output, is_fixture, len(prepared)


def render_runtime_config(
    *,
    mode: ToolVQAMode,
    data_root: Path,
    toolmeta: Path,
    api_base: str,
    api_model: str,
    tool_url: str | None,
    limit: int | None,
) -> str:
    """Render the minimal config consumed by ToolVQA's modified OpenCompass."""
    if mode not in {"step", "e2e"}:
        raise ValueError(f"Unsupported ToolVQA mode: {mode!r}")
    if mode == "e2e" and not tool_url:
        raise ValueError("ToolVQA e2e mode requires an AgentLego tool URL.")
    if limit is not None and limit <= 0:
        raise ValueError("ToolVQA limit must be positive.")
    if not api_model.strip():
        raise ValueError("ToolVQA API model must be non-empty.")

    infer_mode = "every_with_gt" if mode == "step" else "every"
    max_turn = 6 if mode == "step" else 8
    range_line = "" if limit is None else f"    test_range={f'[0:{limit}]'!r},\n"
    tool_line = (
        f"        tool_meta={str(toolmeta.resolve())!r},\n"
        if mode == "step"
        else f"        tool_server={tool_url!r},\n"
    )
    normalized_base = normalize_api_base(api_base)

    return f'''from lagent.agents import ReAct
from opencompass.datasets.gta_bench import GTABenchDataset, GTABenchEvaluator
from opencompass.models import OpenAIVisual
from opencompass.models.lagent import LagentAgentVisual
from opencompass.openicl.icl_inferencer import AgentInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

gta_bench_reader_cfg = dict(
    input_columns=["dialogs", "resources"],
    output_column="gt_answer",
    train_split="test",
    test_split="test",
{range_line})

gta_bench_infer_cfg = dict(
    prompt_template=dict(type=PromptTemplate, template="{{questions}}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=AgentInferencer, infer_mode={infer_mode!r}),
)

gta_bench_eval_cfg = dict(
    evaluator=dict(type=GTABenchEvaluator, mode={infer_mode!r}),
)

datasets = [dict(
    abbr="gta_bench",
    type=GTABenchDataset,
    path={str(data_root.resolve())!r},
    reader_cfg=gta_bench_reader_cfg,
    infer_cfg=gta_bench_infer_cfg,
    eval_cfg=gta_bench_eval_cfg,
)]

models = {{
    "{MODEL_ABBR}": dict(
        abbr="{MODEL_ABBR}",
        type=LagentAgentVisual,
        agent_type=ReAct,
        max_turn={max_turn},
        llm=dict(
            type=OpenAIVisual,
            path={api_model.strip()!r},
            key="ENV",
            openai_api_base={normalized_base!r},
            query_per_second=1,
            max_seq_len=4096,
            retry=3,
            temperature=0.0,
        ),
{tool_line}        batch_size=1,
    ),
}}

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=4000, gen_task_coef=1),
    runner=dict(type=LocalRunner, task=dict(type=OpenICLInferTask)),
)
'''


def map_metrics(
    upstream: Mapping[str, Any],
    mode: ToolVQAMode,
) -> dict[str, float]:
    """Map official ToolVQA metrics without changing their values."""
    if mode not in {"step", "e2e"}:
        raise ValueError(f"Unsupported ToolVQA mode: {mode!r}")
    mapping = STEP_METRICS if mode == "step" else E2E_METRICS
    metrics: dict[str, float] = {}
    for source, target in mapping.items():
        value = upstream.get(source)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            raise ValueError(
                f"Official ToolVQA result requires finite metric {source!r}."
            )
        metrics[target] = float(value)
    return metrics


def _latest_official_result(upstream_root: Path) -> Path:
    matches = [
        path
        for path in upstream_root.rglob("gta_bench.json")
        if path.parent.name == MODEL_ABBR and path.parent.parent.name == "results"
    ]
    if not matches:
        raise FileNotFoundError(
            f"Official ToolVQA gta_bench.json not found under {upstream_root}"
        )
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def run_toolvqa(
    *,
    mode: ToolVQAMode,
    toolvqa_root: Path,
    data_root: Path,
    output_dir: Path,
    api_base: str,
    api_model: str,
    tool_url: str | None = None,
    media_root: Path | None = None,
    python_executable: str = sys.executable,
    workers: int = 1,
    limit: int | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Run ToolVQA's upstream OpenCompass CLI and parse its official JSON."""
    if mode not in {"step", "e2e"}:
        raise ValueError(f"Unsupported ToolVQA mode: {mode!r}")
    if workers <= 0:
        raise ValueError("ToolVQA workers must be positive.")
    if limit is not None and limit <= 0:
        raise ValueError("ToolVQA limit must be positive.")

    toolvqa_root = toolvqa_root.resolve()
    data_root = data_root.resolve()
    output_dir = output_dir.resolve()
    opencompass_root = toolvqa_root / "opencompass"
    run_py = opencompass_root / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(
            f"ToolVQA OpenCompass entry point does not exist: {run_py}"
        )
    if not (data_root / "dataset.json").is_file():
        raise FileNotFoundError(
            f"ToolVQA dataset.json does not exist: {data_root / 'dataset.json'}"
        )

    toolmeta = resolve_toolmeta(data_root, toolvqa_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dataset, is_fixture, sample_count = prepare_data(
        data_root,
        output_dir / "data",
        media_root=media_root.resolve() if media_root is not None else None,
    )
    runtime_config = output_dir / "runtime_config.py"
    runtime_config.write_text(
        render_runtime_config(
            mode=mode,
            data_root=prepared_dataset.parent,
            toolmeta=toolmeta,
            api_base=api_base,
            api_model=api_model,
            tool_url=tool_url,
            limit=limit,
        ),
        encoding="utf-8",
        newline="\n",
    )

    upstream_root = output_dir / "upstream"
    command = [
        python_executable,
        str(run_py),
        str(runtime_config),
        "--work-dir",
        str(upstream_root),
        "--eval_mode",
        MODEL_ABBR,
        "--max-num-workers",
        str(workers),
        "--debug",
    ]
    try:
        completed = command_runner(
            command,
            cwd=str(opencompass_root),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        (output_dir / "stdout.log").write_text("", encoding="utf-8")
        (output_dir / "stderr.log").write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"Unable to start ToolVQA OpenCompass: {exc}") from exc

    (output_dir / "stdout.log").write_text(
        completed.stdout or "",
        encoding="utf-8",
    )
    (output_dir / "stderr.log").write_text(
        completed.stderr or "",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ToolVQA OpenCompass failed with exit code {completed.returncode}; "
            f"see {output_dir / 'stderr.log'}"
        )

    official_result = _latest_official_result(upstream_root)
    upstream_metrics = load_json(official_result)
    metrics = map_metrics(upstream_metrics, mode)
    result = {
        "mode": mode,
        "dataset_kind": "train_fixture" if is_fixture else "official_test",
        "comparable": not is_fixture,
        "warning": "Not an official ToolVQA test score." if is_fixture else None,
        "sample_count": (
            min(sample_count, limit) if limit is not None else sample_count
        ),
        "upstream_result": official_result.as_posix(),
        "upstream_metrics": upstream_metrics,
        "metrics": metrics,
        "command": shlex.join(command),
        "exit_code": completed.returncode,
    }
    write_json(output_dir / "result.json", result)
    return result
