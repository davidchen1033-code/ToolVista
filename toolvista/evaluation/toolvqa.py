"""Minimal bridge helpers for ToolVQA's upstream OpenCompass evaluator."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit, urlunsplit

from toolvista.evaluation.io import load_json, write_json


ToolVQAMode = Literal["step", "e2e"]
MODEL_ABBR = "toolvista_qwen3vl"


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
