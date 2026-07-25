"""Build the isolated C3 datafixed agent terminal-retention track.

This track keeps explicit Router-mode prompts while adding ToolVQA transitions
that match the Lagent ReAct protocol used by the step benchmark.  All ToolVQA
official-test questions are excluded before any derived artifact is written.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts import prepare_c3_datafixed_agentretention as base
    from scripts.build_sft_replay import encode_jsonl, write_transaction
except ModuleNotFoundError:  # Direct execution sets sys.path[0] to scripts/.
    import prepare_c3_datafixed_agentretention as base
    from build_sft_replay import encode_jsonl, write_transaction


VARIANT = "datafixed-agentretention-terminalfix"
REACT_CALL_PROTOCOL = """You are a assistant who can utilize external tools.
{tool_description}
To use a tool, please use the following format:
```
Thought:Think what you need to solve, do you need to use tools?
Action:the tool name, should be one of [{action_names}]
Action Input:the input to the action
```
The response after utilizing tools should using the following format:
```
Response:the results after call the tool.
```
If you already know the answer, or you do not need to use tools,
please using the following format to reply:
```
Thought:the thought process to get the final answer
Final Answer:final answer
```
Begin!"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--light-limit", type=int, default=200)
    parser.add_argument("--video-light-min", type=int, default=16)
    parser.add_argument("--toolvqa-max-rows", type=int, default=12000)
    parser.add_argument("--abstention-max-rows", type=int, default=2000)
    parser.add_argument(
        "--base-report",
        type=Path,
        default=Path("data/reports/c3/router_datafixed.report.json"),
    )
    parser.add_argument(
        "--router-train-source",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/mllm_tool_router_datafixed_train.jsonl"
        ),
    )
    parser.add_argument(
        "--router-full-source",
        type=Path,
        default=Path("data/eval/mllm_router_datafixed/full.jsonl"),
    )
    parser.add_argument(
        "--direct-answer-source",
        type=Path,
        default=Path(
            "data/exports/msswift_sft/toolvqa_direct_answer_datafixed_train.jsonl"
        ),
    )
    parser.add_argument(
        "--toolvqa-source",
        type=Path,
        default=Path("data/exports/msswift_sft/toolvqa_train.jsonl"),
    )
    parser.add_argument(
        "--toolvqa-test-source",
        type=Path,
        default=Path("data/raw/ToolVQA/test.jsonl"),
    )
    parser.add_argument(
        "--toolvqa-eval-source",
        type=Path,
        default=Path(
            "/root/autodl-tmp/ToolVQA-release/opencompass/data/ToolVQA/dataset.json"
        ),
    )
    parser.add_argument(
        "--xlam-source",
        type=Path,
        default=Path("data/exports/msswift_sft/xlam_train.jsonl"),
    )

    prefix = "datafixed-agentretention-terminalfix"
    parser.add_argument(
        "--router-train",
        type=Path,
        default=Path(f"data/exports/msswift_sft/mllm_tool_router_{prefix}_train.jsonl"),
    )
    parser.add_argument(
        "--router-full",
        type=Path,
        default=Path(f"data/eval/mllm_router_{prefix}/full.jsonl"),
    )
    parser.add_argument(
        "--router-light",
        type=Path,
        default=Path(f"data/eval/mllm_router_{prefix}/light.jsonl"),
    )
    parser.add_argument(
        "--direct-answer",
        type=Path,
        default=Path(
            f"data/exports/msswift_sft/toolvqa_direct_answer_{prefix}_train.jsonl"
        ),
    )
    parser.add_argument(
        "--toolvqa-complete",
        type=Path,
        default=Path(
            f"data/exports/msswift_sft/toolvqa_complete_{prefix}_train.jsonl"
        ),
    )
    parser.add_argument(
        "--toolvqa-terminal",
        type=Path,
        default=Path(
            f"data/exports/msswift_sft/toolvqa_terminal_answer_{prefix}_train.jsonl"
        ),
    )
    parser.add_argument(
        "--toolvqa-react-action",
        type=Path,
        default=Path(
            f"data/exports/msswift_sft/toolvqa_react_action_{prefix}_train.jsonl"
        ),
    )
    parser.add_argument(
        "--toolvqa-react-terminal",
        type=Path,
        default=Path(
            f"data/exports/msswift_sft/toolvqa_react_terminal_{prefix}_train.jsonl"
        ),
    )
    parser.add_argument(
        "--xlam-abstention",
        type=Path,
        default=Path(
            f"data/exports/msswift_sft/xlam_abstention_{prefix}_train.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(f"data/reports/c3/router_{prefix}.report.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _require_variant_output(path: Path) -> None:
    if VARIANT not in str(path).casefold() or VARIANT not in str(_resolved(path)).casefold():
        raise ValueError(f"output must remain in the {VARIANT!r} namespace: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).replace("<image>", " ")
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text or None


def _test_question_keys(raw_test: Path, canonical_eval: Path) -> set[str]:
    keys: set[str] = set()
    for _, row in base._iter_jsonl(raw_test):
        for field in ("question", "ori_question"):
            key = _question_key(row.get(field))
            if key:
                keys.add(key)

    canonical = json.loads(canonical_eval.read_text(encoding="utf-8"))
    rows: Iterable[Any]
    if isinstance(canonical, dict):
        rows = canonical.values()
    elif isinstance(canonical, list):
        rows = canonical
    else:
        raise ValueError("canonical ToolVQA eval manifest must be a mapping or list")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        dialogs = row.get("dialogs")
        if not isinstance(dialogs, list):
            continue
        for message in dialogs:
            if isinstance(message, Mapping) and message.get("role") == "user":
                key = _question_key(message.get("content"))
                if key:
                    keys.add(key)
                break
    if not keys:
        raise ValueError("ToolVQA official-test exclusion set is empty")
    return keys


def _first_user_key(row: Mapping[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "user":
            return _question_key(message.get("content"))
    return None


def _filter_test_rows(
    rows: Iterable[dict[str, Any]],
    excluded_questions: set[str],
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        if _first_user_key(row) in excluded_questions:
            excluded += 1
        else:
            kept.append(row)
    if not kept:
        raise ValueError("test exclusion removed every source row")
    return kept, excluded


def _parse_tools(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError("ToolVQA row has no valid tool catalog")
    tools = [dict(item) for item in value if isinstance(item, Mapping)]
    if not tools:
        raise ValueError("ToolVQA row has an empty tool catalog")
    return tools


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    value = function if isinstance(function, Mapping) else tool
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("ToolVQA tool is missing a name")
    return name.strip()


def _react_system(tools: list[dict[str, Any]], images: list[str]) -> str:
    names = [_tool_name(tool) for tool in tools]
    description = json.dumps(tools, ensure_ascii=False, sort_keys=True)
    system = REACT_CALL_PROTOCOL.format(
        tool_description=description,
        action_names=", ".join(names),
    )
    if images:
        paths = ", ".join(f"`{path}`" for path in images)
        system += f"\n\nThe related files are at {paths}"
    return system


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("content"), str):
        return str(value["content"])
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_tool_call(value: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("ToolVQA tool call must be a JSON object")
    name = value.get("name")
    arguments = value.get("arguments", {})
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, Mapping):
        raise ValueError("ToolVQA tool call has an invalid name or arguments")
    return name.strip(), dict(arguments)


def _react_target(message: Mapping[str, Any]) -> tuple[str, str]:
    role = message.get("role")
    if role == "tool_call":
        name, arguments = _parse_tool_call(message.get("content"))
        return (
            "action",
            f"Action:{name}\nAction Input:{json.dumps(arguments, ensure_ascii=False)}",
        )
    if role == "assistant":
        answer = message.get("content")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("ToolVQA final answer is empty")
        return (
            "terminal",
            "Thought:Based on the tool results, I can answer the question.\n"
            f"Final Answer:{answer.strip()}",
        )
    raise ValueError(f"unsupported ToolVQA target role: {role!r}")


def _react_prefix(
    row: Mapping[str, Any],
    *,
    target_index: int,
    source_row: int,
) -> tuple[str, dict[str, Any]]:
    messages = row["messages"]
    images = [str(value) for value in row.get("images", [])]
    tools = _parse_tools(row.get("tools"))
    converted: list[dict[str, Any]] = [
        {"role": "system", "content": _react_system(tools, images)}
    ]
    for index, message in enumerate(messages[: target_index + 1]):
        if not isinstance(message, Mapping):
            raise ValueError("ToolVQA message must be an object")
        role = message.get("role")
        if index == target_index:
            transition, target = _react_target(message)
            converted.append(
                {"role": "assistant", "content": target, "loss_scale": 1.0}
            )
            break
        if role == "user":
            converted.append(
                {
                    "role": "user",
                    "content": str(message.get("content", "")).replace("<image>", ""),
                }
            )
        elif role == "tool_call":
            _, target = _react_target(message)
            converted.append(
                {"role": "assistant", "content": target, "loss_scale": 0.0}
            )
        elif role in {"tool_response", "tool"}:
            converted.append(
                {"role": "user", "content": _text_content(message.get("content"))}
            )
        elif role == "assistant":
            raise ValueError("unexpected final answer before the selected transition")
        else:
            raise ValueError(f"unsupported ToolVQA history role: {role!r}")

    last_user = max(
        (index for index, message in enumerate(converted) if message["role"] == "user"),
        default=-1,
    )
    if last_user < 0:
        raise ValueError("ReAct transition has no user context")
    if images:
        converted[last_user]["content"] = "<image>" + converted[last_user]["content"]
    row_out: dict[str, Any] = {
        "messages": converted,
        "metadata": {
            "agent_retention_source": f"toolvqa_react_{transition}",
            "agent_retention_source_row": source_row,
            "agent_retention_target_index": target_index,
            "agent_retention_contract": "toolvqa-lagent-react-transition-v1",
            "supervision": "current_assistant_transition_only",
        },
    }
    if images:
        row_out["images"] = images
    return transition, row_out


def _toolvqa_rows(
    source: Path,
    *,
    excluded_questions: set[str],
    seed: int,
    max_rows: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    eligible: dict[str, tuple[int, dict[str, Any]]] = {}
    skipped = Counter()
    input_rows = 0
    excluded_rows = 0
    for source_row, row in base._iter_jsonl(source):
        input_rows += 1
        if _first_user_key(row) in excluded_questions:
            excluded_rows += 1
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            skipped["invalid_messages"] += 1
            continue
        roles = [m.get("role") for m in messages if isinstance(m, Mapping)]
        last = messages[-1]
        if (
            "tool_call" not in roles
            or "tool_response" not in roles
            or not isinstance(last, Mapping)
            or last.get("role") != "assistant"
            or not isinstance(last.get("content"), str)
            or not last["content"].strip()
        ):
            skipped["not_complete_agent_trajectory"] += 1
            continue
        try:
            _parse_tools(row.get("tools"))
            images = row.get("images")
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError("expected exactly one image")
        except (TypeError, ValueError, json.JSONDecodeError):
            skipped["invalid_tools_or_images"] += 1
            continue
        eligible.setdefault(base._identity(row), (source_row, row))

    selected_raw = base._deterministic_select(
        [row for _, row in eligible.values()],
        count=min(max_rows, len(eligible)),
        seed=seed,
        namespace="terminalfix-toolvqa-trajectories",
    )
    source_by_identity = {
        identity: source_row for identity, (source_row, _) in eligible.items()
    }
    complete_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    react_actions: list[dict[str, Any]] = []
    react_terminals: list[dict[str, Any]] = []
    for row in selected_raw:
        source_row = source_by_identity[base._identity(row)]
        complete = copy.deepcopy(row)
        complete["metadata"] = {
            "agent_retention_source": "toolvqa_complete_terminalfix",
            "agent_retention_source_row": source_row,
            "agent_retention_contract": "hermes-complete-agent-trajectory-v1",
        }
        complete_rows.append(complete)

        terminal = copy.deepcopy(complete)
        terminal["metadata"] = {
            **complete["metadata"],
            "agent_retention_source": "toolvqa_terminal_terminalfix",
            "supervision": "final_assistant_only",
        }
        for index, message in enumerate(terminal["messages"]):
            if isinstance(message, dict) and message.get("role") in {
                "assistant",
                "tool_call",
            }:
                message["loss_scale"] = (
                    1.0 if index == len(terminal["messages"]) - 1 else 0.0
                )
        terminal_rows.append(terminal)

        for target_index, message in enumerate(row["messages"]):
            if not isinstance(message, Mapping) or message.get("role") not in {
                "tool_call",
                "assistant",
            }:
                continue
            transition, converted = _react_prefix(
                row,
                target_index=target_index,
                source_row=source_row,
            )
            if transition == "action":
                react_actions.append(converted)
            else:
                react_terminals.append(converted)

    if not complete_rows or not react_actions or not react_terminals:
        raise ValueError("ToolVQA terminal-fix conversion produced an empty artifact")
    report = {
        "input_rows": input_rows,
        "official_test_excluded_rows": excluded_rows,
        "eligible_unique_rows": len(eligible),
        "selected_trajectories": len(complete_rows),
        "react_action_transitions": len(react_actions),
        "react_terminal_transitions": len(react_terminals),
        "max_rows": max_rows,
        "skipped": dict(sorted(skipped.items())),
    }
    return complete_rows, terminal_rows, react_actions, react_terminals, report


def _validate_zero_overlap(
    rows: Iterable[Mapping[str, Any]], excluded_questions: set[str], label: str
) -> None:
    overlaps = sum(_first_user_key(row) in excluded_questions for row in rows)
    if overlaps:
        raise ValueError(f"{label} still overlaps {overlaps} ToolVQA test questions")


def build_outputs(args: argparse.Namespace) -> tuple[dict[Path, bytes], dict[str, Any]]:
    if args.toolvqa_max_rows <= 0 or args.abstention_max_rows <= 0:
        raise ValueError("ToolVQA and abstention row limits must be positive")
    inputs = {
        "base_report": args.base_report.resolve(strict=True),
        "router_train": args.router_train_source.resolve(strict=True),
        "router_full": args.router_full_source.resolve(strict=True),
        "direct_answer": args.direct_answer_source.resolve(strict=True),
        "toolvqa": args.toolvqa_source.resolve(strict=True),
        "toolvqa_test": args.toolvqa_test_source.resolve(strict=True),
        "toolvqa_eval": args.toolvqa_eval_source.resolve(strict=True),
        "xlam": args.xlam_source.resolve(strict=True),
    }
    outputs = {
        "router_train": _resolved(args.router_train),
        "router_full": _resolved(args.router_full),
        "router_light": _resolved(args.router_light),
        "direct_answer": _resolved(args.direct_answer),
        "toolvqa_complete": _resolved(args.toolvqa_complete),
        "toolvqa_terminal": _resolved(args.toolvqa_terminal),
        "toolvqa_react_action": _resolved(args.toolvqa_react_action),
        "toolvqa_react_terminal": _resolved(args.toolvqa_react_terminal),
        "xlam_abstention": _resolved(args.xlam_abstention),
        "report": _resolved(args.report),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("terminal-fix output paths must be unique")
    for path in outputs.values():
        _require_variant_output(path)
    if set(inputs.values()) & set(outputs.values()):
        raise ValueError("terminal-fix outputs must not overwrite input artifacts")

    base._validate_base_report(
        inputs["base_report"],
        router_train=inputs["router_train"],
        router_full=inputs["router_full"],
        direct_answer=inputs["direct_answer"],
    )
    excluded_questions = _test_question_keys(
        inputs["toolvqa_test"], inputs["toolvqa_eval"]
    )
    router_train_rows = base._read_jsonl(inputs["router_train"])
    router_full_rows = base._read_jsonl(inputs["router_full"])
    router_light_rows, light_quotas = base._balanced_router_light(
        router_full_rows,
        limit=args.light_limit,
        video_minimum=args.video_light_min,
        seed=args.seed,
    )
    direct_rows, direct_excluded = _filter_test_rows(
        base._read_jsonl(inputs["direct_answer"]), excluded_questions
    )
    (
        complete_rows,
        terminal_rows,
        react_action_rows,
        react_terminal_rows,
        toolvqa_report,
    ) = _toolvqa_rows(
        inputs["toolvqa"],
        excluded_questions=excluded_questions,
        seed=args.seed,
        max_rows=args.toolvqa_max_rows,
    )
    abstention_rows, abstention_report = base._abstention_rows(
        inputs["xlam"],
        seed=args.seed,
        max_rows=args.abstention_max_rows,
    )

    for label, rows in (
        ("direct_answer", direct_rows),
        ("toolvqa_complete", complete_rows),
        ("toolvqa_terminal", terminal_rows),
        ("toolvqa_react_action", react_action_rows),
        ("toolvqa_react_terminal", react_terminal_rows),
    ):
        _validate_zero_overlap(rows, excluded_questions, label)

    row_artifacts = {
        "router_train": router_train_rows,
        "router_full": router_full_rows,
        "router_light": router_light_rows,
        "direct_answer": direct_rows,
        "toolvqa_complete": complete_rows,
        "toolvqa_terminal": terminal_rows,
        "toolvqa_react_action": react_action_rows,
        "toolvqa_react_terminal": react_terminal_rows,
        "xlam_abstention": abstention_rows,
    }
    encoded = {
        outputs[name]: encode_jsonl(rows) for name, rows in row_artifacts.items()
    }
    report = {
        "schema_version": "1.0",
        "variant": VARIANT,
        "seed": args.seed,
        "contract": "c3-datafixed-agentretention-terminalfix-v1",
        "inputs": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in inputs.items()
        },
        "official_test_exclusion": {
            "question_keys": len(excluded_questions),
            "direct_answer_excluded_rows": direct_excluded,
            "toolvqa_excluded_rows": toolvqa_report[
                "official_test_excluded_rows"
            ],
            "postbuild_overlap_rows": 0,
        },
        "router": {
            "train_rows": len(router_train_rows),
            "full_rows": len(router_full_rows),
            "light_rows": len(router_light_rows),
            "light_by_modality": light_quotas,
            "video_light_minimum": args.video_light_min,
        },
        "toolvqa": toolvqa_report,
        "react_contract": {
            "action_marker": "Action:",
            "action_input_marker": "Action Input:",
            "terminal_marker": "Final Answer:",
            "history_supervision": 0.0,
            "target_supervision": 1.0,
            "benchmark_protocol": "Lagent ReAct every_with_gt",
        },
        "xlam_abstention": abstention_report,
        "artifacts": {name: str(path) for name, path in outputs.items()},
        "audit": {
            name: {
                "rows": len(row_artifacts[name]),
                "sha256": hashlib.sha256(encoded[outputs[name]]).hexdigest(),
            }
            for name in row_artifacts
        },
    }
    report_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    encoded[outputs["report"]] = report_payload
    return encoded, report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.overwrite and args.verify_only:
        raise SystemExit("--overwrite and --verify-only are mutually exclusive")
    try:
        encoded, report = build_outputs(args)
        if args.verify_only:
            mismatches = [
                str(path)
                for path, expected in encoded.items()
                if not path.is_file() or path.read_bytes() != expected
            ]
            if mismatches:
                raise ValueError(
                    f"terminal-fix deterministic verification failed: {mismatches}"
                )
            print(
                json.dumps(
                    {
                        "variant": VARIANT,
                        "verification": "passed",
                        "artifacts": len(encoded),
                    },
                    indent=2,
                )
            )
            return
        existing = [path for path in encoded if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"outputs already exist; pass --overwrite: {existing}"
            )
        write_transaction(encoded)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
