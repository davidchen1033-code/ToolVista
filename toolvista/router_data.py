"""Shared prompt and candidate-catalog contract for MLLM tool routing."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


ROUTER_CONTRACT_VERSION = "toolvista-router-v2"
ROUTER_MODE_MARKER = "[TOOL_ROUTER_MODE]"
DEFAULT_ROUTER_CATALOG_SIZE = 32


def normalize_tool_ids(values: Sequence[Any]) -> list[str]:
    return sorted(
        {
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        }
    )


def build_router_catalog(
    target_tools: Sequence[Any],
    candidates: Sequence[Any],
    *,
    identity: str,
    seed: int,
    max_size: int = DEFAULT_ROUTER_CATALOG_SIZE,
) -> list[str]:
    """Return all targets plus deterministic hard negatives from one modality."""
    if max_size <= 1:
        raise ValueError("Router catalog size must be greater than one.")
    targets = normalize_tool_ids(target_tools)
    if not targets:
        raise ValueError("Router catalog requires at least one target tool.")
    if len(targets) > max_size:
        raise ValueError(
            f"Router target count exceeds catalog size: {len(targets)} > {max_size}"
        )
    pool = normalize_tool_ids(candidates)
    missing = sorted(set(targets) - set(pool))
    if missing:
        raise ValueError(f"Router targets are absent from the candidate pool: {missing}")

    negatives = [tool_id for tool_id in pool if tool_id not in targets]
    negatives.sort(
        key=lambda tool_id: (
            hashlib.sha256(
                f"{seed}:{identity}:{tool_id}".encode("utf-8")
            ).hexdigest(),
            tool_id,
        )
    )
    selected = targets + negatives[: max(0, max_size - len(targets))]
    return sorted(selected)


def router_system_prompt(catalog: Sequence[Any]) -> str:
    normalized = normalize_tool_ids(catalog)
    if not normalized:
        raise ValueError("Router prompt requires a non-empty catalog.")
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return (
        f"{ROUTER_MODE_MARKER}\n"
        "Route the user's request instead of answering it. Select exactly one "
        f"tool ID from this candidate catalog: {encoded}\n"
        "Return only the selected tool ID, with no explanation or extra text."
    )


def with_router_system_prompt(
    messages: Sequence[Mapping[str, Any]],
    catalog: Sequence[Any],
) -> list[dict[str, Any]]:
    """Apply the Router contract idempotently while preserving source messages."""
    retained: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if (
            copied.get("role") == "system"
            and isinstance(content, str)
            and ROUTER_MODE_MARKER in content
        ):
            continue
        retained.append(copied)
    return [
        {"role": "system", "content": router_system_prompt(catalog)},
        *retained,
    ]
