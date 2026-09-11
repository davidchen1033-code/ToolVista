"""Strict, small GUI action protocol. Never execute model-supplied Python."""
from __future__ import annotations

import json
import math

PROTOCOL_VERSION = "toolvista.gui.v1"
KEYS = {"Enter", "Tab", "Escape", "Backspace", "Delete", "Space", "ArrowUp",
        "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "Control+A"}
FIELDS = {"click": {"action", "x", "y"}, "type": {"action", "text"},
          "scroll": {"action", "dx", "dy"}, "press": {"action", "key"},
          "finish": {"action", "status"}}


class ActionError(ValueError):
    def __init__(self, message: str, *, parsed: bool = True):
        super().__init__(message)
        self.parsed = parsed


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_action(text: str) -> dict:
    try:
        if not isinstance(text, str) or len(text) > 8192:
            raise ValueError("action must be at most 8192 characters")
        value = json.loads(text, object_pairs_hook=_unique,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError) as exc:
        raise ActionError(f"invalid JSON: {exc}", parsed=False) from exc
    if not isinstance(value, dict):
        raise ActionError("action must be a JSON object")
    kind = value.get("action")
    if not isinstance(kind, str) or kind not in FIELDS or set(value) != FIELDS[kind]:
        raise ActionError("unknown action or incorrect fields")
    bounds = {"click": ("x", "y", 0, 1), "scroll": ("dx", "dy", -1000, 1000)}
    if kind in bounds:
        first, second, low, high = bounds[kind]
        for key in (first, second):
            number = value[key]
            if type(number) not in (int, float) or not math.isfinite(number) or not low <= number <= high:
                raise ActionError(f"{key} must be a finite number in [{low}, {high}]")
    if kind == "type" and (not isinstance(value["text"], str) or not 1 <= len(value["text"]) <= 2000):
        raise ActionError("text must contain 1..2000 characters")
    if kind == "press" and (not isinstance(value["key"], str) or value["key"] not in KEYS):
        raise ActionError("unsupported key")
    if kind == "finish" and value["status"] not in ("success", "failure"):
        raise ActionError("finish status must be success or failure")
    return value


def browsergym_action(action: dict, width: int, height: int) -> str:
    action = parse_action(json.dumps(action))
    kind = action["action"]
    if kind == "click":
        return f"mouse_click({round(action['x'] * (width - 1))}, {round(action['y'] * (height - 1))})"
    if kind == "type":
        return f"keyboard_type({action['text']!r})"
    if kind == "scroll":
        return f"scroll({action['dx']}, {action['dy']})"
    if kind == "press":
        return f"keyboard_press({action['key']!r})"
    return "noop(0)"  # Re-run the environment evaluator; finish cannot award itself success.
