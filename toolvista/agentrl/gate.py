"""Fixed, checkpoint-bound GUI cold-start gate."""
from __future__ import annotations

import hashlib
import json

from .actions import ActionError, PROTOCOL_VERSION, parse_action
from .rewards import episode_reward

TASKS = ("browsergym/miniwob.click-test", "browsergym/miniwob.click-button", "browsergym/miniwob.enter-text")
# Pilot/main AgentRL tasks are deliberately limited to the action protocol implemented
# here. Some other MiniWoB++ tasks need unsupported actions or reward auditing.
# click-checkboxes is excluded: its partial-credit reward is binarized by BrowserGym.
TRAIN_TASKS = (
    "browsergym/miniwob.click-test", "browsergym/miniwob.click-button",
    "browsergym/miniwob.click-link", "browsergym/miniwob.click-option",
    "browsergym/miniwob.login-user", "browsergym/miniwob.click-tab",
    "browsergym/miniwob.click-dialog", "browsergym/miniwob.click-button-sequence",
    "browsergym/miniwob.enter-text", "browsergym/miniwob.enter-text-2",
    "browsergym/miniwob.enter-password", "browsergym/miniwob.focus-text",
)
SMOKE_SUITE = tuple((task, seed) for task in TASKS for seed in range(10))
SUITE_SHA256 = hashlib.sha256(json.dumps(SMOKE_SUITE).encode()).hexdigest()


def evaluate_gate(records: list[dict], manifest_sha256: str, *, expected_suite=SMOKE_SUITE) -> dict:
    actions = parsed = valid = successes = 0
    infrastructure_failures = []
    for record in records:
        if record.get("protocol") != PROTOCOL_VERSION:
            infrastructure_failures.append("action_protocol_mismatch")
        reason = record.get("failure_reason") or ""
        if reason.startswith(("runtime:", "close:")):
            infrastructure_failures.append(reason)
        for step in record.get("steps", []):
            actions += 1
            try:
                parse_action(step["assistant_action_text"])
                parsed += 1
                valid += 1
            except ActionError as exc:
                parsed += int(exc.parsed)
        successes += int(record.get("success") is True and episode_reward(record) > 0)
    observed = [(record["task_id"], record["seed"]) for record in records]
    coverage = sorted(observed) == sorted(expected_suite)
    parse_rate = parsed / actions if actions else 0.0
    validity = valid / parsed if parsed else 0.0
    passed = coverage and not infrastructure_failures and parse_rate >= 0.95 and validity >= 0.95 and successes > 0
    return {"schema": "toolvista.gui_gate.v1", "protocol": PROTOCOL_VERSION,
            "checkpoint_manifest_sha256": manifest_sha256, "suite_sha256": SUITE_SHA256,
            "episodes": len(records), "actions": actions, "json_parse_rate": parse_rate,
            "action_validity_rate": validity, "successes": successes,
            "success_rate": successes / len(records) if records else 0.0,
            "suite_complete": coverage, "infrastructure_failures": infrastructure_failures,
            "passed": passed, "recommendation": "ready_for_agentrl" if passed else
            ("fix_environment_or_inference" if infrastructure_failures or not coverage else "gui_sft_or_curriculum_required")}
