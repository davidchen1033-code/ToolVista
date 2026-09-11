"""Pure, bounded reward computation from recorded environment evidence."""
from __future__ import annotations


def episode_reward(record: dict, *, process_shaping: bool = False, terminal_only: bool = False) -> float:
    steps = record.get("steps", [])
    terminal_success = bool(steps and steps[-1].get("terminated")
                            and steps[-1].get("success") and not record.get("failure_reason"))
    if terminal_only:
        return float(terminal_success)
    invalid = min(sum(not step.get("valid", False) for step in steps), 5)
    noop = min(sum(bool(step.get("no_op")) for step in steps), 5)
    looping = 0.05 if record.get("loop_detected") else 0.0
    shaping = 0.0
    if process_shaping:
        shaping = min(0.2, sum(max(0.0, min(0.2, float(step.get("verified_intermediate_reward", 0))))
                               for step in steps if not step.get("terminated")))
    return round(float(terminal_success) - 0.02 * invalid - 0.02 * noop - looping + shaping, 6)
