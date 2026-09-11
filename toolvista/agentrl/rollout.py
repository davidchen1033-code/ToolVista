"""Backend-independent bounded episode driver; observations never expose DOM data."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from uuid import uuid4

from .actions import ActionError, PROTOCOL_VERSION, parse_action
from .rewards import episode_reward


class ContextLimit(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    token_ids: list[int] = field(default_factory=list)


async def run_episode(env, policy, *, task_id: str, seed: int, max_steps: int = 15,
                      episode_id: str | None = None, record_path: Path | None = None,
                      reward_mode: str = "legacy") -> dict:
    if type(max_steps) is not int or not 1 <= max_steps <= 100:
        raise ValueError("max_steps must be in 1..100")
    if reward_mode not in ("legacy", "terminal"):
        raise ValueError("reward_mode must be legacy or terminal")
    record = {"schema": "toolvista.episode.v1", "protocol": PROTOCOL_VERSION,
              "episode_id": episode_id or uuid4().hex, "task_id": task_id, "seed": seed,
              "max_steps": max_steps, "reward_mode": reward_mode, "steps": [], "success": False,
              "failure_reason": None, "loop_detected": False}
    history = []
    repeated = {}
    try:
        observation = await env.reset(task_id=task_id, seed=seed)
        record["environment"] = observation.get("environment", {})
        for index in range(max_steps):
            observation = {**observation, "step": index}
            generated = await policy.generate(observation, history)
            step = {"step_index": index, "observation": observation,
                    "assistant_action_text": generated.text, "parsed": False, "valid": False,
                    "action": None, "terminated": False, "truncated": False, "success": False}
            record["steps"].append(step)
            try:
                action = parse_action(generated.text)
                step.update(parsed=True, valid=True, action=action)
            except ActionError as exc:
                step.update(parsed=exc.parsed, error=str(exc))
                observation = {**observation, "feedback": str(exc)}
                history.append(generated.text)
                continue
            result = await env.step(action)
            step.update({key: result.get(key, default) for key, default in
                         (("terminated", False), ("truncated", False), ("success", False),
                          ("environment_reward", 0.0), ("failure_reason", None))})
            if result.get("invalid_action"):
                step.update(valid=False, error=result.get("feedback", "environment action error"))
            next_observation = result["observation"]
            step["next_observation"] = next_observation
            step["no_op"] = (observation["image_sha256"] == next_observation["image_sha256"]
                              and not step["terminated"])
            key = (observation["image_sha256"], json.dumps(action, sort_keys=True))
            repeated[key] = repeated.get(key, 0) + 1
            observation = next_observation
            history.append(generated.text)
            if step["failure_reason"]:
                record["failure_reason"] = step["failure_reason"]
                break
            if step["terminated"] or step["truncated"]:
                record["success"] = bool(step["terminated"] and step["success"])
                if step["truncated"] and not step["terminated"]:
                    record["failure_reason"] = "environment_timeout"
                break
            if action["action"] == "finish":
                record["failure_reason"] = "premature_finish"
                break
            if repeated[key] >= 3:
                record.update(loop_detected=True, failure_reason="action_loop")
                break
        else:
            record["failure_reason"] = "step_limit"
    except ContextLimit:
        record["failure_reason"] = "context_limit"
    except Exception as exc:
        record["failure_reason"] = f"runtime:{type(exc).__name__}"
        record["error"] = str(exc)
    finally:
        try:
            await env.close()
        except Exception as exc:
            record.update(failure_reason=f"close:{type(exc).__name__}", error=str(exc), success=False)
        record["reward"] = episode_reward(record, terminal_only=reward_mode == "terminal")
        if reward_mode == "terminal":
            for index, step in enumerate(record["steps"]):
                step["reward"] = record["reward"] if index == len(record["steps"]) - 1 else 0.0
        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record
