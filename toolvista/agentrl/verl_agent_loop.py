"""Hydra factory for the pinned verl AgentLoop; verl is imported only on use."""
from __future__ import annotations

from pathlib import Path
import json
import time
from uuid import uuid4

from .dependencies import load_verl_api
from .gigpo import episode_metadata
from .miniwob_env import AsyncMiniWoBEnv
from .prompts import SYSTEM_PROMPT, observation_message
from .rollout import ContextLimit, Generation, run_episode


class VerlPolicy:
    """Append observations without retokenizing any sampled action tokens."""
    def __init__(self, loop, sampling_params: dict, *, max_action_tokens: int = 256):
        self.loop = loop
        self.sampling_params = dict(sampling_params)
        self.max_action_tokens = max_action_tokens
        self.prompt_ids = []
        self.response_ids = []
        self.response_mask = []
        self.action_spans = []
        self.logprobs = []
        self.has_logprobs = None
        self.images = []
        self.request_id = uuid4().hex
        self.generation_seconds = 0.0
        self.prompt_length = loop.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = loop.config.actor_rollout_ref.rollout.response_length

    async def generate(self, observation: dict, history: list[str]) -> Generation:
        from PIL import Image

        with Image.open(observation["image_path"]) as image:
            image = image.convert("RGB")
        message = observation_message(observation)
        message["content"][0]["image"] = image
        vision = await self.loop.process_vision_info([message])
        new_images = vision.get("images", [])
        if len(new_images) != 1:
            raise RuntimeError("Qwen3-VL observation must produce exactly one image")
        first = not self.prompt_ids
        messages = ([{"role": "system", "content": SYSTEM_PROMPT}] if first else []) + [message]
        observation_ids = await self.loop.apply_chat_template(
            messages, images=new_images, remove_system_prompt=not first)
        remaining = self.response_length - len(self.response_ids) - (0 if first else len(observation_ids))
        if remaining < 1 or (first and len(observation_ids) > self.prompt_length):
            raise ContextLimit("screenshot history exceeds token budget")
        # Stage observations until generation succeeds. On a failed follow-up, the
        # episode reward must still land on the last policy token, not an observation.
        candidate_ids = observation_ids if first else self.prompt_ids + self.response_ids + observation_ids
        candidate_images = self.images + new_images
        params = {**self.sampling_params, "max_tokens": min(self.max_action_tokens, remaining)}
        started = time.monotonic()
        output = await self.loop.server_manager.generate(
            request_id=self.request_id, prompt_ids=candidate_ids,
            sampling_params=params, image_data=candidate_images)
        self.generation_seconds += time.monotonic() - started
        ids = list(output.token_ids)
        if not ids or len(ids) > params["max_tokens"]:
            raise RuntimeError("verl returned an empty or over-budget generation")
        has_logprobs = output.log_probs is not None
        if self.has_logprobs is not None and self.has_logprobs != has_logprobs:
            raise RuntimeError("verl changed log-probability availability between turns")
        if has_logprobs and len(output.log_probs) != len(ids):
            raise RuntimeError("verl token/log-probability lengths differ")
        if first:
            self.prompt_ids = list(observation_ids)
        else:
            self.response_ids.extend(observation_ids)
            self.response_mask.extend([0] * len(observation_ids))
            self.logprobs.extend([0.0] * len(observation_ids))
        self.images.extend(new_images)
        self.has_logprobs = has_logprobs
        self.action_spans.append((len(self.response_ids), len(self.response_ids) + len(ids)))
        self.response_ids.extend(ids)
        self.response_mask.extend([1] * len(ids))
        self.logprobs.extend(output.log_probs if has_logprobs else [0.0] * len(ids))
        return Generation(self.loop.tokenizer.decode(ids, skip_special_tokens=True), ids)


def create_agent_loop(*args, max_steps: int = 15, max_action_tokens: int = 256,
                      artifact_dir: str = "runs/agentrl/rollouts", reward_mode: str = "terminal", **kwargs):
    if reward_mode != "terminal":
        raise ValueError("GUI GRPO/GiGPO ablation requires terminal reward")
    api = load_verl_api()
    kwargs.pop("name", None)  # Registry key, not an AgentLoopBase argument.

    class BrowserGymAgentLoop(api.AgentLoopBase):
        async def run(self, sampling_params, **row):
            if self.processor is None:
                raise RuntimeError("MiniWoB AgentRL requires a Qwen3-VL multimodal processor")
            extra = row.get("extra_info", {})
            task_id, seed = extra["task_id"], int(extra["seed"])
            episode_id = uuid4().hex
            directory = Path(artifact_dir).resolve() / episode_id
            policy = VerlPolicy(self, sampling_params, max_action_tokens=max_action_tokens)
            env = AsyncMiniWoBEnv(directory, max_steps=max_steps)
            started = time.monotonic()
            record = await run_episode(env, policy, task_id=task_id, seed=seed, max_steps=max_steps,
                                       episode_id=episode_id, record_path=directory / "episode.json",
                                       reward_mode=reward_mode)
            if (record["failure_reason"] or "").startswith(("runtime:", "close:")):
                raise RuntimeError(f"GUI infrastructure failure: {record['failure_reason']}: {record.get('error', '')}")
            if not policy.prompt_ids or not any(policy.response_mask):
                # No policy tokens exist: a valid training sample cannot be fabricated.
                raise RuntimeError(f"MiniWoB episode produced no policy tokens: {record['failure_reason']}: {record.get('error', '')}")
            metadata = episode_metadata(record, policy.action_spans)
            (directory / "training.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return api.AgentLoopOutput(
                prompt_ids=policy.prompt_ids, response_ids=policy.response_ids,
                response_mask=policy.response_mask,
                response_logprobs=policy.logprobs if policy.has_logprobs else None,
                multi_modal_data={"images": policy.images}, reward_score=record["reward"],
                num_turns=2 * len(record["steps"]),
                metrics={"generate_sequences": policy.generation_seconds,
                         "tool_calls": max(0.0, time.monotonic() - started - policy.generation_seconds)},
                extra_fields={"episode_path": str(directory / "episode.json"),
                              "gui_episode": metadata,
                              "reward_extra_info": {"success": float(record["success"]),
                                                    "failure_reason": record["failure_reason"] or ""}})

    return BrowserGymAgentLoop(*args, **kwargs)
