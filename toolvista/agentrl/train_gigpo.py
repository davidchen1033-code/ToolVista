"""Isolated verl 0.7.0 entry point; no fork installation or upstream file edits."""
from __future__ import annotations

from functools import partial
import json
import math
from pathlib import Path
from unittest.mock import patch

from .gigpo import step_advantages, validate_metadata


def compute_advantage(original, data, adv_estimator, *, step_weight=1.0, **kwargs):
    if adv_estimator != "gigpo":
        return original(data, adv_estimator=adv_estimator, **kwargs)
    if not math.isfinite(step_weight) or step_weight < 0:
        raise ValueError("step_weight must be finite and nonnegative")
    episodes = data.non_tensor_batch.get("gui_episode", [])
    mask = data.batch["response_mask"]
    rewards = data.batch["token_level_rewards"]
    validate_metadata(episodes, mask.tolist(), rewards.sum(-1).tolist())
    # Exact upstream episode normalization / loss mask, one vote per trajectory.
    data = original(data, adv_estimator="grpo", **kwargs)
    values, metrics = step_advantages(
        episodes, data.non_tensor_batch["uid"], gamma=kwargs.get("gamma", 1.0),
        normalize=kwargs.get("norm_adv_by_std_in_grpo", True))
    advantages = data.batch["advantages"].clone()
    for i, (episode, row) in enumerate(zip(episodes, values)):
        for step, value in zip(episode["steps"], row):
            advantages[i, step["start"]:step["end"]] += step_weight * value
    data.batch["advantages"] = advantages * mask
    data.batch["returns"] = data.batch["advantages"].clone()
    print("toolvista/gigpo " + json.dumps({**metrics, "step_weight": step_weight}, sort_keys=True))
    return data


def main():
    from .dependencies import load_verl_api
    load_verl_api()
    import hydra
    import ray
    from verl.trainer.main_ppo import TaskRunner, run_ppo
    from verl.trainer.ppo import ray_trainer
    from verl.utils.device import auto_set_device

    class GuiTaskRunner(TaskRunner):
        def run(self, config):
            if config.algorithm.adv_estimator != "gigpo" or config.algorithm.use_kl_in_reward:
                raise ValueError("GiGPO entry requires gigpo and terminal-only task rewards")
            wrapped = partial(compute_advantage, ray_trainer.compute_advantage,
                              step_weight=config.gigpo.step_weight)
            # The upstream registry cannot receive non-tensor step metadata. Scope
            # this compatibility hook to this Ray controller only, restoring on exit.
            with patch.object(ray_trainer, "compute_advantage", wrapped):
                return super().run(config)

    @hydra.main(config_path=str(Path(__file__).resolve().parents[2] / "configs/verl"),
                config_name="miniwob_gigpo", version_base=None)
    def entry(config):
        auto_set_device(config)
        run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(GuiTaskRunner))

    entry()


if __name__ == "__main__":
    main()
