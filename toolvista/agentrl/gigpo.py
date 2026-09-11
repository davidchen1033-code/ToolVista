"""GiGPO's same-state Monte Carlo comparison, independent of torch/verl.

Equations 5–8: https://arxiv.org/abs/2505.10978. The episode baseline is
computed by upstream GRPO, with one vote per trajectory for a controlled ablation.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from statistics import mean, stdev


# Only used inside the browser worker. No DOM/privileged state enters the prompt.
# Strip BrowserGym instrumentation IDs; retain actual DOM IDs and live form state.
BROWSER_STATE_JS = """() => {
  const root = document.documentElement.cloneNode(true);
  for (const el of [root, ...root.querySelectorAll('*')]) {
    for (const attr of [...el.attributes]) {
      if (attr.name === 'bid' || attr.name.startsWith('browsergym_'))
        el.removeAttribute(attr.name);
    }
  }
  const elements = [...document.querySelectorAll('*')];
  return {
    url: location.href, dom: root.outerHTML,
    focus: elements.indexOf(document.activeElement),
    scroll: [scrollX, scrollY],
    controls: elements.map((e, i) => ({i, value: e.value ?? null,
      checked: e.checked ?? null, selected: e.selected ?? null,
      selectionStart: e.selectionStart ?? null, selectionEnd: e.selectionEnd ?? null,
      scroll: [e.scrollLeft, e.scrollTop]}))
  };
}"""


def state_anchor(observation: dict, browser_state: dict) -> str:
    if not browser_state.get("dom") or not observation.get("image_sha256"):
        raise ValueError("GiGPO requires DOM state and a screenshot hash")
    payload = {"version": 1, "goal": observation["goal"],
               "image": observation["image_sha256"], "feedback": observation.get("feedback", "ready"),
               "browser": browser_state}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def episode_metadata(record: dict, spans: list[tuple[int, int]]) -> dict:
    if record.get("reward_mode") != "terminal" or len(record["steps"]) != len(spans):
        raise ValueError("GUI steps must match generated action spans and use terminal reward")
    steps = []
    for step, (start, end) in zip(record["steps"], spans):
        anchor = step["observation"].get("state_anchor")
        if not anchor:
            raise ValueError("missing GUI state anchor")
        steps.append({"start": start, "end": end, "anchor": anchor, "reward": step["reward"]})
    return {"task_id": record["task_id"], "seed": record["seed"], "steps": steps}


def validate_metadata(episodes, masks, scores):
    """Reject silent step loss, reward mismatch, overlap, or observation supervision."""
    if not len(episodes) == len(masks) == len(scores):
        raise ValueError("GUI metadata batch length mismatch")
    for episode, mask, score in zip(episodes, masks, scores):
        if not isinstance(episode, dict) or not episode.get("steps"):
            raise ValueError("missing GUI episode metadata")
        steps = episode["steps"]
        covered = [0] * len(mask)
        previous_end = 0
        for step in steps:
            start, end = step["start"], step["end"]
            if (type(start) is not int or type(end) is not int
                    or not previous_end <= start < end <= len(mask)
                    or not isinstance(step["anchor"], str) or not step["anchor"]):
                raise ValueError("invalid GUI action span or state anchor")
            covered[start:end] = [1] * (end - start)
            previous_end = end
        if covered != list(mask):
            raise ValueError("GUI action spans do not exactly cover response_mask")
        rewards = [step["reward"] for step in steps]
        if any(r != 0 for r in rewards[:-1]) or rewards[-1] not in (0, 1) or rewards[-1] != score:
            raise ValueError("expected matching terminal-only GUI rewards")


def step_advantages(episodes, uids, *, gamma=1.0, normalize=True):
    """One group per (rollout uid, exact state), including repeated state visits."""
    if not math.isfinite(gamma) or not 0 <= gamma <= 1 or len(episodes) != len(uids):
        raise ValueError("invalid discount or episode/uid count")
    groups, identities = defaultdict(list), {}
    result = [[0.0] * len(ep["steps"]) for ep in episodes]
    for i, (episode, uid) in enumerate(zip(episodes, uids)):
        identity = (episode["task_id"], episode["seed"])
        if uid in identities and identities[uid] != identity:
            raise ValueError("rollout uid mixes different tasks/seeds")
        identities[uid] = identity
        value = 0.0
        for j in reversed(range(len(episode["steps"]))):
            step = episode["steps"][j]
            value = float(step["reward"]) + gamma * value
            groups[(uid, step["anchor"])].append((i, j, value))
    matched = cross_trajectory = nonzero = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        values = [item[2] for item in members]
        center = mean(values)
        scale = stdev(values) + 1e-6 if normalize else 1.0
        matched += len(members)
        cross_trajectory += len(members) if len({i for i, _, _ in members}) > 1 else 0
        for i, j, value in members:
            result[i][j] = (value - center) / scale
            nonzero += int(abs(result[i][j]) > 1e-8)
    count = sum(map(len, result))
    return result, {"steps": count, "groups": len(groups),
                    "matched_fraction": matched / max(count, 1),
                    "cross_trajectory_fraction": cross_trajectory / max(count, 1),
                    "nonzero_fraction": nonzero / max(count, 1)}
