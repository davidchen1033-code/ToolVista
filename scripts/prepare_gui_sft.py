"""Convert verified successful GUI episodes to ms-swift SFT with ToolVista replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolvista.agentrl.actions import PROTOCOL_VERSION, parse_action
from toolvista.agentrl.checkpoints import sha256
from toolvista.agentrl.prompts import SYSTEM_PROMPT, observation_text
from toolvista.agentrl.rewards import episode_reward


def gui_rows(records: list[dict]) -> list[dict]:
    result, seen = [], set()
    for record in records:
        if record.get("protocol") != PROTOCOL_VERSION or record.get("success") is not True or episode_reward(record) <= 0:
            raise ValueError("GUI SFT requires verified successful episode records")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        images, hashes = [], []
        for step in record["steps"]:
            if not step.get("valid"):
                raise ValueError("successful episode contains invalid actions; curate before SFT")
            action = parse_action(step["assistant_action_text"])
            observation = step["observation"]
            path = Path(observation["image_path"]).resolve(strict=True)
            if sha256(path) != observation["image_sha256"]:
                raise ValueError(f"screenshot hash mismatch: {path}")
            images.append(str(path))
            hashes.append(observation["image_sha256"])
            messages.append({"role": "user", "content": "<image>\n" + observation_text(observation)})
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))})
            key = json.dumps([hashes, messages], sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                sample_messages = [dict(message) for message in messages]
                for message in sample_messages[:-1]:
                    if message["role"] == "assistant":
                        message["loss_scale"] = 0.0
                sample_messages[-1]["loss_scale"] = 1.0
                result.append({"messages": sample_messages, "images": list(images)})
    if not result:
        raise ValueError("no verified GUI state-action examples")
    return result


def mix_replay(gui: list[dict], replay: list[dict], fraction: float = 0.2, seed: int = 42) -> list[dict]:
    if not replay or not 0 < fraction < 1:
        raise ValueError("nonempty ToolVista replay and a fraction in (0,1) are required")
    for row in replay:
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise ValueError("replay must use the ms-swift messages format")
    rng = random.Random(seed)
    count = max(1, round(len(gui) * fraction / (1 - fraction)))
    result = [*gui, *rng.choices(replay, k=count)]
    rng.shuffle(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--replay-fraction", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gui-examples", type=int, default=20000)
    args = parser.parse_args()
    if args.output.exists() or args.max_gui_examples < 1:
        parser.error("output must be new and max-gui-examples positive")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.episodes]
    from toolvista.agentrl.gate import SMOKE_SUITE
    if any((record.get("task_id"), record.get("seed")) in SMOKE_SUITE for record in records):
        parser.error("GUI SFT demonstrations must not overlap the fixed cold-start gate task/seeds")
    rows = gui_rows(records)
    random.Random(42).shuffle(rows)
    rows = rows[:args.max_gui_examples]
    # Existing exporters use absolute image paths. Resolve any relative replay image path against its JSONL.
    replay = [json.loads(line) for line in args.replay.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in replay:
        if row.get("images"):
            row["images"] = [str((args.replay.parent / image).resolve(strict=True)) for image in row["images"]]
    mixed = mix_replay(rows, replay, args.replay_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mixed), encoding="utf-8")
    print(json.dumps({"gui_examples": len(rows), "total_examples": len(mixed),
                      "recommended_gui_examples": "5000-20000", "recommended_epochs": "0.5-1"}))


if __name__ == "__main__":
    main()
