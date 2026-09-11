"""Run one checkpoint/browser rollout or the fixed GUI SFT gate."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolvista.agentrl.checkpoints import verify_checkpoint
from toolvista.agentrl.dependencies import check_chromium, check_versions, load_verl_api
from toolvista.agentrl.gate import SMOKE_SUITE, TASKS, evaluate_gate
from toolvista.agentrl.miniwob_env import AsyncMiniWoBEnv, local_miniwob_url
from toolvista.agentrl.rollout import run_episode


async def smoke(args, manifest):
    from toolvista.agentrl.hf_policy import TransformersPolicy
    args.output.mkdir(parents=True, exist_ok=False)
    policy = TransformersPolicy(args.checkpoint)
    records = []
    suite = SMOKE_SUITE if args.gate else [(args.task, args.seed)]
    for index, (task_id, seed) in enumerate(suite):
        policy.reset()
        directory = args.output / f"episode-{index:03d}"
        record = await run_episode(AsyncMiniWoBEnv(directory, max_steps=args.max_steps), policy,
                                   task_id=task_id, seed=seed, max_steps=args.max_steps,
                                   record_path=directory / "episode.json")
        records.append(record)
        print(json.dumps({"episode": index, "task": task_id, "success": record["success"],
                          "reward": record["reward"], "failure_reason": record["failure_reason"]}), flush=True)
    if args.gate:
        report = evaluate_gate(records, manifest["manifest_sha256"])
        report["max_steps"] = args.max_steps
        (args.output / "gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit("GUI cold-start gate failed; read gate.json before AgentRL training")
    elif any((r["failure_reason"] or "").startswith(("runtime:", "close:")) or not r["steps"] for r in records):
        raise SystemExit("Smoke rollout failed; inspect episode.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--task", choices=TASKS, default=TASKS[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15)
    args = parser.parse_args()
    try:
        versions = check_versions()
        load_verl_api()
        local_miniwob_url()
        check_chromium()
        if args.preflight:
            print(json.dumps(versions, indent=2))
            return
        if args.checkpoint is None or args.output is None:
            parser.error("--checkpoint and --output are required")
        if not 1 <= args.max_steps <= 100:
            parser.error("--max-steps must be in 1..100")
        manifest = verify_checkpoint(args.checkpoint)
        asyncio.run(smoke(args, manifest))
    except (ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f"AgentRL preflight failed: {exc}") from exc


if __name__ == "__main__":
    main()
