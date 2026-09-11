"""Reviewable launch commands and fail-closed training preflight."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .checkpoints import file_info, verify_checkpoint
from .dependencies import MINIWOB_REVISION, check_chromium, check_versions, load_verl_api
from .gate import SMOKE_SUITE, SUITE_SHA256, TRAIN_TASKS
from .miniwob_env import local_miniwob_url

ROOT = Path(__file__).resolve().parents[2]


def launch_plan(checkpoint: Path, data_dir: Path, output: Path, *, mode: str, python: str = sys.executable,
                gpus: int = 1, steps: int = 100, algorithm: str = "grpo") -> dict:
    checkpoint, data_dir, output = checkpoint.resolve(), data_dir.resolve(), output.resolve()
    if mode not in {"dry-run", "smoke", "gate", "rollout", "train"}:
        raise ValueError("invalid launch mode")
    if algorithm not in ("grpo", "gigpo"):
        raise ValueError("algorithm must be grpo or gigpo")
    if gpus not in (1, 2, 4) or steps < 1:
        raise ValueError("gpus must be 1, 2, or 4; steps must be positive")
    if output.exists():
        raise FileExistsError(f"AgentRL output already exists: {output}")
    if output.is_relative_to(checkpoint) or checkpoint.is_relative_to(output):
        raise ValueError("training output must not overlap the merged checkpoint")
    manifest = verify_checkpoint(checkpoint)
    for filename in ("train.parquet", "val.parquet"):
        if not (data_dir / filename).is_file():
            raise ValueError(f"missing {filename}; run scripts/prepare_agentrl_data.py")
    gate_command = [python, str(ROOT / "scripts/agentrl_smoke.py"), "--checkpoint", str(checkpoint),
                    "--output", str(output / "cold-start"), "--gate"]
    module = "verl.trainer.main_ppo" if algorithm == "grpo" else "toolvista.agentrl.train_gigpo"
    command = [python, "-m", module, "--config-path", str(ROOT / "configs/verl"),
               "--config-name", f"miniwob_{algorithm}", f"trainer.n_gpus_per_node={gpus}",
               f"trainer.total_training_steps={steps}"]
    if mode == "rollout":
        command += ["trainer.val_only=true", "actor_rollout_ref.rollout.n=1", "data.val_batch_size=1"]
    environment = {"AGENTRL_MODEL": str(checkpoint), "AGENTRL_TRAIN_DATA": str(data_dir / "train.parquet"),
                   "AGENTRL_VAL_DATA": str(data_dir / "val.parquet"),
                   "AGENTRL_AGENT_CONFIG": str(ROOT / "configs/verl/miniwob_agent.yaml"),
                   "AGENTRL_OUTPUT_DIR": str(output / "checkpoints"),
                   "AGENTRL_ROLLOUT_DIR": str(output / "rollouts"),
                   "PYTHONPATH": str(ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
    return {"checkpoint_manifest_sha256": manifest["manifest_sha256"], "mode": mode,
            "algorithm": algorithm, "reward_mode": "terminal",
            "output_dir": str(output), "gate_command": gate_command, "training_command": command,
            "environment": environment}


def validate_dataset(data_dir: Path):
    import pyarrow.parquet as pq
    splits = []
    for filename in ("train.parquet", "val.parquet"):
        rows = pq.read_table(data_dir / filename).to_pylist()
        if not rows:
            raise ValueError(f"empty {filename}")
        identities = []
        for row in rows:
            extra = row.get("extra_info", {})
            identity = (extra.get("task_id"), extra.get("seed"))
            if (row.get("agent_name") != "toolvista_miniwob" or not row.get("prompt")
                    or extra.get("asset_revision") != MINIWOB_REVISION
                    or extra.get("browsergym_version") != "0.13.3"
                    or type(identity[1]) is not int or identity[1] < 0
                    or str(identity[0]) not in TRAIN_TASKS):
                raise ValueError(f"invalid AgentRL row in {filename}")
            identities.append(identity)
        if len(set(identities)) != len(identities) or set(identities).intersection(SMOKE_SUITE):
            raise ValueError("dataset duplicates tasks/seeds or overlaps the fixed gate")
        splits.append(set(identities))
    if splits[0].intersection(splits[1]):
        raise ValueError("training and validation tasks/seeds overlap")
    if os.environ.get("MINIWOB_ASSET_REVISION") != MINIWOB_REVISION:
        raise ValueError(f"set MINIWOB_ASSET_REVISION={MINIWOB_REVISION} after checking local assets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dry-run", "smoke", "gate", "rollout", "train"], default="dry-run")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--algorithm", choices=["grpo", "gigpo"], default="grpo")
    args = parser.parse_args()
    try:
        plan = launch_plan(args.checkpoint, args.data_dir, args.output, mode=args.mode,
                           gpus=args.gpus, steps=args.steps, algorithm=args.algorithm)
        # Dry-run validates artifacts and prints exact commands without requiring CUDA packages.
        if args.mode == "dry-run":
            print(json.dumps(plan, indent=2))
            return
        plan["versions"] = check_versions()
        load_verl_api()
        check_chromium()
        plan["miniwob_url"] = local_miniwob_url()
        validate_dataset(args.data_dir)
        plan["datasets"] = {name: file_info(args.data_dir / f"{name}.parquet") for name in ("train", "val")}
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        env = {**os.environ, **plan["environment"]}
        if args.mode in ("smoke", "gate", "train"):
            command = list(plan["gate_command"])
            if args.mode == "smoke":
                command.remove("--gate")
            subprocess.run(command, env=env, cwd=ROOT, check=True)
            if args.mode != "train":
                return
            report = json.loads((args.output / "cold-start/gate.json").read_text(encoding="utf-8"))
            if (report.get("passed") is not True or report.get("suite_sha256") != SUITE_SHA256
                    or report.get("checkpoint_manifest_sha256") != plan["checkpoint_manifest_sha256"]):
                raise ValueError("cold-start gate did not authorize this checkpoint")
            if verify_checkpoint(args.checkpoint)["manifest_sha256"] != plan["checkpoint_manifest_sha256"]:
                raise ValueError("checkpoint changed while running cold-start gate")
            if plan["datasets"] != {name: file_info(args.data_dir / f"{name}.parquet") for name in ("train", "val")}:
                raise ValueError("dataset changed while running cold-start gate")
        if args.mode == "rollout":
            import pyarrow as pa
            import pyarrow.parquet as pq
            from scripts.prepare_agentrl_data import task_rows
            path = args.output / "one-episode.parquet"
            pq.write_table(pa.Table.from_pylist(task_rows(range(100000, 100001), tasks=(SMOKE_SUITE[0][0],))), path)
            env["AGENTRL_VAL_DATA"] = str(path.resolve())
            plan["environment"]["AGENTRL_VAL_DATA"] = str(path.resolve())
            plan["rollout_dataset"] = file_info(path)
            (args.output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        subprocess.run(plan["training_command"], env=env, cwd=ROOT, check=True)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"AgentRL launch refused/failed: {exc}") from exc


if __name__ == "__main__":
    main()
