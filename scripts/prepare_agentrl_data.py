"""Build deterministic MiniWoB task/seed rows for verl; no demonstrations required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolvista.agentrl.gate import TRAIN_TASKS
from toolvista.agentrl.dependencies import MINIWOB_REVISION


def task_rows(seeds: range, tasks=TRAIN_TASKS) -> list[dict]:
    return [{"data_source": "toolvista_miniwob", "agent_name": "toolvista_miniwob",
             "prompt": [{"role": "user", "content": "Complete the task in the browser screenshot."}],
             "ability": "gui", "reward_model": {"style": "rule", "ground_truth": "environment"},
             "extra_info": {"index": index, "task_id": task, "seed": seed,
                            "browsergym_version": "0.13.3", "asset_revision": MINIWOB_REVISION}}
            for index, (task, seed) in enumerate((task, seed) for task in tasks for seed in seeds)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true", help="print dataset manifest without pyarrow or file writes")
    parser.add_argument("--tasks", default=",".join(TRAIN_TASKS),
                        help="comma-separated registered MiniWoB task ids")
    args = parser.parse_args()
    if not 1 <= args.train_seeds <= 10000:
        parser.error("--train-seeds must be in 1..10000")
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in TRAIN_TASKS for task in tasks):
        parser.error("--tasks must be unique and only contain the supported first-stage TRAIN_TASKS")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    manifest = {"schema": "toolvista.miniwob12.v1", "tasks": tasks,
        "browsergym_version": "0.13.3", "asset_revision": MINIWOB_REVISION,
        "reward": "terminal_success_1_failure_0", "observation": "screenshot_goal_history",
        "train_seed_start": 100, "train_seeds": args.train_seeds,
        "validation_seeds": [100000, 100009], "gate_seeds": [0, 9],
        "train_rows": len(tasks) * args.train_seeds, "validation_rows": len(tasks) * 10}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    args.output_dir.mkdir(parents=True)
    for name, seeds in (("train", range(100, 100 + args.train_seeds)), ("val", range(100000, 100010))):
        rows = task_rows(seeds, tasks=tasks)
        pq.write_table(pa.Table.from_pylist(rows), args.output_dir / f"{name}.parquet")
    manifest["file_bytes"] = {name: (args.output_dir / name).stat().st_size
                              for name in ("train.parquet", "val.parquet")}
    (args.output_dir / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
