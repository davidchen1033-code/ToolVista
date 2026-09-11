# AutoDL experiment entrypoints

This directory has five experiment entrypoints. New scripts are named by action,
not by execution order:

```text
prepare_data.sh   Build or verify datasets.
train_sft.sh      Run C1-C3 SFT, optionally in tmux.
train_rl.sh       Run C4 GRPO-family training in tmux.
train_agent_rl.sh Run isolated verl + BrowserGym GRPO/GiGPO (--algorithm).
benchmark.sh      Merge, serve, and evaluate C3/C4 checkpoints in tmux.
```

Setup and resource downloads remain separate because they are not experiment
variants:

```text
setup_env.sh
setup_toolvqa_eval.sh
download_qwen3_vl.sh
download_toolvqa.sh
patch_msswift_vlmeval.sh
```

`lib/` contains implementation details shared by the entrypoints. Do not launch
those files directly for a new experiment. `legacy/` contains the old numbered
and one-off scripts for historical comparison; they are archived source, not the
current command interface.

## Variant names

Every new command uses one of these exact names:

| Variant | Meaning | Historical internal C3 name |
| --- | --- | --- |
| `raw` | Original uncorrected training/evaluation lineage | `raw` |
| `datafixed` | Explicit router-mode prompt plus direct-answer retention | `datafixed` |
| `datafixed-withnegative` | Datafixed C3 with agent/tool retention, followed by C4 negative samples | `datafixed-agentretention` |
| `datafixed-withnegative-terminal` | Previous variant plus the C3 terminal/action-format correction | `datafixed-agentretention-terminalfix` |

The variant names describe the end-to-end experiment lineage. C3 does not train
on Visual-RFT negative images; `withnegative` means that the C3 checkpoint is the
authorized starting point for the C4 dataset containing negative samples.

The historical names remain inside existing reports, gates, checkpoints, and
run summaries so previous results keep their identity. The unified launchers map
them to the canonical names without rewriting artifacts.

## Data preparation

Prepare the shared raw assets:

```bash
bash scripts/autodl/prepare_data.sh --stage c0 --variant raw
```

Prepare a C3 lineage:

```bash
bash scripts/autodl/prepare_data.sh --stage c3 --variant datafixed
bash scripts/autodl/prepare_data.sh --stage c3 --variant datafixed-withnegative
bash scripts/autodl/prepare_data.sh --stage c3 --variant datafixed-withnegative-terminal
```

Verify the canonical C4 negative dataset without rebuilding it:

```bash
bash scripts/autodl/prepare_data.sh \
  --stage c4 \
  --variant datafixed-withnegative-terminal \
  --verify-only
```

Use `--overwrite` only for deterministic builders whose existing outputs are
intentionally being regenerated.

## SFT

C1 and C2 use the raw lineage:

```bash
bash scripts/autodl/train_sft.sh --stage c1 --variant raw
bash scripts/autodl/train_sft.sh --stage c2 --variant raw
```

C3 accepts all four variants:

```bash
bash scripts/autodl/train_sft.sh \
  --stage c3 \
  --variant datafixed-withnegative-terminal
```

The command launches tmux and enters the `train` window by default. Use
`--detach` for a background launch. The default windows are:

```text
0 train   Live step/max_steps, percentage, loss, reward, and ETA.
1 log     The persisted training log.
2 gpu     nvidia-smi refreshed every five seconds.
```

The completed formal C2 checkpoint is the default C3 base. Override
`C1_CHECKPOINT`, `C2_CHECKPOINT`, `OUTPUT_DIR`, `MAX_STEPS`, or
`LEARNING_RATE` explicitly when starting a different controlled run.

## RL

One launcher covers all four algorithms:

```bash
bash scripts/autodl/train_rl.sh \
  --variant datafixed-withnegative-terminal \
  --algorithm dapo
```

Valid algorithms and their only algorithm-specific settings are:

| Algorithm | ms-swift `loss_type` | Additional settings |
| --- | --- | --- |
| `grpo` | `grpo` | none |
| `bnpo` | `bnpo` | none |
| `dapo` | `dapo` | `epsilon_high=0.28`, dynamic resampling, overlong filtering |
| `drgrpo` | `dr_grpo` | `scale_rewards=none` |

Examples for a strict four-way comparison:

```bash
bash scripts/autodl/train_rl.sh --variant datafixed-withnegative-terminal --algorithm grpo
bash scripts/autodl/train_rl.sh --variant datafixed-withnegative-terminal --algorithm bnpo
bash scripts/autodl/train_rl.sh --variant datafixed-withnegative-terminal --algorithm dapo
bash scripts/autodl/train_rl.sh --variant datafixed-withnegative-terminal --algorithm drgrpo
```

The terminal comparison fixes `seed=42`, `max_steps=6000`, the C3 step-1700
checkpoint, the canonical negative dataset, and the two with-negative reward
functions. The dataset SHA-256 is:

```text
95116f0ca0598bd6b587576992d209ad69677e2e7600447e18798e1d1e503f16
```

Each run refuses to overwrite its output, log, or provenance manifest. Use
`RUN_TAG`, `SESSION`, `OUTPUT_DIR`, and `TRAIN_LOG` to name a controlled rerun.

## Benchmark

C3 benchmark:

```bash
bash scripts/autodl/benchmark.sh \
  --stage c3 \
  --variant datafixed-withnegative-terminal
```

C4 benchmark:

```bash
bash scripts/autodl/benchmark.sh \
  --stage c4 \
  --variant datafixed-withnegative-terminal \
  --algorithm dapo \
  --scope light
```

C4 also supports the corresponding full suite. C3 remains light-only:

```bash
bash scripts/autodl/benchmark.sh \
  --stage c4 \
  --variant datafixed-withnegative-terminal \
  --algorithm dapo \
  --scope full
```

`--scope` defaults to `light`. Full runs do not inherit a light C4 baseline.
For datafixed variants, the passed C3 light summary is still checked as the
checkpoint/judge authorization chain, but it is not used as a full-metric
history comparison. Set `BASELINE_SUMMARY` explicitly only to a matching full
run.

Completed formal C3, DAPO, GRPO, and BNPO checkpoints have reuse defaults.
For a new checkpoint, including the currently running DRGRPO experiment, set
`ADAPTER` explicitly after `checkpoint-6000` exists. `MERGED` may point to an
existing verified merged model; otherwise the launcher creates it atomically.

The benchmark tmux session opens directly on the live six-component percentage
monitor:

```text
0 run       Merge, candidate service, judge service, and evaluator lifecycle.
1 monitor   Per-component completed/total counts and percentages.
2 logs      Orchestrator, candidate, judge, and benchmark logs.
3 gpu       nvidia-smi refreshed every five seconds.
```

Common overrides are `ADAPTER`, `MERGED`, `HISTORY_SUMMARY`,
`BASELINE_SUMMARY`, `RUN_ID`, `SESSION`, and `API_MODEL`. The launcher refuses
to start while training, inference, evaluation, or another GPU process is active.

## Non-mutating checks

All train and benchmark launchers support a preflight-only check:

```bash
DRY_RUN=1 bash scripts/autodl/train_rl.sh \
  --variant datafixed-withnegative-terminal \
  --algorithm dapo \
  --detach

DRY_RUN=1 bash scripts/autodl/benchmark.sh \
  --stage c4 \
  --variant datafixed-withnegative-terminal \
  --algorithm dapo \
  --detach
```

Dry-run validates the selected artifacts and prints the resolved contract without
starting tmux, changing a run directory, or allocating the GPU.

## Environment-feedback AgentRL (verl)

See [verl + BrowserGym MiniWoB usage](../../docs/agentrl_verl_usage.md) for the isolated
`train_agent_rl.sh` launcher, checkpoint export, GUI cold-start gate, and optional GUI SFT.
Use `--algorithm gigpo` or the thin `train_agent_gigpo.sh` wrapper for GiGPO on the same data.
Install the shared pinned AgentRL environment with `bash scripts/autodl/setup_verl_agent.sh`.
