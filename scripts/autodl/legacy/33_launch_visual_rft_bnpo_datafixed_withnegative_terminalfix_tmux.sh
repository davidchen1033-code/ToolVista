#!/usr/bin/env bash
set -Eeuo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
RUN_TAG="${RUN_TAG:-terminalfix-bnpo$(date -u +%Y%m%dT%H%M%SZ)}"

export ALGORITHM=bnpo
export LOSS_TYPE=bnpo
export TRAIN_LAUNCHER="${TOOLVISTA_ROOT}/scripts/autodl/09_train_visual_rft_bnpo.sh"
export SESSION="${SESSION:-c4-datafixed-withnegative-terminalfix-bnpo}"
export RUN_TAG
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/bnpo_datafixed-withnegative_seed42_run${RUN_TAG}}"
export TRAIN_LOG="${TRAIN_LOG:-${TOOLVISTA_ROOT}/logs/c4/c4-bnpo-datafixed-withnegative-${RUN_TAG}-seed42.train.log}"
export STAGE="${STAGE:-rl-bnpo-datafixed-withnegative-terminalfix}"
export WANDB_GROUP="${WANDB_GROUP:-rl-bnpo-datafixed-withnegative-terminalfix}"

exec bash "${TOOLVISTA_ROOT}/scripts/autodl/31_launch_visual_rft_grpo_datafixed_withnegative_terminalfix_tmux.sh"
