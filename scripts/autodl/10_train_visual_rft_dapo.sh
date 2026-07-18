#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
export TOOLVISTA_ROOT
export LOSS_TYPE="dapo"
export STAGE="${STAGE:-rl-dapo}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/dapo_seed${SEED:-42}}"
export WANDB_GROUP="${WANDB_GROUP:-rl-ab}"
ALGORITHM_ARGS=(
  --epsilon_high 0.28
  --dynamic_sample true
  --max_resample_times 3
  --overlong_filter true
)

source "${TOOLVISTA_ROOT}/scripts/autodl/rl_common.sh"
