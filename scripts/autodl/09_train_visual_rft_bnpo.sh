#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
export TOOLVISTA_ROOT
export LOSS_TYPE="bnpo"
export STAGE="${STAGE:-rl-bnpo}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/rl/bnpo_seed${SEED:-42}}"
export WANDB_GROUP="${WANDB_GROUP:-rl-ab}"
ALGORITHM_ARGS=()

source "${TOOLVISTA_ROOT}/scripts/autodl/rl_common.sh"
