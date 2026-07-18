#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
export TOOLVISTA_ROOT
export STAGE="${STAGE:-c1-xlam}"
export DATASET="${XLAM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_train.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c1_xlam_sft}"
export WANDB_GROUP="${WANDB_GROUP:-c1-xlam}"
export PREVIOUS_CHECKPOINT=""

source "${TOOLVISTA_ROOT}/scripts/autodl/sft_common.sh"
