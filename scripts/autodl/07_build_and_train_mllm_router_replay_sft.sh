#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C2_CHECKPOINT="${C2_CHECKPOINT:?Set C2_CHECKPOINT to the completed C2 checkpoint-* directory}"
MLLM_DATASET="${MLLM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_tool_router_train.jsonl}"
TOOLVQA_DATASET="${TOOLVQA_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_train.jsonl}"
XLAM_DATASET="${XLAM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_train.jsonl}"
REPLAY_DATASET="${REPLAY_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_router_replay_train.jsonl}"
REPLAY_REPORT="${REPLAY_REPORT:-${REPLAY_DATASET%.jsonl}.report.json}"
REPLAY_SEED="${REPLAY_SEED:-${SEED:-42}}"
MLLM_WEIGHT="${MLLM_WEIGHT:-0.6}"
TOOLVQA_WEIGHT="${TOOLVQA_WEIGHT:-0.2}"
XLAM_WEIGHT="${XLAM_WEIGHT:-0.2}"

test -d "${C2_CHECKPOINT}"
test -s "${MLLM_DATASET}"
test -s "${TOOLVQA_DATASET}"
test -s "${XLAM_DATASET}"

cd "${TOOLVISTA_ROOT}"
# Default weights: mllm=0.6 toolvqa=0.2 xlam=0.2
"${PYTHON_BIN}" scripts/build_sft_replay.py \
  --source "mllm=${MLLM_DATASET}" \
  --source "toolvqa=${TOOLVQA_DATASET}" \
  --source "xlam=${XLAM_DATASET}" \
  --weight "mllm=${MLLM_WEIGHT}" \
  --weight "toolvqa=${TOOLVQA_WEIGHT}" \
  --weight "xlam=${XLAM_WEIGHT}" \
  --anchor mllm \
  --seed "${REPLAY_SEED}" \
  --output "${REPLAY_DATASET}" \
  --report "${REPLAY_REPORT}" \
  --overwrite

export TOOLVISTA_ROOT
export STAGE="${STAGE:-c3-router-replay}"
export DATASET="${REPLAY_DATASET}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_replay_sft}"
export WANDB_GROUP="${WANDB_GROUP:-c3-router-replay}"
export PREVIOUS_CHECKPOINT="${C2_CHECKPOINT}"

source "${TOOLVISTA_ROOT}/scripts/autodl/sft_common.sh"
