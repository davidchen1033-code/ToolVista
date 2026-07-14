#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-python}"
C1_CHECKPOINT="${C1_CHECKPOINT:?Set C1_CHECKPOINT to the completed C1 checkpoint-* directory}"
TOOLVQA_DATASET="${TOOLVQA_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_train.jsonl}"
XLAM_DATASET="${XLAM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_train.jsonl}"
REPLAY_DATASET="${REPLAY_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_replay_train.jsonl}"
REPLAY_REPORT="${REPLAY_REPORT:-${REPLAY_DATASET%.jsonl}.report.json}"
REPLAY_SEED="${REPLAY_SEED:-${SEED:-42}}"
TOOLVQA_WEIGHT="${TOOLVQA_WEIGHT:-0.8}"
XLAM_WEIGHT="${XLAM_WEIGHT:-0.2}"

test -d "${C1_CHECKPOINT}"
test -s "${TOOLVQA_DATASET}"
test -s "${XLAM_DATASET}"

cd "${TOOLVISTA_ROOT}"
# Default weights: toolvqa=0.8 xlam=0.2
"${PYTHON_BIN}" scripts/build_sft_replay.py \
  --source "toolvqa=${TOOLVQA_DATASET}" \
  --source "xlam=${XLAM_DATASET}" \
  --weight "toolvqa=${TOOLVQA_WEIGHT}" \
  --weight "xlam=${XLAM_WEIGHT}" \
  --anchor toolvqa \
  --seed "${REPLAY_SEED}" \
  --output "${REPLAY_DATASET}" \
  --report "${REPLAY_REPORT}" \
  --overwrite

export TOOLVISTA_ROOT
export STAGE="${STAGE:-c2-toolvqa-replay}"
export DATASET="${REPLAY_DATASET}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c2_toolvqa_replay_sft}"
export WANDB_GROUP="${WANDB_GROUP:-c2-toolvqa-replay}"
export PREVIOUS_CHECKPOINT="${C1_CHECKPOINT}"

source "${TOOLVISTA_ROOT}/scripts/autodl/sft_common.sh"
