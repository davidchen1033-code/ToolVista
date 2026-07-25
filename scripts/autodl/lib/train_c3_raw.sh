#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
C2_CHECKPOINT="${C2_CHECKPOINT:?Set C2_CHECKPOINT to the completed C2 checkpoint-* directory}"
MLLM_DATASET="${MLLM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_tool_router_v2_train.jsonl}"
DIRECT_ANSWER_DATASET="${DIRECT_ANSWER_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_direct_answer_retention_train.jsonl}"
TOOLVQA_DATASET="${TOOLVQA_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/toolvqa_train.jsonl}"
XLAM_DATASET="${XLAM_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/xlam_train.jsonl}"
REPLAY_DATASET="${REPLAY_DATASET:-${TOOLVISTA_ROOT}/data/exports/msswift_sft/mllm_router_v2_replay_train.jsonl}"
REPLAY_REPORT="${REPLAY_REPORT:-${REPLAY_DATASET%.jsonl}.report.json}"
REPLAY_SEED="${REPLAY_SEED:-${SEED:-42}}"
MLLM_WEIGHT="${MLLM_WEIGHT:-0.4}"
DIRECT_ANSWER_WEIGHT="${DIRECT_ANSWER_WEIGHT:-0.3}"
TOOLVQA_WEIGHT="${TOOLVQA_WEIGHT:-0.15}"
XLAM_WEIGHT="${XLAM_WEIGHT:-0.15}"

test -d "${C2_CHECKPOINT}"
test -s "${MLLM_DATASET}"
test -s "${DIRECT_ANSWER_DATASET}"
test -s "${TOOLVQA_DATASET}"
test -s "${XLAM_DATASET}"

cd "${TOOLVISTA_ROOT}"
# Router mode is explicit in MLLM v2. Direct-answer rehearsal protects ordinary
# VQA behavior when no router marker/catalog is present.
"${PYTHON_BIN}" scripts/build_sft_replay.py \
  --source "mllm=${MLLM_DATASET}" \
  --source "direct=${DIRECT_ANSWER_DATASET}" \
  --source "toolvqa=${TOOLVQA_DATASET}" \
  --source "xlam=${XLAM_DATASET}" \
  --weight "mllm=${MLLM_WEIGHT}" \
  --weight "direct=${DIRECT_ANSWER_WEIGHT}" \
  --weight "toolvqa=${TOOLVQA_WEIGHT}" \
  --weight "xlam=${XLAM_WEIGHT}" \
  --anchor mllm \
  --seed "${REPLAY_SEED}" \
  --output "${REPLAY_DATASET}" \
  --report "${REPLAY_REPORT}" \
  --overwrite

export TOOLVISTA_ROOT
export STAGE="${STAGE:-c3-router-v2-retention}"
export DATASET="${REPLAY_DATASET}"
export OUTPUT_DIR="${OUTPUT_DIR:-${TOOLVISTA_ROOT}/outputs/c3_mllm_router_v2_sft}"
export WANDB_GROUP="${WANDB_GROUP:-c3-router-v2-retention}"
export PREVIOUS_CHECKPOINT="${C2_CHECKPOINT}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export MAX_STEPS="${MAX_STEPS:-800}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"

source "${TOOLVISTA_ROOT}/scripts/autodl/lib/sft_common.sh"
