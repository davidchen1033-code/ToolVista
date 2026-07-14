#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the model currently served by the OpenAI-compatible API}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID for this evaluation}"
STAGE="${STAGE:?Set the training stage name}"
SUITE="${SUITE:-c3_light}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
API_MODEL="${API_MODEL:?Set API_MODEL to the model name exposed by the serving endpoint}"
SEED="${SEED:-42}"
WANDB_PROJECT="${WANDB_PROJECT:-toolvista-qwen3vl-eval}"
WANDB_GROUP="${WANDB_GROUP:-${STAGE}}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"
HISTORY_SUMMARY="${HISTORY_SUMMARY:-}"
EVAL_URL="${EVAL_URL:-${API_BASE}}"

test -d "${TOOLVISTA_ROOT}"
test -e "${CHECKPOINT}"

comparison_args=()
if [[ -n "${BASELINE_SUMMARY}" ]]; then
  test -s "${BASELINE_SUMMARY}"
  comparison_args+=(--baseline-summary "${BASELINE_SUMMARY}")
fi
if [[ -n "${HISTORY_SUMMARY}" ]]; then
  test -s "${HISTORY_SUMMARY}"
  comparison_args+=(--history-summary "${HISTORY_SUMMARY}")
fi

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/evaluate.py \
  --suite "${SUITE}" \
  --checkpoint "${CHECKPOINT}" \
  --run-id "${RUN_ID}" \
  --stage "${STAGE}" \
  --api-base "${API_BASE}" \
  --api-model "${API_MODEL}" \
  --eval-url "${EVAL_URL}" \
  --seed "${SEED}" \
  "${comparison_args[@]}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}"
