#!/usr/bin/env bash
set -euo pipefail

TOOLVISTA_ROOT="${TOOLVISTA_ROOT:-/root/autodl-tmp/ToolVista}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/toolvista/bin/python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the model currently served by the OpenAI-compatible API}"
RUN_ID="${RUN_ID:?Set a unique RUN_ID for this evaluation}"
STAGE="${STAGE:?Set the training stage name}"
ALGORITHM="${ALGORITHM:-}"
SUITE="${SUITE:?Set SUITE to c0_light, c1_light, c2_light, c3_light, c4_light, c0_full or c4_full}"
API_BASE="${API_BASE:-http://127.0.0.1:8000/v1}"
API_MODEL="${API_MODEL:?Set API_MODEL to the model name exposed by the serving endpoint}"
SEED="${SEED:-42}"
RESUME="${RESUME:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-toolvista-qwen3vl-eval}"
WANDB_GROUP="${WANDB_GROUP:-${STAGE}}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"
HISTORY_SUMMARY="${HISTORY_SUMMARY:-}"
EVAL_URL="${EVAL_URL:-${API_BASE}}"
JUDGE_API_BASE="${JUDGE_API_BASE:?Set JUDGE_API_BASE to the fixed C0-C4 judge endpoint}"
JUDGE_MODEL="${JUDGE_MODEL:?Set JUDGE_MODEL to the fixed C0-C4 judge model ID}"
LMUData="${LMUData:-/root/autodl-tmp/cache/vlmevalkit}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
TOOLVQA_PYTHON="${TOOLVQA_PYTHON:-/root/autodl-tmp/venvs/toolvqa/bin/python}"
BFCL_PYTHON="${BFCL_PYTHON:-/root/autodl-tmp/venvs/bfcl/bin/python}"
C3_ADAPTER_CHECKPOINT="${C3_ADAPTER_CHECKPOINT:-}"
C3_RETENTION_GATE="${C3_RETENTION_GATE:-}"

test -d "${TOOLVISTA_ROOT}"
test -e "${CHECKPOINT}"
test -x "${TOOLVQA_PYTHON}"
"${TOOLVQA_PYTHON}" -c "from lagent.agents import ReAct; from opencompass.datasets.gta_bench import GTABenchEvaluator; from opencompass.models.lagent import LagentAgentVisual"
test -x "${BFCL_PYTHON}"
"${BFCL_PYTHON}" -c "import bfcl_eval"
mkdir -p "${LMUData}"
export LMUData
export HF_HOME
export TOOLVQA_PYTHON
export BFCL_PYTHON

resume_args=()
if [[ "${RESUME}" == "true" ]]; then
  resume_args+=(--resume)
elif [[ "${RESUME}" != "false" ]]; then
  echo "RESUME must be true or false" >&2
  exit 2
fi

algorithm_args=()
if [[ -n "$ALGORITHM" ]]; then
  algorithm_args+=(--algorithm "$ALGORITHM")
fi

comparison_args=()
if [[ -n "${BASELINE_SUMMARY}" ]]; then
  test -s "${BASELINE_SUMMARY}"
  comparison_args+=(--baseline-summary "${BASELINE_SUMMARY}")
fi
if [[ -n "${HISTORY_SUMMARY}" ]]; then
  test -s "${HISTORY_SUMMARY}"
  comparison_args+=(--history-summary "${HISTORY_SUMMARY}")
fi

if [[ "${SUITE}" == "c3_light" ]]; then
  test -n "${HISTORY_SUMMARY}" || {
    echo "c3_light requires HISTORY_SUMMARY pointing to the C2 light summary." >&2
    exit 2
  }
  test -n "${C3_ADAPTER_CHECKPOINT}" || {
    echo "c3_light requires C3_ADAPTER_CHECKPOINT for the C4 authorization gate." >&2
    exit 2
  }
  test -d "${C3_ADAPTER_CHECKPOINT}"
  if [[ "$(realpath "${CHECKPOINT}")" != "$(realpath "${C3_ADAPTER_CHECKPOINT}")" ]]; then
    echo "c3_light CHECKPOINT must equal C3_ADAPTER_CHECKPOINT so the retention gate authorizes the evaluated adapter." >&2
    exit 2
  fi
  C3_RETENTION_GATE="${C3_RETENTION_GATE:-${C3_ADAPTER_CHECKPOINT}/toolvista_c3_retention_gate.json}"
  rm -f "${C3_RETENTION_GATE}"
fi

CHECK_ONLY=true \
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-/root/autodl-tmp/ms-swift}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${TOOLVISTA_ROOT}/scripts/autodl/patch_msswift_vlmeval.sh"

cd "${TOOLVISTA_ROOT}"
"${PYTHON_BIN}" scripts/evaluate.py \
  --suite "${SUITE}" \
  --checkpoint "${CHECKPOINT}" \
  --run-id "${RUN_ID}" \
  --stage "${STAGE}" \
  --api-base "${API_BASE}" \
  --api-model "${API_MODEL}" \
  --eval-url "${EVAL_URL}" \
  --judge-api-base "${JUDGE_API_BASE}" \
  --judge-model "${JUDGE_MODEL}" \
  --seed "${SEED}" \
  "${resume_args[@]}" \
  "${algorithm_args[@]}" \
  "${comparison_args[@]}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}"

if [[ "${SUITE}" == "c3_light" ]]; then
  "${PYTHON_BIN}" scripts/check_c3_retention.py \
    --candidate-summary "${TOOLVISTA_ROOT}/runs/${RUN_ID}/summary.json" \
    --baseline-summary "${HISTORY_SUMMARY}" \
    --authorized-checkpoint "${C3_ADAPTER_CHECKPOINT}" \
    --output "${C3_RETENTION_GATE}"
fi
